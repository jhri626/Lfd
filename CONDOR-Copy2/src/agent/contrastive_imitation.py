import numpy as np
import torch
import torch.nn.functional as F
from agent.neural_network import NeuralNetwork
from agent.utils.ranking_losses import ContrastiveLoss, TripletLoss
from agent.dynamical_system import DynamicalSystem
from agent.utils.dynamical_system_operations import normalize_state


class ContrastiveImitation:
    """
    Computes CONDOR losses and optimizes Neural Network
    """
    def __init__(self, data, params):
        # Params file parameters
        self.dim_workspace = params.workspace_dimensions
        self.dynamical_system_order = params.dynamical_system_order
        self.dim_state = self.dim_workspace * self.dynamical_system_order
        self.imitation_window_size = params.imitation_window_size
        self.batch_size = params.batch_size
        self.save_path = params.results_path
        self.multi_motion = params.multi_motion
        self.stabilization_loss = params.stabilization_loss
        self.generalization_window_size = params.stabilization_window_size
        self.imitation_loss_weight = params.imitation_loss_weight
        self.stabilization_loss_weight = params.stabilization_loss_weight
        self.boundary_loss_weight = params.boundary_loss_weight
        self.load_model = params.load_model
        self.results_path = params.results_path
        self.interpolation_sigma = params.interpolation_sigma
        self.delta_t = params.delta_t  # used for training, can be anything
        print("contrastive time" , self.delta_t)

        # Parameters data processor
        self.primitive_ids = np.array(data['demonstrations primitive id']) 
        self.n_primitives = data['n primitives']
        self.goals_tensor = torch.FloatTensor(data['goals training']).cuda()
        self.demonstrations_train = data['demonstrations train']
        self.n_demonstrations = data['n demonstrations']
        self.demonstrations_length = data['demonstrations length']
        self.min_vel = torch.from_numpy(data['vel min train'].reshape([1, self.dim_workspace])).float().cuda()
        self.max_vel = torch.from_numpy(data['vel max train'].reshape([1, self.dim_workspace])).float().cuda()
        if data['acc min train'] is not None:
            min_acc = torch.from_numpy(data['acc min train'].reshape([1, self.dim_workspace])).float().cuda()
            max_acc = torch.from_numpy(data['acc max train'].reshape([1, self.dim_workspace])).float().cuda()
        else:
            min_acc = None
            max_acc = None

        # Dynamical-system-only params
        self.params_dynamical_system = {'saturate transition': params.saturate_out_of_boundaries_transitions,
                                        'x min': data['x min'],
                                        'x max': data['x max'],
                                        'vel min train': self.min_vel,
                                        'vel max train': self.max_vel,
                                        'acc min train': min_acc,
                                        'acc max train': max_acc}

        # Initialize Neural Network losses
        self.mse_loss = torch.nn.MSELoss()
        self.triplet_loss_latent = TripletLoss(margin=1e-4, swap=True)
        self.triplet_loss_goal = TripletLoss(margin=params.triplet_margin, swap=False)
        self.contrastive_loss = ContrastiveLoss(margin=params.contrastive_margin)

        # Initialize Neural Network
        self.model = NeuralNetwork(dim_state=self.dim_state,
                                   dynamical_system_order=self.dynamical_system_order,
                                   n_primitives=self.n_primitives,
                                   multi_motion=self.multi_motion,
                                   latent_gain_lower_limit=params.latent_gain_lower_limit,
                                   latent_gain_upper_limit=params.latent_gain_upper_limit,
                                   latent_gain=params.latent_gain,
                                   latent_space_dim=params.latent_space_dim,
                                   neurons_hidden_layers=params.neurons_hidden_layers,
                                   adaptive_gains=params.adaptive_gains).cuda()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=params.learning_rate,
                                           weight_decay=params.weight_decay)

        # Load Neural Network if requested
        if self.load_model:
            self.model.load_state_dict(torch.load(self.results_path + 'model'), strict=False)

        # Initialize latent goals
        self.model.update_goals_latent_space(self.goals_tensor)

    def init_dynamical_system(self, initial_states, primitive_type=None, delta_t=1):
        """
        Creates dynamical system using the parameters/variables of the learning policy
        """
        # If no primitive type, assume single-model learning
        if primitive_type is None:
            primitive_type = torch.FloatTensor([1])

        # Create dynamical system
        dynamical_system = DynamicalSystem(x_init=initial_states,
                                           model=self.model,
                                           primitive_type=primitive_type,
                                           order=self.dynamical_system_order,
                                           min_state_derivative=[self.params_dynamical_system['vel min train'],
                                                                 self.params_dynamical_system['acc min train']],
                                           max_state_derivative=[self.params_dynamical_system['vel max train'],
                                                                 self.params_dynamical_system['acc max train']],
                                           saturate_transition=self.params_dynamical_system['saturate transition'],
                                           dim_state=self.dim_state,
                                           delta_t=delta_t,
                                           x_min=self.params_dynamical_system['x min'],
                                           x_max=self.params_dynamical_system['x max'])

        return dynamical_system

    def imitation_cost(self, state_sample, primitive_type_sample):
        """
        Imitation cost
        """
        # Create dynamical system
        dynamical_system = self.init_dynamical_system(initial_states=state_sample[:, :, 0],
                                                      primitive_type=primitive_type_sample,delta_t=self.delta_t)

        # Compute imitation error for transition window
        imitation_error_accumulated = 0

        for i in range(self.imitation_window_size - 1):
            # Compute transition
            x_t_d = dynamical_system.transition(space='task')['desired state']

            # Compute and accumulate error
            imitation_error_accumulated += self.mse_loss(x_t_d[:, :self.dim_workspace], state_sample[:, :self.dim_workspace, i + 1].cuda())
        imitation_error_accumulated = imitation_error_accumulated / (self.imitation_window_size - 1)

        return imitation_error_accumulated * self.imitation_loss_weight

    def contrastive_matching(self, y_traj, state_sample, primitive_type_sample):
        """
        Transition matching cost
        """
        # Create dynamical systems
        dynamical_system_task = self.init_dynamical_system(initial_states=state_sample,
                                                           primitive_type=primitive_type_sample,delta_t=self.delta_t)

        dynamical_system_latent = self.init_dynamical_system(initial_states=state_sample,
                                                             primitive_type=primitive_type_sample,delta_t=self.delta_t)

        # Compute cost over trajectory
        contrastive_matching_cost = 0
        batch_size = state_sample.shape[0]

        for i in range(self.generalization_window_size):
            # Do transition
            y_t_task_prev = dynamical_system_task.y_t['task']
            y_t_task = dynamical_system_task.transition(space='task')['latent state']
            _, y_t_latent = dynamical_system_latent.transition_latent_system(y_traj=y_traj)

            if i > 0:  # we need at least one iteration to have a previous point to push the current one away from
                # Transition matching cost
                if self.stabilization_loss == 'contrastive':
                    # Anchor
                    anchor_samples = torch.cat((y_t_task, y_t_task))

                    # Positive/Negative samples
                    contrastive_samples = torch.cat((y_t_latent, y_t_task_prev))

                    # Contrastive label
                    contrastive_label_pos = torch.ones(batch_size).cuda()
                    contrastive_label_neg = torch.zeros(batch_size).cuda()
                    contrastive_label = torch.cat((contrastive_label_pos, contrastive_label_neg))

                    # Compute cost
                    contrastive_matching_cost += self.contrastive_loss(anchor_samples, contrastive_samples, contrastive_label)

                elif self.stabilization_loss == 'triplet':
                    contrastive_matching_cost += self.triplet_loss(y_t_task, y_t_latent, y_t_task_prev)
                elif self.stabilization_loss == 'triplet_goal':
                    # latent_trip = 0.3 * self.mse_loss(y_t_task, y_t_latent)
                    latent_trip=0
                    y_goal = self.model.get_goals_latent_space_batch(primitive_type_sample)
                    trip_loss = self.triplet_loss_goal(y_goal, y_t_task, y_t_task_prev)
                    # trip_loss=0
    
                    contrastive_matching_cost += latent_trip + 0.1 * trip_loss
                    # print(trip_loss)
                    
        contrastive_matching_cost = contrastive_matching_cost / (self.generalization_window_size - 1)

        return contrastive_matching_cost * self.stabilization_loss_weight

    

    def boundary_constrain_loss(self, state_sample, primitive_type_sample):
        """
        Forces dynamical system to respect workspace boundaries
        Adapted from PUMA to CONDOR
        """
        # Force states to start at the boundary
        selected_axis = torch.randint(low=0, high=self.dim_state, size=[self.batch_size])
        selected_limit = torch.randint(low=0, high=2, size=[self.batch_size])
        limit_options = torch.FloatTensor([-1, 1])
        limits = limit_options[selected_limit]
        replaced_samples = torch.arange(start=0, end=self.batch_size)
        state_sample[replaced_samples, selected_axis] = limits.cuda()
    
        # Create dynamical systems with saturation disabled
        params_dynamical_system_backup = self.params_dynamical_system['saturate transition']
        self.params_dynamical_system['saturate transition'] = False
        dynamical_system = self.init_dynamical_system(initial_states=state_sample,
                                                     primitive_type=primitive_type_sample,
                                                     delta_t=self.delta_t)
        self.params_dynamical_system['saturate transition'] = params_dynamical_system_backup
    
        # Do one transition at the boundary and get velocity
        transition_info = dynamical_system.transition(space='task')
        x_t_d = transition_info['desired state']
        dx_t_d = transition_info['desired velocity']
    
        # Iterate through every dimension
        epsilon = 5e-2
        loss = 0
        states_boundary = self.dim_workspace  # CONDOR uses workspace dimensions
    
        for i in range(states_boundary):
            distance_upper = torch.abs(x_t_d[:, i] - 1)
            distance_lower = torch.abs(x_t_d[:, i] + 1)
    
            # Get velocities for points in the boundary
            dx_axis_upper = dx_t_d[distance_upper < epsilon]
            dx_axis_lower = dx_t_d[distance_lower < epsilon]
    
            # Compute normal vectors for lower and upper limits
            normal_upper = torch.zeros(dx_axis_upper.shape).cuda()
            normal_upper[:, i] = 1
            normal_lower = torch.zeros(dx_axis_lower.shape).cuda()
            normal_lower[:, i] = -1
    
            # Compute dot product between boundary velocities and normal vectors
            dot_product_upper = torch.bmm(dx_axis_upper.view(-1, 1, self.dim_workspace),
                                         normal_upper.view(-1, self.dim_workspace, 1)).reshape(-1)
    
            dot_product_lower = torch.bmm(dx_axis_lower.view(-1, 1, self.dim_workspace),
                                         normal_lower.view(-1, self.dim_workspace, 1)).reshape(-1)
    
            # Concat with zero in case no points sampled in boundaries, to avoid nans
            dot_product_upper = torch.cat([dot_product_upper, torch.zeros(1).cuda()])
            dot_product_lower = torch.cat([dot_product_lower, torch.zeros(1).cuda()])
    
            # Compute losses - penalize velocities pointing outward
            loss += F.relu(dot_product_upper).mean()
            loss += F.relu(dot_product_lower).mean()
    
        loss = loss / (2 * self.dim_workspace)
    
        return loss * self.boundary_loss_weight  # 파라미터 추가 필요
   
    def demo_sample(self):
        """
        Samples a batch of windows from the demonstrations
        """

        # Select demonstrations randomly
        selected_demos = np.random.choice(range(self.n_demonstrations), self.batch_size)

        # Get random points inside trajectories
        i_samples = []
        for i in range(self.n_demonstrations):
            selected_demo_batch_size = sum(selected_demos == i)
            demonstration_length = self.demonstrations_train.shape[1]
            i_samples = i_samples + list(np.random.randint(0, demonstration_length, selected_demo_batch_size, dtype=int))

        # Get sampled positions from training data
        position_sample = self.demonstrations_train[selected_demos, i_samples]
        position_sample = torch.FloatTensor(position_sample).cuda()

        # Create empty state
        state_sample = torch.empty([self.batch_size, self.dim_state, self.imitation_window_size]).cuda()

        # Fill first elements of the state with position
        state_sample[:, :self.dim_workspace, :] = position_sample[:, :, (self.dynamical_system_order - 1):]

        # Fill rest of the elements with velocities for second order systems
        if self.dynamical_system_order == 2:
            velocity = (position_sample[:, :, 1:] - position_sample[:, :, :-1]) / self.delta_t
            velocity_norm = normalize_state(velocity,
                                            x_min=self.min_vel.reshape(1, self.dim_workspace, 1),
                                            x_max=self.max_vel.reshape(1, self.dim_workspace, 1))
            state_sample[:, self.dim_workspace:, :] = velocity_norm

        # Finally, get primitive ids of sampled batch (necessary when multi-motion learning)
        primitive_type_sample = self.primitive_ids[selected_demos]
        primitive_type_sample = torch.FloatTensor(primitive_type_sample).cuda()

        return state_sample, primitive_type_sample

    def curvature_regulation_loss(self,y_traj):
        traj_len = y_traj.shape[0]
        y_traj_0 = y_traj[:-2,:]
        y_traj_1 = y_traj[1:-1,:]
        y_traj_2 = y_traj[2:,:]

        ddy = (y_traj_2 - 2* y_traj_1 + y_traj_0).pow(2).sum()

        return 1e-1 * ddy
            
    def space_sample(self):
        """
        Samples a batch of windows from the state space
        """
        with torch.no_grad():
            # Sample state
            state_sample_gen = torch.Tensor(self.batch_size, self.dim_state).uniform_(-1, 1).cuda()

            # Choose sampling methods
            if not self.multi_motion:
                primitive_type_sample_gen = torch.randint(0, self.n_primitives, (self.batch_size,)).cuda()
            else:
                # If multi-motion learning also sample in interpolation space
                # sigma of the samples are in the demonstration spaces
                encodings = torch.eye(self.n_primitives).cuda()
                primitive_type_sample_gen_demo = encodings[torch.randint(0, self.n_primitives, (round(self.batch_size * self.interpolation_sigma),)).cuda()]

                # 1 - sigma  of the samples are in the interpolation space
                primitive_type_sample_gen_inter = torch.rand(round(self.batch_size * (1 - self.interpolation_sigma)), self.n_primitives).cuda()

                # Concatenate both samples
                primitive_type_sample_gen = torch.cat((primitive_type_sample_gen_demo, primitive_type_sample_gen_inter), dim=0)

        return state_sample_gen, primitive_type_sample_gen

    def compute_loss(self, state_sample_IL, primitive_type_sample_IL, state_sample_gen, primitive_type_sample_gen, y_traj):
        """
        Computes total cost
        """
        loss_list = []  # list of losses
        losses_names = []

        # Learning from demonstrations outer loop
        if self.imitation_loss_weight != 0:
            imitation_cost = self.imitation_cost(state_sample_IL, primitive_type_sample_IL)
            loss_list.append(imitation_cost)
            losses_names.append('Imitation')

        # Transition matching
        if self.stabilization_loss_weight != 0:
            contrastive_matching_cost = self.contrastive_matching(y_traj,state_sample_gen, primitive_type_sample_gen)
            loss_list.append(contrastive_matching_cost)
            losses_names.append('Stability')
       
        if self.boundary_loss_weight != 0:
            state_sample_gen_bound = torch.clone(state_sample_gen)
            boundary_cost = self.boundary_constrain_loss(state_sample_gen_bound, primitive_type_sample_gen)
            loss_list.append(boundary_cost)
            losses_names.append('Boundary')

        # Sum losses
        loss = 0
        for i in range(len(loss_list)):
            loss += loss_list[i]

        # loss += self.curvature_regulation_loss(y_traj)

        return loss, loss_list, losses_names

    def update_model(self, loss):
        """
        Updates Neural Network with computed cost
        """
        self.optimizer.zero_grad()
        loss.backward()
        
        max_grad_norm = 0.01  # Set as needed
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        self.optimizer.step()

        # Update goal in latent space
        self.model.update_goals_latent_space(self.goals_tensor)

    def train_step(self):
        """
        Samples data and trains Neural Network
        """
        # Sample from space
        state_sample_gen, primitive_type_sample_gen = self.space_sample()

        # Sample from trajectory
        state_sample_IL, primitive_type_sample_IL = self.demo_sample()

        y_traj = self.model.encoder(self.demo_traj,self.primitive_ids)
        
        # Get loss from CONDOR
        loss, loss_list, losses_names = self.compute_loss(state_sample_IL,
                                                          primitive_type_sample_IL,
                                                          state_sample_gen,
                                                          primitive_type_sample_gen,
                                                          y_traj
                                                          )

        # Update model
        self.update_model(loss)

        return loss, loss_list, losses_names

    def traj_generator(self):
        
        mean_traj = np.mean(self.demonstrations_train, axis=0)
        full_traj = mean_traj[...,0]
        next_traj = mean_traj[...,1]
        vel_traj = (next_traj - full_traj)/self.delta_t
        y_traj = torch.cat([torch.from_numpy(full_traj) , torch.from_numpy(vel_traj)],dim=1)
        y_traj = y_traj.to(dtype=next(self.model.parameters()).dtype, device=next(self.model.parameters()).device)
        
        self.demo_traj = y_traj

        
