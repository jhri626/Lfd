import numpy as np
import torch
import torch.nn.functional as F
from agent.neural_network import NeuralNetwork
from agent.utils.ranking_losses import ContrastiveLoss, TripletLoss, TrajectoryLoss
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
        self.latent_space_dim = params.latent_space_dim
        self.encoder_only_epochs = params.encoder_only_epochs
        self.current_epoch = 0
        self.encoder_frozen = False
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
        self.triplet_loss_demo = TripletLoss(margin=1e-3, swap=False)
        self.triplet_loss_goal = TripletLoss(margin=params.triplet_margin, swap=False)
        self.contrastive_loss = ContrastiveLoss(margin=params.contrastive_margin)
        self.curvature_loss = TrajectoryLoss(eps=1e-6)

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
        
        self.encoder_params = []
        self.decoder_params = []
        
        # model의 파라미터를 encoder와 decoder로 분리
        for name, param in self.model.named_parameters():
            if 'encoder' in name.lower():  # encoder 관련 파라미터
                self.encoder_params.append(param)
            else:  # decoder 관련 파라미터
                self.decoder_params.append(param)
                
        self.encoder_optimizer = torch.optim.AdamW(self.encoder_params,
                                                   lr=params.learning_rate,
                                                   weight_decay=params.weight_decay)
        
        self.decoder_optimizer = torch.optim.AdamW(self.decoder_params,
                                                   lr=params.learning_rate,
                                                   weight_decay=params.weight_decay)
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=params.learning_rate,
                                           weight_decay=params.weight_decay)

        # Load Neural Network if requested
        if self.load_model:
            self.model.load_state_dict(torch.load(self.results_path + 'model'), strict=False)

        # Initialize latent goals
        self.model.update_goals_latent_space(self.goals_tensor)
    
    
    def freeze_encoder(self):
        """Encoder 파라미터를 freeze"""
        for param in self.encoder_params:
            param.requires_grad = False
        self.encoder_frozen = True
        print("Encoder frozen - only decoder will be trained")

    def unfreeze_encoder(self):
        """Encoder 파라미터를 unfreeze"""
        for param in self.encoder_params:
            param.requires_grad = True
        self.encoder_frozen = False
        print("Encoder unfrozen - both encoder and decoder will be trained")
    
    def update_epoch(self, epoch):
        """에폭 업데이트 및 학습 모드 전환"""
        self.current_epoch = epoch
        
        if epoch < self.encoder_only_epochs and self.encoder_frozen:
            # Encoder만 학습하는 단계
            self.unfreeze_encoder()
            # Decoder freeze
            for param in self.decoder_params:
                param.requires_grad = True
            print(f"Epoch {epoch}: Training both")
            
        elif epoch >= self.encoder_only_epochs and not self.encoder_frozen:
            # Encoder freeze하고 decoder만 학습하는 단계
            self.freeze_encoder()
            # Decoder unfreeze
            for param in self.decoder_params:
                param.requires_grad = True
            print(f"Epoch {epoch}: Training decoder only (encoder frozen)")
            
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
            # print("x_t_d", x_t_d.shape)
            # Compute and accumulate error
            imitation_error_accumulated += self.mse_loss(x_t_d[:, :self.dim_workspace], state_sample[:, :self.dim_workspace, i + 1].cuda())
            # imitation_error_accumulated += self.mse_loss(x_t_d[:, :], state_sample[:, :, i + 1].cuda())
        imitation_error_accumulated = imitation_error_accumulated / (self.imitation_window_size - 1)

        return imitation_error_accumulated * self.imitation_loss_weight

    def demo_cost(self, y_traj, primitive_type_sample):
        """
        demo cost - GPU optimized version
        """        
        
        y_goal = self.model.get_goals_latent_space_batch(primitive_type_sample)
        # print("y_traj", y_traj.shape)
        
        if y_traj.shape[0] <= 1:
            return torch.tensor(0.0, device=y_traj.device)
        
        batch_size = y_traj.shape[0] - 1
        
        # y_goal을 trajectory의 각 시간 스텝에 맞게 복제
        # y_goal: [n_primitives, latent_dim] -> [batch_size, latent_dim]
        if y_goal.dim() == 2:
            # 첫 번째 primitive type만 사용 (단순화)
            y_goal_expanded = y_goal[0].unsqueeze(0).expand(batch_size, -1).contiguous()
        else:
            y_goal_expanded = y_goal.expand(batch_size, -1).contiguous()
        
        # 연속된 trajectory 포인트들
        y_current = y_traj[1:]   # [batch_size, latent_dim]
        y_previous = y_traj[:-1] # [batch_size, latent_dim]


        # print("y_current", y_current.shape)
        # print("y_previous", y_previous.shape)

        # triplet loss 계산 (모든 차원이 [batch_size, latent_dim]로 동일)
        trip_loss = self.triplet_loss_demo(
            y_goal_expanded,  # anchor: [batch_size, latent_dim]
            y_current,        # positive: [batch_size, latent_dim]
            y_previous,        # negative: [batch_size, latent_dim]
        )
        
        return  1 * trip_loss 

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
        # print("start")
        for i in range (self.generalization_window_size):
            # Do transition
            y_t_task_prev = dynamical_system_task.y_t['task']
            y_t_task = dynamical_system_task.transition(space='task')['latent state']
            _, y_t_latent = dynamical_system_latent.transition_latent_system(y_traj=y_traj)
            
            # if i <5:
            #     print(i)
            #     print(y_t_task_prev[0])
            #     print(y_t_task[0])
            #     print(y_t_latent[0])

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
                    contrastive_matching_cost += self.triplet_loss_latent(y_t_task, y_t_latent, y_t_task_prev)
                elif self.stabilization_loss == 'triplet_goal':
                    latent_trip =  self.mse_loss(y_t_task, y_t_latent)
                    # latent_trip=0
                    y_goal = self.model.get_goals_latent_space_batch(primitive_type_sample)
                    
                    trip_loss= self.triplet_loss_goal(y_goal, y_t_task, y_t_task_prev)
                    contrastive_matching_cost += 0.5 * latent_trip + 0.1 * trip_loss
                    # contrastive_matching_cost += 0.5 * latent_trip 
                    # contrastive_matching_cost +=  0.1 * trip_loss
                    # print(trip_loss)
                else:
                    raise ValueError('Unknown stabilization loss type: ' + self.stabilization_loss)
                    
        contrastive_matching_cost = contrastive_matching_cost / (self.generalization_window_size - 1)
        # print("end")
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
    
    def linarization_loss(self,y_traj):
        
        
        loss = self.mse_loss(self.y_traj_latent, y_traj)
        # loss = self.curvature_loss(y_traj)

        return 1e-2 * loss
   
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
        
        # demo_cost = self.linarization_loss(y_traj)
        # loss_list.append(demo_cost)
        # losses_names.append('Demo')

        # Sum losses
        loss = 0
        for i in range(len(loss_list)):
            loss += loss_list[i]



        return loss, loss_list, losses_names

    def update_model(self, loss):
        """
        Updates Neural Network with computed cost
        """
        self.optimizer.zero_grad()
        loss.backward()
        
        max_grad_norm = 1e-3
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
        # print(y_traj[-1,:].cpu().detach().numpy())
        
        # Update model
        self.update_model(loss)

        return loss, loss_list, losses_names

    def compute_targets_from_traj_4d(
        self,
        traj: torch.Tensor,
        dim: int = 4,
        start_val: float = -0.75,
        end_val:   float = 0.75
    ) -> torch.Tensor:
        """
        Map a 4D trajectory to a 4D line segment using normalized cumulative distance.
        Parameters
        ----------
        self : any
            Unused class instance reference.
        traj : torch.Tensor, shape (N, 4)
            Original trajectory states, e.g. [x, y, dx, dy].
        start_val : float
            Value at the start of the target line (applied to all 4 dims).
        end_val : float
            Value at the end of the target line (applied to all 4 dims).
        Returns
        -------
        targets : torch.Tensor, shape (N, 4)
            Mapped points on the line from [start_val,...] to [end_val,...].
        """
        # 1. Compute 4D differences between consecutive points
        delta = traj[1:] - traj[:-1]               # shape (N-1, 4)

        # 2. Compute Euclidean distances in 4D
        d = torch.norm(delta, dim=1)               # shape (N-1,)

        # 3. Build cumulative distances and normalize to [0,1]
        s = torch.cat((torch.zeros(1, device=traj.device), torch.cumsum(d, dim=0)))  # shape (N,)
        u = s / s[-1]                              # shape (N,)

        # 4. Define 4D start and end vectors
        start = torch.full((1, dim), start_val, device=traj.device)
        end   = torch.full((1, dim), end_val,   device=traj.device)

        # 5. Interpolate along the 4D line for each normalized u
        targets = start + u.unsqueeze(1) * (end - start)  # shape (N, 4)

        return targets



    def traj_generator(self):
        
        mean_traj = np.mean(self.demonstrations_train, axis=0)
        mean_traj = torch.from_numpy(mean_traj).cuda()
        full_traj = mean_traj[...,0]
        next_traj = mean_traj[...,1]
        vel_traj = (next_traj - full_traj)/self.delta_t
        
        vel_traj = normalize_state(vel_traj,
                                            x_min=self.min_vel.reshape(1, self.dim_workspace),
                                            x_max=self.max_vel.reshape(1, self.dim_workspace))
        
        
        y_traj = torch.cat([full_traj , vel_traj],dim=1)
        y_traj = y_traj.to(dtype=next(self.model.parameters()).dtype, device=next(self.model.parameters()).device)
        
        self.y_traj_latent = self.compute_targets_from_traj_4d(y_traj,dim=self.latent_space_dim).cuda()
        self.demo_traj = y_traj


