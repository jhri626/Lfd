import importlib
import numpy as np
import torch
from simple_parsing import ArgumentParser
from data_preprocessing.data_preprocessor import DataPreprocessor
from agent.utils.dynamical_system_operations import normalize_state , denormalize_state
from state_dynamic import StateDynamic
import os
import time
from agent.utils.utils import *
from agent.utils.distance import riemann_anisotropic_distance
from agent.utils.R2_to_S2 import map_R2_curve_to_S2_q_torch, qtraj_to_qdot_forward_consistent
from agent.utils.plot_utils import *
from agent.utils.S2_functions import xdot_to_qdot, q_to_x
from initializer import initialize_framework
from fastdtw import fastdtw

def get_workspace_bounds(demonstrations, params):
    """
    Get workspace bounds from demonstration data
    """
    all_positions = []
    for demo in demonstrations:
        positions = demo[:params.workspace_dimensions, :]
        all_positions.append(positions.T)  # Transpose to get [time_steps, workspace_dim]
    
    all_positions = np.vstack(all_positions)  # Combine all demonstrations
    
    min_bounds = np.min(all_positions, axis=0)
    max_bounds = np.max(all_positions, axis=0)
    
    # Add some margin
    margin = 0.1 * (max_bounds - min_bounds)
    min_bounds -= margin
    max_bounds += margin
    
    return min_bounds, max_bounds

def cal_lyapunov_exponent(distance_traj, terminal_time=1, eps=1e-6, delta_t=0.005):
    '''
    From BCSDM curve_analysis.py
    '''
    # print("terminal_time",terminal_time)
    # input = ((nd), nt)
    input_size = distance_traj.shape
    len_traj = input_size[-1]
    if len(input_size) == 2:
        multitraj=True
    else:
        multitraj=False
        ## added ##
        distance_traj = distance_traj.unsqueeze(0)
    
    ## original ##
    # t_linspace = torch.linspace(0, terminal_time, len_traj).to(distance_traj)
    # if multitraj:
    #     t_linspace = t_linspace.unsqueeze(0).repeat(input_size[0], 1).to(distance_traj)
    # log_dist_traj = torch.log(distance_traj).unsqueeze(-1) # ((nd), nt, 1)
    # log_d0 = log_dist_traj[..., 0:1, 0:1]
    # lamb = (torch.pinverse(t_linspace.unsqueeze(-1)) 
    #         @ (log_d0 - log_dist_traj)).squeeze().squeeze()
    
    ## fixed to truncate distance traj ##
    lamb_list = []
    for i in range(len(distance_traj)):
        for j in range(len(distance_traj[i])):
            if distance_traj[i][j] < eps:
                distance_truncated = distance_traj[i][:j+1]
                # print(j)
                # print(distance_truncated)
                break
            else:
                distance_truncated = distance_traj[i]

        
        if len(distance_truncated) < 2:
            lamb_list.append(float('nan'))
            continue
        # if j < 100:
        #     lamb_list.append(float('nan')) # to short traj could make wrong lyapunov
        #     continue 
        

        terminal_time_traj = j * terminal_time * delta_t
        terminal_time_traj = torch.as_tensor(terminal_time_traj, dtype=torch.float32)
        # print(i,terminal_time_traj)
        t_linspace = torch.linspace(0, terminal_time_traj, len(distance_truncated)).to(distance_traj)
        log_dist_traj = torch.log(distance_truncated).unsqueeze(-1) # ((nd), nt, 1)
        log_d0 = log_dist_traj[..., 0:1, 0:1]
        lamb = (torch.pinverse(t_linspace.unsqueeze(-1)) 
                @ (log_d0 - log_dist_traj)).squeeze().squeeze()
        lamb_list.append(lamb)
        # print(f"{i}th traj, lyapunov: {lamb:.4e}")
    lamb = torch.tensor(lamb_list)
    
    # shape: (nd, ) or ()
    return lamb


def main(num = int):
    # Get arguments
    # TODO we need more baseline (SEDS, euclidean flow)
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, default='2nd_order_2D', help='Parameter file name')
    parser.add_argument('--results-base-directory', type=str, default='./', help='Base directory for results')
    parser.add_argument('--simulation-steps', type=int, default=5500, help='Number of simulation steps')
    parser.add_argument('--sample', type=int, default=25, help='Number of random initial states to generate')
    parser.add_argument('--use-model', type=str, default='false', choices=['true', 'false'], 
                        help='Whether to use learned model for simulation (true/false)')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=== State Dynamics Simulation ===")
    print(f"Using parameters: {args.params}")
    print(f"Number of random initial states: {args.sample}")
    
    # Import parameters
    Params = getattr(importlib.import_module('params.' + args.params), 'Params')
    params = Params(args.results_base_directory)
    params.selected_primitives_ids = str(num)
    
    # Set load_model flag if using model
    if params.name in ["PUMA","CONDOR"]:
        params.load_model = True
        print("Using learned model for simulation")
    
    
    if params.name in ["PUMA","CONDOR"]:
        params.save_path = 'exp_result/state_dynamics/'+ params.space +'/baseline/' + params.name + '/' + params.selected_primitives_ids + '/'
        params.delta_t = 1
    else:
        # params.save_path = 'exp_result/state_dynamics/'+ params.space +'/test'+'/eta' + str(params.eta) +'/'+ params.selected_primitives_ids + '/'
        # params.save_path = 'exp_result/state_dynamics/'+ params.space +'/new_metric'+'/eta' + str(params.eta) +'/'+ params.selected_primitives_ids + '/'
        params.save_path = 'exp_result/state_dynamics/'+ params.space +'/dir_test/'+str(params.trajectories_resample_length) + '/dir'+'/eta' + str(params.eta) +'/'+ params.selected_primitives_ids + '/'
        # params.save_path = 'exp_result/state_dynamics/'+ params.space +'/1st_order'+'/eta' + str(params.eta) +'/'+ params.selected_primitives_ids + '/'
    
    # Create results directory
    
    create_directories(params.save_path)
    
    print(f"Dataset: {params.dataset_name}")
    print(f"Selected primitives: {params.selected_primitives_ids}")
    print(f"Workspace dimensions: {params.workspace_dimensions}")
    print(f"Dynamical system order: {params.dynamical_system_order}")
    print(f"Results will be saved to: {params.save_path}")
    
    # Load and preprocess data using existing framework
    print("\nLoading and preprocessing data...")
    data_preprocessor = DataPreprocessor(params=params, verbose=True)
    data = data_preprocessor.run()
    
    # Extract relevant data
    # demonstrations = data['demonstrations raw']
    demonstrations_norm = data['demonstrations train'][:,:,:,0]
    time_scale_factor = data['demonstrations raw'][0].shape[1]
    demonstrations_norm = lpf_demonstrations(demonstrations_norm,
                                        fs=1.0,   # if samples are unit-spaced
                                        L_min=30, # remove variations shorter than 30 samples
                                        order=4)

    
    x_min, x_max = data['x min'], data['x max']
    vel_min, vel_max = data['vel min train'], data['vel max train']
    demonstrations_denorm = denormalize_state(demonstrations_norm, x_min, x_max)

    # demonstrations = np.array(demonstrations)  # Ensure it's a numpy array
    
    # demonstrations = np.transpose(demonstrations, (0, 2, 1))  # (B, D, T)

    
    # Analyze demonstrations
    analyze_demonstrations(demonstrations_denorm, params)
    
    # Plot demonstrations only first to check data loading
    demo_plot_path = params.save_path + 'images/demonstrations_only.png'
    plot_demonstrations_only(demonstrations_denorm, params, demo_plot_path)
    
    print(f"Loaded {len(demonstrations_denorm)} demonstrations")

    if params.space == "sphere":
        demonstrations_denorm = torch.from_numpy(demonstrations_denorm).float()
        q_traj, extras = map_R2_curve_to_S2_q_torch(
            demonstrations_norm,
            coverage=0.25,
            center=(0.0, 0.0),
            align_front=True,
            front_target=torch.tensor([1.2, -0.8, 0.8]),
            R_user=None,
            return_intermediates=True
        )
        
        q_traj, x_traj = lpf_then_rematch_q(
            q_traj, fs=1/params.delta_t, fc=3.0, numtaps=101, refine_mean=True
        )
        
        # Save x_traj as npy file
        if True:
            x_data_path = 'data/' + params.dataset_name + '/' + params.selected_primitives_ids + '/x_traj/'
            x_traj_save_path = x_data_path + 'stats/x_traj.npy'
            os.makedirs(os.path.dirname(x_traj_save_path), exist_ok=True)
            np.save(x_traj_save_path, x_traj.numpy() if isinstance(x_traj, torch.Tensor) else x_traj)
            print(f"x_traj saved to: {x_traj_save_path}")
            
            # Save q_traj as npy file as well
            q_data_path = 'data/' + params.dataset_name + '/' + params.selected_primitives_ids + '/q_traj/'
            q_traj_save_path = q_data_path + 'stats/q_traj.npy'
            os.makedirs(os.path.dirname(q_traj_save_path), exist_ok=True)
            np.save(q_traj_save_path, q_traj.numpy() if isinstance(q_traj, torch.Tensor) else q_traj)
            print(f"q_traj saved to: {q_traj_save_path}")
        
    


    if params.space == "euclidean":
        if params.dynamical_system_order == 2:
            print("\nNote: Dynamical system order is set to 2, using second-order dynamics.")
            velocity = (demonstrations_norm[:, 1:, :] - demonstrations_norm[:, :-1, :])/ params.delta_t
            dummy_vel = 1
            
            # while np.linalg.norm(velocity[:,0,:]) < 1e-5:
            #     velocity[:,0,:] = (demonstrations_norm[:, dummy_vel, :] - demonstrations_norm[:, 0, :])/ dummy_vel*params.delta_t
            #     dummy_vel += 1
            
            # vel_start = (demonstrations_norm[:, 0, :] - demonstrations_norm[:, 1, :])/ params.delta_t
            # vel_start = np.expand_dims(vel_start, axis=1)  # Add as first point
            velocity = np.concatenate((velocity ,np.zeros((velocity.shape[0], 1, params.workspace_dimensions))), axis=1)  # Add zero velocity for last point
            # print(velocity[0,:10,:])
            # print(demonstrations_norm[0,:10,:])
            
            # velocity = np.concatenate((vel_start,velocity ,np.zeros((velocity.shape[0], 1, params.workspace_dimensions))), axis=1)  # Add zero velocity for last point
            traj = np.concatenate((demonstrations_norm, velocity), axis=2)  # Combine position and velocity
        else:
            print("\nNote: Dynamical system order is set to 1, using first-order dynamics.")
            traj = demonstrations_norm
    elif params.space == "sphere":
        q_dot = qtraj_to_qdot_forward_consistent(q_traj, params.delta_t)
        traj = torch.cat([q_traj, q_dot], dim=-1)  # Combine position and velocity
        
                
    
    # Initialize state dynamics
    print("\nInitializing state dynamics...")
    init_state_dynamics = StateDynamic(params ,traj,vel_norm_stat=[vel_min, vel_max], sample=args.sample, max_steps=args.simulation_steps)
    # grid_state_dynamics = StateDynamic(params ,traj,vel_norm_stat=[vel_min, vel_max], sample=25, max_steps=args.simulation_steps)
    # all_state_dynamics = StateDynamic(params ,traj,vel_norm_stat=[vel_min, vel_max], sample=1, max_steps=args.simulation_steps)

    # Generate random initial states
    
    print(f"Generating {args.sample} random initial states...")
    
    if params.space == "euclidean":
        random_initial_states = generate_random_initial_states(args.sample, params, demonstrations_norm, mode='init', sampling_std=0.1)
        grid_initial_states = generate_random_initial_states(25, params, demonstrations_norm, mode='grid')
        all_initial_states = generate_random_initial_states(1, params, demonstrations_norm, mode='all')
    elif params.space == "sphere":
        random_initial_states = generate_random_initial_states_s2(args.sample, params, q_traj, mode='init')
        # grid_initial_states = generate_random_initial_states_s2(25, params, q_traj, mode='grid')
        # all_initial_states = generate_random_initial_states_s2(50, params, q_traj, mode='all')


    # print("init_state",random_initial_states.shape,grid_initial_states.shape, all_initial_states.shape)
    
    # Save initial points as npy files
    random_initial_save_path = params.save_path + 'stats/random_initial_states.npy'
    # grid_initial_save_path = params.save_path + 'stats/grid_initial_states.npy'
    # all_initial_save_path = params.save_path + 'stats/all_initial_states.npy'
    
    np.save(random_initial_save_path, random_initial_states)
    # np.save(grid_initial_save_path, grid_initial_states)
    # np.save(all_initial_save_path, all_initial_states)
    
    print(f"Random initial states saved to: {random_initial_save_path}")
    # print(f"Grid initial states saved to: {grid_initial_save_path}")
    # print(f"All initial states saved to: {all_initial_save_path}")

    # Run initial point simulation
    print(f"\Goal position: {init_state_dynamics.goal[:params.workspace_dimensions]}")
    print(f"\nRunning initial point simulation for {args.simulation_steps} steps...")
    
    start_time = time.time()
    
    # Check if we should use the learned model for simulation
    if params.name in ["PUMA","CONDOR"]:
        print("Using learned model for trajectory simulation...")
        
        # Initialize framework to get the learner
        params.results_path += params.selected_primitives_ids + '/'
        print("Loading model from:", params.results_path)
        if params.multi:
            params.dataset_name = 'LAIR'
            params.save_path = 'exp_result/state_dynamics/'+ params.space +'/baseline/' + params.name + '/original/' + params.selected_primitives_ids + '/'
            create_directories(params.save_path)
        learner, _, model_data = initialize_framework(params, args.params, verbose=False)
        
        # Prepare initial states for model simulation
        x_t_init_random = random_initial_states.copy()
        x_t_init_grid = grid_initial_states.copy()
        x_t_init_all = all_initial_states.copy()
        
        # Function to simulate with model
        def simulate_with_model(initial_states, simulation_steps):
            # Initialize dynamical system with the learned model
            dynamical_system = learner.init_dynamical_system(
                initial_states=torch.FloatTensor(initial_states).cuda()
            )
            # print(initial_states[0])
            # Simulate trajectories
            simulated_trajectories = [initial_states]
            for i in range(simulation_steps):
                # Do transition
                x_t = dynamical_system.transition(space='task')['desired state']
                simulated_trajectories.append(x_t.cpu().detach().numpy())
                # if i < 20:
                #     print(x_t[0])
            print("Simulation with model completed.")
            
            # Convert to numpy array: shape (time_steps, n_trajectories, state_dim)
            simulated_trajectories = np.array(simulated_trajectories)
            # Transpose to match expected format: (n_trajectories, time_steps, state_dim)
            return simulated_trajectories.transpose(1, 0, 2)
        
        # Simulate with model
        batch_trajectories = simulate_with_model(x_t_init_random, args.simulation_steps -1)
        grid_trajectories = simulate_with_model(x_t_init_grid, args.simulation_steps -1)
        all_trajectories = simulate_with_model(x_t_init_all, args.simulation_steps -1)

        for traj_sample in [batch_trajectories, grid_trajectories, all_trajectories]:  
            vel = traj_sample[:,1:,:params.workspace_dimensions] - traj_sample[:,:-1,:params.workspace_dimensions]
            vel_scaled = vel / 0.005
            zeros = np.zeros((vel.shape[0], 1, vel.shape[2]))
            vel = np.concatenate((vel_scaled, zeros), axis=1)            
            traj_sample[:,:,params.workspace_dimensions:] = vel
    else:
        # Use existing StateDynamic simulation
        batch_trajectories = init_state_dynamics.simulate(random_initial_states)
        # quit()
        print(f"\nRunning grid point simulation for {args.simulation_steps} steps...")
        # grid_trajectories = grid_state_dynamics.simulate(grid_initial_states)
        
        print(f"\nRunning all point simulation for {args.simulation_steps} steps...")
        # all_trajectories = all_state_dynamics.simulate(all_initial_states)
    
    simulation_time = time.time() - start_time
    
    print(f"Batch simulation completed in {simulation_time:.2f} seconds")
    print(f"Batch trajectories shape: {batch_trajectories.shape}")
    
    # Convert batch trajectories to list of individual trajectories for easier handling
    individual_trajectories = [batch_trajectories[i] for i in range(len(batch_trajectories))]
    individual_trajectories = np.array(individual_trajectories)  # Convert to numpy array if needed
    
    # Convert grid trajectories to list of individual trajectories
    # individual_grid_trajectories = [grid_trajectories[i] for i in range(len(grid_trajectories))]
    # individual_grid_trajectories = np.array(individual_grid_trajectories)  # Convert to numpy array if needed

    # individual_all_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
    # individual_all_trajectories = np.array(individual_all_trajectories)  # Convert to numpy array if needed

    # Analyze batch results
    if params.name not in ["PUMA","CONDOR"]:
        # Only analyze with StateDynamic objects when not using model
        analyze_batch_trajectories(individual_trajectories, params, init_state_dynamics)
        # analyze_batch_trajectories(individual_grid_trajectories, params, grid_state_dynamics)
        # analyze_batch_trajectories(individual_all_trajectories, params, all_state_dynamics)

        init_dist = torch.from_numpy(init_state_dynamics.get_dist_history())
        # grid_dist = torch.from_numpy(grid_state_dynamics.get_dist_history())
        # all_dist = torch.from_numpy(all_state_dynamics.get_dist_history())
        # print("init_dist",init_dist.shape)
        # Save distance histories as npy files
        suffix = "_model" if args.use_model.lower() == 'true' else ""
        init_dist_save_path = params.save_path + f'stats/init_dist_history{suffix}.npy'
        # grid_dist_save_path = params.save_path + f'stats/grid_dist_history{suffix}.npy'
        # all_dist_save_path = params.save_path + f'stats/all_dist_history{suffix}.npy'
        
        np.save(init_dist_save_path, init_dist.numpy())
        # np.save(grid_dist_save_path, grid_dist.numpy())
        # np.save(all_dist_save_path, all_dist.numpy())
        
        # print(init_dist)
        
        print(f"Init distance history saved to: {init_dist_save_path}")
        # print(f"Grid distance history saved to: {grid_dist_save_path}")
        # print(f"All distance history saved to: {all_dist_save_path}")
    else:
        # print("Model simulation: Distance history analysis skipped (not available with learned model)")
        # Create dummy distance tensors for plotting compatibility
        if params.dist == True:
            init_dist = torch.zeros(args.sample, args.simulation_steps)
            # grid_dist = torch.zeros(25, args.simulation_steps)
            # all_dist = torch.zeros(100, args.simulation_steps)
            traj_expanded_init = np.tile(traj, (args.sample, 1, 1))  # (B, T, D)
            # traj_expanded_grid = np.tile(traj, (25, 1, 1))  # (B, T, D)
            # traj_expanded_all = np.tile(traj, (100, 1, 1))  # (B, T, D)

            for t in range(individual_trajectories.shape[1]):
                point = individual_trajectories[:,t,:]
                # print(individual_trajectories.shape, point.shape)
                point_expanded = np.tile(point[:, None, :], (1, traj.shape[1], 1))
                # print(traj.shape[1])
                # print(point_expanded.shape,traj_expanded.shape)
                dist, _ , _ = riemann_anisotropic_distance(traj_expanded_init, point_expanded)
                # print(dist.shape)
                dist_min_idx = np.argmin(dist, axis=1)
                dist_min = np.linalg.norm(traj_expanded_init[np.arange(dist.shape[0]), dist_min_idx][:,:2] - point[:,:2], axis=1)
                init_dist[:,t] = torch.from_numpy(dist_min)
                # print(point,dist_min_idx,dist_min)
            init_dist = init_dist.T
            
            # for t in range(individual_grid_trajectories.shape[1]):
            #     point = individual_grid_trajectories[:,t,:]
            #     point_expanded = np.tile(point[:, None, :], (1, traj.shape[1], 1))
            #     dist, _ , _ = riemann_anisotropic_distance(traj_expanded_grid, point_expanded)
            #     dist_min_idx = np.argmin(dist, axis=1)
            #     dist_min = np.linalg.norm(traj_expanded_grid[np.arange(dist.shape[0]), dist_min_idx][:,:2] - point[:,:2], axis=1)
            #     grid_dist[:,t] = torch.from_numpy(dist_min)
            # grid_dist = grid_dist.T
            
            # for t in range(individual_all_trajectories.shape[1]):
            #     point = individual_all_trajectories[:,t,:]
            #     point_expanded = np.tile(point[:, None, :], (1, traj.shape[1], 1))
            #     dist, _ , _ = riemann_anisotropic_distance(traj_expanded_all, point_expanded)
            #     dist_min_idx = np.argmin(dist, axis=1)
            #     dist_min = np.linalg.norm(traj_expanded_all[np.arange(dist.shape[0]), dist_min_idx][:,:2] - point[:,:2], axis=1)
            #     all_dist[:,t] = torch.from_numpy(dist_min)
            # all_dist = all_dist.T
            
            suffix = "_model" if args.use_model.lower() == 'true' else ""
            init_dist_save_path = params.save_path + f'stats/init_dist_history{suffix}.npy'
            # grid_dist_save_path = params.save_path + f'stats/grid_dist_history{suffix}.npy'
            # all_dist_save_path = params.save_path + f'stats/all_dist_history{suffix}.npy'
            
            np.save(init_dist_save_path, init_dist.numpy())
            # np.save(grid_dist_save_path, grid_dist.numpy())
            # np.save(all_dist_save_path, all_dist.numpy())
        else:
            init_dist = None
            grid_dist = None
            all_dist = None
        
    
    if params.name in ["PUMA","CONDOR"]:
        delta_t = 10 / time_scale_factor
    else:
        delta_t = params.delta_t
    
    if init_dist is not None:
        init_lyapunov = cal_lyapunov_exponent(init_dist.T,eps=1e-2,delta_t=delta_t)
    else:
        init_lyapunov = None
    
    # grid_lyapunov = cal_lyapunov_exponent(grid_dist.T,eps=1e-2)
    # if all_dist is not None:
    #     all_lyapunov = cal_lyapunov_exponent(all_dist.T,eps=1e-2,delta_t=delta_t)
    # else:    
        # all_lyapunov = None

    # print("\nLyapunov Exponents:")
    # print(init_dist.shape)
    # print(init_lyapunov.mean())
    # print(init_dist.min())

    # Generate plots
    print("\nGenerating plots...")
    
    # Batch 2D trajectory plot
    if params.workspace_dimensions == 2:
        batch_plot_path_2d_init = params.save_path + 'images/batch_trajectories_2d_init.png'
        batch_plot_path_2d_grid = params.save_path + 'images/batch_trajectories_2d_grid.png'
        batch_plot_path_2d_all = params.save_path + 'images/batch_trajectories_2d_all.png'
        if params.space == "euclidean":
            individual_trajectories[:,:,:2] = denormalize_state(individual_trajectories[:,:,:2], x_min, x_max)
            # individual_grid_trajectories[:,:,:2] = denormalize_state(individual_grid_trajectories[:,:,:2], x_min, x_max)
            individual_all_trajectories[:,:,:2] = denormalize_state(individual_all_trajectories[:,:,:2], x_min, x_max)
            plot_batch_trajectories_2d(demonstrations_denorm, individual_trajectories, batch_plot_path_2d_init, 
                                    f"Batch State Dynamics Simulation ({args.sample} random starts)\n", init_dist)
            # plot_batch_trajectories_2d(demonstrations_denorm, individual_grid_trajectories, batch_plot_path_2d_grid, 
            #                           f"Batch State Dynamics Simulation ({args.sample} grid starts)\n", grid_dist)
            plot_batch_trajectories_2d(demonstrations_denorm, individual_all_trajectories, batch_plot_path_2d_all, 
                                      f"Batch State Dynamics Simulation ({args.sample} all starts)\n", all_dist)
        elif params.space == "sphere":            
            plot_batch_trajectories_2d_sphere(q_traj, individual_trajectories, batch_plot_path_2d_init, 
                                    f"Batch State Dynamics Simulation ({args.sample} random starts)\n", init_dist)
            # plot_batch_trajectories_2d_sphere(q_traj, individual_grid_trajectories, batch_plot_path_2d_grid, 
            #                           f"Batch State Dynamics Simulation ({args.sample} grid starts)\n", grid_dist)
            # plot_batch_trajectories_2d_sphere(q_traj, individual_all_trajectories, batch_plot_path_2d_all, 
            #                           f"Batch State Dynamics Simulation ({args.sample} all starts)\n", all_dist)

    # Batch state evolution plot
    batch_plot_path_evolution = params.save_path + 'images/batch_state_evolution.png'
    plot_batch_state_evolution(individual_trajectories, batch_plot_path_evolution, params.delta_t)
    
    # Save batch trajectory data
    suffix = "_model" if args.use_model.lower() == 'true' else ""
    batch_trajectory_save_path = params.save_path + f'stats/batch_trajectories{suffix}.npy'
    np.save(batch_trajectory_save_path, batch_trajectories)
    print(f"Batch trajectory data saved to: {batch_trajectory_save_path}")
    
    # Save individual trajectories
    if params.individual_save is True:
        individual_traj_dir = params.save_path + 'stats/individual_trajectories/'
        if not os.path.exists(individual_traj_dir):
            os.makedirs(individual_traj_dir)
        
        for i, trajectory in enumerate(individual_trajectories):
            individual_traj_path = individual_traj_dir + f'trajectory_random_{i+1:03d}.npy'
            np.save(individual_traj_path, trajectory)
        
        print(f"Individual random trajectories saved to: {individual_traj_dir}")
        print(f"Saved {len(individual_trajectories)} individual random trajectory files")
        
        # Save individual trajectory plots (Random)
        individual_plots_dir = params.save_path + 'images/individual_trajectories/'
        if not os.path.exists(individual_plots_dir):
            os.makedirs(individual_plots_dir)
        
        print(f"\nGenerating individual trajectory plots...")
        for i, trajectory in enumerate(individual_trajectories):
            plot_path = individual_plots_dir + f'trajectory_random_{i+1:03d}.png'
            # Get distance data for this trajectory
            trajectory_dist = init_dist[:,i] if i < init_dist.shape[1] else None
            if params.space == "euclidean":
                plot_individual_trajectory_2d(demonstrations_denorm, trajectory, plot_path, i+1, "Random", trajectory_dist)
            elif params.space == "sphere":
                plot_individual_trajectory_2d_sphere(q_traj, trajectory, plot_path, i+1, "Random", trajectory_dist)

        print(f"Individual random trajectory plots saved to: {individual_plots_dir}")
        print(f"Generated {len(individual_trajectories)} individual random trajectory plots\n")
        
        # Save individual grid trajectories
        # individual_grid_traj_dir = params.save_path + 'stats/individual_grid_trajectories/'
        # if not os.path.exists(individual_grid_traj_dir):
        #     os.makedirs(individual_grid_traj_dir)
        
        # for i, trajectory in enumerate(individual_grid_trajectories):
        #     individual_grid_traj_path = individual_grid_traj_dir + f'trajectory_grid_{i+1:03d}.npy'
        #     np.save(individual_grid_traj_path, trajectory)
        
        # print(f"Individual grid trajectories saved to: {individual_grid_traj_dir}")
        # print(f"Saved {len(individual_grid_trajectories)} individual grid trajectory files\n")
        
        # # Save individual grid trajectory plots
        # individual_grid_plots_dir = params.save_path + 'images/individual_grid_trajectories/'
        # if not os.path.exists(individual_grid_plots_dir):
        #     os.makedirs(individual_grid_plots_dir)
        
        # for i, trajectory in enumerate(individual_grid_trajectories):
        #     plot_path = individual_grid_plots_dir + f'trajectory_grid_{i+1:03d}.png'
        #     # Get distance data for this trajectory
        #     trajectory_dist = grid_dist[:,i] if i < grid_dist.shape[1] else None
        #     if params.space == "euclidean":
        #         plot_individual_trajectory_2d(demonstrations_denorm, trajectory, plot_path, i+1, "Random", trajectory_dist)
        #     elif params.space == "sphere":
        #         plot_individual_trajectory_2d_sphere(q_traj, trajectory, plot_path, i+1, "Random", trajectory_dist)

        
        # print(f"Individual grid trajectory plots saved to: {individual_grid_plots_dir}")
        # print(f"Generated {len(individual_grid_trajectories)} individual grid trajectory plots\n")
        
        
    


        # individual_all_traj_dir = params.save_path + 'stats/individual_all_trajectories/'
        # if not os.path.exists(individual_all_traj_dir):
        #     os.makedirs(individual_all_traj_dir)

        # for i, trajectory in enumerate(individual_all_trajectories):
        #     individual_all_traj_path = individual_all_traj_dir + f'trajectory_all_{i+1:03d}.npy'
        #     np.save(individual_all_traj_path, trajectory)

        # print(f"Individual all trajectories saved to: {individual_all_traj_dir}")
        # print(f"Saved {len(individual_all_trajectories)} individual all trajectory files")

        # # Save individual all trajectory plots
        # individual_all_plots_dir = params.save_path + 'images/individual_all_trajectories/'
        # if not os.path.exists(individual_all_plots_dir):
        #     os.makedirs(individual_all_plots_dir)

        # for i, trajectory in enumerate(individual_all_trajectories):
        #     plot_path = individual_all_plots_dir + f'trajectory_all_{i+1:03d}.png'
        #     # Get distance data for this trajectory
        #     trajectory_dist = all_dist[:,i] if i < all_dist.shape[1] else None
        #     if params.space == "euclidean":
        #         plot_individual_trajectory_2d(demonstrations_denorm, trajectory, plot_path, i+1, "Random", trajectory_dist)
        #     elif params.space == "sphere":
        #         plot_individual_trajectory_2d_sphere(q_traj, trajectory, plot_path, i+1, "Random", trajectory_dist)


        # # print(f"Individual all trajectory plots saved to: {individual_all_plots_dir}")
        # print(f"Generated {len(individual_all_trajectories)} individual all trajectory plots")

    # Save random initial states
    initial_states_save_path = params.save_path + 'stats/random_initial_states.npy'
    np.save(initial_states_save_path, random_initial_states)
    print(f"Random initial states saved to: {initial_states_save_path}\n")
    
    # Save grid initial states
    # grid_states_save_path = params.save_path + 'stats/grid_initial_states.npy'
    # np.save(grid_states_save_path, grid_initial_states)
    # print(f"Grid initial states saved to: {grid_states_save_path}\n")

    
    # all_states_save_path = params.save_path + 'stats/all_initial_states.npy'
    # np.save(all_states_save_path, all_initial_states)
    # print(f"All initial states saved to: {all_states_save_path}\n")

    # Also run single trajectory simulation for comparison (using first random state)
    # print("\nRunning single trajectory simulation for comparison...")
    # single_initial_state = random_initial_states[0]
    # single_trajectory = init_state_dynamics.simulate(single_initial_state)

    # # Single trajectory plots
    # if params.workspace_dimensions == 2:
    #     single_plot_path_2d = params.save_path + 'images/single_trajectory_2d.png'
    #     plot_trajectory_2d(demonstrations_denorm, single_trajectory, single_plot_path_2d, 
    #                       "Single State Dynamics Simulation")
    
    # single_plot_path_evolution = params.save_path + 'images/single_state_evolution.png'
    # plot_state_evolution(single_trajectory, single_plot_path_evolution, params.delta_t)
    
    dtw_results = []
    if params.space == "euclidean":
        for idx , init_traj in enumerate(individual_trajectories):
            traj_list = []
            positions = init_traj[:, :2]
            # print(positions[-5:,:],demonstrations_denorm[0,-5:,:])
            goal = demonstrations_denorm[0,-1,:2]
            # positions_denorm = denormalize_state(positions, x_min, x_max)
            if torch.is_tensor(goal):
                goal = goal.detach().cpu().numpy()
            if torch.is_tensor(positions):
                positions = positions.detach().cpu().numpy()
            
            for position in positions:
                if torch.is_tensor(position):
                    position = position.detach().cpu().numpy()
                
                if np.linalg.norm(position-goal) > 10:
                    traj_list.append(position)
                else:
                    traj_list.append(position)
                    break
            traj_list = np.array(traj_list).reshape(-1, 2)
            distance, path = fastdtw(demonstrations_denorm[0], traj_list, dist=2)

            
            normalized_distance = distance / len(path)
            dtw_results.append(normalized_distance)
    elif params.space == "sphere":
        for idx , init_traj in enumerate(individual_trajectories):
            # print(init_traj.shape, q_traj.shape)
            traj_list = []
            positions_q = init_traj[:,:2]
            positions = q_to_x(torch.from_numpy(positions_q))
            # print(x_traj.shape,q_traj.shape)/
            goal = x_traj[-1,:3]
            if torch.is_tensor(goal):
                goal = goal.detach().cpu().numpy()
            if torch.is_tensor(positions):
                positions = positions.detach().cpu().numpy()
            
            for position in positions:
                if torch.is_tensor(position):
                    position = position.detach().cpu().numpy()
                
                if np.linalg.norm(position-goal) > 1:
                    traj_list.append(position)
                else:
                    traj_list.append(position)
                    break
            traj_list = np.array(traj_list).reshape(-1, 3)
            from scipy.spatial.distance import cosine
            # print(x_traj[0].numpy().shape,traj_list.shape)
            distance, path = fastdtw(x_traj.numpy(), traj_list, dist=cosine)
            
            normalized_distance = distance / len(path)
            dtw_results.append(normalized_distance)
        
    dtw_results = np.array(dtw_results)
    mean_dtw = np.mean(dtw_results)
    std_dtw = np.std(dtw_results)
    if params.space == "euclidean":
        threshold_dtw = min(max(mean_dtw + 2 * std_dtw , mean_dtw + 10.0),20) # trajectories with DTW > mean + 2*std
    elif params.space == "sphere":
        threshold_dtw = min(max(mean_dtw + 2 * std_dtw , mean_dtw + 0.1),0.3) # trajectories with DTW > mean + 2*std
        
    print(f"\nDTW Analysis:")
    print(f"  Mean DTW: {mean_dtw:.4f}")
    print(f"  Std DTW: {std_dtw:.4f}")
    print(f"  Threshold (mean + 2*std): {threshold_dtw:.4f}")

    # Find high DTW trajectories
    high_dtw_indices = np.where(dtw_results > threshold_dtw)[0]
    print(f"  High DTW trajectories: {len(high_dtw_indices)}/{len(dtw_results)}")
    print(f"  High DTW trajectory indices: {high_dtw_indices + 1}")  # +1 for 1-based indexing
    print(f"  Their DTW values: {dtw_results[high_dtw_indices]}")
    
    dtw_path = params.save_path + 'dtw_results.txt'
    with open(dtw_path, 'w') as f:
        f.write("DTW Analysis Results\n")
        f.write(f"Mean DTW: {mean_dtw:.4f}\n")
        f.write(f"Std DTW: {std_dtw:.4f}\n")
        f.write(f"Threshold (mean + 2*std): {threshold_dtw:.4f}\n")
        f.write(f"High DTW trajectories: {len(high_dtw_indices)}/{len(dtw_results)}\n")
        f.write(f"High DTW trajectory indices (1-based): {high_dtw_indices + 1}\n")
        f.write(f"Their DTW values: {dtw_results[high_dtw_indices]}\n")
        # if all_lyapunov is not None:
        #     arr = all_lyapunov.cpu().numpy()
        #     mean_val = np.nanmean(arr)
        #     std_val = np.nanstd(arr)
            # print(arr)

            # f.write(f"Random sample Lyapunov exponents: {mean_val:.4f} +- {std_val:.4f}\n")
        if init_lyapunov is not None:    
            for idx, dtw_value in enumerate(dtw_results):
                f.write(f"Trajectory {idx + 1}: DTW = {dtw_value:.4f} Lyapunov = {init_lyapunov[idx]:.4f}\n")
            

    print("\n=== Simulation completed successfully! ===")
    print(f"Results saved in: {params.save_path}")
    simulation_method = "Learned model transition" if args.use_model.lower() == 'true' else "dx/dt = (x_goal - x)"
    print(f"Simulation method: {simulation_method}")
    if args.use_model.lower() != 'true':
        print(f"Goal position: {init_state_dynamics.goal[:params.workspace_dimensions]}")
    print(f"Number of random trajectories simulated: {args.sample}")
    print(f"Simulation time: {simulation_time:.2f} seconds")

if __name__ == "__main__":
    for i in range(11):
    # i = 8
        main(i)