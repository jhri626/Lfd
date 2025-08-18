import os
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import importlib
import argparse
from initializer import initialize_framework
import numpy as np
from agent.utils.dynamical_system_operations import normalize_state, denormalize_state
import similaritymeasures as sm
from sklearn.metrics.pairwise import cosine_similarity





ALLOW_RANGE_3 = [
            (0, 240),
            (300, 430),
            (440, 500)
        ]

# Parse command line arguments
parser = argparse.ArgumentParser(description='Simulate dynamical system with evaluation style plotting')
parser.add_argument('--random', type=str, default='false', choices=['true', 'false'],
                    help='Whether to use random sampling around initial points (true/false)')
parser.add_argument('--num-samples', type=int, default=14,
                    help='Number of random samples to generate when --random=true')
parser.add_argument('--noise-std', type=float, default=0.1,
                    help='Standard deviation of noise for random sampling')
parser.add_argument('--sampling-mode', type=str, default='all', choices=['all', 'init'],
                    help='Sampling mode: "all" for trajectory sampling, "init" for initial points only')
args = parser.parse_args()

# Parameters
title = 'Dynamic system with CONDOR framework latent dim 4'
eta = 4
params_name = '2nd_order_2D'
simulation_length = 1500
results_base_directory = './'
np.random.seed(42)  # For reproducibility

# Original initial states (base points) - fallback for non-random mode
x_t_init_base = np.array([[0.5, 0.6], [-0.75, 0.9], [0.9, -0.9], [-0.9, -0.9], [0.9, 0.9], [0.9, 0.3], [-0.9, -0.1],
                         [-0.9, 0.0], [0.4, 0.4], [0.9, -0.1], [-0.9, -0.5], [0.9, -0.5], [-0.25, 0.5], [-0.5, -0.5]])

# Load parameters first to get demonstration data
Params = getattr(importlib.import_module('params.' + params_name), 'Params')
params = Params(results_base_directory)
params.results_path += params.selected_primitives_ids + '/'
params.load_model = True

results_directory = results_base_directory + "test8/cos_sim_99/" + params.name + "/" + params.selected_primitives_ids + "/"
os.makedirs(results_directory, exist_ok=True)
# Initialize framework to get demonstration data
learner, _, data = initialize_framework(params, params_name, verbose=False)

# Get demonstration trajectories for uniform sampling
demonstrations_eval = data['demonstrations train'][..., 0]
x_min = np.array(data['x min'])
x_max = np.array(data['x max'])
vel_x_min = np.array(data['vel min train'])
vel_x_max = np.array(data['vel max train'])

demonstrations_eval_velocities = demonstrations_eval[:, 1:, :] - demonstrations_eval[:, :-1, :]  # Calculate velocities   
demonstrations_eval_velocities = np.concatenate((demonstrations_eval_velocities, np.zeros((demonstrations_eval_velocities.shape[0], 1, 2))), axis=1)  # Add zero velocity for last point


# Generate initial states based on random flag
if args.random.lower() == 'true':
    print(f"Using uniform sampling along demonstration trajectories with {args.num_samples} samples")
    print(f"Found {len(demonstrations_eval)} demonstration trajectories")
    print(f"Sampling mode: {args.sampling_mode}")
    
    # Collect all trajectory points and their velocities
    all_traj_points = []
    all_traj_velocities = []
    
    if args.sampling_mode == 'init':
        # Sample only from initial points of demonstrations with zero velocities
        print("Mode: Sampling from demonstration starting points with zero velocities")
        
        for demo in demonstrations_eval:
            # Get initial point (first point of demonstration)
            initial_position = demo[0, :2]  # Position coordinates only
            initial_velocity = demo[2,:2] - demo[0,:2]   # Zero velocity for init mode
            
            # all_traj_points.append(initial_position)
            # all_traj_velocities.append(initial_velocity)
        
        # Convert to numpy arrays
        all_traj_points = np.array(all_traj_points)
        all_traj_velocities = np.array(all_traj_velocities)
        
        # If we need more samples than available demos, duplicate with noise
        if len(all_traj_points) < args.num_samples:
            print(f"Need more samples ({args.num_samples}) than demos ({len(all_traj_points)}), duplicating with noise")
            
            while len(all_traj_points) < args.num_samples:
                remaining = args.num_samples - len(all_traj_points)
                repeat_count = min(remaining, len(demonstrations_eval))
                
                # Add duplicates with small noise
                duplicate_positions = initial_position.copy()
                duplicate_velocities = initial_velocity.copy()
                

                # Add small random noise to avoid exact duplicates
                small_noise = abs(1 * np.random.normal(0, args.noise_std, 1))
                direction = np.random.uniform(3.14/6, 5 * 3.14/6, 1)
                sign = np.random.choice([-1, 1], size=1)
                direction *= sign
                min_noise_factor = 0.05 # Minimum absolute value for noise
                max_noise_factor = 0.15  # Maximum absolute value for noise
                for i in range(small_noise.shape[0]):
                    if abs(small_noise[i]) < min_noise_factor:
                        # Keep the sign but ensure minimum absolute value
                        small_noise[i] = min_noise_factor if small_noise[i] >= 0 else -min_noise_factor
                    if abs(small_noise[i]) > max_noise_factor:
                        small_noise[i] = max_noise_factor if small_noise[i] >= 0 else -max_noise_factor

                norm_init_vel = np.linalg.norm(duplicate_velocities, keepdims=True)
                normalized_init_velocities = duplicate_velocities / np.where(norm_init_vel == 0, 1, norm_init_vel)  # Avoid division by zero

                normal_to_vel = np.stack([-normalized_init_velocities[1], normalized_init_velocities[0]])
                
                duplicate_positions += small_noise * (normal_to_vel * np.sin(direction) - normalized_init_velocities * np.cos(direction))
                # small_noise = np.random.normal(0, args.noise_std, duplicate_positions.shape)
                
                # for i in range(small_noise.shape[0]):
                #     # Ensure minimum absolute value for noise
                #     if abs(small_noise[i]) < min_noise_factor:
                #         # Keep the sign but ensure minimum absolute value
                #         small_noise[i] = min_noise_factor if small_noise[i] >= 0 else -min_noise_factor
                # duplicate_positions += small_noise
                
                if len(all_traj_points) == 0:
                    all_traj_points = duplicate_positions
                    all_traj_velocities = duplicate_velocities
                else:
                    all_traj_points = np.vstack([all_traj_points, duplicate_positions])
                    all_traj_velocities = np.vstack([all_traj_velocities, duplicate_velocities])
        
        # Select exactly args.num_samples points
        
        x_t_init_positions = all_traj_points
        x_t_init_velocities = all_traj_velocities
        
        unique_indices = np.zeros(len(x_t_init_positions),dtype=int)  # No unique indices in init mode

    elif args.sampling_mode == 'all':
        # Original trajectory sampling logic
        print("Mode: Sampling from all trajectory points with calculated velocities")
        
        for demo in demonstrations_eval:
            traj_length = len(demo)
            # Sample more heavily from the beginning of each trajectory
            num_samples_per_traj = max(1, args.num_samples // len(demonstrations_eval))
            print(f"Sampling {num_samples_per_traj} points from trajectory of length {traj_length}")

            # Create weighted sampling indices - more samples at the beginning
            # Use exponential decay: more weight at the beginning, less at the end
            decay_factor = 1.5  # Higher value = more bias toward beginning
            weights = np.exp(-decay_factor * np.linspace(0, 1, traj_length))
            weights = weights / weights.sum()  # Normalize to probabilities
            
            # allowed_indices = []
            # for start_idx, end_idx in ALLOW_RANGE_3:
            #     allowed_indices.extend(range(start_idx, end_idx))
            
            # allowed_indices = list(set(allowed_indices))  
            # allowed_indices.sort()
            
            # # Sample indices based on weights (with replacement for better beginning bias)
            # if len(allowed_indices) == 0:
            #     allowed_indices = list(range(traj_length))
            
            # allowed_weights = weights[allowed_indices]
            # allowed_weights = allowed_weights / allowed_weights.sum()
            allowed_indices = np.arange(traj_length)  # Use all indices for sampling
            
            sampled_indices = np.random.choice(allowed_indices, size=num_samples_per_traj, 
                                             replace=False, p=weights)
            # Remove duplicates but keep order bias
            unique_indices = np.unique(sampled_indices)
            
            
            # If we don't have enough unique samples, add more from beginning
            if len(unique_indices) < num_samples_per_traj:
                beginning_samples = min(num_samples_per_traj - len(unique_indices), 
                                      traj_length // 4)  # Sample from first quarter
                additional_indices = np.random.choice(traj_length // 4, 
                                                    size=beginning_samples, 
                                                    replace=False)
                unique_indices = np.unique(np.concatenate([unique_indices, additional_indices]))
            
            for idx in unique_indices:
                # Position (x, y)
                position = demo[idx, :2]
                all_traj_points.append(position)
                
                # Calculate velocity at this point
                if idx == 0:
                    # Forward difference for first point
                    if traj_length > 1:
                        velocity = (demo[1, :2] - demo[0, :2]) / params.delta_t
                    else:
                        velocity = np.zeros(2)
                elif idx == traj_length - 1:
                    # Backward difference for last point
                    velocity = np.zeros(2) 
                else:
                    # Central difference for middle points
                    velocity = (demo[idx+2, :2] - demo[idx, :2]) / (2 * params.delta_t)
                
                all_traj_velocities.append(velocity)

        # Convert to numpy arrays
        all_traj_points = np.array(all_traj_points)
        all_traj_velocities = np.array(all_traj_velocities)
        
        # Select exactly args.num_samples points
        if len(all_traj_points) > args.num_samples:
            # Randomly select from uniformly sampled points
            selected_indices = np.random.choice(len(all_traj_points), args.num_samples, replace=False)
            x_t_init_positions = all_traj_points[selected_indices]
            x_t_init_velocities = all_traj_velocities[selected_indices]
            
        else:
            # If we don't have enough points, use all and pad with repetition
            x_t_init_positions = all_traj_points
            x_t_init_velocities = all_traj_velocities
            
            while len(x_t_init_positions) < args.num_samples:
                remaining = args.num_samples - len(x_t_init_positions)
                repeat_indices = np.random.choice(len(all_traj_points), 
                                                min(remaining, len(all_traj_points)), 
                                                replace=True)
                x_t_init_positions = np.vstack([x_t_init_positions, all_traj_points[repeat_indices]])
                x_t_init_velocities = np.vstack([x_t_init_velocities, all_traj_velocities[repeat_indices]])
    
    # Add noise to positions if specified
        if args.noise_std > 0:
            noise_factor = np.random.normal(0, args.noise_std, x_t_init_positions.shape[0])
            min_noise_factor = 0.03  # Minimum absolute value for noise
            for i in range(noise_factor.shape[0]):
                if abs(noise_factor[i]) < min_noise_factor:
                    # Keep the sign but ensure minimum absolute value
                    noise_factor[i] = min_noise_factor if noise_factor[i] >= 0 else -min_noise_factor
            normal_to_vel = np.stack([-x_t_init_velocities[:, 1], x_t_init_velocities[:, 0]], axis=1)
            norms = np.linalg.norm(normal_to_vel, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            perpendicular_normalized = normal_to_vel / norms
            noise = noise_factor[:, np.newaxis] * perpendicular_normalized
            x_t_init_positions = x_t_init_positions + noise
            # print("nosie",noise)
            # print("x_t_init_positions after noise", x_t_init_positions)
            # x_init_positions = normalize_state(x_t_init_positions_denorm, x_min, x_max)
            print(f"Added noise with std={args.noise_std}")

        # Clip to stay within reasonable bounds [-1, 1] (since demos are normalized)
        x_t_init_positions = np.clip(x_t_init_positions, -1, 1)
        
        title += f' ({args.sampling_mode} sampling: {args.num_samples} samples, std={args.noise_std})'
else:
    print("Using predefined initial states")
    x_t_init_positions = x_t_init_base.copy()
    # Add zero velocities for predefined states (original behavior)
    x_t_init_velocities = np.zeros((x_t_init_positions.shape[0], 2))

# Combine positions and velocities




x_t_init = np.hstack((x_t_init_positions, x_t_init_velocities))
print(x_t_init[0:5,:])
print(unique_indices)




# Get other data for plotting
primitive_ids = np.array(data['demonstrations primitive id'])
goals = data['goals training'][0]

# Initialize dynamical system
dynamical_system = learner.init_dynamical_system(initial_states=torch.FloatTensor(x_t_init).cuda(), delta_t=params.delta_t)

# Simulate trajectories
simulated_trajectories = [x_t_init]
for i in range(simulation_length):
    # Do transition
    x_t = dynamical_system.transition(space='task')['desired state']
    simulated_trajectories.append(x_t.cpu().detach().numpy())

# Convert to numpy array for easier handling
simulated_trajectories = np.array(simulated_trajectories)  # shape: (simulation_length, n_trajectories, state_dim)

# Prepare variables for analysis and plotting
goal_denorm = denormalize_state(goals.reshape(1, -1), x_min[:2], x_max[:2])
save_filename = f'simulate_ds_eval_style_dim_{eta}'
if args.random.lower() == 'true':
    save_filename += f'_random_{args.num_samples}samples_std{args.noise_std}'

# Calculate Lyapunov exponent for R^2 trajectories
print("\nCalculating Lyapunov exponent for R^2 trajectories...")

# Import necessary functions for curve analysis
import sys
import os
sys.path.append('/home/jhri626/Lfd/submodule/BCSDM')

# Define R^2 functions locally if import fails
def get_closest_dist_traj_r2_local(sample_traj, demo_traj, search_window=50, unique_indices=None):
    """Calculate closest distance trajectory for lyapunov exponent in R^2
    
    Args:
        sample_traj: (nb, nt1, 2) - simulated trajectories
        demo_traj: (nt2, 2) - demonstration trajectory
        search_window: int - window size around previous closest point to search
    """
    eps = 1e-10
    
    nb, nt1, _ = sample_traj.shape
    nt2, _ = demo_traj.shape

    # Initialize distance trajectory
    dist_min = torch.zeros(nb, nt1)
    closest_idx_list = torch.zeros(nb, nt1, dtype=torch.long)
    
    for batch_idx in range(nb):
        prev_closest_idx = 0  # Start from beginning of demo trajectory
        
        for time_idx in range(nt1):
            
            current_point = sample_traj[batch_idx, time_idx]  # (2,)
            
            if time_idx == 0:
                # For first time step, search entire demo trajectory
                search_start = max(0, unique_indices - search_window)
                search_end = min(nt2, unique_indices + search_window + 40)
                demo_window = demo_traj[search_start:search_end]
            else:
                # Define search window around previous closest point
                search_start = max(0, prev_closest_idx - search_window)
                search_end = min(nt2, prev_closest_idx + search_window + 1)
                
                # Calculate distances only within search window
                demo_window = demo_traj[search_start:search_end]  # (window_size, 2)
    
            # Calculate distances to points in window
            distances = torch.norm(current_point.unsqueeze(0) - demo_window, dim=-1)  # (window_size,)
            
            # Find minimum distance and its index within window
            min_idx_in_window = torch.argmin(distances)
            min_distance = distances[min_idx_in_window]
            # print(min_distance,min_idx_in_window)
            
            # Convert back to global index
            if time_idx == 0:
                closest_idx = search_start + min_idx_in_window
            else:
                closest_idx = search_start + min_idx_in_window
        
            # Store distance and update previous closest index
            dist_min[batch_idx, time_idx] = min_distance
            prev_closest_idx = closest_idx
            closest_idx_list[batch_idx, time_idx] = closest_idx

    dist_min = torch.clamp(dist_min, min=eps)
    return dist_min ,closest_idx_list

def calc_lyapunov_exponent_local(distance_traj, terminal_time=1, eps=1e-6, 
                                sim_trajectories=None, sim_trajectories_velocities=None, vel_at_closest=None,goal_position=None, goal_threshold=10):
    """Calculate lyapunov exponent from distance trajectory
    
    Args:
        distance_traj: Distance trajectory for Lyapunov calculation
        terminal_time: Terminal time for normalization
        eps: Distance threshold for truncation
        sim_trajectories: Simulated trajectories to check goal arrival
        goal_position: Goal position for checking arrival
        goal_threshold: Distance threshold to consider goal reached
    """
    input_size = distance_traj.shape
    len_traj = input_size[-1]
    
    if len(input_size) == 2:
        multitraj = True
    else:
        multitraj = False
        distance_traj = distance_traj.unsqueeze(0)
    
    lamb_list = []
    cos_sim = cosine_similarity(sim_trajectories_velocities, vel_at_closest).diagonal()
    cos_dist = 1 - cos_sim
    for i in range(len(distance_traj)):
        truncation_point = len(distance_traj[i])  # Default: use full trajectory
        
        # Check for eps-based truncation
        for j in range(len(distance_traj[i])):
            if distance_traj[i][j] < eps:
                if cos_sim[j] > 0.99:
                    print(f"Trajectory {i}: eps truncation at step {j}, dist={distance_traj[i][j]}, cos_sim={cos_sim[j]:.2f}")
                    truncation_point = j + 1
                    break
        
        # for j in range(len(distance_traj[i])):
        #     if distance_traj[i][j] + 1 * cos_dist[j] < eps:
        #         print(f"Trajectory {i}: eps truncation at step {j}, dist={distance_traj[i][j]}, cos_sim={cos_sim[j]:.2f}")
        #         truncation_point = j + 1
        #         break
        
        # Check for goal arrival truncation (if sim_trajectories and goal provided)
        if sim_trajectories is not None and goal_position is not None:
            for j in range(len(sim_trajectories[i])):
                # Calculate distance to goal
                current_pos = sim_trajectories[i][j]  # (2,) position
                goal_dist = torch.norm(current_pos - goal_position)
                
                if goal_dist < goal_threshold:
                    goal_truncation_point = j + 1
                    # Use the earlier truncation point (eps or goal)
                    if goal_truncation_point < truncation_point:
                        truncation_point = goal_truncation_point
                        print(f"Trajectory {i}: goal arrival truncation at step {j}, dist={goal_dist:.2f}")
                    break
        
        # Apply truncation
        distance_truncated = distance_traj[i][:truncation_point]
        
        
        if len(distance_truncated) < 2:
            lamb_list.append(float('nan'))
            continue
            
        # Calculate actual time based on truncation point
        # actual_time = (len(distance_truncated) - 1) * (terminal_time / (len_traj - 1))
        
        
        log_dist_traj = torch.log(distance_truncated).unsqueeze(-1)
        log_d0 = log_dist_traj[..., 0:1, 0:1]
        t_linspace = torch.linspace(0, terminal_time, len(distance_truncated), dtype=torch.float64).to(log_d0)
        
        try:
            lamb = (torch.pinverse(t_linspace.unsqueeze(-1)) @ (log_d0 - log_dist_traj)).squeeze().squeeze()
            lamb_list.append(lamb)
        except:
            lamb_list.append(float('nan'))
            
    
    lamb = torch.tensor(lamb_list)
    return lamb, truncation_point, cos_dist

# Use local curve analysis functions
get_closest_dist_traj_r2 = get_closest_dist_traj_r2_local
cal_lyapunov_exponent = calc_lyapunov_exponent_local

# Convert simulated trajectories to torch tensors for analysis
# Only use position coordinates (first 2 dimensions) and denormalize
sim_trajectories_positions = simulated_trajectories[:, :, :2]  # (time, n_trajectories, 2)
sim_trajectories_velocities = simulated_trajectories[1:, :, :2] - simulated_trajectories[:-1, :, :2]  # (time, n_trajectories, 2)
sim_trajectories_velocities = np.concatenate((sim_trajectories_velocities, np.zeros((1, sim_trajectories_velocities.shape[1], 2))), axis=0)  # Add zero velocity for last point
  # (time, n_trajectories, 2)
sim_trajectories_denorm = np.zeros_like(sim_trajectories_positions)

# Denormalize each trajectory
for i in range(sim_trajectories_positions.shape[1]):  # for each trajectory
    sim_trajectories_denorm[:, i, :] = denormalize_state(sim_trajectories_positions[:, i, :], x_min[:2], x_max[:2])
sim_trajectories_torch = torch.FloatTensor(sim_trajectories_denorm)  # (time, n_trajectories, 2)
# sim_trajectories_torch = torch.FloatTensor(sim_trajectories_positions)  # (time, n_trajectories, 2)
sim_trajectories_torch = sim_trajectories_torch.transpose(0, 1)  # (n_trajectories, time, 2)

# Use demonstration trajectories as reference (also denormalize them)
demo_trajectories_denorm = np.zeros_like(demonstrations_eval[:, :, :2])
for i in range(len(demonstrations_eval)):
    demo_trajectories_denorm[i] = denormalize_state(demonstrations_eval[i, :, :2], x_min[:2], x_max[:2])

demo_trajectories_torch = torch.FloatTensor(demo_trajectories_denorm)  # (n_demos, time, 2)
# demo_trajectories_torch = torch.FloatTensor(demonstrations_eval)  # (n_demos, time, 2)

# print('compare',sim_trajectories_torch[:,-1,:], demo_trajectories_torch[:,-1,:])
# Calculate Lyapunov exponent for each simulated trajectory against all demonstrations
lyapunov_results = []
arrival_points = []
eps = 1
terminal_time = simulation_length * params.delta_t

# Prepare goal position for truncation
goal_denorm_tensor = torch.FloatTensor(goal_denorm.squeeze())  # Convert to tensor
goal_threshold = 10  # Same as fixed_point_iteration_thr from params

closest_distances_for_all_trajectories = []  # Store closest distances for each trajectory
# print(demonstrations_eval_velocities[0,:,:])
print(demo_trajectories_torch.shape)

cos_dist_list = []

window_size = demonstrations_eval.shape[1] // 10  # Search window size for closest distance calculation
for i, sim_traj in enumerate(sim_trajectories_torch):
    print(f"\nProcessing trajectory {i+1}/{len(sim_trajectories_torch)}")
    # print("sim_traj_vel", sim_trajectories_velocities[15:20,:])
    # Find closest distances to all demonstration trajectories
    closest_distances = []
    all_closest_indices = []  # Store indices for each demo
    for demo_traj in demo_trajectories_torch:
        # Reshape for get_closest_dist_traj_r2 function
        sim_traj_expanded = sim_traj.unsqueeze(0)  # (1, time, 2)
        dist_traj, closest_idx_list = get_closest_dist_traj_r2(sim_traj_expanded, demo_traj, search_window=window_size, unique_indices=unique_indices[i])  # (time,)
        closest_distances.append(dist_traj.squeeze(0))  # (time,)
        all_closest_indices.append(closest_idx_list.squeeze(0))

    # Use minimum distance across all demonstrations
    min_distances = torch.stack(closest_distances, dim=0)  # (n_demos, time)
    min_dist_traj, _ = torch.min(min_distances, dim=0)  # (time,)
    
    # Store the minimum distances for this trajectory for later color mapping
    closest_distances_for_all_trajectories.append(min_dist_traj.cpu().numpy())

    closest_indices = torch.stack(all_closest_indices, dim=0).squeeze(0)  # (n_demos, time)
    
    vel_at_closest = demonstrations_eval_velocities[0,closest_indices,:]  # Initialize velocity array
    
    # print(vel_at_closest[15:20,:])
    # print(vel_at_closest[100:120,:])
    # print(sim_trajectories_velocities[100:120,i,:])
    # print(cosine_similarity(sim_trajectories_velocities[:, i, :], vel_at_closest).diagonal()[100:120])
    # print(min_dist_traj[100:120])

    # Calculate Lyapunov exponent with goal arrival consideration
    try:
        lyapunov_exp, arrival_point, cos_dist = cal_lyapunov_exponent(min_dist_traj, 
                                           terminal_time=terminal_time, 
                                           eps=eps,
                                           sim_trajectories=sim_traj.unsqueeze(0),  # Add batch dimension
                                           sim_trajectories_velocities=sim_trajectories_velocities[:, i, :],  # Use velocities for this trajectory
                                           vel_at_closest=vel_at_closest,
                                           goal_position=goal_denorm_tensor,
                                           goal_threshold=goal_threshold)
        lyapunov_results.append(lyapunov_exp.item())
        arrival_points.append(arrival_point)
        cos_dist_list.append(cos_dist)
        print(f"  Lyapunov exponent: {lyapunov_exp.item():.6f}")
    except Exception as e:
        print(f"  Error calculating Lyapunov exponent: {e}")
        lyapunov_results.append(float('nan'))

# Calculate statistics
lyapunov_results = np.array(lyapunov_results)
arrival_points = np.array(arrival_points)
valid_results = lyapunov_results[~np.isnan(lyapunov_results)]
valid_arrival_points = arrival_points[~np.isnan(lyapunov_results)]

from fastdtw import fastdtw
print(demo_trajectories_denorm.shape)

# Store DTW results for analysis
dtw_results = []
dtw_trajectories = []
dtw_demo_refs = []
traj_len = 0
for idx, traj in enumerate(sim_trajectories_positions.transpose(1, 0, 2)):
    traj_list = []
    positions = traj[:, :2]  # Only position coordinates
    positions_denorm = denormalize_state(positions, x_min[:2], x_max[:2])
    for position in positions_denorm:
        if np.linalg.norm(position-goal_denorm) > 10:
            traj_list.append(position)
        else:
            traj_list.append(position)
            break
    
    traj_list = np.array(traj_list).reshape(-1, 2)
    demo_fre = np.array(demo_trajectories_denorm[0,unique_indices[idx]:,:]).reshape(-1, 2)
    print(traj_list.shape, demo_fre.shape)
    distance, path = fastdtw(traj_list, demo_fre)
    normalized_distance = distance / len(path)
    print(f"index {idx+1} ,DTW distance: {normalized_distance}")
    traj_len += traj_list.shape[0]
    
    # Store results for analysis
    dtw_results.append(normalized_distance)
    dtw_trajectories.append(traj_list)
    dtw_demo_refs.append(demo_fre)
# print(f"Total number of simulated trajectories: {sim_trajectories_positions.shape}")
print(f"Total trajectory length: {traj_len/sim_trajectories_positions.shape[1]}")
# Analyze DTW results to identify high-distance trajectories
dtw_results = np.array(dtw_results)
mean_dtw = np.mean(dtw_results)
std_dtw = np.std(dtw_results)
threshold_dtw = min(max(mean_dtw + 2 * std_dtw , mean_dtw + 10.0),8) # trajectories with DTW > mean + 2*std

print(f"\nDTW Analysis:")
print(f"  Mean DTW: {mean_dtw:.4f}")
print(f"  Std DTW: {std_dtw:.4f}")
print(f"  Threshold (mean + 2*std): {threshold_dtw:.4f}")

# Find high DTW trajectories
high_dtw_indices = np.where(dtw_results > threshold_dtw)[0]
print(f"  High DTW trajectories: {len(high_dtw_indices)}/{len(dtw_results)}")
print(f"  High DTW trajectory indices: {high_dtw_indices + 1}")  # +1 for 1-based indexing
print(f"  Their DTW values: {dtw_results[high_dtw_indices]}")

# Now plot using evaluate.py style with DTW information available
plt.rcParams.update({'font.size': 14,
                     'figure.figsize': (8, 8)})

fig, ax = plt.subplots()
ax.grid(linestyle='--')

# Define slightly darker (less pale) red and green for the colormap
medium_red = "#ff1a1a"   # moderately bright red (high distance)
medium_green = "#1aff1a" # moderately bright green (low distance)

# Create a custom colormap with these medium tones
medium_cmap = mcolors.LinearSegmentedColormap.from_list("medium_red_green", [medium_green, medium_red])

# Create directory for individual trajectory plots
per_traj_dir = f'{results_directory + save_filename}/per'
os.makedirs(per_traj_dir, exist_ok=True)

# Plot demonstrations (background trajectories in light gray)
for i in range(len(demonstrations_eval)):
    # Denormalize demonstration data for plotting
    demo_denorm = denormalize_state(demonstrations_eval[i], x_min, x_max)
    plt.plot(demo_denorm[:, 0], demo_denorm[:, 1], color='lightgray', alpha=1.0, linewidth=6, zorder=1)

# Plot simulated trajectories with colors based on closest distances
all_line_collections = []  # Store all line collections for unified colorbar

# Plot each trajectory with individual min-max normalization
for i in range(x_t_init.shape[0]):
    # Extract trajectory for i-th initial state and denormalize
    traj = simulated_trajectories[:, i, :2]  # only position coordinates
    traj_denorm = denormalize_state(traj, x_min[:2], x_max[:2])
    
    # Get the closest distances for this trajectory
    converge_point = arrival_points[i]  # Get arrival point for this trajectory
    pos_distances = closest_distances_for_all_trajectories[i]
    cos_dists = cos_dist_list[i]  # cosine distances for this trajectory
    
    pos_distances[converge_point-1:] = pos_distances[converge_point-1]  # Set distances after convergence point to the last value
    # cos_dists[converge_point:] = np.min(cos_dists)  # Set cosine distances after convergence point to the last value
    cos_dists[converge_point-1:] = cos_dists[converge_point]-1  # Set cosine distances after convergence point to the last value
    # distances = pos_distances + cos_dists
    distances = pos_distances  + 1 * cos_dists  # Use only position distances for color mapping
    
    

    # Create line segments for LineCollection
    points = traj_denorm.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Individual trajectory min-max normalization (0 = green, 1 = red)
    # Use the distances corresponding to each segment (use average of start and end point distances)
    segment_distances = (distances[:-1])
    
    # Individual normalization for this trajectory
    traj_min_dist = np.min(distances)
    traj_max_dist = np.max(distances)
    
    # Normalize distances for this trajectory only (0 = green for this trajectory's min, 1 = red for this trajectory's max)
    if traj_max_dist - traj_min_dist > 1e-8:  # Avoid division by zero
        norm_distances = np.log((segment_distances - traj_min_dist) / (traj_max_dist - traj_min_dist) + 1 + 1e-8)  # Add small value to avoid log(0)
    else:
        norm_distances = np.zeros_like(segment_distances)  # All segments same color if no variation
    
    # Create LineCollection with gradient colors
    lc = LineCollection(segments, cmap=medium_cmap, array=norm_distances, linewidth=3, zorder=11)
    line = ax.add_collection(lc)
    all_line_collections.append(lc)
    
    # Create individual figure for this trajectory
    # fig_single, ax_single = plt.subplots(figsize=(8, 8))
    # ax_single.grid(linestyle='--')
    
    # # Plot demonstrations in the individual figure
    # for j in range(len(demonstrations_eval)):
    #     demo_denorm = denormalize_state(demonstrations_eval[j], x_min, x_max)
    #     ax_single.plot(demo_denorm[:, 0], demo_denorm[:, 1], color='lightgray', alpha=1.0, linewidth=6, zorder=1)
    
    # # Plot goal in the individual figure
    # ax_single.scatter(goal_denorm[:, 0], goal_denorm[:, 1], linewidth=8, color='blue', zorder=12, edgecolors='black', s=100)
    
    # # Plot initial position for this trajectory
    # x_init_single = denormalize_state(x_t_init[i:i+1, :2], x_min[:2], x_max[:2])
    # ax_single.scatter(x_init_single[:, 0], x_init_single[:, 1], color='green', s=50, zorder=10, alpha=0.7, edgecolors='darkgreen')
    
    # # Add the trajectory to individual figure
    # lc_single = LineCollection(segments, cmap=medium_cmap, array=norm_distances, linewidth=3, zorder=11)
    # ax_single.add_collection(lc_single)
    
    # # Set labels and title for individual figure
    # ax_single.set_xlabel('x (px)')
    # ax_single.set_ylabel('y (px)')
    # ax_single.set_title(f'Trajectory {i+1} - DTW: {dtw_results[i]:.4f}, Lyapunov: {lyapunov_results[i]:.6f}')
    # ax_single.spines['right'].set_color('none')
    # ax_single.spines['top'].set_color('none')
    # ax_single.spines['bottom'].set_color('none')
    # ax_single.spines['left'].set_color('none')
    
    # # Save individual trajectory plot
    # plt.savefig(f'{per_traj_dir}/trajectory_{i+1}.png', bbox_inches='tight', dpi=300)
    # plt.close(fig_single)  # Close the individual figure to free memory

# Note: No unified colorbar since each trajectory has its own normalization scale
# Instead, add a text explanation
# ax.text(0.02, 0.98, 'Each trajectory: Green=Local Min Distance, Red=Local Max Distance', 
#         transform=ax.transAxes, fontsize=10, verticalalignment='top',
#         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Plot goal (attractor)
plt.scatter(goal_denorm[:, 0], goal_denorm[:, 1], linewidth=8, color='blue', zorder=12, edgecolors='black', s=100)

# Plot initial positions
x_init_positions = x_t_init[:, :2]  # only position coordinates
x_init_denorm = denormalize_state(x_init_positions, x_min[:2], x_max[:2])
plt.scatter(x_init_denorm[:, 0], x_init_denorm[:, 1], color='green', s=50, zorder=10, alpha=0.7, edgecolors='darkgreen')

# Plot details/info following evaluate.py style
ax.set_xlabel('x (px)')
ax.set_ylabel('y (px)')
ax.set_title(title)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')

# Save figure
save_path = results_directory + save_filename + '.pdf'
print(f'Saving image to {save_path}...')
plt.savefig(save_path, bbox_inches='tight')

# Also save as PNG for convenience
png_path = results_directory + save_filename + '.png'
plt.savefig(png_path, dpi=300, bbox_inches='tight')

# Show plot
plt.show()

# Close figure
plt.clf()
plt.close()

print(f"Simulation completed! Trajectories plotted with evaluate.py style.")
print(f"Random sampling: {args.random}")
if args.random.lower() == 'true':
    print(f"Number of samples: {args.num_samples}, Noise std: {args.noise_std}")
print(f"Files saved: {save_path} and {png_path}")

# Plot high DTW trajectories separately
if len(high_dtw_indices) > 0:
    plt.rcParams.update({'font.size': 14, 'figure.figsize': (12, 8)})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: All trajectories with high DTW ones highlighted
    ax1.grid(linestyle='--')
    ax1.set_title(f'All Trajectories (High DTW highlighted)')
    
    # Plot demonstrations
    for i in range(len(demonstrations_eval)):
        demo_denorm = denormalize_state(demonstrations_eval[i], x_min, x_max)
        ax1.plot(demo_denorm[:, 0], demo_denorm[:, 1], color='lightgray', alpha=1.0, linewidth=6, zorder=1)
    
    # Plot all simulated trajectories
    cm = plt.get_cmap('gist_rainbow')
    num_colors = x_t_init.shape[0]
    
    for i in range(x_t_init.shape[0]):
        traj = simulated_trajectories[:, i, :2]
        traj_denorm = denormalize_state(traj, x_min[:2], x_max[:2])
        
        # Highlight high DTW trajectories in red, others in light colors
        if i in high_dtw_indices:
            ax1.plot(traj_denorm[:, 0], traj_denorm[:, 1], color='red', linewidth=4, zorder=11, alpha=0.8)
            label_pos = traj_denorm[0]
            ax1.text(label_pos[0], label_pos[1], f'{i+1}', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
                    ha='center', va='center', zorder=15, color='white', weight='bold')
        else:
            color = cm(1. * i / num_colors)
            ax1.plot(traj_denorm[:, 0], traj_denorm[:, 1], color=color, linewidth=2, zorder=10, alpha=0.3)
    
    # Plot goal
    ax1.scatter(goal_denorm[:, 0], goal_denorm[:, 1], linewidth=8, color='blue', zorder=12, edgecolors='black', s=100)
    
    # Plot initial positions
    x_init_denorm = denormalize_state(x_t_init[:, :2], x_min[:2], x_max[:2])
    ax1.scatter(x_init_denorm[:, 0], x_init_denorm[:, 1], color='green', s=50, zorder=10, alpha=0.7, edgecolors='darkgreen')
    
    ax1.set_xlabel('x (px)')
    ax1.set_ylabel('y (px)')
    
    # Right plot: Only high DTW trajectories with their reference demonstrations
    ax2.grid(linestyle='--')
    ax2.set_title(f'High DTW Trajectories (DTW > {threshold_dtw:.4f})')
    
    # Plot goal
    ax2.scatter(goal_denorm[:, 0], goal_denorm[:, 1], linewidth=8, color='blue', zorder=12, edgecolors='black', s=100)
    
    # Different colors for each high DTW trajectory
    colors = plt.cm.Set1(np.linspace(0, 1, len(high_dtw_indices)))
    
    for plot_idx, traj_idx in enumerate(high_dtw_indices):
        color = colors[plot_idx]
        
        # Plot the high DTW trajectory
        traj_denorm = dtw_trajectories[traj_idx]
        ax2.plot(traj_denorm[:, 0], traj_denorm[:, 1], color=color, linewidth=4, zorder=11, 
                label=f'Traj {traj_idx+1} (DTW: {dtw_results[traj_idx]:.4f})')
        
        # Plot corresponding demo reference
        demo_ref = dtw_demo_refs[traj_idx]
        ax2.plot(demo_ref[:, 0], demo_ref[:, 1], color=color, linewidth=2, linestyle='--', alpha=0.7, zorder=10)
        
        # Mark initial position
        ax2.scatter(traj_denorm[0, 0], traj_denorm[0, 1], color=color, s=100, zorder=12, 
                   edgecolors='black', marker='o')
        
        # Add trajectory label
        label_pos = traj_denorm[0]
        ax2.text(label_pos[0], label_pos[1], f'{traj_idx+1}', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                ha='center', va='center', zorder=15, weight='bold')
    
    ax2.set_xlabel('x (px)')
    ax2.set_ylabel('y (px)')
    ax2.legend(loc='best', fontsize=10)
    
    # Save the high DTW analysis plot
    high_dtw_save_path = results_directory + save_filename + '_high_dtw_analysis.pdf'
    plt.tight_layout()
    plt.savefig(high_dtw_save_path, bbox_inches='tight')
    
    # Also save as PNG
    high_dtw_png_path = results_directory + save_filename + '_high_dtw_analysis.png'
    plt.savefig(high_dtw_png_path, dpi=300, bbox_inches='tight')
    
    print(f"\nHigh DTW analysis plots saved:")
    print(f"  PDF: {high_dtw_save_path}")
    print(f"  PNG: {high_dtw_png_path}")
    
    plt.show()
    plt.close()
else:
    print("No trajectories with high DTW found.")





if len(valid_results) > 0:
    lyap_mean = np.mean(valid_results)
    lyap_std = np.std(valid_results)
    lyap_median = np.median(valid_results)
    
    # Calculate arrival points mean for all valid trajectories
    arrival_points_mean = np.mean(valid_arrival_points)
    print(f"Mean arrival point (all valid): {arrival_points_mean:.4f}")
        
    lyap_mean_positive = np.mean(valid_results[valid_results > 0])
    lyap_std_positive = np.std(valid_results[valid_results > 0])
    lyap_median_positive = np.median(valid_results[valid_results > 0])
    
    print(f"\nLyapunov Exponent Statistics:")
    print(f"  Mean: {lyap_mean:.6f}")
    print(f"  Std:  {lyap_std:.6f}")
    print(f"  Median: {lyap_median:.6f}")
    print(f"  Valid trajectories: {len(valid_results)}/{len(lyapunov_results)}")
    
    # Print valid trajectories (low DTW) statistics
    valid_dtw_indices = np.where(dtw_results <= threshold_dtw)[0]
    valid_dtw_lyapunov = []
    valid_dtw_arrival_points = []
    for idx in valid_dtw_indices:
        if not np.isnan(lyapunov_results[idx]):
            valid_dtw_lyapunov.append(lyapunov_results[idx])
            valid_dtw_arrival_points.append(arrival_points[idx])
    
    if len(valid_dtw_lyapunov) > 0:
        valid_dtw_lyapunov = np.array(valid_dtw_lyapunov)
        valid_dtw_arrival_points = np.array(valid_dtw_arrival_points)
        valid_dtw_mean = np.mean(valid_dtw_lyapunov)
        valid_dtw_std = np.std(valid_dtw_lyapunov)
        valid_dtw_median = np.median(valid_dtw_lyapunov)
        
        # Calculate arrival points mean for valid low DTW trajectories
        valid_dtw_arrival_mean = np.mean(valid_dtw_arrival_points)
        
        print(f"\nValid Trajectories (Low DTW <= {threshold_dtw:.4f}) Lyapunov Statistics:")
        print(f"  Mean: {valid_dtw_mean:.6f}")
        print(f"  Std:  {valid_dtw_std:.6f}")
        print(f"  Median: {valid_dtw_median:.6f}")
        print(f"  Valid low DTW trajectories: {len(valid_dtw_lyapunov)}/{len(valid_dtw_indices)}")
        print(f"  Mean arrival point (valid low DTW): {valid_dtw_arrival_mean:.4f}")
    else:
        print(f"\nNo valid trajectories with low DTW found.")
    
    # Save results
    lyapunov_save_path = results_directory +  save_filename + '_lyapunov_results.txt'
    with open(lyapunov_save_path, 'w') as f:
        f.write(f"Lyapunov Exponent Analysis Results\n")
        f.write(f"==================================\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Random sampling: {args.random}\n")
        f.write(f"  Number of samples: {args.num_samples if args.random.lower() == 'true' else len(x_t_init_base)}\n")
        if args.random.lower() == 'true':
            f.write(f"  Noise std: {args.noise_std}\n")
        f.write(f"  Simulation length: {simulation_length}\n")
        f.write(f"  Delta t: {params.delta_t}\n")
        f.write(f"  Terminal time: {terminal_time}\n")
        f.write(f"  Epsilon: {eps}\n\n")
        f.write(f"All Trajectories Results:\n")
        f.write(f"  Mean: {lyap_mean:.6f}\n")
        f.write(f"  Std:  {lyap_std:.6f}\n")
        f.write(f"  Median: {lyap_median:.6f}\n")
        f.write(f"  Mean (positive): {lyap_mean_positive:.6f}\n")
        f.write(f"  Std (positive):  {lyap_std_positive:.6f}\n")
        f.write(f"  Median (positive): {lyap_median_positive:.6f}\n")
        f.write(f"  Valid trajectories: {len(valid_results)}/{len(lyapunov_results)}\n\n")
        
        # Add DTW analysis results
        f.write(f"DTW Analysis Results:\n")
        f.write(f"  Mean DTW: {mean_dtw:.4f}\n")
        f.write(f"  Std DTW: {std_dtw:.4f}\n")
        f.write(f"  Threshold DTW: {threshold_dtw:.4f}\n")
        f.write(f"  High DTW trajectories: {len(high_dtw_indices)}/{len(dtw_results)}\n")
        if len(high_dtw_indices) > 0:
            f.write(f"  High DTW trajectory indices: {list(high_dtw_indices + 1)}\n")
            f.write(f"  High DTW values: {[f'{val:.4f}' for val in dtw_results[high_dtw_indices]]}\n")
        f.write(f"\n")
        
        f.write(f"Mean Arrival Point (all valid): {arrival_points_mean:.4f}\n\n")
        
        # Add valid trajectories (low DTW) lyapunov statistics
        valid_dtw_indices = np.where(dtw_results <= threshold_dtw)[0]
        valid_dtw_lyapunov = []
        valid_dtw_arrival_points = []
        for idx in valid_dtw_indices:
            if not np.isnan(lyapunov_results[idx]):
                valid_dtw_lyapunov.append(lyapunov_results[idx])
                valid_dtw_arrival_points.append(arrival_points[idx])
        
        if len(valid_dtw_lyapunov) > 0:
            valid_dtw_lyapunov = np.array(valid_dtw_lyapunov)
            valid_dtw_arrival_points = np.array(valid_dtw_arrival_points)
            valid_dtw_mean = np.mean(valid_dtw_lyapunov)
            valid_dtw_std = np.std(valid_dtw_lyapunov)
            valid_dtw_median = np.median(valid_dtw_lyapunov)
            
            # Calculate arrival points mean for valid low DTW trajectories
            valid_dtw_arrival_mean = np.mean(valid_dtw_arrival_points)
            
            valid_dtw_positive = valid_dtw_lyapunov[valid_dtw_lyapunov > 0]
            if len(valid_dtw_positive) > 0:
                valid_dtw_mean_positive = np.mean(valid_dtw_positive)
                valid_dtw_std_positive = np.std(valid_dtw_positive)
                valid_dtw_median_positive = np.median(valid_dtw_positive)
            else:
                valid_dtw_mean_positive = float('nan')
                valid_dtw_std_positive = float('nan')
                valid_dtw_median_positive = float('nan')
            
            f.write(f"Valid Trajectories (Low DTW <= {threshold_dtw:.4f}) Lyapunov Results:\n")
            f.write(f"  Mean: {valid_dtw_mean:.6f}\n")
            f.write(f"  Std:  {valid_dtw_std:.6f}\n")
            f.write(f"  Median: {valid_dtw_median:.6f}\n")
            f.write(f"  Mean (positive): {valid_dtw_mean_positive:.6f}\n")
            f.write(f"  Std (positive):  {valid_dtw_std_positive:.6f}\n")
            f.write(f"  Median (positive): {valid_dtw_median_positive:.6f}\n")
            f.write(f"  Valid low DTW trajectories: {len(valid_dtw_lyapunov)}/{len(valid_dtw_indices)}\n")
            f.write(f"  Valid low DTW trajectory indices: {list(valid_dtw_indices + 1)}\n")
            f.write(f"  Mean Arrival Point (valid low DTW): {valid_dtw_arrival_mean:.4f}\n\n")
        else:
            f.write(f"Valid Trajectories (Low DTW <= {threshold_dtw:.4f}) Lyapunov Results:\n")
            f.write(f"  No valid trajectories with low DTW found\n\n")
        
        f.write(f"Individual Results:\n")
        # Sort results by selected_indices for better readability
        # sorted_results = sorted(zip(unique_indices, lyapunov_results), key=lambda x: x[0])
        for idx, result in enumerate(lyapunov_results):
            dtw_val = dtw_results[idx]
            dtw_status = "High DTW" if dtw_val > threshold_dtw else "Low DTW"
            f.write(f"  Trajectory {idx+1}: Lyap={result:.6f}, DTW={dtw_val:.4f} ({dtw_status})\n")

    print(f"\nLyapunov results saved to: {lyapunov_save_path}")
else:
    print("\nNo valid Lyapunov exponent calculations!")

print(f"\nAnalysis completed!")
