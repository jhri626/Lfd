import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser
from data_preprocessing.data_preprocessor import DataPreprocessor
from agent.utils.dynamical_system_operations import normalize_state , denormalize_state
from state_dynamic import StateDynamic
import os
import time
from scipy.signal import butter, filtfilt




def create_directories(results_path):
    """
    Creates the requested directory and subfolders
    """
    try:
        if not os.path.exists(results_path + 'images/'):
            os.makedirs(results_path + 'images/')
            os.makedirs(results_path + 'stats/')
        print('Results directory created:', results_path)
    except FileExistsError:
        print('Results directory already exists:', results_path)

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

def generate_random_initial_states(num_states, params, demonstrations, mode='init', sampling_std=0.1):
    """
    Generate initial states based on mode
    mode='init': random states near starting point (existing logic)
    mode='grid': 25 points in a 5x5 grid between (-1,-1) and (1,1)
    mode='all': random sampling from all demonstration points with gaussian noise
    """
    if mode == 'init':
        # Original logic: random states near starting point
        demo = demonstrations[0]  # Single demonstration
        start_point = demo[0, :params.workspace_dimensions]  # First time step, position only
        
        print(f"Demonstration start point: {start_point}")
        print(f"Sampling radius around start point: {sampling_std}")
        
        # Generate random positions near the start point
        random_positions = []
        
        for i in range(num_states):
            # Generate random offset within the sampling radius
            random_offset = np.random.normal(
                loc=0.0, 
                scale=sampling_std, 
                size=params.workspace_dimensions
            )
            
            # Add offset to start point
            random_position = start_point + random_offset
            random_positions.append(random_position)
        
        random_positions = np.array(random_positions)
        
    elif mode == 'grid':
        # Grid mode: 25 points in 5x5 grid between (-1,-1) and (1,1)
        print("Generating 5x5 grid points between (-1,-1) and (1,1)")
        
        if params.workspace_dimensions == 2:
            # Create 5x5 grid
            x = np.linspace(-1, 1, 5)
            y = np.linspace(-1, 1, 5)
            xx, yy = np.meshgrid(x, y)
            
            # Flatten to get 25 points
            grid_x = xx.flatten()
            grid_y = yy.flatten()
            random_positions = np.column_stack([grid_x, grid_y])
            
            print(f"Generated 25 grid points (using first {min(num_states, 25)} points)")
            # If num_states < 25, use only the first num_states points
            # If num_states > 25, repeat the pattern or use only 25
            if num_states <= 25:
                random_positions = random_positions[:num_states]
            else:
                print(f"Warning: num_states ({num_states}) > 25, using only 25 grid points")
                random_positions = random_positions[:25]
                
        else:
            raise ValueError(f"Grid mode currently only supports 2D workspace, got {params.workspace_dimensions}D")
    
    elif mode == 'all':
        # Random sampling from all demonstration points with gaussian noise
        print(f"Generating {num_states} states by random sampling from all demonstration points with gaussian noise")
        print(f"Gaussian noise std: {sampling_std}")
        
        # Collect all positions from all demonstrations
        all_positions = []
        for demo in demonstrations:
            # Extract positions (first workspace_dimensions columns) from all time steps
            positions = demo[:, :params.workspace_dimensions]
            all_positions.append(positions)
        
        # Concatenate all positions
        all_positions = np.vstack(all_positions)  # Shape: (total_points, workspace_dimensions)
        print(f"Total demonstration points available: {len(all_positions)}")
        
        # Random sampling from all demonstration points
        random_positions = []
        for i in range(num_states):
            # Randomly select a point from all demonstration points
            random_idx = np.random.randint(0, len(all_positions))
            selected_point = all_positions[random_idx].copy()
            
            # Add gaussian noise
            gaussian_noise = np.random.normal(
                loc=0.0, 
                scale=sampling_std, 
                size=params.workspace_dimensions
            )
            
            # Add noise to selected point
            noisy_position = selected_point + gaussian_noise
            random_positions.append(noisy_position)
        
        random_positions = np.array(random_positions)
        
        print(f"Sample of selected demonstration points (before noise):")
        sample_indices = np.random.choice(len(all_positions), min(3, len(all_positions)), replace=False)
        for i, idx in enumerate(sample_indices):
            print(f"  Demo point {idx}: {all_positions[idx]}")
        print("\n")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'init', 'grid', or 'all'")
    
    # Add velocities if second order system
    if params.dynamical_system_order == 1:
        # First order: only positions
        initial_states = random_positions
    elif params.dynamical_system_order == 2:
        # Second order: positions + velocities (start with zero velocity)
        random_velocities = np.zeros((len(random_positions), params.workspace_dimensions))
        initial_states = np.concatenate([random_positions, random_velocities], axis=1)
    else:
        raise ValueError(f"Unsupported dynamical system order: {params.dynamical_system_order}")
    
    print(f"Generated {len(initial_states)} initial states with shape: {initial_states.shape}")
    if mode == 'grid':
        print("Grid points:")
        for i, state in enumerate(initial_states):
            if params.dynamical_system_order == 1:
                print(f"  Point {i+1}: ({state[0]:.2f}, {state[1]:.2f})")
            else:
                print(f"  Point {i+1}: pos=({state[0]:.2f}, {state[1]:.2f}), vel=({state[2]:.2f}, {state[3]:.2f})")
        print("\n")
    elif mode == 'all':
        print("Sample of generated initial states (with noise):")
        for i in range(min(5, len(initial_states))):
            if params.dynamical_system_order == 1:
                print(f"  State {i+1}: pos=({initial_states[i][0]:.3f}, {initial_states[i][1]:.3f})")
            else:
                print(f"  State {i+1}: pos=({initial_states[i][0]:.3f}, {initial_states[i][1]:.3f}), vel=({initial_states[i][2]:.3f}, {initial_states[i][3]:.3f})")
        print("\n")
    else:
        print("Sample initial states:")
        for i in range(min(3, len(initial_states))):
            print(f"  State {i+1}: {initial_states[i]}")
        print("\n")
    return initial_states

def plot_batch_trajectories_2d(demonstrations, batch_trajectories, save_path, title="Batch State Dynamics Simulation"):
    """
    Plot demonstrations and multiple simulated trajectories for 2D case
    """
    plt.figure(figsize=(12, 10))
    plt.rcParams.update({'font.size': 12})
    
    print(f"Plotting {len(demonstrations)} demonstrations and {len(batch_trajectories)} simulated trajectories")
    
    # Plot demonstrations in light gray
    for i, demo in enumerate(demonstrations):
        if demo.shape[0] >= 2:  # Ensure we have at least x, y coordinates
            plt.plot(demo[:, 0], demo[:, 1], 'lightgray', alpha=1.0, linewidth=10, 
                    label='Demonstrations' if i == 0 else "")
    
    # Plot simulated trajectories with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_trajectories)))
    
    for i, trajectory in enumerate(batch_trajectories):
        if trajectory.shape[1] >= 2:
            color = colors[i]
            plt.plot(trajectory[:, 0], trajectory[:, 1], color=color,
                    linewidth=2, alpha=0.8, label=f'Simulation {i+1}' if i < 5 else "")
            
            # Mark start points
            plt.plot(trajectory[0, 0], trajectory[0, 1], 'o', color=color, 
                    markersize=8, markerfacecolor='white', markeredgewidth=2)
            # Mark end points
            plt.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=color, 
                    markersize=8, markerfacecolor=color, alpha=0.8)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Batch trajectories plot saved to: {save_path}")

def plot_batch_state_evolution(batch_trajectories, save_path, dt=0.01):
    """
    Plot state evolution over time for multiple trajectories
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_trajectories)))
    
    for i, trajectory in enumerate(batch_trajectories):
        time_steps = np.arange(len(trajectory)) * dt
        color = colors[i]
        
        # Position plots
        if trajectory.shape[1] >= 2:
            axes[0, 0].plot(time_steps, trajectory[:, 0], color=color, linewidth=2, 
                           alpha=0.7, label=f'Traj {i+1}' if i < 5 else "")
            axes[0, 1].plot(time_steps, trajectory[:, 1], color=color, linewidth=2, 
                           alpha=0.7, label=f'Traj {i+1}' if i < 5 else "")
        
        # Velocity plots (if 2nd order system)
        if trajectory.shape[1] >= 4:
            axes[1, 0].plot(time_steps, trajectory[:, 2], color=color, linewidth=2, 
                           alpha=0.7, linestyle='--', label=f'Traj {i+1}' if i < 5 else "")
            axes[1, 1].plot(time_steps, trajectory[:, 3], color=color, linewidth=2, 
                           alpha=0.7, linestyle='--', label=f'Traj {i+1}' if i < 5 else "")
    
    axes[0, 0].set_title('X Position')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Y Position')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position')
    axes[0, 1].grid(True, alpha=0.3)
    
    if batch_trajectories[0].shape[1] >= 4:
        axes[1, 0].set_title('X Velocity')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Velocity')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Y Velocity')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Velocity')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Hide unused subplots for 1st order systems
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Batch state evolution plot saved to: {save_path}")

def analyze_batch_trajectories(batch_trajectories, params, state_dynamics):
    """
    Analyze batch trajectory results
    """
    print("\n=== Batch Trajectory Analysis ===")
    print(f"Number of trajectories: {len(batch_trajectories)}")
    
    final_distances = []
    trajectory_lengths = []
    
    for i, trajectory in enumerate(batch_trajectories):
        final_pos = trajectory[-1, :params.workspace_dimensions]
        goal_pos = state_dynamics.goal[:params.workspace_dimensions]
        distance = np.linalg.norm(final_pos - goal_pos)
        final_distances.append(distance)
        
        # Calculate trajectory length
        positions = trajectory[:, :params.workspace_dimensions]
        step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_length = np.sum(step_distances)
        trajectory_lengths.append(total_length)
        
        print(f"Trajectory {i+1}:")
        print(f"  Final position: {final_pos}")
        print(f"  Distance to goal: {distance:.6f}")
        print(f"  Total path length: {total_length:.3f}")
        print(f"  Simulation steps: {len(trajectory)}\n")
    
    print(f"\nSummary Statistics:")
    print(f"  Average final distance to goal: {np.mean(final_distances):.6f}")
    print(f"  Min final distance to goal: {np.min(final_distances):.6f}")
    print(f"  Max final distance to goal: {np.max(final_distances):.6f}")
    print(f"  Average trajectory length: {np.mean(trajectory_lengths):.3f}")
    print(f"  Goal position: {state_dynamics.goal[:params.workspace_dimensions]}\n")

def plot_trajectory_2d(demonstrations, simulated_trajectory, save_path, title="State Dynamics Simulation"):
    """
    Plot demonstrations and simulated trajectory for 2D case
    """
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    
    print(f"Plotting trajectory with {len(demonstrations)} demonstrations and simulated trajectory of shape {simulated_trajectory.shape}")
    
    # Plot demonstrations
    for i, demo in enumerate(demonstrations):
        if demo.shape[1] >= 2:  # Ensure we have at least x, y coordinates
            plt.plot(demo[:, 0], demo[:, 1], 'lightgray', alpha=0.7, linewidth=3, 
                    label='Demonstrations' if i == 0 else "")
    
    # Plot simulated trajectory
    if simulated_trajectory.shape[1] >= 2:
        plt.plot(simulated_trajectory[:, 0], simulated_trajectory[:, 1], 'r-', 
                linewidth=2, label='State Dynamics Simulation')
        
        # Mark start and end points
        plt.plot(simulated_trajectory[0, 0], simulated_trajectory[0, 1], 'go', 
                markersize=10, label='Start')
        plt.plot(simulated_trajectory[-1, 0], simulated_trajectory[-1, 1], 'ro', 
                markersize=10, label='End')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")

def plot_individual_trajectory_2d(demonstrations, trajectory, save_path, trajectory_idx, trajectory_type="Random"):
    """
    Plot demonstrations and a single trajectory for 2D case
    """
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    
    # Plot demonstrations in light gray
    for i, demo in enumerate(demonstrations):
        if demo.shape[0] >= 2:  # Ensure we have at least x, y coordinates
            plt.plot(demo[:, 0], demo[:, 1], 'lightgray', alpha=0.7, linewidth=3, 
                    label='Demonstrations' if i == 0 else "")
    
    # Plot individual trajectory
    if trajectory.shape[1] >= 2:
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', 
                linewidth=3, label=f'{trajectory_type} Trajectory {trajectory_idx}')
        
        # Mark start and end points
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', 
                markersize=12, label='Start')
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', 
                markersize=12, label='End')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'{trajectory_type} Trajectory {trajectory_idx} - State Dynamics Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_state_evolution(simulated_trajectory, save_path, dt=0.01):
    """
    Plot state evolution over time
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    time_steps = np.arange(len(simulated_trajectory)) * dt
    
    # Position plots
    if simulated_trajectory.shape[1] >= 2:
        axes[0, 0].plot(time_steps, simulated_trajectory[:, 0], 'b-', linewidth=2)
        axes[0, 0].set_title('X Position')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time_steps, simulated_trajectory[:, 1], 'r-', linewidth=2)
        axes[0, 1].set_title('Y Position')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Position')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Velocity plots (if 2nd order system)
    if simulated_trajectory.shape[1] >= 4:
        axes[1, 0].plot(time_steps, simulated_trajectory[:, 2], 'b--', linewidth=2)
        axes[1, 0].set_title('X Velocity')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Velocity')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time_steps, simulated_trajectory[:, 3], 'r--', linewidth=2)
        axes[1, 1].set_title('Y Velocity')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Velocity')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Hide unused subplots for 1st order systems
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"State evolution plot saved to: {save_path}")

def analyze_demonstrations(demonstrations, params):
    """
    Analyze demonstration data and print statistics
    """
    print("\\n=== Demonstration Analysis ===")
    print(f"Number of demonstrations: {len(demonstrations)}")
    
    for i, demo in enumerate(demonstrations):
        print(f"\\nDemo {i+1}:")
        print(f"  Shape: {demo.shape}")
        print(f"  Length: {len(demo)} points")
        
        if params.workspace_dimensions == 2:
            start_pos = demo[0, :2]
            end_pos = demo[-1, :2]
            print(f"  Start position: [{start_pos[0]:.3f}, {start_pos[1]:.3f}]")
            print(f"  End position: [{end_pos[0]:.3f}, {end_pos[1]:.3f}]")
            
            # Calculate trajectory statistics
            positions = demo[:, :params.workspace_dimensions]
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_length = np.sum(distances)
            print(f"  Total path length: {total_length:.3f}")
            print(f"  Average step size: {np.mean(distances):.6f}")

def plot_demonstrations_only(demonstrations, params, save_path):
    """
    Plot only the demonstrations to check data loading
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, demo in enumerate(demonstrations):
        if params.workspace_dimensions >= 2:
            # Extract positions (first workspace_dimensions columns)
            positions = demo[:, :params.workspace_dimensions]
            
            color = colors[i % len(colors)]
            plt.plot(positions[:, 0], positions[:, 1], color=color, linewidth=0.1, 
                    label=f'Demo {i+1}', marker='o', markersize=1)
            
            # Mark start and end
            plt.plot(positions[0, 0], positions[0, 1], color=color, marker='s', 
                    markersize=8, markerfacecolor='white', markeredgewidth=2)
            plt.plot(positions[-1, 0], positions[-1, 1], color=color, marker='*', 
                    markersize=12, markerfacecolor='white', markeredgewidth=2)
            plt.plot(positions[88, 0], positions[88, 1], color=color, marker='*', 
                    markersize=12, markerfacecolor='white', markeredgewidth=2)
            plt.plot(positions[355, 0], positions[355, 1], color=color, marker='*', 
                    markersize=12, markerfacecolor='white', markeredgewidth=2)
    
    print("positions shape:", positions.shape)
    print("demo shape:", demo.shape)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Original Demonstrations (Dataset: {params.dataset_name})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Demonstrations plot saved to: {save_path}")
    
def butter_lpf_uniform(signal, fs=1.0, L_min=20, order=4):
    """
    Apply zero-phase Butterworth LPF to a 1D signal.
    signal : 1D array (length T)
    fs     : sampling frequency (1/dt). If dt=1, set fs=1.
    L_min  : smooth out variations shorter than L_min samples
    order  : filter order
    """
    fc = fs / float(L_min)
    wn = fc / (0.5 * fs)
    if not (0 < wn < 1):
        raise ValueError("Cutoff must satisfy 0 < fc < fs/2.")
    b, a = butter(order, wn, btype="low", analog=False)
    return filtfilt(b, a, signal, method="pad")

def lpf_demonstrations(demo_array, fs=1.0, L_min=20, order=4):
    """
    Apply LPF to demonstrations array of shape (N, T, D).
    Returns filtered array of the same shape.
    """
    N, T, D = demo_array.shape
    demo_f = np.zeros_like(demo_array)
    for n in range(N):
        for d in range(D):
            demo_f[n, :, d] = butter_lpf_uniform(demo_array[n, :, d],
                                                fs=fs, L_min=L_min, order=order)
    return demo_f

def cal_lyapunov_exponent(distance_traj, terminal_time=1, eps=1e-6):
    '''
    From BCSDM curve_analysis.py
    '''
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
                break
            else:
                distance_truncated = distance_traj[i]
                print
        
        t_linspace = torch.linspace(0, terminal_time, len(distance_truncated)).to(distance_traj)
        log_dist_traj = torch.log(distance_truncated).unsqueeze(-1) # ((nd), nt, 1)
        log_d0 = log_dist_traj[..., 0:1, 0:1]
        lamb = (torch.pinverse(t_linspace.unsqueeze(-1)) 
                @ (log_d0 - log_dist_traj)).squeeze().squeeze()
        lamb_list.append(lamb)
    lamb = torch.tensor(lamb_list)
    # shape: (nd, ) or ()
    return lamb


def main():
    # Get arguments
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, default='2nd_order_2D', help='Parameter file name')
    parser.add_argument('--results-base-directory', type=str, default='./', help='Base directory for results')
    parser.add_argument('--simulation-steps', type=int, default=6000, help='Number of simulation steps')
    parser.add_argument('--sample', type=int, default=25, help='Number of random initial states to generate')
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
    params.results_path = 'exp_result/state_dynamics/'+ params.metric +'/new_metric'+'/eta' + str(params.eta) +'/'+ params.selected_primitives_ids + '/'
    
    # Create results directory
    create_directories(params.results_path)
    
    print(f"Dataset: {params.dataset_name}")
    print(f"Selected primitives: {params.selected_primitives_ids}")
    print(f"Workspace dimensions: {params.workspace_dimensions}")
    print(f"Dynamical system order: {params.dynamical_system_order}")
    print(f"Results will be saved to: {params.results_path}")
    
    # Load and preprocess data using existing framework
    print("\nLoading and preprocessing data...")
    data_preprocessor = DataPreprocessor(params=params, verbose=True)
    data = data_preprocessor.run()
    
    # Extract relevant data
    # demonstrations = data['demonstrations raw']
    demonstrations_norm = data['demonstrations train'][:,:,:,0]
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
    demo_plot_path = params.results_path + 'images/demonstrations_only.png'
    plot_demonstrations_only(demonstrations_denorm, params, demo_plot_path)
    
    print(f"Loaded {len(demonstrations_denorm)} demonstrations")
    
    if params.dynamical_system_order == 2:
        print("\nNote: Dynamical system order is set to 2, using second-order dynamics.")
        velocity = (demonstrations_norm[:, 1:, :] - demonstrations_norm[:, :-1, :])/ params.delta_t
        # vel_start = (demonstrations_norm[:, 0, :] - demonstrations_norm[:, 1, :])/ params.delta_t
        # vel_start = np.expand_dims(vel_start, axis=1)  # Add as first point
        velocity = np.concatenate((velocity ,np.zeros((velocity.shape[0], 1, params.workspace_dimensions))), axis=1)  # Add zero velocity for last point
        # velocity = np.concatenate((vel_start,velocity ,np.zeros((velocity.shape[0], 1, params.workspace_dimensions))), axis=1)  # Add zero velocity for last point
        traj = np.concatenate((demonstrations_norm, velocity), axis=2)  # Combine position and velocity
    else:
        print("\nNote: Dynamical system order is set to 1, using first-order dynamics.")
        traj = demonstrations_norm
            
    
    # Initialize state dynamics
    print("\nInitializing state dynamics...")
    init_state_dynamics = StateDynamic(params ,traj,vel_norm_stat=[vel_min, vel_max], sample=args.sample, max_steps=args.simulation_steps)
    grid_state_dynamics = StateDynamic(params ,traj,vel_norm_stat=[vel_min, vel_max], sample=args.sample, max_steps=args.simulation_steps)
    all_state_dynamics = StateDynamic(params ,traj,vel_norm_stat=[vel_min, vel_max], sample=args.sample, max_steps=args.simulation_steps)

    # Generate random initial states
    print
    print(f"Generating {args.sample} random initial states...")
    random_initial_states = generate_random_initial_states(args.sample, params, demonstrations_norm, mode='init', sampling_std=0.1)
    grid_initial_states = generate_random_initial_states(args.sample, params, demonstrations_norm, mode='grid')
    all_initial_states = generate_random_initial_states(args.sample, params, demonstrations_norm, mode='all')
    
    print("init_state",random_initial_states.shape,grid_initial_states.shape, all_initial_states.shape)

    # Run batch simulation
    print(f"\nRunning batch simulation for {args.simulation_steps} steps...")
    print(f"Goal position: {init_state_dynamics.goal[:params.workspace_dimensions]}")
    
    start_time = time.time()
    batch_trajectories = init_state_dynamics.simulate(random_initial_states )
    grid_trajectories = grid_state_dynamics.simulate(grid_initial_states)
    all_trajectories = all_state_dynamics.simulate(all_initial_states)
    simulation_time = time.time() - start_time
    
    print(f"Batch simulation completed in {simulation_time:.2f} seconds")
    print(f"Batch trajectories shape: {batch_trajectories.shape}")
    
    # Convert batch trajectories to list of individual trajectories for easier handling
    individual_trajectories = [batch_trajectories[i] for i in range(len(batch_trajectories))]
    individual_trajectories = np.array(individual_trajectories)  # Convert to numpy array if needed
    
    # Convert grid trajectories to list of individual trajectories
    individual_grid_trajectories = [grid_trajectories[i] for i in range(len(grid_trajectories))]
    individual_grid_trajectories = np.array(individual_grid_trajectories)  # Convert to numpy array if needed

    individual_all_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
    individual_all_trajectories = np.array(individual_all_trajectories)  # Convert to numpy array if needed

    # Analyze batch results
    analyze_batch_trajectories(individual_trajectories, params, init_state_dynamics)

    init_dist = torch.from_numpy(init_state_dynamics.get_dist_history())
    grid_dist = torch.from_numpy(grid_state_dynamics.get_dist_history())
    all_dist = torch.from_numpy(all_state_dynamics.get_dist_history())

    # Save distance histories as npy files
    init_dist_save_path = params.results_path + 'stats/init_dist_history.npy'
    grid_dist_save_path = params.results_path + 'stats/grid_dist_history.npy'
    all_dist_save_path = params.results_path + 'stats/all_dist_history.npy'
    
    np.save(init_dist_save_path, init_dist.numpy())
    np.save(grid_dist_save_path, grid_dist.numpy())
    np.save(all_dist_save_path, all_dist.numpy())
    
    print(f"Init distance history saved to: {init_dist_save_path}")
    print(f"Grid distance history saved to: {grid_dist_save_path}")
    print(f"All distance history saved to: {all_dist_save_path}")

    init_lyapunov = cal_lyapunov_exponent(init_dist.T,eps=1e-2)
    # grid_lyapunov = cal_lyapunov_exponent(grid_dist.T,eps=1e-2)
    # all_lyapunov = cal_lyapunov_exponent(all_dist.T,eps=1e-2)

    print("\nLyapunov Exponents:")
    print(init_dist.shape)
    print(init_lyapunov.mean())
    print(init_dist.min())

    # Generate plots
    print("\nGenerating plots...")
    
    individual_trajectories[:,:,:2] = denormalize_state(individual_trajectories[:,:,:2], x_min, x_max)
    individual_grid_trajectories[:,:,:2] = denormalize_state(individual_grid_trajectories[:,:,:2], x_min, x_max)
    individual_all_trajectories[:,:,:2] = denormalize_state(individual_all_trajectories[:,:,:2], x_min, x_max)
    # Batch 2D trajectory plot
    if params.workspace_dimensions == 2:
        batch_plot_path_2d_init = params.results_path + 'images/batch_trajectories_2d_init.png'
        batch_plot_path_2d_grid = params.results_path + 'images/batch_trajectories_2d_grid.png'
        batch_plot_path_2d_all = params.results_path + 'images/batch_trajectories_2d_all.png'
        plot_batch_trajectories_2d(demonstrations_denorm, individual_trajectories, batch_plot_path_2d_init, 
                                  f"Batch State Dynamics Simulation ({args.sample} random starts)\n")
        plot_batch_trajectories_2d(demonstrations_denorm, individual_grid_trajectories, batch_plot_path_2d_grid, 
                                  f"Batch State Dynamics Simulation ({args.sample} grid starts)\n")
        plot_batch_trajectories_2d(demonstrations_denorm, individual_all_trajectories, batch_plot_path_2d_all, 
                                  f"Batch State Dynamics Simulation ({args.sample} all starts)\n")

    # Batch state evolution plot
    batch_plot_path_evolution = params.results_path + 'images/batch_state_evolution.png'
    plot_batch_state_evolution(individual_trajectories, batch_plot_path_evolution, params.delta_t)
    
    # Save batch trajectory data
    batch_trajectory_save_path = params.results_path + 'stats/batch_trajectories.npy'
    np.save(batch_trajectory_save_path, batch_trajectories)
    print(f"Batch trajectory data saved to: {batch_trajectory_save_path}")
    
    # Save individual trajectories
    individual_traj_dir = params.results_path + 'stats/individual_trajectories/'
    if not os.path.exists(individual_traj_dir):
        os.makedirs(individual_traj_dir)
    
    for i, trajectory in enumerate(individual_trajectories):
        individual_traj_path = individual_traj_dir + f'trajectory_random_{i+1:03d}.npy'
        np.save(individual_traj_path, trajectory)
    
    print(f"Individual random trajectories saved to: {individual_traj_dir}")
    print(f"Saved {len(individual_trajectories)} individual random trajectory files")
    
    # Save individual trajectory plots (Random)
    individual_plots_dir = params.results_path + 'images/individual_trajectories/'
    if not os.path.exists(individual_plots_dir):
        os.makedirs(individual_plots_dir)
    
    print(f"\nGenerating individual trajectory plots...")
    for i, trajectory in enumerate(individual_trajectories):
        plot_path = individual_plots_dir + f'trajectory_random_{i+1:03d}.png'
        plot_individual_trajectory_2d(demonstrations_denorm, trajectory, plot_path, i+1, "Random")
    
    print(f"Individual random trajectory plots saved to: {individual_plots_dir}")
    print(f"Generated {len(individual_trajectories)} individual random trajectory plots\n")
    
    # Save individual grid trajectories
    individual_grid_traj_dir = params.results_path + 'stats/individual_grid_trajectories/'
    if not os.path.exists(individual_grid_traj_dir):
        os.makedirs(individual_grid_traj_dir)
    
    for i, trajectory in enumerate(individual_grid_trajectories):
        individual_grid_traj_path = individual_grid_traj_dir + f'trajectory_grid_{i+1:03d}.npy'
        np.save(individual_grid_traj_path, trajectory)
    
    print(f"Individual grid trajectories saved to: {individual_grid_traj_dir}")
    print(f"Saved {len(individual_grid_trajectories)} individual grid trajectory files\n")
    
    # Save individual grid trajectory plots
    individual_grid_plots_dir = params.results_path + 'images/individual_grid_trajectories/'
    if not os.path.exists(individual_grid_plots_dir):
        os.makedirs(individual_grid_plots_dir)
    
    for i, trajectory in enumerate(individual_grid_trajectories):
        plot_path = individual_grid_plots_dir + f'trajectory_grid_{i+1:03d}.png'
        plot_individual_trajectory_2d(demonstrations_denorm, trajectory, plot_path, i+1, "Grid")
    
    print(f"Individual grid trajectory plots saved to: {individual_grid_plots_dir}")
    print(f"Generated {len(individual_grid_trajectories)} individual grid trajectory plots\n")


    individual_all_traj_dir = params.results_path + 'stats/individual_all_trajectories/'
    if not os.path.exists(individual_all_traj_dir):
        os.makedirs(individual_all_traj_dir)

    for i, trajectory in enumerate(individual_all_trajectories):
        individual_all_traj_path = individual_all_traj_dir + f'trajectory_all_{i+1:03d}.npy'
        np.save(individual_all_traj_path, trajectory)

    print(f"Individual all trajectories saved to: {individual_all_traj_dir}")
    print(f"Saved {len(individual_all_trajectories)} individual all trajectory files")

    # Save individual all trajectory plots
    individual_all_plots_dir = params.results_path + 'images/individual_all_trajectories/'
    if not os.path.exists(individual_all_plots_dir):
        os.makedirs(individual_all_plots_dir)

    for i, trajectory in enumerate(individual_all_trajectories):
        plot_path = individual_all_plots_dir + f'trajectory_all_{i+1:03d}.png'
        plot_individual_trajectory_2d(demonstrations_denorm, trajectory, plot_path, i+1, "All")

    print(f"Individual all trajectory plots saved to: {individual_all_plots_dir}")
    print(f"Generated {len(individual_all_trajectories)} individual all trajectory plots")

    # Save random initial states
    initial_states_save_path = params.results_path + 'stats/random_initial_states.npy'
    np.save(initial_states_save_path, random_initial_states)
    print(f"Random initial states saved to: {initial_states_save_path}\n")
    
    # Save grid initial states
    grid_states_save_path = params.results_path + 'stats/grid_initial_states.npy'
    np.save(grid_states_save_path, grid_initial_states)
    print(f"Grid initial states saved to: {grid_states_save_path}\n")

    
    all_states_save_path = params.results_path + 'stats/all_initial_states.npy'
    np.save(all_states_save_path, all_initial_states)
    print(f"All initial states saved to: {all_states_save_path}\n")

    # Also run single trajectory simulation for comparison (using first random state)
    print("\nRunning single trajectory simulation for comparison...")
    single_initial_state = random_initial_states[0]
    single_trajectory = init_state_dynamics.simulate(single_initial_state)

    # Single trajectory plots
    if params.workspace_dimensions == 2:
        single_plot_path_2d = params.results_path + 'images/single_trajectory_2d.png'
        plot_trajectory_2d(demonstrations_denorm, single_trajectory, single_plot_path_2d, 
                          "Single State Dynamics Simulation")
    
    single_plot_path_evolution = params.results_path + 'images/single_state_evolution.png'
    plot_state_evolution(single_trajectory, single_plot_path_evolution, params.delta_t)
    
    print("\n=== Simulation completed successfully! ===")
    print(f"Results saved in: {params.results_path}")
    print("\nState dynamics used: dx/dt = (x_goal - x)")
    print(f"Goal position: {init_state_dynamics.goal[:params.workspace_dimensions]}")
    print(f"Number of random trajectories simulated: {args.sample}")
    print(f"Simulation time: {simulation_time:.2f} seconds")

if __name__ == "__main__":
    main()