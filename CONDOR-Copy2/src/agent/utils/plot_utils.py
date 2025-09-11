import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import numpy as np

def plot_batch_trajectories_2d(demonstrations, batch_trajectories, save_path, title="Batch State Dynamics Simulation", dist_set=None):
    """
    Plot demonstrations and multiple simulated trajectories for 2D case
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.grid(linestyle='--')
    plt.rcParams.update({'font.size': 12})
    
    print(f"Plotting {len(demonstrations)} demonstrations and {len(batch_trajectories)} simulated trajectories")
    
    # Define brighter and more vibrant red and green for the colormap
    vibrant_red = "#ff4d4d"   # brighter red (high distance)
    vibrant_green = "#4dff4d" # brighter green (low distance)

    # Create a custom colormap with these vibrant tones
    vibrant_cmap = mcolors.LinearSegmentedColormap.from_list("vibrant_red_green", [vibrant_green, vibrant_red])
    
    # Plot demonstrations (background trajectories in light gray)
    for i, demo in enumerate(demonstrations):
        if demo.shape[0] >= 2:  # Ensure we have at least x, y coordinates
            ax.plot(demo[:, 0], demo[:, 1], color='lightgray', alpha=1.0, linewidth=6, 
                   label='Demonstrations' if i == 0 else "", zorder=1)
    
    # Plot simulated trajectories with colors based on distances
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_trajectories)))
    all_line_collections = []  # Store all line collections for unified colorbar
    
    for i, trajectory in enumerate(batch_trajectories):
        if trajectory.shape[1] >= 2:
            color = colors[i]
            
            if dist_set is not None:
                # Get distance data for this trajectory
                trajectory_dist = dist_set[:,i] if i < len(dist_set) else None
                if trajectory_dist is not None and len(trajectory_dist) > 0:
                    # Convert tensor to numpy if needed
                    if hasattr(trajectory_dist, 'detach'):
                        trajectory_dist = trajectory_dist.detach().cpu().numpy()
                    
                    # Create line segments for LineCollection
                    
                    points = trajectory[:, :2]  # Use only x, y coordinates
                    points = points.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1,:], points[1:,:]], axis=1)
                    
                    # Use distances corresponding to each segment (use distances of start points)
                    segment_distances = trajectory_dist[:-1]
                    
                    # Individual trajectory min-max normalization
                    traj_min_dist = np.min(trajectory_dist)
                    traj_max_dist = np.max(trajectory_dist)
                    
                    # Normalize distances for this trajectory only (0 = green, 1 = red)
                    if traj_max_dist - traj_min_dist > 1e-8:  # Avoid division by zero
                        norm_distances = np.log((segment_distances - traj_min_dist) / (traj_max_dist - traj_min_dist) + 0.1 + 1e-8)
                    else:
                        norm_distances = np.zeros_like(segment_distances)  # All segments same color if no variation
                    
                    # Create LineCollection with gradient colors
                    
                    lc = LineCollection(segments, cmap=vibrant_cmap, array=norm_distances, linewidth=3, zorder=11)
                    line = ax.add_collection(lc)
                    all_line_collections.append(lc)
                else:
                    # No distance data, plot with solid color
                    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color,
                           linewidth=2, alpha=0.8, label=f'Simulation {i+1}' if i < 5 else "")
                
                # Mark start points
                ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', color=color, 
                       markersize=8, markerfacecolor='white', markeredgewidth=2, zorder=12)
                # Mark end points
                ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=color, 
                       markersize=8, markerfacecolor=color, alpha=0.8, zorder=12)
            else:
                # No distance set provided, plot with solid color
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=color,
                       linewidth=2, alpha=0.8, label=f'Simulation {i+1}' if i < 5 else "")
                
                # Mark start points
                ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', color=color, 
                       markersize=8, markerfacecolor='white', markeredgewidth=2)
                # Mark end points
                ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=color, 
                       markersize=8, markerfacecolor=color, alpha=0.8)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    # ax.set_aspect('equal')
    
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
        traj_np = _to_numpy(trajectory)
        final_pos = traj_np[-1, :params.workspace_dimensions]
        goal_pos = _to_numpy(state_dynamics.goal)[:params.workspace_dimensions]
        distance = np.linalg.norm(final_pos - goal_pos)
        final_distances.append(distance)
        
        # Calculate trajectory length
        positions = traj_np[:, :params.workspace_dimensions]
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
    
    vibrant_red = "#ff4d4d"   # brighter red (high distance)
    vibrant_green = "#4dff4d" # brighter green (low distance)

    # Create a custom colormap with these vibrant tones
    vibrant_cmap = mcolors.LinearSegmentedColormap.from_list("vibrant_red_green", [vibrant_green, vibrant_red])
    
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
    # plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")

def plot_individual_trajectory_2d(demonstrations, trajectory, save_path, trajectory_idx, trajectory_type="Random", dist_set=None):
    """
    Plot demonstrations and a single trajectory for 2D case
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    
    # Define brighter and more vibrant red and green for the colormap
    vibrant_red = "#ff4d4d"   # brighter red (high distance)
    vibrant_green = "#4dff4d" # brighter green (low distance)
    
    # Create a custom colormap with these vibrant tones
    vibrant_cmap = mcolors.LinearSegmentedColormap.from_list("vibrant_red_green", [vibrant_green, vibrant_red])
    
    # Plot demonstrations in light gray
    for i, demo in enumerate(demonstrations):
        if demo.shape[0] >= 2:  # Ensure we have at least x, y coordinates
            ax.plot(demo[:, 0], demo[:, 1], 'lightgray', alpha=0.7, linewidth=3, 
                    label='Demonstrations' if i == 0 else "")
    
    # Plot individual trajectory
    if trajectory.shape[1] >= 2:
        if dist_set is not None and len(dist_set) > 0:
            # Convert tensor to numpy if needed
            if hasattr(dist_set, 'detach'):
                trajectory_dist = dist_set.detach().cpu().numpy()
            else:
                trajectory_dist = dist_set
            
            # Create line segments for LineCollection
            points = trajectory[:, :2]  # Use only x, y coordinates
            points = points.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1,:], points[1:,:]], axis=1)
            
            # Use distances corresponding to each segment
            segment_distances = trajectory_dist[:-1]
            
            # Individual trajectory min-max normalization
            traj_min_dist = np.min(trajectory_dist)
            traj_max_dist = np.max(trajectory_dist)
            
            # Normalize distances for this trajectory only (0 = green, 1 = red)
            if traj_max_dist - traj_min_dist > 1e-8:  # Avoid division by zero
                norm_distances = np.log((segment_distances - traj_min_dist) / (traj_max_dist - traj_min_dist) + 0.1 + 1e-8)
            else:
                norm_distances = np.zeros_like(segment_distances)  # All segments same color if no variation
            
            # Create LineCollection with gradient colors
            lc = LineCollection(segments, cmap=vibrant_cmap, array=norm_distances, linewidth=3, zorder=11)
            line = ax.add_collection(lc)
        else:
            # No distance data, plot with solid color
            ax.plot(trajectory[:, 0], trajectory[:, 1], color='#ff4d4d', 
                    linewidth=3, label=f'{trajectory_type} Trajectory {trajectory_idx}')
        
        # Mark start and end points
        ax.plot(trajectory[0, 0], trajectory[0, 1], color='#4dff4d', marker='o',
                markersize=12, label='Start', zorder=12)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], color='#ff4d4d', marker='o',
                markersize=12, label='End', zorder=12)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'{trajectory_type} Trajectory {trajectory_idx} - State Dynamics Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
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
            # plt.plot(positions[88, 0], positions[88, 1], color=color, marker='*', 
            #         markersize=12, markerfacecolor='white', markeredgewidth=2)
            # plt.plot(positions[355, 0], positions[355, 1], color=color, marker='*', 
            #         markersize=12, markerfacecolor='white', markeredgewidth=2)
    
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

# -----------------------------
# helpers
# -----------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from agent.utils.utils import _to_numpy

# -----------------------------
# helpers
# -----------------------------

def _q_to_xyz(q):
    """q = (..., 2) with (theta, phi) in radians -> (..., 3) on S^2 (normalized)."""
    q = _to_numpy(q)
    if q.shape[-1] < 2:
        raise ValueError("q must have at least 2 columns: (theta, phi).")
    theta = q[..., 0]
    phi   = q[..., 1]
    st = np.sin(theta)
    x = st * np.cos(phi)
    y = st * np.sin(phi)
    z = np.cos(theta)
    xyz = np.stack([x, y, z], axis=-1)
    # normalize to unit sphere
    n = np.linalg.norm(xyz, axis=-1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return xyz / n

from mpl_toolkits.mplot3d import proj3d
def _project_xyz_to_fig_coords(xyz, ax3d, fig):
    """Project Nx3 world coords to normalized figure coords in [0,1]^2."""
    M = ax3d.get_proj()
    x2d, y2d, _ = proj3d.proj_transform(xyz[:, 0], xyz[:, 1], xyz[:, 2], M)
    disp = ax3d.transData.transform(np.column_stack([x2d, y2d]))      # to display pixels
    fig_coords = fig.transFigure.inverted().transform(disp)           # to [0,1]^2
    return fig_coords  # (N, 2)

def _set_axes_equal_3d(ax):
    """Equal aspect for 3D axes."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_mid = np.mean(z_limits)
    R = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - R, x_mid + R])
    ax.set_ylim3d([y_mid - R, y_mid + R])
    ax.set_zlim3d([z_mid - R, z_mid + R])

def _draw_sphere(ax,
                 surface_alpha=0.5,
                 surface_color="lightblue",
                 light_dir=(0.25, 0.35, 0.9),
                 ambient=0.65,
                 diffuse=0.45,
                 add_wire=True,
                 wire_alpha=0.15,
                 wire_lw=0.3):
    """Render a lit unit sphere plus optional wireframe.
    Returns (surface_handle, wire_handle) so caller can tweak zorders if needed.
    """
    nu, nv = 160, 80
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    # simple lambert lighting
    N = np.stack([xs, ys, zs], axis=-1)
    N /= np.clip(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12, None)
    L = np.asarray(light_dir, dtype=float); L /= np.linalg.norm(L)
    lambert = np.maximum(0.0, N @ L)
    intensity = np.clip(ambient + diffuse * lambert, 0.0, 1.0)
    base_rgb = np.asarray(matplotlib.colors.to_rgb(surface_color), dtype=float).reshape(1, 1, 3)
    facecolors = intensity[..., None] * base_rgb
    if surface_alpha < 1.0:
        fc = np.concatenate([facecolors, np.full((*facecolors.shape[:2], 1), surface_alpha)], axis=-1)
    else:
        fc = facecolors

    # Draw surface with very low zorder so it stays behind everything else
    surf = ax.plot_surface(xs, ys, zs,
                           facecolors=fc, rstride=1, cstride=1,
                           linewidth=0, antialiased=True, shade=False,
                           zorder=-1_000_000)

    # Hint the internal polygon sorting to push the surface further back
    # ('min' tends to make it sort as farther away)
    if hasattr(surf, "set_zsort"):
        surf.set_zsort("min")

    wire = None
    if add_wire:
        u_w = np.linspace(0, 2*np.pi, 24)
        v_w = np.linspace(0, np.pi, 12)
        xw = np.outer(np.cos(u_w), np.sin(v_w))
        yw = np.outer(np.sin(u_w), np.cos(v_w)) * 0.0  # not used, will reset below
        # correct yw / zw
        yw = np.outer(np.sin(u_w), np.sin(v_w))
        zw = np.outer(np.ones_like(u_w), np.cos(v_w))

        # Wireframe with slightly higher (but still very low) zorder
        wire = ax.plot_wireframe(xw, yw, zw,
                                 color="k", linewidth=wire_lw,
                                 alpha=wire_alpha, zorder=-900_000)

    return surf, wire


# -----------------------------
# S^2 plotter (same I/O spirit as R^2 version)
# -----------------------------
def plot_batch_trajectories_2d_sphere(
    demonstrations,
    batch_trajectories,
    save_path,
    title="Batch State Dynamics Simulation (S^2)",
    dist_set=None,
    traj_lift=1.00,
):
    vibrant_red = "#ff4d4d"
    vibrant_green = "#4dff4d"
    vibrant_cmap = mcolors.LinearSegmentedColormap.from_list(
        "vibrant_red_green", [vibrant_green, vibrant_red]
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(linestyle="--", alpha=0.2)
    plt.rcParams.update({"font.size": 12})
    ax.set_title(title)

    # Draw opaque sphere using your existing function
    surf, wire = _draw_sphere(
        ax,
        surface_alpha=1.0,
        surface_color="lightblue",
        light_dir=(0.25, 0.35, 0.9),
        ambient=0.65,
        diffuse=0.45,
        add_wire=True,
        wire_alpha=0.15,
        wire_lw=0.3,
    )

    # 2D overlay for trajectories
    plt.tight_layout()
    fig.canvas.draw()  # ensure ax position is finalized
    overlay = fig.add_axes(ax.get_position(), frameon=False)
    overlay.set_axis_off()
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.invert_yaxis()

    # Demos on overlay
    for demo in demonstrations:
        if demo is None:
            continue
        q_demo = _to_numpy(demo)
        if q_demo.ndim != 2 or q_demo.shape[1] < 2:
            continue
        q_demo = q_demo[:, :2]
        xyz_demo = _q_to_xyz(q_demo)
        xyz_demo /= np.linalg.norm(xyz_demo, axis=1, keepdims=True)
        xyz_demo *= float(traj_lift)
        P_demo = _project_xyz_to_fig_coords(xyz_demo, ax, fig)
        overlay.plot(P_demo[:, 0], P_demo[:, 1],
                     color="red", linewidth=2.5, zorder=10000)
        overlay.scatter(P_demo[-1, 0], P_demo[-1, 1],
                        color="red", marker="*", s=60, zorder=10000)

    # Distances (optional)
    dist_np = None
    if dist_set is not None:
        dist_np = _to_numpy(dist_set)
        if dist_np.ndim == 1 and len(batch_trajectories) == 1:
            dist_np = dist_np[:, None]

    # Trajectories on overlay
    for i, traj in enumerate(batch_trajectories):
        q_traj = _to_numpy(traj)
        if q_traj.ndim != 2 or q_traj.shape[1] < 2:
            continue
        q_traj = q_traj[:, :2]
        xyz = _q_to_xyz(q_traj)
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        xyz *= float(traj_lift)

        P = _project_xyz_to_fig_coords(xyz, ax, fig)

        if dist_np is not None:
            d_all = dist_np[:, i] if dist_np.shape[1] > i else None
            if d_all is not None:
                d_seg = d_all[:-1]
                dmin, dmax = np.min(d_all), np.max(d_all)
                if dmax - dmin > 1e-8:
                    norm_vals = np.log((d_seg - dmin) /
                                       (dmax - dmin) + 0.1 + 1e-8)
                else:
                    norm_vals = np.zeros_like(d_seg)

                seg2d = np.stack([P[:-1, :], P[1:, :]], axis=1)
                lc = LineCollection(seg2d, cmap=vibrant_cmap,
                                    linewidths=1.0, zorder=9999)
                lc.set_array(norm_vals)
                overlay.add_collection(lc)
            else:
                overlay.plot(P[:, 0], P[:, 1],
                             color="blue", linewidth=0.8, zorder=9998)
        else:
            overlay.plot(P[:, 0], P[:, 1],
                         color="blue", linewidth=0.8, zorder=9998)

        # start/end markers
        overlay.scatter(P[0, 0], P[0, 1],
                        s=10, color="blue", edgecolors="white",
                        linewidths=1, zorder=10000)
        overlay.scatter(P[-1, 0], P[-1, 1],
                        s=10, color="blue", zorder=10000)

    # Poles (still in 3D, optional)
    ax.scatter(0, 0, 1.02, s=70, color="blue", zorder=15)
    ax.scatter(0, 0, -1.02, s=70, color="green", zorder=15)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    try:
        _set_axes_equal_3d(ax)
    except Exception:
        pass

    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] S^2 batch trajectories plot saved to: {save_path}")

def plot_individual_trajectory_2d_sphere(
    demonstrations,
    trajectory,
    save_path,
    trajectory_idx,
    trajectory_type="Random",
    dist_set=None,
    title=None,
    traj_lift=1.00  # multiply Cartesian coords to slightly lift curves above the surface
):
    """
    Plot demonstrations and a single trajectory on the unit sphere S^2.

    Parameters
    ----------
    demonstrations : iterable of arrays
        Each demo is (T,2) or (T,k>=2) with columns [theta, phi] in radians.
        If (T,k>=2), only the first two columns are used.
    trajectory : array-like, shape (T,2) or (T,k>=2)
        Single trajectory in spherical angles [theta, phi] in radians.
    save_path : str
        Output image path.
    trajectory_idx : int
        Index used in the title/legend.
    trajectory_type : str
        Label prefix (e.g., "Random", "Greedy", etc.).
    dist_set : array-like or tensor, optional
        Per-timestep distances of shape (T,) for the given trajectory
        used to color each segment (low=green, high=red).
    title : str, optional
        Figure title. If None, a default based on trajectory_type/idx is used.
    traj_lift : float
        Factor applied to Cartesian coordinates to avoid z-fighting with the sphere surface.
        Use 1.0 to lie exactly on the unit sphere.
    """
    # Colormap consistent with your 2D version
    vibrant_red = "#ff4d4d"    # high distance
    vibrant_green = "#4dff4d"  # low distance
    vibrant_cmap = mcolors.LinearSegmentedColormap.from_list(
        "vibrant_red_green", [vibrant_green, vibrant_red]
    )

    # Create 3D figure/axes
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    plt.rcParams.update({'font.size': 12})
    ax.grid(linestyle="--", alpha=0.2)

    # Title
    if title is None:
        title = f"{trajectory_type} Trajectory {trajectory_idx} - State Dynamics Simulation (S^2)"
    ax.set_title(title)

    # Background sphere
    _draw_sphere(
        ax,
        surface_alpha=0.12,       # subtle background
        surface_color="lightblue",
        add_wire=True,
        wire_alpha=0.10,
        wire_lw=0.4
    )

    # Plot demonstrations as light gray geodesic-like polylines
    for i, demo in enumerate(demonstrations):
        if demo is None:
            continue
        q_demo = _to_numpy(demo)
        if q_demo.ndim != 2 or q_demo.shape[1] < 2:
            continue
        q_demo = q_demo[:, :2]
        xyz_demo = _q_to_xyz(q_demo) * float(traj_lift)

        if xyz_demo.shape[0] >= 2:
            ax.plot(
                xyz_demo[:, 0], xyz_demo[:, 1], xyz_demo[:, 2],
                color="lightgray", alpha=0.9, linewidth=2.5, zorder=10,
                label="Demonstrations" if i == 0 else None
            )

    # Plot single trajectory
    q_traj = _to_numpy(trajectory)
    assert q_traj.ndim == 2 and q_traj.shape[1] >= 2, "trajectory must be (T,2+) with (theta, phi)"
    q_traj = q_traj[:, :2]
    xyz = _q_to_xyz(q_traj) * float(traj_lift)
    T = xyz.shape[0]

    if dist_set is not None and T >= 2:
        # Convert distance to numpy array and handle shape (T,)
        d = dist_set
        if hasattr(d, "detach"):  # torch tensor
            d = d.detach().cpu().numpy()
        d = np.asarray(d).reshape(-1)
        

        # Build 3D segments between consecutive points
        segs = np.stack([xyz[:-1, :], xyz[1:, :]], axis=1)  # (T-1, 2, 3)

        # Log-based per-trajectory normalization (consistent with your 2D)
        dmin, dmax = np.min(d), np.max(d)
        if dmax - dmin > 1e-8:
            # Use distances at segment starts to color each segment
            d_seg = d[:-1]
            norm_vals = np.log((d_seg - dmin) / (dmax - dmin) + 0.1 + 1e-8)
        else:
            norm_vals = np.zeros(T-1, dtype=float)

        # Add colored segments
        lc = Line3DCollection(segs, cmap=vibrant_cmap, linewidth=3.0, zorder=50)
        lc.set_array(norm_vals)
        ax.add_collection(lc)
    else:
        # Solid polyline if no distance is provided
        ax.plot(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            color="#ff4d4d", linewidth=3.0, alpha=0.95, zorder=50,
            label=f"{trajectory_type} Trajectory {trajectory_idx}"
        )

    # Start / End markers
    ax.scatter(
        xyz[0, 0], xyz[0, 1], xyz[0, 2],
        s=70, color=vibrant_green, edgecolors="white", linewidths=1.2, zorder=100,
        label="Start"
    )
    ax.scatter(
        xyz[-1, 0], xyz[-1, 1], xyz[-1, 2],
        s=70, color=vibrant_red, edgecolors="white", linewidths=1.2, zorder=100,
        label="End"
    )

    # Poles for reference
    ax.scatter(0.0, 0.0, 1.02, s=60, color="blue",  alpha=0.9, zorder=5)
    ax.scatter(0.0, 0.0, -1.02, s=60, color="green", alpha=0.9, zorder=5)

    # Labels and equal aspect
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    _set_axes_equal_3d(ax)

    # Legend and save
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # /print(f"[INFO] S^2 individual trajectory plot saved to: {save_path}")
