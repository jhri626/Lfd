import numpy as np
import torch
import torch.nn.functional as F
from agent.utils.R2_functions import gvf_R2_np 
from agent.utils.dynamical_system_operations import normalize_state
from agent.utils.S2_functions import exp_sphere, xdot_to_qdot, x_to_q, q_to_x
from agent.utils.gvf import cvf_sphere_2nd_order
from tqdm import tqdm


class StateDynamic:
    """
    Direct state dynamics implementation using simple goal-based dynamics
    """
    def __init__(self, params,traj=None, vel_norm_stat=None, sample=None, max_steps=None):
        self.params = params
        self.workspace_dimensions = params.workspace_dimensions
        self.dynamical_system_order = params.dynamical_system_order
        self.delta_t = params.delta_t
        self.traj = traj
        self.eta = params.eta
        self.metric = 'sasaki'
        self.vel_norm_stat = vel_norm_stat
        self.dist = 2 * np.ones(sample) if sample is not None else None
        self.space = params.space

        self.max_steps = max_steps if max_steps is not None else 10000  
        self.dist_history = np.zeros((self.max_steps, sample if sample else 1))
        self.current_step = 0

        # Set goal to origin for simplicity
        if traj is not None:
            self.goal = traj[0, -1, :] 
        else:
            # Default goal at origin
            self.goal = np.zeros(self.workspace_dimensions)
        
    def compute_dynamics(self, state):
        """
        Compute state dynamics: dx/dt = (x_goal - x)
        Args:
            state: current state [state_dim] or [batch_size, state_dim]
        Returns:
            next_state: predicted next state
        """
        if isinstance(state, torch.Tensor):
            if self.space == "euclidean":
                state = state.detach().cpu().numpy().copy()
            elif self.space == "sphere":
                pass
        else:
            state = np.array(state, dtype=float, copy=True)
        
        # Handle both single state and batch processing
        if state.ndim == 1:
            # Single state processing (original logic)
            if self.dynamical_system_order == 1:
                # First order: dx/dt = (x_goal - x)
                position = state[:self.workspace_dimensions]
                velocity = self.goal[:self.workspace_dimensions] - position  # Simple goal-directed dynamics
                
                # Integration: x_next = x + dt * dx/dt
                next_position = position + self.delta_t * velocity
                next_state = next_position
                
            elif self.dynamical_system_order == 2:
                position = state[:self.workspace_dimensions]
                velocity = 0.1*(self.goal[:self.workspace_dimensions] - position)  # Simple goal-directed dynamics
                
                # Integration: x_next = x + dt * dx/dt
                next_position = position + self.delta_t * velocity
                next_velocity = velocity  # No acceleration in this simple model
                
                next_state = np.concatenate([next_position, next_velocity], axis=0)
            else:
                raise ValueError(f"Unsupported order: {self.dynamical_system_order}")
                
        elif state.ndim == 2:
            # Batch processing with numpy vectorization
            batch_size = state.shape[0]
            
            if self.space == "euclidean":
                # Second order batch processing
                positions = state[:, :self.workspace_dimensions]  # [batch_size, workspace_dim]
                
                # Broadcast goal to match batch size
                goal_batch = np.tile(self.goal[:self.workspace_dimensions], (batch_size, 1))
                # velocities = 0.1 * (goal_batch - positions)  # [batch_size, workspace_dim]
                xtraj = self.traj[:,:, :self.workspace_dimensions]
                xdottraj = self.traj[:,:, self.workspace_dimensions:]
                vel_norm_stat = self.vel_norm_stat
                # print("state",state)
                velocities, dist  = gvf_R2_np(state, self.eta , xtraj, xdottraj, self.metric, self.dist)  # Goal-directed dynamics
                
                if self.current_step < self.max_steps:
                    self.dist_history[self.current_step] = dist
                    self.current_step += 1
                self.dist = np.maximum(dist, 1e-2)
                # Integration: x_next = x + dt * dx/dt
                
                next_positions = positions + self.delta_t * velocities
                # print("next_positions",next_positions)
                
                # Concatenate positions and velocities
                next_state = np.concatenate([next_positions, velocities], axis=1)
                # next_state = np.concatenate([next_positions, velocities], axis=1)

            elif self.space == "sphere":
                # Spherical space dynamics (placeholder)
                positions = state[:, :self.workspace_dimensions].float()  # [batch_size, workspace_dim]
                
                # Broadcast goal to match batch size
                goal_batch = self.goal[:self.workspace_dimensions].repeat(batch_size, 1)
                # velocities = 0.1 * (goal_batch - positions)  # [batch_size, workspace_dim]
                xtraj = self.traj[:,:, :self.workspace_dimensions]
                xdottraj = self.traj[:,:, self.workspace_dimensions:]
                # print("state",state)
    
                velocities, dist, check, idx = cvf_sphere_2nd_order(state, self.eta , xtraj, xdottraj, self.dist)  # Goal-directed dynamics
                # print(dist.shape)
                if self.current_step < self.max_steps:
                    self.dist_history[self.current_step] = dist
                    self.current_step += 1
                self.dist = np.maximum(dist, 1e-3)
                # print(dist,q_to_x(positions))
                # Integration: x_next = x + dt * dx/dt
                # velocities = xdot_to_qdot(velocities,positions)
                # print("velocities",velocities,positions)
                
                next_positions = exp_sphere(positions, self.delta_t * velocities)
                
                # print(next_positions.shape)
                # print("next_positions",next_positions)
                next_positions = x_to_q(next_positions)  # Ensure next positions are on the sphere
                
                # if check:
                #     print(idx)
                #     xyz_pos = q_to_x(positions[idx])
                #     xyz_pos_next = q_to_x(next_positions[idx])
                #     print(xyz_pos, xyz_pos_next, velocities[idx], xdot_to_qdot(velocities, next_positions)[idx])
                velocities = xdot_to_qdot(velocities, next_positions)  # Convert velocities to spherical space  
                # if check:
                #     xyz_pos = q_to_x(positions[batch_list])
                #     xyz_pos_next = q_to_x(next_positions[batch_list])
                #     print("check",xyz_pos, velocities[batch_list], xyz_pos_next)
                
                # if check:
                #     print(positions[idx], next_positions[idx], velocities[idx])
                
                

                # Concatenate positions and velocities
                next_state = torch.cat([next_positions, velocities], dim=-1)
                
            else:
                raise ValueError(f"Unsupported order: {self.dynamical_system_order}")
        else:
            raise ValueError(f"Unsupported state dimensions: {state.ndim}")

        return next_state
    
    def simulate(self, initial_state):
        """
        Simulate trajectory using state dynamics
        Args:
            initial_state: starting state [state_dim] or [batch_size, state_dim]
            steps: number of simulation steps
        Returns:
            trajectory: simulated trajectory [steps, state_dim] or [batch_size, steps, state_dim]
        """
        if isinstance(initial_state, torch.Tensor):
            if self.space == "euclidean":
                initial_state = initial_state.numpy()
        
            
        # Handle both single state and batch processing
        if initial_state.ndim == 1:
            # Single state simulation (original logic)
            # print("why?")
            return self._simulate_single(initial_state)
        elif initial_state.ndim == 2:
            # Batch simulation
            self.dist = 2 * np.ones(initial_state.shape[0]) if self.dist is None else self.dist
            traj = self._simulate_batch(initial_state)
            # print("*"*100,traj[:,-1,:])
            return traj
        else:
            raise ValueError(f"Unsupported initial_state dimensions: {initial_state.ndim}")
    
    def _simulate_single(self, initial_state):
        """
        Simulate single trajectory
        Args:
            initial_state: starting state [state_dim]
            steps: number of simulation steps
        Returns:
            trajectory: simulated trajectory [steps, state_dim]
        """
        trajectory = np.zeros((self.max_steps, len(initial_state)))
        trajectory[0] = initial_state
        
        current_state = initial_state.copy()


        for i in range(1, self.max_steps):
            next_state = self.compute_dynamics(current_state)
            
            
            # Ensure next_state has the same shape as initial_state
            if isinstance(next_state, np.ndarray):
                if next_state.ndim > 1:
                    next_state = next_state.flatten()
                if len(next_state) != len(initial_state):
                    # Pad or truncate to match
                    if len(next_state) < len(initial_state):
                        padded_state = np.zeros(len(initial_state))
                        padded_state[:len(next_state)] = next_state
                        next_state = padded_state
                    else:
                        next_state = next_state[:len(initial_state)]
            
            trajectory[i] = next_state
            current_state = next_state.copy()
            
            # Early stopping if converged
            pos = current_state[:self.workspace_dimensions]
            if np.linalg.norm(pos - self.goal[:self.workspace_dimensions]) < 1e-3:
                trajectory = trajectory[:i+1]
                break
                
        return trajectory
    
    def _simulate_batch(self, initial_states):
        """
        Simulate multiple trajectories in batch
        Args:
            initial_states: starting states [batch_size, state_dim]
            steps: number of simulation steps
        Returns:
            trajectories: simulated trajectories [batch_size, steps, state_dim]
        """
        if self.space == "euclidean":
            batch_size, state_dim = initial_states.shape
            trajectories = np.zeros((batch_size, self.max_steps, state_dim))
            trajectories[:, 0, :] = initial_states
            
            current_states = initial_states.copy()
            active_mask = np.ones(batch_size, dtype=bool)  # Track which trajectories are still active

            for i in tqdm(range(1, self.max_steps)):
                # print("current states 0",current_states, "step",i)
                if not np.any(active_mask):
                    # All trajectories have converged
                    trajectories = trajectories[:, :i, :]
                    break
                    
                # Only compute dynamics for active trajectories
                # if np.all(active_mask):
                    # All trajectories are active, process all at once
                # print("current states",current_states, "step",i)
                next_states = self.compute_dynamics(current_states.copy())
                # print("next state",next_states, "step",i)

                # print(f"next_states shape: {next_states}")
                # else:
                #     # Some trajectories have converged, process only active ones
                #     active_states = current_states[active_mask]
                #     next_active_states = self.compute_dynamics(active_states)
                    
                #     # Update only active trajectories
                #     next_states = current_states.copy()
                #     next_states[active_mask] = next_active_states
                
                trajectories[:, i, :] = next_states
                current_states = next_states.copy()
                # print("current_states 2",current_states, "step",i,"\n")
                
                
                # Check for convergence
                positions = current_states[:, :self.workspace_dimensions]
                goal_batch = np.tile(self.goal[:self.workspace_dimensions], (batch_size, 1))
                distances = np.linalg.norm(positions - goal_batch, axis=1)
                
                # Update active mask (mark converged trajectories as inactive)
                active_mask = active_mask & (distances >= 1e-3)
            # print("current_states 3",current_states,"\n")
        elif self.space == "sphere":
            device, dtype = initial_states.device, initial_states.dtype
            batch_size, state_dim = initial_states.shape

            # Allocate tensor for trajectories
            trajectories = torch.zeros((batch_size, self.max_steps, state_dim),
                                    device=device, dtype=dtype)
            trajectories[:, 0, :] = initial_states

            current_states = initial_states.clone()
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)  # Track active trajectories

            for i in tqdm(range(1, self.max_steps)):
            # for i in range(1, self.max_steps):
                if not torch.any(active_mask):
                    # All trajectories have converged ¡æ truncate
                    trajectories = trajectories[:, :i, :]
                    break
                
                # Compute dynamics for all active states
                # print("\ncurrent",current_states[5,:], "step",i)
                next_states = self.compute_dynamics(current_states.clone())
                # print("next",next_states[5,:], "step",i)

                # Save trajectory
                trajectories[:, i, :] = next_states
                current_states = next_states.clone()

                # Check convergence (distance to goal)
                positions = current_states[:, :self.workspace_dimensions]
                goal_batch = self.goal[:self.workspace_dimensions].to(device).repeat(batch_size, 1)
                distances = 1- F.cosine_similarity(q_to_x(positions),q_to_x(goal_batch), dim=1)
                # if i == self.max_steps - 1:
                #     print(positions,goal_batch)
                

                # Update active mask
                active_mask = active_mask & (distances >= 1e-3)
        
        return trajectories
    
    def get_dist_history(self):
        return self.dist_history[:self.current_step]
    