#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motion Raw Visualization Utilities

This module provides trajectory visualization functionality for motion primitive models.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from models.motion_primitive_model import MotionPrimitiveModel


def normalize_state(state, x_min, x_max):
    """
    State Normalization

    Args:
        state: State vector to normalize
        x_min: Minimum value
        x_max: Maximum value

    Returns:
        Normalized state [-1, 1]
    """
    return 2.0 * (state - x_min) / (x_max - x_min) - 1.0

def denormalize_state(state_norm, x_min, x_max):
    """
    Restore normalized state to original scale

    Args:
        state_norm: Normalized state [-1, 1]  
        x_min: Minimum value
        x_max: Maximum value

    Returns:
        State in original scale
    """
    return 0.5 * (state_norm + 1.0) * (x_max - x_min) + x_min

def visualize_from_dataset(
    model: MotionPrimitiveModel,
    dataset: Dict[str, np.ndarray],
    prim_id: int,
    traj_id: int,
    n_samples: int = 5,
    steps: int = 100,
    sample_radius: float = 0.1,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show_original: bool = True,
    seed: Optional[int] = None
):
    """
    Find a trajectory with specific primitive ID and trajectory ID from dataset,
    sample n points near its starting point and visualize trajectories following the model.

    Args:
        model: Motion primitive model instance
        dataset: Dataset dictionary
        prim_id: Primitive ID
        traj_id: Trajectory ID within that primitive  
        n_samples: Number of points to sample
        steps: Number of steps for generated trajectory
        sample_radius: Sampling radius around start point
        save_path: Path to save figure (None for no save)
        title: Graph title (None for default title)
        show_original: Whether to show original trajectory
        seed: Random seed (for reproducibility)

    Returns:
        matplotlib Figure object
    """
    # 난수 시드 설정
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 시각화를 위한 색상
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    
    # 제목 설정
    if title is None:
        title = f"Dataset Trajectory Visualization (Primitive {prim_id}, Trajectory {traj_id})"
    
    # 1. 데이터셋에서 해당 프리미티브 ID와 궤적 ID에 해당하는 궤적 찾기
    prim_ids = np.array(dataset['demonstrations primitive id'])
    prim_trajectories = np.nonzero(prim_ids == prim_id)[0]
    model.encoder.eval()
    model.decoder.eval()
    model.state_dynamics.eval()
    
    
    if len(prim_trajectories) == 0:
        raise ValueError(f"No trajectories found for primitive ID {prim_id}")
    
    if traj_id >= len(prim_trajectories):
        raise ValueError(f"Trajectory ID {traj_id} out of range. Valid range: 0-{len(prim_trajectories)-1}")
    
    # 해당 궤적의 실제 인덱스
    traj_idx = prim_trajectories[traj_id]
      # 원본 데모 데이터 추출
    demos = dataset['demonstrations train']  # (n_traj, n_steps, dim_ws, window)
    
    # 해당 궤적의 데이터 추출
    orig_trajectory = demos[traj_idx]  # (n_steps, dim_ws, window)
    n_steps, dim_ws, window = orig_trajectory.shape
    
    # 동역학계 차수 확인
    order = model.dynamical_system_order
    
    
    vel_min = dataset['vel min train']
    vel_max = dataset['vel max train']
    
    if order == 2 and 'acc min train' in dataset and 'acc max train' in dataset:
        acc_min = dataset['acc min train'].reshape(1, -1) if len(dataset['acc min train'].shape) == 1 else dataset['acc min train']
        acc_max = dataset['acc max train'].reshape(1, -1) if len(dataset['acc max train'].shape) == 1 else dataset['acc max train']
    else:
        acc_min = None
        acc_max = None
        
    # 전체 궤적 데이터 준비 (위치 + 속도)
    if order == 2 and window > 1:
        # 위치와 속도 데이터 추출
        print("orig_trajectory shape:", orig_trajectory.shape)
        orig_positions = orig_trajectory[:, :, 0]  # (n_steps, dim_ws)
        print("orig_positions:", orig_trajectory[0,:, 0])
        print("orig_positions:", orig_trajectory[0,:, 1])
        velocities = orig_trajectory[:, :, 1] - orig_trajectory[:, :, 0] if window > 1 else np.zeros_like(orig_positions)  # (n_steps, dim_ws)
        
        # 정규화 파라미터 준비
        if len(vel_min.shape) == 1:
            vel_min_expanded = vel_min.reshape(1, -1)
            vel_max_expanded = vel_max.reshape(1, -1)
        else:
            vel_min_expanded = vel_min
            vel_max_expanded = vel_max

        # 속도 정규화
        velocities_normalized = normalize_state(velocities, vel_min_expanded, vel_max_expanded)
        
        # 전체 상태 벡터 구성 (위치 + 정규화된 속도)
        orig_states = np.concatenate([orig_positions, velocities_normalized], axis=1)  # (n_steps, 2*dim_ws)
        
    else:
        # 1차 시스템: 위치만 사용
        orig_positions = orig_trajectory[:, :, 0]  # (n_steps, dim_ws)
        orig_states = orig_positions
    
      # 3. 시작점(첫 번째 상태) 근처에서 n개의 포인트 샘플링
    start_state = orig_states[0]  # 첫 번째 상태 (위치 + 속도)
    print("start_state:", start_state)
    
    # 샘플링된 초기 상태 저장 텐서
    sampled_states = torch.zeros(n_samples, model.dim_state, device=model.device)
    
    if order == 2:
        # 위치와 속도 분리
        start_pos = start_state[:dim_ws]
        start_vel = start_state[dim_ws:]
        
        # 위치 샘플링
        random_pos_offsets = torch.randn(n_samples, dim_ws, device=model.device) * sample_radius
        sampled_positions = torch.tensor(start_pos, device=model.device).unsqueeze(0) + random_pos_offsets
        
        # 속도 샘플링 (시작 속도 기준)
        random_vel_offsets = torch.randn(n_samples, dim_ws, device=model.device) * (sample_radius * 0.01)
        sampled_velocities = torch.tensor(start_vel, device=model.device).unsqueeze(0) + random_vel_offsets
    

            
        # 정규화된 상태 구성
        sampled_states[:, :dim_ws] = sampled_positions
        sampled_states[:, dim_ws:] = sampled_velocities
    else:
        # 1차 시스템: 위치만 샘플링
        random_offsets = torch.randn(n_samples, dim_ws, device=model.device) * sample_radius
        sampled_positions = torch.tensor(start_state, device=model.device).unsqueeze(0) + random_offsets
        sampled_states[:, :dim_ws] = sampled_positions
    
    # 2. 모델에 정규화 파라미터 설정
    vel_min = torch.from_numpy(dataset['vel min train'].reshape(1, -1)).float().to(model.device)
    vel_max = torch.from_numpy(dataset['vel max train'].reshape(1, -1)).float().to(model.device)

    if 'acc min train' in dataset and 'acc max train' in dataset:
        acc_min = torch.from_numpy(dataset['acc min train'].reshape(1, -1)).float().to(model.device)
        acc_max = torch.from_numpy(dataset['acc max train'].reshape(1, -1)).float().to(model.device)
        model.set_normalization_params(vel_min, vel_max, acc_min, acc_max)
    else:
        model.set_normalization_params(vel_min, vel_max)

    # 4. 각 샘플 포인트에서 모델을 통해 궤적 생성
    # 모든 샘플에 동일한 프리미티브 ID 할당
    sampled_prim_ids = torch.full((n_samples,), prim_id, dtype=torch.long, device=model.device)
    
    state_traj=[sampled_states]
    states=sampled_states
    # 모델을 통해 궤적 생성
    for _ in range(steps):
        latent_state = model.encoder(states, sampled_prim_ids)
        decoder_out= model.decoder(latent_state,sampled_prim_ids)
        task_state = model.state_dynamics(states,decoder_out,sampled_prim_ids)
        states = task_state
        state_traj.append(states)
        
    state_traj = torch.stack(state_traj, dim=1)  # (n_samples, steps, dim_state)
    # 5. 시각화
    fig = plt.figure(figsize=(12, 10))
    
    # 2D 또는 3D 플롯 설정
    if dim_ws == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    # 원본 궤적 그리기 (선택적)
    if show_original:
        if dim_ws == 2:
            ax.plot(
                orig_positions[:, 0], 
                orig_positions[:, 1], 
                'k--', 
                linewidth=2, 
                alpha=0.7, 
                label='Original Trajectory'
            )
            ax.plot(
                orig_positions[0, 0], 
                orig_positions[0, 1], 
                'ko', 
                markersize=8, 
                label='Original Start'
            )
        elif dim_ws == 3:
            ax.plot(
                orig_positions[:, 0], 
                orig_positions[:, 1], 
                orig_positions[:, 2], 
                'k--', 
                linewidth=2, 
                alpha=0.7, 
                label='Original Trajectory'
            )
            ax.plot(
                [orig_positions[0, 0]], 
                [orig_positions[0, 1]], 
                [orig_positions[0, 2]], 
                'ko', 
                markersize=8, 
                label='Original Start'
            )
    
    # 샘플링된 시작점 표시
    for i in range(n_samples):
        color = colors[i % len(colors)]
        pos = sampled_positions[i].cpu().numpy()
        
        if dim_ws == 2:
            ax.plot(
                pos[0], 
                pos[1], 
                'o', 
                color=color, 
                markersize=8, 
                label=f'Sample {i+1} Start'
            )
        elif dim_ws == 3:
            ax.plot(
                [pos[0]], 
                [pos[1]], 
                [pos[2]], 
                'o', 
                color=color, 
                markersize=8, 
                label=f'Sample {i+1} Start'
            )
    
    # 생성된 궤적 그리기
    for i in range(n_samples):
        color = colors[i % len(colors)]
        traj = state_traj[ i, :,:dim_ws].detach().cpu().numpy()
        
        if dim_ws == 2:
            ax.plot(
                traj[:, 0], 
                traj[:, 1], 
                color=color, 
                marker='.', 
                markersize=3, 
                alpha=0.8,
                label=f'Generated Trajectory {i+1}'
            )
        elif dim_ws == 3:
            ax.plot(
                traj[:, 0], 
                traj[:, 1], 
                traj[:, 2], 
                color=color, 
                marker='.', 
                markersize=3, 
                alpha=0.8,
                label=f'Generated Trajectory {i+1}'
            )
    
    # 목표점 그리기
    goals = model.state_dynamics.goals.cpu().numpy()
    prim_goal = goals[prim_id]
    
    if dim_ws == 2:
        ax.plot(
            prim_goal[0], 
            prim_goal[1], 
            '*', 
            color='green', 
            markersize=15, 
            label=f'Goal (Primitive {prim_id})'
        )
    elif dim_ws == 3:
        ax.plot(
            [prim_goal[0]], 
            [prim_goal[1]], 
            [prim_goal[2]], 
            '*', 
            color='green', 
            markersize=15, 
            label=f'Goal (Primitive {prim_id})'
        )
    
    # 그래프 설정
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    # 축 레이블
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if dim_ws == 3:
        ax.set_zlabel('Z')
    
    fig.tight_layout()
    
    # 그림 저장 (선택적)
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Figure saved: {save_path}")
    
    return fig


def evaluate_and_visualize_trajectories(
    model: MotionPrimitiveModel,
    dataset: Dict[str, np.ndarray],
    save_dir: str,
    prim_ids: List[int] = None,
    traj_per_prim: int = 3,
    n_samples: int = 5,
    steps: int = 100,
    sample_radius: float = 0.1,
    seed: Optional[int] = None
):
    """
    Perform evaluation and visualization for multiple primitives and trajectories.

    Args:
        model: Motion primitive model instance
        dataset: Dataset dictionary
        save_dir: Directory path to save results
        prim_ids: List of primitive IDs to evaluate (None for all primitives)
        traj_per_prim: Number of trajectories to evaluate per primitive
        n_samples: Number of points to sample per trajectory
        steps: Number of steps for generated trajectory
        sample_radius: Sampling radius around start point
        seed: Random seed (for reproducibility)
    """
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 난수 시드 설정
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # 데이터셋의 모든 프리미티브 ID 가져오기
    all_prim_ids = np.unique(dataset['demonstrations primitive id'])
    
    # 평가할 프리미티브 ID 선택
    if prim_ids is None:
        prim_ids = all_prim_ids
    
    print(f"Selected Primitive IDs: {prim_ids}")
    
    # 각 프리미티브 ID에 대해
    for prim_id in prim_ids:
        # 해당 프리미티브의 궤적 인덱스 목록 가져오기
        prim_trajectories = np.nonzero(dataset['demonstrations primitive id'] == prim_id)[0]
        
        if len(prim_trajectories) == 0:
            print(f"No trajectories found for primitive ID {prim_id}. Skipping.")
            continue
        
        print(f"\nPrimitive ID {prim_id}: Evaluating {min(traj_per_prim, len(prim_trajectories))} out of {len(prim_trajectories)} trajectories")
        
        # 궤적 수 제한
        max_traj = min(traj_per_prim, len(prim_trajectories))
        
        # 각 궤적에 대해
        for t_idx in range(max_traj):
            # 궤적 ID 인덱스 (프리미티브 내에서)
            traj_id = t_idx
            
            print(f"  Evaluating primitive {prim_id}, trajectory {traj_id}...")
            
            # 저장 경로
            save_path = os.path.join(save_dir, f"prim{prim_id}_traj{traj_id}.png")
            
            try:
                # 시각화 수행
                visualize_from_dataset(
                    model=model,
                    dataset=dataset,
                    prim_id=prim_id,
                    traj_id=traj_id,
                    n_samples=n_samples,
                    steps=steps,
                    sample_radius=sample_radius,
                    save_path=save_path,
                    seed=seed
                )
                
                print(f"  Saved: {save_path}")
            except Exception as e:
                print(f"  Error: {e}")


def visualize_trajectories(
    model: MotionPrimitiveModel,
    init_state: torch.Tensor,
    primitive_type: torch.Tensor,
    steps: int = 100,
    latent: bool = False,
    title: str = "Generated Trajectory",
    save_path: Optional[str] = None
):
    """
    궤적 시각화

    Args:
        model: 모델 인스턴스
        init_state: (B, dim_state) 초기 상태
        primitive_type: (B,) 프리미티브 인덱스
        steps: 생성할 스텝 수
        latent: 잠재 공간 궤적 사용 여부
        title: 그래프 제목
        save_path: 저장 경로 (None이면 저장 안함)
        
    Returns:
        matplotlib Figure 객체
    """
    # 궤적 생성
    if latent:
        traj, _ = model.generate_trajectory_latent(init_state, primitive_type, steps)
    else:
        traj = model.generate_trajectory_state(init_state, primitive_type, steps)
        
    # 위치만 추출
    pos_traj = traj[:, :, :model.workspace_dim]
    
    # 프리미티브 별 색상
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    
    # 그림 생성
    fig = plt.figure(figsize=(10, 8))
    
    # 2D 또는 3D 플롯 설정
    if model.workspace_dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    # 각 배치 궤적 그리기
    batch_size = pos_traj.shape[0]
    
    for i in range(batch_size):
        prim = primitive_type[i].item()
        color = colors[prim % len(colors)]
        label = f"Primitive {prim}" if i == 0 or primitive_type[i] != primitive_type[i-1] else ""
        
        # 데이터 준비
        x = pos_traj[i, :, 0].cpu().numpy()
        y = pos_traj[i, :, 1].cpu().numpy()
        
        # 2D 또는 3D 플롯
        if model.workspace_dim == 2:
            ax.plot(x, y, color=color, marker='o', markersize=3, label=label)
            ax.plot(x[0], y[0], 'o', color=color, markersize=8)
        elif model.workspace_dim == 3:
            z = pos_traj[i, :, 2].cpu().numpy()
            ax.plot(x, y, z, color=color, marker='o', markersize=3, label=label)
            ax.plot([x[0]], [y[0]], [z[0]], 'o', color=color, markersize=8)
    
    # 목표점 그리기
    goals = model.state_dynamics.goals.cpu().numpy()
    n_primitives = goals.shape[0]
    
    for p in range(n_primitives):
        goal = goals[p]
        
        if model.workspace_dim == 2:
            ax.plot(goal[0], goal[1], '*', color=colors[p % len(colors)], markersize=15)
        elif model.workspace_dim == 3:
            ax.plot([goal[0]], [goal[1]], [goal[2]], '*', color=colors[p % len(colors)], markersize=15)

    # 그래프 설정
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    
    # 저장
    if save_path is not None:
        plt.savefig(save_path)
        print(f"그림이 저장되었습니다: {save_path}")
    
    return fig
