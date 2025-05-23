#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
테스트 데이터 생성 및 유틸리티 함수

이 모듈은 모션 프리미티브 모델을 위한 테스트 데이터 생성 함수를 제공합니다.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Union, Optional


def create_test_data(
    workspace_dim: int = 2, 
    velocity_dim: int = 2, 
    n_samples: int = 1000, 
    n_primitives: int = 2,
    dt: float = 0.01
) -> Dict[str, torch.Tensor]:
    """
    테스트용 데이터 생성

    Args:
        workspace_dim: 작업 공간 차원
        velocity_dim: 속도 차원 (1차 시스템이면 0)
        n_samples: 샘플 수
        n_primitives: 프리미티브 수
        dt: 시간 간격

    Returns:
        데이터 딕셔너리
    """
    # 임의의 목표점 생성
    goals = torch.randn(n_primitives, workspace_dim)
    
    # 상태 차원 계산
    dim_state = workspace_dim + velocity_dim
    
    # 임의의 시작 위치
    init_pos = torch.randn(n_samples, workspace_dim) * 2.0
    
    # 프리미티브 할당 (균일하게)
    primitive_types = torch.randint(0, n_primitives, (n_samples,))
    
    # 상태 벡터 초기화
    states = torch.zeros(n_samples, dim_state)
    states[:, :workspace_dim] = init_pos
    
    # 속도가 있는 경우 초기화
    if velocity_dim > 0:
        init_vel = torch.randn(n_samples, velocity_dim) * 0.5
        states[:, workspace_dim:] = init_vel
    
    # 다음 상태 생성
    next_states = torch.zeros_like(states)
    
    for i in range(n_samples):
        prim = primitive_types[i].item()
        pos = states[i, :workspace_dim]
        
        # 목표 방향으로 이동 생성
        direction = goals[prim] - pos
        norm = torch.norm(direction, dim=0) + 1e-8  # dim=0 명시적으로 지정
        normed_dir = direction / norm
        
        if velocity_dim > 0:
            # 2차 시스템
            vel = states[i, workspace_dim:]
            # 간단한 물리: 목표 방향으로 가속
            acc = normed_dir * 0.5 - vel * 0.1  # 목표 방향 + 감쇠
            
            # 상태 업데이트: 2차 운동 방정식
            next_vel = vel + acc * dt
            next_pos = pos + vel * dt + 0.5 * acc * dt ** 2
            
            next_states[i, :workspace_dim] = next_pos
            next_states[i, workspace_dim:] = next_vel
        else:
            # 1차 시스템
            # 목표로 직접 이동
            next_pos = pos + normed_dir * 0.1
            next_states[i, :workspace_dim] = next_pos
    
    return {
        "states": states,
        "next_states": next_states,
        "primitive_types": primitive_types,
        "goals": goals
    }


def create_test_data(
    n_primitives: int, 
    n_samples_per_primitive: int = 3, 
    workspace_dim: int = 2, 
    order: int = 2, 
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    수정된 테스트 데이터 생성 함수

    Args:
        n_primitives: 프리미티브 수
        n_samples_per_primitive: 프리미티브당 샘플 수
        workspace_dim: 작업 공간 차원
        order: 시스템 차수 (1 또는 2)
        device: 계산 장치
    
    Returns:
        states: (n_primitives * n_samples_per_primitive, dim_state) 초기 상태
        primitive_types: (n_primitives * n_samples_per_primitive,) 프리미티브 인덱스
    """
    if isinstance(device, str):
        device = torch.device(device)

    # 속도 차원 계산
    velocity_dim = workspace_dim if order == 2 else 0
    
    # 상태 차원
    dim_state = workspace_dim + velocity_dim
    
    # 총 샘플 수
    n_samples = n_primitives * n_samples_per_primitive
    
    # 초기 위치 (각 프리미티브당 n_samples_per_primitive 개씩)
    states = torch.zeros(n_samples, dim_state, device=device)
    init_pos = torch.randn(n_samples, workspace_dim, device=device) * 2.0
    states[:, :workspace_dim] = init_pos
    
    # 초기 속도 (2차 시스템인 경우)
    if order == 2:
        init_vel = torch.randn(n_samples, workspace_dim, device=device) * 0.5
        states[:, workspace_dim:] = init_vel
    
    # 프리미티브 타입 할당
    primitive_types = torch.zeros(n_samples, dtype=torch.long, device=device)
    for p in range(n_primitives):
        start_idx = p * n_samples_per_primitive
        end_idx = (p + 1) * n_samples_per_primitive
        primitive_types[start_idx:end_idx] = p
    
    return states, primitive_types


def sample_near_trajectory_start(
    dataset: Dict[str, np.ndarray],
    primitive_id: int,
    trajectory_id: int,
    n_points: int = 5,
    radius: float = 0.1,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    데이터셋에서 특정 프리미티브 ID와 궤적 ID에 해당하는 궤적의 시작점 근처에서 포인트를 샘플링합니다.

    Args:
        dataset: 전처리된 데이터셋 딕셔너리
        primitive_id: 프리미티브 ID
        trajectory_id: 궤적 ID (해당 프리미티브 내에서의 인덱스)
        n_points: 샘플링할 포인트 개수
        radius: 시작점 주변 샘플링 반경
        device: 계산 장치
    
    Returns:
        states: (n_points, dim_state) 샘플링된 초기 상태
        primitive_types: (n_points,) 프리미티브 인덱스 (모두 동일)
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    # 해당 프리미티브 ID의 궤적들 추출
    prim_ids = dataset['demonstrations primitive id']
    prim_trajectories = np.where(prim_ids == primitive_id)[0]
    
    if len(prim_trajectories) == 0:
        raise ValueError(f"프리미티브 ID {primitive_id}에 해당하는 궤적이 없습니다.")
    
    if trajectory_id >= len(prim_trajectories):
        raise ValueError(f"궤적 ID {trajectory_id}가 범위를 벗어납니다. 유효한 범위: 0-{len(prim_trajectories)-1}")
    
    # 해당 궤적 인덱스
    traj_idx = prim_trajectories[trajectory_id]
    
    # 데모 데이터 추출
    demos = dataset['demonstrations train']  # (n_traj, n_steps, dim_ws, window)
    
    # 해당 궤적의 시작점
    start_point = demos[traj_idx, 0, :, 0]  # (dim_ws,)
    
    # 작업 공간 차원 계산
    workspace_dim = start_point.shape[0]
    
    # 동역학계 차수에 따른 상태 차원 계산
    order = 1
    if 'dynamical_system_order' in dataset:
        order = dataset['dynamical_system_order']
    elif len(demos.shape) > 3 and demos.shape[3] > 1:
        order = 2
        
    velocity_dim = workspace_dim if order == 2 else 0
    dim_state = workspace_dim + velocity_dim
    
    # 시작점 주변에서 n개의 포인트 랜덤 샘플링
    states = torch.zeros(n_points, dim_state, device=device)
    
    # 위치 샘플링 (시작점 주변)
    random_offsets = torch.randn(n_points, workspace_dim, device=device) * radius
    sampled_positions = torch.tensor(start_point, device=device).unsqueeze(0) + random_offsets
    states[:, :workspace_dim] = sampled_positions
    
    # 속도가 있는 경우 (2차 시스템) 샘플링
    if order == 2:
        # 시작 속도도 추출하여 비슷한 속도 사용
        if demos.shape[3] > 1:
            start_velocity = (demos[traj_idx, 1, :, 0] - start_point) / 0.01  # 속도 추정 (dt=0.01 가정)
            # 비슷한 속도 샘플링
            random_vel_offsets = torch.randn(n_points, workspace_dim, device=device) * (radius * 0.5)
            sampled_velocities = torch.tensor(start_velocity, device=device).unsqueeze(0) + random_vel_offsets
            states[:, workspace_dim:] = sampled_velocities
        else:
            # 속도 데이터가 없으면 작은 랜덤 속도 사용
            random_velocities = torch.randn(n_points, workspace_dim, device=device) * 0.1
            states[:, workspace_dim:] = random_velocities
    
    # 모든 샘플에 동일한 프리미티브 ID 할당
    primitive_types = torch.full((n_points,), primitive_id, dtype=torch.long, device=device)
    
    return states, primitive_types
