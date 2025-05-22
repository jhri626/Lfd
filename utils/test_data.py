#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
테스트 데이터 생성 및 유틸리티 함수

이 모듈은 모션 프리미티브 모델을 위한 테스트 데이터 생성 함수를 제공합니다.
"""

import torch
from typing import Dict


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
