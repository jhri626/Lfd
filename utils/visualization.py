#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
동작 원시 시각화 유틸리티

이 모듈은 모션 프리미티브 모델의 궤적 시각화 기능을 제공합니다.
"""

import torch
import matplotlib.pyplot as plt
from models.motion_primitive_model import MotionPrimitiveModel


def visualize_trajectories(
    model: MotionPrimitiveModel,
    init_state: torch.Tensor,
    primitive_type: torch.Tensor,
    steps: int = 100,
    latent: bool = False,
    title: str = "생성된 궤적"
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
    _plot_trajectories(model.workspace_dim, ax, pos_traj, primitive_type, colors)
    
    # 목표점 그리기
    _plot_goals(model.workspace_dim, ax, model.state_dynamics.goals, colors, model.n_primitives)

    # 그래프 설정
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    
    return fig


def _plot_trajectories(workspace_dim, ax, pos_traj, primitive_type, colors):
    """궤적을 그리는 헬퍼 함수"""
    batch_size = pos_traj.shape[0]
    
    for i in range(batch_size):
        prim = primitive_type[i].item()
        color = colors[prim % len(colors)]
        label = f"Primitive {prim}" if i == 0 or primitive_type[i] != primitive_type[i-1] else ""
        
        # 데이터 준비
        x = pos_traj[i, :, 0].cpu().numpy()
        y = pos_traj[i, :, 1].cpu().numpy()
        
        # 2D 또는 3D 플롯
        if workspace_dim == 2:
            ax.plot(x, y, color=color, marker='o', markersize=3, label=label)
            ax.plot(x[0], y[0], 'o', color=color, markersize=8)
        elif workspace_dim == 3:
            z = pos_traj[i, :, 2].cpu().numpy()
            ax.plot(x, y, z, color=color, marker='o', markersize=3, label=label)
            ax.plot([x[0]], [y[0]], [z[0]], 'o', color=color, markersize=8)


def _plot_goals(workspace_dim, ax, goals, colors, n_primitives):
    """목표점을 그리는 헬퍼 함수"""
    for p in range(n_primitives):
        goal = goals[p].cpu().numpy()
        
        if workspace_dim == 2:
            ax.plot(goal[0], goal[1], '*', color=colors[p % len(colors)], markersize=15)
        elif workspace_dim == 3:
            ax.plot([goal[0]], [goal[1]], [goal[2]], '*', color=colors[p % len(colors)], markersize=15)
