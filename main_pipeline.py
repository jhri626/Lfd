#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CONDOR 영감을 받은 동작 원시(Motion Primitive) 학습을 위한 메인 스크립트

이 스크립트는 다음 구성 요소를 통합합니다:
- 인코더 (encoder): 상태 공간을 잠재 공간으로 매핑 (사전 훈련된 모델 로딩 지원)
- 디코더 (decoder): 잠재 공간을 상태 공간으로 매핑
- 잠재 다이나믹스 모델 (latent dynamics): 잠재 공간에서의 동작 모델링
- 상태 다이나믹스 모델 (state dynamics): 상태 공간에서의 동작 모델링
- 다양한 동역학 차수 지원 (1차, 2차)
- 다중 동작 원시 기능
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import argparse
from dataclasses import asdict
import json
import yaml  # YAML 파일 처리를 위한 모듈 추가

# 모델 가져오기
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_dynamics import LatentDynamics
from models.state_dynamics import StateDynamics

# 데이터 파이프라인 가져오기
from data_pipeline import DataPipeline, DataPipelineParams


class MotionPrimitiveModel:
    """
    동작 원시 학습 및 생성을 위한 통합 모델

    이 클래스는 인코더, 디코더, 잠재 및 상태 다이나믹스 모델을 통합하여
    동작 원시를 학습하고 생성하는 기능을 제공합니다.
    """

    def __init__(
        self, 
        dim_state: int, 
        latent_dim: int = 16, 
        n_primitives: int = 1,
        dynamical_system_order: int = 2,
        workspace_dim: int = 2,
        multi_motion: bool = False,
        encoder_hidden_sizes: List[int] = [256, 256],
        decoder_hidden_sizes: List[int] = [256, 256],
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        모델 초기화

        Args:
            dim_state: 상태 차원
            latent_dim: 잠재 공간 차원
            n_primitives: 프리미티브 수
            dynamical_system_order: 동역학계 차수 (1차 또는 2차)
            workspace_dim: 작업 공간 차원
            multi_motion: 다중 동작 원시 지원 여부
            encoder_hidden_sizes: 인코더 은닉층 크기 목록
            decoder_hidden_sizes: 디코더 은닉층 크기 목록
            latent_dynamics_hidden_sizes: 잠재 다이나믹스 은닉층 크기 목록
            state_dynamics_hidden_sizes: 상태 다이나믹스 은닉층 크기 목록
            device: 계산 장치
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dim_state = dim_state
        self.latent_dim = latent_dim
        self.n_primitives = n_primitives
        self.dynamical_system_order = dynamical_system_order
        self.workspace_dim = workspace_dim
        self.multi_motion = multi_motion

        print(f"초기화: 상태 차원={dim_state}, 잠재 차원={latent_dim}, 프리미티브 수={n_primitives}")
        print(f"동역학계 차수: {dynamical_system_order}, 다중 동작: {multi_motion}")

        # 모델 컴포넌트 초기화
        self.encoder = Encoder(
            dim_state=dim_state,
            latent_dim=latent_dim,
            n_primitives=n_primitives,
            hidden_sizes=encoder_hidden_sizes,
            multi_motion=multi_motion,
            device=device
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            dim_state=dim_state,
            n_primitives=n_primitives,
            hidden_sizes=decoder_hidden_sizes,
            multi_motion=multi_motion,
            device=device
        )        
        self.latent_dynamics = LatentDynamics(
            latent_dim=latent_dim,
            n_primitives=n_primitives,
            multi_motion=multi_motion,
            device=device
        )
        # 잠재 다이나믹스 모델 초기화 (학습 없이 오일러 적분만 수행)
        self.latent_dynamics.delta_t = 1.0

        self.state_dynamics = StateDynamics(
            dim_state=dim_state,
            n_primitives=n_primitives,
            multi_motion=multi_motion,
            dynamical_system_order=dynamical_system_order,
            workspace_dim=workspace_dim,
            device=device
        )
        # 상태 다이나믹스 모델 초기화 (학습 없이 오일러 적분만 수행)
        self.state_dynamics.delta_t = 1.0
        if dynamical_system_order == 2:
            self.state_dynamics.damping = 0.1# 인코더와 디코더를 위한 최적화 도구
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        # dynamics 모델은 학습이 필요 없어 옵티마이저를 제거함

    def configure_optimizers(
        self, 
        lr_encoder: float = 1e-4,
        lr_decoder: float = 1e-4, 
        weight_decay: float = 1e-5
    ):
        """
        옵티마이저 설정 (인코더와 디코더만 학습)

        Args:
            lr_encoder: 인코더 학습률
            lr_decoder: 디코더 학습률
            weight_decay: 가중치 감쇠
        """
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=lr_encoder, weight_decay=weight_decay
        )
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=lr_decoder, weight_decay=weight_decay
        )
        # dynamics 모델 학습 제거

    def set_goals(self, goals: torch.Tensor):
        """
        목표 위치 설정

        Args:
            goals: (n_primitives, workspace_dim) 목표점
        """
        # 상태 다이나믹스용 목표 설정
        self.state_dynamics.set_goals(goals)

        # 인코더를 통해 잠재 공간 목표 계산 및 설정
        self.encoder.update_goals_latent(goals)
        self.latent_dynamics.set_goals_latent(self.encoder.goals_latent)

    def load_pretrained_encoder(self, checkpoint_path: str, strict: bool = False):
        """
        사전 학습된 인코더 로드

        Args:
            checkpoint_path: 체크포인트 경로
            strict: 엄격한 로딩 여부
        """
        self.encoder.load_pretrained(checkpoint_path, strict=strict)

    def load_pretrained_models(
        self, 
        encoder_path: Optional[str] = None, 
        decoder_path: Optional[str] = None,
        latent_dynamics_path: Optional[str] = None,
        state_dynamics_path: Optional[str] = None,
        strict: bool = False
    ):
        """
        사전 학습된 모델 로드

        Args:
            encoder_path: 인코더 체크포인트 경로
            decoder_path: 디코더 체크포인트 경로
            latent_dynamics_path: 잠재 다이나믹스 체크포인트 경로
            state_dynamics_path: 상태 다이나믹스 체크포인트 경로
            strict: 엄격한 로딩 여부
        """
        if encoder_path:
            self.encoder.load_pretrained(encoder_path, strict)
        
        if decoder_path:
            self.decoder.load_pretrained(decoder_path, strict)
        
        if latent_dynamics_path:
            self.latent_dynamics.load_state_dict(
                torch.load(latent_dynamics_path, map_location=self.device), 
                strict=strict
            )
            print(f"Loaded latent dynamics checkpoint from {latent_dynamics_path}")
        
        if state_dynamics_path:
            self.state_dynamics.load_pretrained(state_dynamics_path, strict)

    def set_normalization_params(
        self, 
        vel_min: torch.Tensor, 
        vel_max: torch.Tensor, 
        acc_min: Optional[torch.Tensor] = None, 
        acc_max: Optional[torch.Tensor] = None
    ):
        """
        상태 다이나믹스 정규화 파라미터 설정

        Args:
            vel_min: 최소 속도
            vel_max: 최대 속도
            acc_min: 최소 가속도 (2차 시스템용)
            acc_max: 최대 가속도 (2차 시스템용)
        """
        self.state_dynamics.set_normalization_params(vel_min, vel_max, acc_min, acc_max)

    def train_autoencoder(
        self, 
        states: torch.Tensor, 
        primitive_types: torch.Tensor, 
        epochs: int = 100,
        batch_size: int = 128,
        log_interval: int = 10
    ):
        """
        오토인코더(인코더+디코더) 학습

        Args:
            states: (N, dim_state) 상태 데이터
            primitive_types: (N,) 프리미티브 인덱스
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            log_interval: 로깅 간격
        """
        if self.encoder_optimizer is None or self.decoder_optimizer is None:
            self.configure_optimizers()

        dataset_size = states.shape[0]
        indices = torch.randperm(dataset_size)
        
        self.encoder.train()
        self.decoder.train()
        
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            batches = 0
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # 배치 데이터
                state_batch = states[batch_indices].to(self.device)
                prim_batch = primitive_types[batch_indices].to(self.device)
                
                # 오토인코더 순전파
                latent = self.encoder(state_batch, prim_batch)
                recon_state = self.decoder(latent, prim_batch)
                
                # 손실 계산
                loss = nn.functional.mse_loss(recon_state, state_batch)
                
                # 최적화
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            avg_loss = total_loss / batches
            losses.append(avg_loss)
            
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"에포크 {epoch+1}/{epochs}, 손실: {avg_loss:.6f}")
        
        self.encoder.eval()
        self.decoder.eval()
        return losses    
    
    def generate_trajectory_latent(
        self, 
        initial_state: torch.Tensor, 
        primitive_type: torch.Tensor, 
        steps: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        잠재 공간 궤적 생성 (단순 오일러 적분 기반)

        Args:
            initial_state: (B, dim_state) 초기 상태
            primitive_type: (B,) 프리미티브 인덱스
            steps: 생성할 스텝 수

        Returns:
            (B, steps, dim_state) 상태 공간 궤적,
            (B, steps, latent_dim) 잠재 공간 궤적
        """
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            # 초기 상태를 잠재 공간으로 매핑
            z_init = self.encoder(initial_state, primitive_type)
            
            # 목표 잠재 상태 가져오기
            if primitive_type.ndim == 2:
                idx = torch.argmax(primitive_type, dim=1)
            else:
                idx = primitive_type
                
            goals_latent = self.latent_dynamics.goals_latent[idx.long()]
            
            # 잠재 공간 궤적을 저장할 텐서
            batch_size = initial_state.shape[0]
            latent_traj = torch.zeros(batch_size, steps, self.latent_dim, device=self.device)
            state_traj = torch.zeros(batch_size, steps, self.dim_state, device=self.device)
            
            # 초기 상태
            z_t = z_init
            
            # 순수 오일러 적분으로 궤적 생성
            for t in range(steps):
                latent_traj[:, t] = z_t
                state_traj[:, t] = self.decoder(z_t, primitive_type)
                
                # 목표와의 차이를 계산
                delta_z = goals_latent - z_t
                
                # 오일러 적분
                z_t = z_t + delta_z * self.latent_dynamics.delta_t
            
        return state_traj, latent_traj    
    
    def generate_trajectory_state(
        self, 
        initial_state: torch.Tensor, 
        primitive_type: torch.Tensor, 
        steps: int = 100
    ) -> torch.Tensor:
        """
        상태 공간 궤적 생성 (단순 오일러 적분 기반)

        Args:
            initial_state: (B, dim_state) 초기 상태
            primitive_type: (B,) 프리미티브 인덱스
            steps: 생성할 스텝 수

        Returns:
            (B, steps, dim_state) 상태 공간 궤적
        """
        with torch.no_grad():
            # 목표 가져오기
            if primitive_type.ndim == 2:
                idx = torch.argmax(primitive_type, dim=1)
            else:
                idx = primitive_type
                
            goals = self.state_dynamics.goals[idx.long()]
            
            # 궤적 저장용 텐서
            batch_size = initial_state.shape[0]
            state_traj = torch.zeros(batch_size, steps, self.dim_state, device=self.device)
            
            # 초기 상태
            x_t = initial_state
            
            # 순수 오일러 적분으로 궤적 생성
            for t in range(steps):
                state_traj[:, t] = x_t
                
                if self.dynamical_system_order == 2:
                    # 2차 역학계: 위치와 속도가 있음
                    pos = x_t[:, :self.workspace_dim]
                    vel = x_t[:, self.workspace_dim:]
                    
                    # 목표로 향하는 가속도 계산
                    pos_goals = goals[:, :self.workspace_dim]
                    dir_to_goal = pos_goals - pos
                    acc = dir_to_goal - self.state_dynamics.damping * vel
                    
                    # 오일러 적분
                    delta_t = self.state_dynamics.delta_t
                    new_vel = vel + acc * delta_t
                    new_pos = pos + vel * delta_t
                    
                    # 상태 업데이트
                    x_t = torch.cat([new_pos, new_vel], dim=1)
                else:
                    # 1차 역학계: 위치만 있음
                    pos = x_t
                    pos_goals = goals
                    dir_to_goal = pos_goals - pos
                    
                    # 오일러 적분
                    new_pos = pos + dir_to_goal * self.state_dynamics.delta_t
                    x_t = new_pos
                
        return state_traj

    def save_models(self, save_dir: str):
        """
        모든 모델 저장

        Args:
            save_dir: 저장 디렉토리
        """
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir, "decoder.pt"))
        torch.save(self.latent_dynamics.state_dict(), os.path.join(save_dir, "latent_dynamics.pt"))
        torch.save(self.state_dynamics.state_dict(), os.path.join(save_dir, "state_dynamics.pt"))
        
        print(f"모든 모델이 {save_dir}에 저장됨")


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
        
        # 목표 방향으로 이동 생성        direction = goals[prim] - pos
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


def main():
    """메인 실행 함수"""
    # 인수 파싱
    parser = argparse.ArgumentParser(description='CONDOR 기반 모션 프리미티브 모델')
    parser.add_argument('--latent_dim', type=int, default=8, help='잠재 차원')
    parser.add_argument('--workspace_dim', type=int, default=2, help='작업 공간 차원')
    parser.add_argument('--order', type=int, default=2, help='동역학계 차수 (1 또는 2)')
    parser.add_argument('--multi', action='store_true', help='다중 동작 지원')
    parser.add_argument('--save_dir', type=str, default='results/model', help='저장 디렉토리')
    parser.add_argument('--data_dir', type=str, default='results/data', help='데이터 저장 디렉토리')
    parser.add_argument('--encoder_path', type=str, default=None, help='인코더 체크포인트 경로')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에포크')
    parser.add_argument('--batch_size', type=int, default=128, help='배치 크기')
    parser.add_argument('--use_cached_data', action='store_true', help='캐시된 데이터 사용')
    parser.add_argument('--dataset', type=str, default='LAIR', help='데이터셋 이름')
    parser.add_argument('--primitives', type=str, default='0', help='사용할 프리미티브 ID')
    parser.add_argument('--visualize', action='store_true', help='궤적 시각화')
    parser.add_argument('--config', type=str, default=None, help='YAML 설정 파일 경로')  # YAML 설정 파일 인수 추가
    
    args = parser.parse_args()
      # YAML 설정 파일 로드 (있는 경우)
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        
        # 데이터 파이프라인 설정 적용
        if 'data' in config:
            data_config = config['data']
            if 'dataset_name' in data_config:
                args.dataset = data_config['dataset_name']
            if 'selected_primitives_ids' in data_config:
                args.primitives = data_config['selected_primitives_ids']
            if 'workspace_dimensions' in data_config:
                args.workspace_dim = data_config['workspace_dimensions']
            if 'dynamical_system_order' in data_config:
                args.order = data_config['dynamical_system_order']
            if 'batch_size' in data_config:
                args.batch_size = data_config['batch_size']
            if 'use_cached_data' in data_config:
                args.use_cached_data = data_config['use_cached_data']
            if 'data_dir' in data_config:
                args.data_dir = data_config['data_dir']
        
        # 모델 설정 적용
        if 'model' in config:
            model_config = config['model']
            if 'latent_dim' in model_config:
                args.latent_dim = model_config['latent_dim']
            if 'multi_motion' in model_config:
                args.multi = model_config['multi_motion']
        
        # 학습 설정 적용
        if 'training' in config:
            training_config = config['training']
            if 'epochs' in training_config:
                args.epochs = training_config['epochs']
        
        # 경로 설정 적용
        if 'paths' in config:
            paths_config = config['paths']
            if 'save_dir' in paths_config:
                args.save_dir = paths_config['save_dir']
            if 'encoder_path' in paths_config and paths_config['encoder_path'] is not None:
                args.encoder_path = paths_config['encoder_path']
        
        # 시각화 설정 적용
        if 'visualization' in config:
            vis_config = config['visualization']
            if 'enable' in vis_config:
                args.visualize = vis_config['enable']
                
        print("YAML 설정 파일에서 설정을 불러왔습니다:", args.config)
    
    # 데이터 파이프라인 파라미터 설정
    data_params = DataPipelineParams(
        workspace_dimensions=args.workspace_dim,
        dynamical_system_order=args.order,
        dataset_name=args.dataset,
        selected_primitives_ids=args.primitives,
        batch_size=args.batch_size
    )
    
    # YAML 설정 파일에서 추가 데이터 파이프라인 파라미터 적용
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            
        if 'data' in config:
            data_config = config['data']
            # DataPipelineParams의 모든 필드 확인 및 설정
            for field_name in [f for f in dir(DataPipelineParams) if not f.startswith('_')]:
                if field_name in data_config:
                    setattr(data_params, field_name, data_config[field_name])
    
    # 데이터 캐시 경로
    data_cache_path = os.path.join(args.data_dir, f"{args.dataset}_{args.primitives}_order{args.order}.npz")
    os.makedirs(args.data_dir, exist_ok=True)
    
    # 데이터 로딩 및 전처리
    if args.use_cached_data and os.path.exists(data_cache_path):
        print(f"캐시된 데이터를 로드합니다: {data_cache_path}")
        from data_pipeline import load_preprocessed_data
        preprocessed_data = load_preprocessed_data(data_cache_path)
        
        # 데이터셋 및 로더 생성
        pipeline = DataPipeline(data_params)
        dataset = pipeline.create_dataset(preprocessed_data)
        loader = pipeline.create_data_loader(dataset)
    else:
        print("데이터 파이프라인 실행 중...")
        pipeline = DataPipeline(data_params)
        preprocessed_data, dataset, loader = pipeline.run()
        
        # 데이터 캐싱
        from data_pipeline import save_preprocessed_data
        save_preprocessed_data(preprocessed_data, data_cache_path)
    
    # 기본 설정
    if args.order == 1:
        velocity_dim = 0  # 1차 시스템은 속도 없음
    else:
        velocity_dim = args.workspace_dim  # 2차 시스템은 위치와 동일한 차원의 속도 벡터
        
    # 상태 차원 계산
    dim_state = args.workspace_dim + velocity_dim
    
    # 프리미티브 수 확인
    n_primitives = len(np.unique(preprocessed_data['demonstrations primitive id']))
    print(f"프리미티브 수: {n_primitives}")
      # 모델 초기화를 위한 파라미터 준비
    model_params = {
        'dim_state': dim_state,
        'latent_dim': args.latent_dim,
        'n_primitives': n_primitives,
        'dynamical_system_order': args.order,
        'workspace_dim': args.workspace_dim,
        'multi_motion': args.multi
    }
    
    # YAML 설정 파일에서 모델 구조 파라미터 적용
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        
        if 'model' in config:
            model_config = config['model']
            
            # 은닉층 크기 설정
            if 'encoder_hidden_sizes' in model_config:
                model_params['encoder_hidden_sizes'] = model_config['encoder_hidden_sizes']
                
            if 'decoder_hidden_sizes' in model_config:
                model_params['decoder_hidden_sizes'] = model_config['decoder_hidden_sizes']
                
            if 'latent_dynamics_hidden_sizes' in model_config:
                model_params['latent_dynamics_hidden_sizes'] = model_config['latent_dynamics_hidden_sizes']
                
            if 'state_dynamics_hidden_sizes' in model_config:
                model_params['state_dynamics_hidden_sizes'] = model_config['state_dynamics_hidden_sizes']
    
    # 모델 초기화
    model = MotionPrimitiveModel(**model_params)
    
    # 사전 학습된 인코더 로드 (있는 경우)
    if args.encoder_path:
        model.load_pretrained_encoder(args.encoder_path)
        print(f"인코더가 {args.encoder_path}에서 로드됨")
        
    # 목표 설정 (데이터에서 가져오기)
    goals = torch.from_numpy(preprocessed_data['goals training']).float().to(model.device)
    model.set_goals(goals)
    
    # 속도/가속도 정규화 파라미터 (데이터셋에서 가져옴)
    vel_min = torch.from_numpy(preprocessed_data['vel min train'].reshape(1, -1, 1)).float().to(model.device)
    vel_max = torch.from_numpy(preprocessed_data['vel max train'].reshape(1, -1, 1)).float().to(model.device)
    
    if args.order == 2:
        if 'acc min train' in preprocessed_data and 'acc max train' in preprocessed_data:
            acc_min = torch.from_numpy(preprocessed_data['acc min train'].reshape(1, -1, 1)).float().to(model.device)
            acc_max = torch.from_numpy(preprocessed_data['acc max train'].reshape(1, -1, 1)).float().to(model.device)
        else:
            print("가속도 정규화 파라미터가 없어 기본값을 사용합니다.")
            acc_min = torch.full((1, args.workspace_dim, 1), -1.0, device=model.device)
            acc_max = torch.full((1, args.workspace_dim, 1), 1.0, device=model.device)
            
        model.set_normalization_params(vel_min, vel_max, acc_min, acc_max)
    else:
        model.set_normalization_params(vel_min, vel_max)          # 옵티마이저 설정을 위한 파라미터 준비 (인코더와 디코더만)
    optimizer_params = {
        'lr_encoder': 1e-3,
        'lr_decoder': 1e-3,
        'weight_decay': 1e-5
    }
    
    # YAML 설정 파일에서 옵티마이저 파라미터 적용
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        
        if 'training' in config:
            training_config = config['training']
            
            # 학습률 설정
            if 'learning_rates' in training_config:
                lr_config = training_config['learning_rates']
                if 'encoder' in lr_config:
                    optimizer_params['lr_encoder'] = lr_config['encoder']
                if 'decoder' in lr_config:
                    optimizer_params['lr_decoder'] = lr_config['decoder']
                    
            # 가중치 감쇠 설정
            if 'weight_decay' in training_config:
                optimizer_params['weight_decay'] = training_config['weight_decay']
    
    # 옵티마이저 설정
    model.configure_optimizers(**optimizer_params)
    
    # 학습용 데이터셋 준비
    print("학습 데이터셋 준비 중...")
    # 데이터 로더에서 전체 데이터 추출
    all_states = []
    all_next_states = []
    all_prim_ids = []
    
    for x_t, x_tp1, prim_id, _, _ in loader:
        all_states.append(x_t)
        all_next_states.append(x_tp1)
        all_prim_ids.append(prim_id)
    
    # 데이터 결합
    states = torch.cat(all_states, dim=0).to(model.device)
    next_states = torch.cat(all_next_states, dim=0).to(model.device)
    primitive_types = torch.cat(all_prim_ids, dim=0).to(model.device)
    
    print(f"학습 데이터 준비 완료: {states.shape[0]} 샘플")
      # 학습 에포크 설정
    training_epochs = {
        'autoencoder': args.epochs,
        'latent_dynamics': args.epochs,
        'state_dynamics': args.epochs
    }
    
    # YAML 설정 파일에서 개별 에포크 설정 적용
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        
        if 'training' in config and 'epochs' in config['training']:
            # 기본 에포크 설정
            default_epochs = config['training']['epochs']
            training_epochs = {k: default_epochs for k in training_epochs.keys()}
            
            # 개별 모델별 에포크 설정 (있는 경우)
            if 'epochs_autoencoder' in config['training']:
                training_epochs['autoencoder'] = config['training']['epochs_autoencoder']
            if 'epochs_latent_dynamics' in config['training']:
                training_epochs['latent_dynamics'] = config['training']['epochs_latent_dynamics']
            if 'epochs_state_dynamics' in config['training']:
                training_epochs['state_dynamics'] = config['training']['epochs_state_dynamics']
      # 오토인코더 학습 (인코더와 디코더만 학습)
    print(f"\n오토인코더 학습 시작... ({training_epochs['autoencoder']} 에포크)")
    ae_losses = model.train_autoencoder(states, primitive_types, epochs=training_epochs['autoencoder'])
    
    # 다이나믹스 모델은 학습 없이 초기화
    print("\n다이나믹스 모델 초기화 (학습 없음)...")
    
    # 잠재 다이나믹스 초기화 (오일러 적분용)
    print("단순 오일러 적분 기반 잠재 다이나믹스 모델 초기화 완료")
    model.latent_dynamics.delta_t = 1.0
    ld_losses = [0.0]  # 더미 손실값
    
    # 상태 다이나믹스 초기화 (오일러 적분용)
    print("단순 오일러 적분 기반 상태 다이나믹스 모델 초기화 완료")
    model.state_dynamics.delta_t = 1.0
    if model.dynamical_system_order == 2:
        model.state_dynamics.damping = 0.1
    sd_losses = {"total": [0.0], "pos": [0.0]}
    if model.dynamical_system_order == 2:
        sd_losses["vel"] = [0.0]
    
    # 모델 저장
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_models(args.save_dir)
      # 학습 손실 그래프 저장 (오토인코더만)
    plt.figure(figsize=(6, 4))
    
    plt.plot(ae_losses)
    plt.title('오토인코더 손실')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "training_losses.png"))
      # 궤적 시각화
    if args.visualize:
        print("\n궤적 생성 및 시각화 중...")
        
        # 시각화 설정 파라미터
        viz_params = {
            'max_samples': 6,  # 최대 샘플 수
            'n_samples_per_primitive': 3,  # 프리미티브당 샘플 수
            'steps': 100  # 궤적 생성 스텝 수
        }
        
        # YAML 설정 파일에서 시각화 설정 적용
        if args.config:
            with open(args.config, 'r') as file:
                config = yaml.safe_load(file)
            
            if 'visualization' in config:
                viz_config = config['visualization']
                if 'max_samples' in viz_config:
                    viz_params['max_samples'] = viz_config['max_samples']
                if 'n_samples_per_primitive' in viz_config:
                    viz_params['n_samples_per_primitive'] = viz_config['n_samples_per_primitive']
                if 'steps' in viz_config:
                    viz_params['steps'] = viz_config['steps']
        
        # 테스트 샘플 선택
        n_test = min(viz_params['max_samples'], n_primitives * viz_params['n_samples_per_primitive'])
        test_indices = []
        for p in range(n_primitives):
            # 각 프리미티브에서 n_samples_per_primitive개씩 샘플 선택
            prim_indices = torch.where(primitive_types == p)[0][:viz_params['n_samples_per_primitive']]
            test_indices.extend(prim_indices.tolist())
        
        test_indices = test_indices[:n_test]  # 총 샘플 수 제한
        test_states = states[test_indices]
        test_prims = primitive_types[test_indices]
        
        # 잠재 공간 궤적
        plt_latent = visualize_trajectories(
            model, test_states, test_prims, steps=viz_params['steps'], latent=True,
            title="잠재 공간에서 생성된 궤적"
        )
        plt_latent.savefig(os.path.join(args.save_dir, "latent_trajectories.png"))
        
        # 상태 공간 궤적
        plt_state = visualize_trajectories(
            model, test_states, test_prims, steps=viz_params['steps'], latent=False,
            title="상태 공간에서 생성된 궤적"
        )
        plt_state.savefig(os.path.join(args.save_dir, "state_trajectories.png"))
        
        print(f"궤적 시각화가 {args.save_dir}에 저장됨")


if __name__ == "__main__":
    main()
