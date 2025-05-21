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
        latent_dynamics_hidden_sizes: List[int] = [128, 128],
        state_dynamics_hidden_sizes: List[int] = [256, 256],
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
            hidden_sizes=latent_dynamics_hidden_sizes,
            multi_motion=multi_motion,
            device=device
        )

        self.state_dynamics = StateDynamics(
            dim_state=dim_state,
            n_primitives=n_primitives,
            hidden_sizes=state_dynamics_hidden_sizes,
            multi_motion=multi_motion,
            dynamical_system_order=dynamical_system_order,
            workspace_dim=workspace_dim,
            device=device
        )

        # 최적화 도구
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.latent_dynamics_optimizer = None
        self.state_dynamics_optimizer = None

    def configure_optimizers(
        self, 
        lr_encoder: float = 1e-4,
        lr_decoder: float = 1e-4, 
        lr_latent_dynamics: float = 1e-3,
        lr_state_dynamics: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """
        옵티마이저 설정

        Args:
            lr_encoder: 인코더 학습률
            lr_decoder: 디코더 학습률
            lr_latent_dynamics: 잠재 다이나믹스 학습률
            lr_state_dynamics: 상태 다이나믹스 학습률
            weight_decay: 가중치 감쇠
        """
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=lr_encoder, weight_decay=weight_decay
        )
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=lr_decoder, weight_decay=weight_decay
        )
        self.latent_dynamics_optimizer = optim.Adam(
            self.latent_dynamics.parameters(), lr=lr_latent_dynamics, weight_decay=weight_decay
        )
        self.state_dynamics_optimizer = optim.Adam(
            self.state_dynamics.parameters(), lr=lr_state_dynamics, weight_decay=weight_decay
        )

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

    def train_latent_dynamics(
        self, 
        states: torch.Tensor, 
        next_states: torch.Tensor, 
        primitive_types: torch.Tensor, 
        epochs: int = 100,
        batch_size: int = 128,
        log_interval: int = 10
    ):
        """
        잠재 다이나믹스 모델 학습

        Args:
            states: (N, dim_state) 현재 상태 데이터
            next_states: (N, dim_state) 다음 상태 데이터
            primitive_types: (N,) 프리미티브 인덱스
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            log_interval: 로깅 간격
        """
        if self.latent_dynamics_optimizer is None:
            self.configure_optimizers()

        dataset_size = states.shape[0]
        indices = torch.randperm(dataset_size)
        
        self.encoder.eval()  # 인코더는 고정
        self.latent_dynamics.train()
        
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            batches = 0
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # 배치 데이터
                state_batch = states[batch_indices].to(self.device)
                next_state_batch = next_states[batch_indices].to(self.device)
                prim_batch = primitive_types[batch_indices].to(self.device)
                
                # 인코딩
                with torch.no_grad():
                    latent_curr = self.encoder(state_batch, prim_batch)
                    latent_next = self.encoder(next_state_batch, prim_batch)
                
                # 잠재 다이나믹스 순전파
                latent_next_pred = self.latent_dynamics(latent_curr, prim_batch)
                
                # 손실 계산
                loss = nn.functional.mse_loss(latent_next_pred, latent_next)
                
                # 최적화
                self.latent_dynamics_optimizer.zero_grad()
                loss.backward()
                self.latent_dynamics_optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            avg_loss = total_loss / batches
            losses.append(avg_loss)
            
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"에포크 {epoch+1}/{epochs}, 손실: {avg_loss:.6f}")
        
        self.latent_dynamics.eval()
        return losses

    def train_state_dynamics(
        self, 
        states: torch.Tensor, 
        next_states: torch.Tensor, 
        primitive_types: torch.Tensor, 
        epochs: int = 100,
        batch_size: int = 128,
        log_interval: int = 10
    ):
        """
        상태 다이나믹스 모델 학습

        Args:
            states: (N, dim_state) 현재 상태 데이터
            next_states: (N, dim_state) 다음 상태 데이터
            primitive_types: (N,) 프리미티브 인덱스
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            log_interval: 로깅 간격
        """
        if self.state_dynamics_optimizer is None:
            self.configure_optimizers()

        dataset_size = states.shape[0]
        indices = torch.randperm(dataset_size)
        
        self.state_dynamics.train()
        
        losses = {
            "total": [],
            "pos": [],
        }
        
        if self.dynamical_system_order == 2:
            losses["vel"] = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            pos_loss_sum = 0.0
            vel_loss_sum = 0.0
            batches = 0
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # 배치 데이터
                state_batch = states[batch_indices].to(self.device)
                next_state_batch = next_states[batch_indices].to(self.device)
                prim_batch = primitive_types[batch_indices].to(self.device)
                
                # 손실 계산 (compute_loss 메서드 사용)
                loss_dict = self.state_dynamics.compute_loss(
                    state_batch, next_state_batch, prim_batch
                )
                
                # 최적화
                self.state_dynamics_optimizer.zero_grad()
                loss_dict["total"].backward()
                self.state_dynamics_optimizer.step()
                
                total_loss += loss_dict["total"].item()
                pos_loss_sum += loss_dict["pos"].item()
                if self.dynamical_system_order == 2:
                    vel_loss_sum += loss_dict["vel"].item()
                    
                batches += 1
            
            avg_loss = total_loss / batches
            avg_pos_loss = pos_loss_sum / batches
            losses["total"].append(avg_loss)
            losses["pos"].append(avg_pos_loss)
            
            if self.dynamical_system_order == 2:
                avg_vel_loss = vel_loss_sum / batches
                losses["vel"].append(avg_vel_loss)
                vel_info = f", 속도 손실: {avg_vel_loss:.6f}"
            else:
                vel_info = ""
            
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"에포크 {epoch+1}/{epochs}, 총 손실: {avg_loss:.6f}, 위치 손실: {avg_pos_loss:.6f}{vel_info}")
        
        self.state_dynamics.eval()
        return losses

    def generate_trajectory_latent(
        self, 
        initial_state: torch.Tensor, 
        primitive_type: torch.Tensor, 
        steps: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        잠재 공간 궤적 생성

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
        self.latent_dynamics.eval()

        with torch.no_grad():
            # 초기 상태를 잠재 공간으로 매핑
            z_init = self.encoder(initial_state, primitive_type)
            
            # 잠재 공간에서 궤적 예측
            latent_traj = self.latent_dynamics.multi_step_prediction(
                z_init, primitive_type, steps
            )
            
            # 상태 공간으로 디코딩
            batch_size = initial_state.shape[0]
            state_traj = torch.zeros(batch_size, steps, self.dim_state, device=self.device)
            
            for t in range(steps):
                state_traj[:, t] = self.decoder(latent_traj[:, t], primitive_type)
                
        return state_traj, latent_traj

    def generate_trajectory_state(
        self, 
        initial_state: torch.Tensor, 
        primitive_type: torch.Tensor, 
        steps: int = 100
    ) -> torch.Tensor:
        """
        상태 공간 궤적 생성

        Args:
            initial_state: (B, dim_state) 초기 상태
            primitive_type: (B,) 프리미티브 인덱스
            steps: 생성할 스텝 수

        Returns:
            (B, steps, dim_state) 상태 공간 궤적
        """
        self.state_dynamics.eval()

        with torch.no_grad():
            # 상태 공간에서 궤적 예측
            state_traj = self.state_dynamics.multi_step_prediction(
                initial_state, primitive_type, steps
            )
                
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
        
        # 목표 방향으로 이동 생성
        direction = goals[prim] - pos
        norm = torch.norm(direction) + 1e-8
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
    """
    if latent:
        traj, _ = model.generate_trajectory_latent(init_state, primitive_type, steps)
    else:
        traj = model.generate_trajectory_state(init_state, primitive_type, steps)
        
    # 위치만 추출
    pos_traj = traj[:, :, :model.workspace_dim]
    
    # 프리미티브 별 색상
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    
    plt.figure(figsize=(10, 8))
    
    # 각 배치 궤적 그리기
    batch_size = init_state.shape[0]
    for i in range(batch_size):
        prim = primitive_type[i].item()
        color = colors[prim % len(colors)]
        
        # 2D 또는 3D 플롯
        if model.workspace_dim == 2:
            plt.plot(
                pos_traj[i, :, 0].cpu().numpy(), 
                pos_traj[i, :, 1].cpu().numpy(),
                color=color, 
                marker='o', 
                markersize=3,
                label=f"Primitive {prim}" if i == 0 or primitive_type[i] != primitive_type[i-1] else ""
            )
            plt.plot(
                pos_traj[i, 0, 0].cpu().numpy(), 
                pos_traj[i, 0, 1].cpu().numpy(),
                'o', 
                color=color, 
                markersize=8
            )
        elif model.workspace_dim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.gca(projection='3d')
            ax.plot(
                pos_traj[i, :, 0].cpu().numpy(), 
                pos_traj[i, :, 1].cpu().numpy(),
                pos_traj[i, :, 2].cpu().numpy(),
                color=color, 
                marker='o', 
                markersize=3,
                label=f"Primitive {prim}" if i == 0 or primitive_type[i] != primitive_type[i-1] else ""
            )
            ax.plot(
                [pos_traj[i, 0, 0].cpu().numpy()], 
                [pos_traj[i, 0, 1].cpu().numpy()],
                [pos_traj[i, 0, 2].cpu().numpy()],
                'o', 
                color=color, 
                markersize=8
            )
    
    # 목표점 그리기
    if model.workspace_dim == 2:
        for p in range(model.n_primitives):
            goal = model.state_dynamics.goals[p].cpu().numpy()
            plt.plot(goal[0], goal[1], '*', color=colors[p % len(colors)], markersize=15)
    elif model.workspace_dim == 3:
        for p in range(model.n_primitives):
            goal = model.state_dynamics.goals[p].cpu().numpy()
            ax.plot([goal[0]], [goal[1]], [goal[2]], '*', color=colors[p % len(colors)], markersize=15)

    # 그래프 설정
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()


def main():
    """메인 실행 함수"""
    # 인수 파싱
    parser = argparse.ArgumentParser(description='CONDOR 기반 모션 프리미티브 모델')
    parser.add_argument('--dim_state', type=int, default=4, help='상태 차원')
    parser.add_argument('--latent_dim', type=int, default=8, help='잠재 차원')
    parser.add_argument('--n_primitives', type=int, default=2, help='프리미티브 수')
    parser.add_argument('--order', type=int, default=2, help='동역학계 차수 (1 또는 2)')
    parser.add_argument('--workspace_dim', type=int, default=2, help='작업 공간 차원')
    parser.add_argument('--multi', action='store_true', help='다중 동작 지원')
    parser.add_argument('--save_dir', type=str, default='results', help='저장 디렉토리')
    parser.add_argument('--encoder_path', type=str, default=None, help='인코더 체크포인트 경로')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에포크')
    parser.add_argument('--visualize', action='store_true', help='궤적 시각화')
    parser.add_argument('--data_path', type=str, default='data.json', help='데이터 파일 경로')
    
    args = parser.parse_args()
    
    # 기본 설정
    if args.order == 1:
        velocity_dim = 0  # 1차 시스템은 속도 없음
    else:
        velocity_dim = args.workspace_dim  # 2차 시스템은 위치와 동일한 차원의 속도 벡터
        
    # 상태 차원 확인
    dim_state = args.workspace_dim + velocity_dim
    if dim_state != args.dim_state:
        print(f"주의: 계산된 상태 차원({dim_state})이 제공된 dim_state({args.dim_state})와 다릅니다. 계산된 값을 사용합니다.")
        args.dim_state = dim_state
        
    # 모델 초기화
    model = MotionPrimitiveModel(
        dim_state=args.dim_state,
        latent_dim=args.latent_dim,
        n_primitives=args.n_primitives,
        dynamical_system_order=args.order,
        workspace_dim=args.workspace_dim,
        multi_motion=args.multi
    )
    
    # 사전 학습된 인코더 로드 (있는 경우)
    if args.encoder_path:
        model.load_pretrained_encoder(args.encoder_path)
        print(f"인코더가 {args.encoder_path}에서 로드됨")
        
    # 데이터 파이프라인 설정
    print("데이터 파이프라인 설정 중...")
    with open(args.data_path, 'r') as f:
        data_config = json.load(f)
    
    data_params = DataPipelineParams(
        file_path=data_config["file_path"],
        batch_size=data_config["batch_size"],
        shuffle=data_config["shuffle"],
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
        drop_last=data_config["drop_last"]
    )
    
    data_pipeline = DataPipeline(params=data_params)
    
    # 데이터셋에서 첫 번째 배치 가져오기 (모델 입력 크기 확인용)
    for batch in data_pipeline:
        print(f"첫 번째 배치 크기: {batch['states'].size()}")
        break
    
    # 목표 설정
    model.set_goals(data_pipeline.goals)
    
    # 속도/가속도 정규화 파라미터 (실제로는 데이터셋에서 계산해야 함)
    vel_min = torch.full(
        (1, args.workspace_dim, 1), -2.0, device=model.device
    )
    vel_max = torch.full(
        (1, args.workspace_dim, 1), 2.0, device=model.device
    )
    
    if args.order == 2:
        acc_min = torch.full(
            (1, args.workspace_dim, 1), -1.0, device=model.device
        )
        acc_max = torch.full(
            (1, args.workspace_dim, 1), 1.0, device=model.device
        )
        model.set_normalization_params(vel_min, vel_max, acc_min, acc_max)
    else:
        model.set_normalization_params(vel_min, vel_max)
        
    # 옵티마이저 설정
    model.configure_optimizers(
        lr_encoder=1e-3,
        lr_decoder=1e-3,
        lr_latent_dynamics=1e-3,
        lr_state_dynamics=1e-3
    )
    
    # 오토인코더 학습
    print("\n오토인코더 학습 시작...")
    ae_losses = model.train_autoencoder(states, primitive_types, epochs=args.epochs)
    
    # 잠재 다이나믹스 학습
    print("\n잠재 다이나믹스 학습 시작...")
    ld_losses = model.train_latent_dynamics(states, next_states, primitive_types, epochs=args.epochs)
    
    # 상태 다이나믹스 학습
    print("\n상태 다이나믹스 학습 시작...")
    sd_losses = model.train_state_dynamics(states, next_states, primitive_types, epochs=args.epochs)
    
    # 모델 저장
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_models(args.save_dir)
    
    # 궤적 시각화
    if args.visualize:
        print("\n궤적 생성 및 시각화 중...")
        # 테스트 샘플 선택
        n_test = 6  # 프리미티브당 3개씩
        test_indices = []
        for p in range(args.n_primitives):
            # 각 프리미티브에서 3개씩 샘플 선택
            prim_indices = torch.where(primitive_types == p)[0][:3]
            test_indices.extend(prim_indices.tolist())
        
        test_indices = test_indices[:n_test]  # 총 샘플 수 제한
        test_states = states[test_indices]
        test_prims = primitive_types[test_indices]
        
        # 잠재 공간 궤적
        plt_latent = visualize_trajectories(
            model, test_states, test_prims, steps=100, latent=True,
            title="잠재 공간에서 생성된 궤적"
        )
        plt_latent.savefig(os.path.join(args.save_dir, "latent_trajectories.png"))
        
        # 상태 공간 궤적
        plt_state = visualize_trajectories(
            model, test_states, test_prims, steps=100, latent=False,
            title="상태 공간에서 생성된 궤적"
        )
        plt_state.savefig(os.path.join(args.save_dir, "state_trajectories.png"))
        
        print(f"궤적 시각화가 {args.save_dir}에 저장됨")


if __name__ == "__main__":
    main()
