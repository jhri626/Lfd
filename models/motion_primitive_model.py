#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모션 프리미티브 모델 클래스

이 모듈은 동작 원시 학습 및 생성을 위한 통합 모델을 제공합니다.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any

# 모델 가져오기
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_dynamics import LatentDynamics
from models.state_dynamics import StateDynamics


class MotionPrimitiveModel(nn.Module):
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
            device: 계산 장치
        """
        super(MotionPrimitiveModel, self).__init__()  # register module
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
        
        # 옵티마이저 초기화
        self.encoder_optimizer = None
        self.decoder_optimizer = None

    def configure_optimizers(self, lr_encoder=1e-4, lr_decoder=1e-4, weight_decay=1e-5):
        """Set up optimizers for encoder and decoder."""
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(),
                                            lr=lr_encoder, weight_decay=weight_decay)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                            lr=lr_decoder, weight_decay=weight_decay)
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
        epochs: int = 100,
        log_interval: int = 10,
        dataloader: Optional[torch.utils.data.DataLoader] = None
    ):
        """
        오토인코더(인코더+디코더) 학습 - 윈도우 기반 학습 지원 및 배치 처리 최적화

        Args:
            epochs: 학습 에포크 수
            log_interval: 로깅 간격
            dataloader: 외부에서 제공된 데이터로더 (있는 경우 사용)
                        데이터로더는 (windows, prim_ids, [full_trajectory, traj_idx, t_idx]) 형태를 반환해야 함
                        - windows: (B, dim_ws, window) 윈도우 상태 데이터
                        - prim_ids: (B,) 프리미티브 인덱스
                        - full_trajectory: (B, n_steps, dim_ws, window) 전체 궤적 데이터 (선택 사항)
                        - traj_idx: (B,) 궤적 인덱스 (선택 사항)
                        - t_idx: (B,) 시간 인덱스 (선택 사항)
        """
        if self.encoder_optimizer is None or self.decoder_optimizer is None:
            self.configure_optimizers()
        
        self.encoder.train()
        self.decoder.train()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = 0
            for batch_data in dataloader:
                # 데이터 로더의 반환값 형태에 따라 처리
                if len(batch_data) == 2:  # (windows, prim_ids) 형태 - 이전 형식 호환성
                    batch_windows, batch_prims = batch_data
                    has_trajectory = False
                elif len(batch_data) == 4:  # (windows, prim_ids, traj_idx, t_idx) 형태 - 이전 형식 호환성
                    batch_windows, batch_prims, _, _ = batch_data
                    has_trajectory = False
                elif len(batch_data) == 5:  # (windows, prim_ids, full_trajectory, traj_idx, t_idx) 형태 - 새로운 형식
                    batch_windows, batch_prims, batch_full_trajectories, batch_traj_idx, batch_t_idx = batch_data
                    has_trajectory = True
                else:
                    raise ValueError(f"지원되지 않는 데이터 형식: {len(batch_data)} 항목")
                
                # 배치 데이터를 디바이스로 이동
                batch_windows = batch_windows.to(self.device)  # (B, dim_ws, window)
                batch_prims = batch_prims.to(self.device)      # (B,)
                
                if has_trajectory:
                    batch_full_trajectories = batch_full_trajectories.to(self.device)  # (B, n_steps, dim_ws)
                    batch_traj_idx = batch_traj_idx.to(self.device)  # (B,)
                    batch_t_idx = batch_t_idx.to(self.device)  # (B,)
                
                # 윈도우 크기 확인
                window_size = batch_windows.shape[2]
                
                # 초기화 
                total_loss = 0.0
                triplet_gain = 0.4
                
                # 순차 처리 대신 배치로 처리 (PyTorch 효율적 처리)
                # 시작 위치는 항상 윈도우의 첫 번째 위치
                
                # First step
                
                # encoder 
                states = batch_windows[:, :, 0] + 0.1 * torch.randn_like(batch_windows[:, :, 0])# (B, dim_ws)
                batch_latent_trajectories = torch.cat([self.encoder(batch_full_trajectories[:, i, :]).unsqueeze(1) for i in range(batch_full_trajectories.size(1))], dim=1)
                
                for t in range(window_size - 1):
                    # 인코딩
                    latent_state = self.encoder(states, batch_prims)  # (B, latent_dim)
                    latent_in_latent = latent_state
                    goal_latent = self.compute_goal_latent()  # (B, latent_dim)
                    
                    # latent dynamics 손실 (첫 번째 스텝 제외)
                    if t > 0:
                        latent_in_latent = self.latent_dynamics(latent_in_latent, batch_latent_trajectories ,batch_prims)
                        latent_loss = nn.functional.mse_loss(latent_state, latent_in_latent)
                        total_loss += latent_loss
                        
                        triplet_loss = nn.functional.triplet_margin_loss(
                            goal_latent, latent_state, prev_latent_state, margin=1e-4) 
                        
                        total_loss += triplet_gain * triplet_loss
                    
                    # 디코딩 및 상태 동역학
                    decoder_out = self.decoder(latent_state, batch_prims)  # (B, dim_state)
                    next_states = self.state_dynamics(states, decoder_out, batch_prims)  # (B, dim_ws)
                    
                    # 상태 예측 손실
                    target_positions = batch_windows[:, :, t+1]  # (B, dim_ws)
                    state_loss = nn.functional.mse_loss(next_states, target_positions)
                    total_loss += state_loss
                    
                    # 다음 스텝을 위한 업데이트
                    prev_latent_state = latent_state
                    states = next_states
                
                # 모든 시간 스텝에 대한 평균 손실
                batch_loss = total_loss / (window_size - 1)
                
                # 최적화
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                batch_loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                
                epoch_loss += batch_loss.item()
                batches += 1
            
            avg_loss = epoch_loss / batches
            losses.append(avg_loss)
            
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"에포크 {epoch+1}/{epochs}, state_loss: {state_loss:.6f}, triplet_loss: {triplet_loss:.6f}, latent_loss: {latent_loss:.6f}")
        
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

    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
        
        print(f"모든 모델이 {save_dir}에 저장됨")

    def compute_goal_latent(self):
        """
        상태 다이나믹스의 goal을 encoder에 통과시켜 goal_latent를 계산합니다.
        """
        # 상태 다이나믹스의 goal 가져오기
        goals = self.state_dynamics.goals  # (n_primitives, workspace_dim)

        # goal을 encoder에 통과시켜 goal_latent 계산
        goal_latent = self.encoder(goals, torch.arange(self.n_primitives, device=self.device))  # (n_primitives, latent_dim)

        return goal_latent
