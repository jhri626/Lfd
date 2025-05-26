"""
상태 공간 다이나믹스 모델링 클래스
"""
import torch
import torch.nn as nn
import os
import numpy as np
from typing import Union, Optional, List, Dict, Tuple, Any


class StateDynamics(nn.Module):
    """
    상태 공간 다이나믹스 모델링
    
    * **입력**: 현재 상태 x_t (B, dim_state), primitive_type ids or one-hot (B,) or (B, n_primitives)
    * **출력**: 다음 상태 x_{t+1} (B, dim_state)
    """

    def __init__(
        self,
        dim_state: int,
        n_primitives: int = 1,
        multi_motion: bool = False,
        dynamical_system_order: int = 2,
        workspace_dim: int = 2,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        상태 다이나믹스 모델 초기화
        
        Args:
            dim_state (int): 상태 공간 차원
            n_primitives (int): 프리미티브 수
            multi_motion (bool): 여러 프리미티브 지원 여부
            dynamical_system_order (int): 동역학계 차수 (1차 또는 2차)
            workspace_dim (int): 작업 공간 차원 (예: 2D, 3D)
            device (str): 계산 장치
        """
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.n_primitives = n_primitives
        self.dim_state = dim_state
        self.workspace_dim = workspace_dim
        self.dynamical_system_order = dynamical_system_order
        self.multi_motion = multi_motion

        # 프리미티브 인코딩 (multi-motion인 경우)
        if multi_motion:
            self.register_buffer(
                "primitive_encodings", torch.eye(n_primitives, device=self.device)
            )
        
        # 목표점 저장 버퍼
        self.register_buffer(
            "goals", torch.zeros(n_primitives, workspace_dim * dynamical_system_order, device=self.device)
        )        # 네트워크 부분 제거: 단순 오일러 적분만 수행
        # 더이상 MLP를 사용하지 않음 - 직접 목표 위치와 현재 상태의 차이를 통해 계산
        
        # 상태 정규화/비정규화를 위한 버퍼
        # 속도 및 가속도 최소/최대값
        self.register_buffer("vel_min", torch.zeros(1, workspace_dim,  device=self.device))
        self.register_buffer("vel_max", torch.zeros(1, workspace_dim,  device=self.device))
        self.register_buffer("acc_min", torch.zeros(1, workspace_dim,  device=self.device))
        self.register_buffer("acc_max", torch.zeros(1, workspace_dim,  device=self.device))
        
        # 시간 간격
        self.delta_t = 1.0
        
        self.to(device)

    def encode_primitives(self, primitive_type: torch.Tensor) -> torch.Tensor:
        """
        프리미티브 타입을 원핫 인코딩으로 변환
        
        Args:
            primitive_type: (B,) 또는 (B, n_primitives) 텐서
            
        Returns:
            (B, n_primitives) 원핫 인코딩
        """
        if not self.multi_motion:
            return torch.zeros(primitive_type.shape[0], 0, device=self.device)
            
        if primitive_type.ndim == 1:
            # 인덱스를 원핫으로 변환
            batch_size = primitive_type.shape[0]
            one_hot = torch.zeros(batch_size, self.n_primitives, device=self.device)
            one_hot.scatter_(1, primitive_type.view(-1, 1), 1)
            return one_hot
        else:
            # 이미 원핫 또는 소프트맥스 형태
            return primitive_type

    def normalize_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        속도 정규화
        
        Args:
            velocity: (B, workspace_dim) 속도 벡터
            
        Returns:
            (B, workspace_dim) 정규화된 속도 [-1, 1]
        """
        
            
        # 정규화
        return 2.0 * (velocity - self.vel_min) / (self.vel_max - self.vel_min) - 1.0
    
    def denormalize_velocity(self, velocity_norm: torch.Tensor) -> torch.Tensor:
        """
        정규화된 속도를 원래 스케일로 복원
        
        Args:
            velocity_norm: (B, workspace_dim) 정규화된 속도 [-1, 1]
            
        Returns:
            (B, workspace_dim) 원래 스케일의 속도
        """
        # 3차원으로 변환: (B, D) -> (B, D, 1)
            
        # 비정규화
        velocity = 0.5 * (velocity_norm + 1.0) * (self.vel_max - self.vel_min) + self.vel_min
        
        # 원래 차원으로 되돌리기
        return velocity
    
    def normalize_acceleration(self, acceleration: torch.Tensor) -> torch.Tensor:
        """
        가속도 정규화
        
        Args:
            acceleration: (B, workspace_dim) 가속도 벡터
            
        Returns:
            (B, workspace_dim) 정규화된 가속도 [-1, 1]
        """

            
        # 정규화
        return 2.0 * (acceleration - self.acc_min) / (self.acc_max - self.acc_min) - 1.0
    
    def denormalize_acceleration(self, acceleration_norm: torch.Tensor) -> torch.Tensor:
        """
        정규화된 가속도를 원래 스케일로 복원
        
        Args:
            acceleration_norm: (B, workspace_dim) 정규화된 가속도 [-1, 1]
            
        Returns:
            (B, workspace_dim) 원래 스케일의 가속도
        """
        
        acceleration = 0.5 * (acceleration_norm + 1.0) * (self.acc_max - self.acc_min) + self.acc_min
        
        # 원래 차원으로 되돌리기
        return acceleration
        
    def forward(self, x: torch.Tensor, dx: torch.Tensor, primitive_type: torch.Tensor) -> torch.Tensor:
        """
        순전파: 현재 상태로부터 다음 상태 예측
        
        단순 오일러 적분:
        x_{t+1} = x_t + delta_x * delta_t (1차 시스템)
        또는 
        pos_{t+1} = pos_t + vel_t * delta_t + 0.5 * acc_t * delta_t^2
        vel_{t+1} = vel_t + acc_t * delta_t (2차 시스템)
        
        Args:
            x: (B, dim_state) 현재 상태
            primitive_type: (B,) 또는 (B, n_primitives) 프리미티브 ID 또는 원핫
            
        Returns:
            (B, dim_state) 다음 상태
        """
        x = x.to(self.device)
        primitive_type = primitive_type.to(self.device)
        
        # 1차 또는 2차 시스템에 따라 다음 상태 계산
        if self.dynamical_system_order == 1:
            # 1차 시스템: 단순 오일러 적분
            # 현재 위치에서 목표 위치로의 방향만 사용
            
            # 현재 위치
            pos_t = x
            
            # 목표 방향으로의 벡터
            vel_t = self.denormalize_velocity(dx)
            
            # 오일러 적분
            pos_tp1 = pos_t + vel_t * self.delta_t
            return pos_tp1
        elif self.dynamical_system_order == 2:
            pos_t = x[:, :self.workspace_dim]
            vel_t = x[:, self.workspace_dim:]
            acc_t= dx[:, self.workspace_dim:]
            
            # 정규화된 속도와 가속도를 실제 물리량으로 변환 (필요한 경우)
            vel_t = torch.clamp(vel_t, -1.0, 1.0)  # 속도를 [-1, 1] 범위로 클리핑
            acc_t = torch.clamp(acc_t, -1.0, 1.0)
            
            vel_t_denorm = self.denormalize_velocity(vel_t)
            acc_t_denorm = self.denormalize_acceleration(acc_t)
       
            
            # 오일러 적분
            vel_tp1_denorm = vel_t_denorm + acc_t_denorm * self.delta_t   
            pos_tp1 = pos_t + vel_tp1_denorm * self.delta_t
            
            # 속도 다시 정규화 (상태 벡터에 저장하기 위해)
            
            vel_tp1_norm = self.normalize_velocity(vel_tp1_denorm)
            vel_tp1_norm = vel_tp1_norm.squeeze(-1)
            
            
            # 다음 상태 벡터 구성
            next_state = torch.cat([pos_tp1, vel_tp1_norm], dim=1)
            
            return next_state
            
        else:
            raise ValueError("dynamical_system_order는 1 또는 2여야 합니다.")
            
    def multi_step_prediction(
        self, 
        x_init: torch.Tensor, 
        primitive_type: torch.Tensor, 
        steps: int = 100
    ) -> torch.Tensor:
        """
        여러 스텝의 상태 예측
        
        Args:
            x_init: (B, dim_state) 초기 상태
            primitive_type: (B,) 프리미티브 인덱스
            steps: 예측할 스텝 수
            
        Returns:
            (B, steps, dim_state) 예측된 궤적
        """
        batch_size = x_init.shape[0]
        trajectories = torch.zeros(batch_size, steps, self.dim_state, device=self.device)
        
        # 초기 상태
        x_t = x_init
        
        # 타깃 프리미티브의 목표 가져오기
        if primitive_type.ndim == 2:
            idx = torch.argmax(primitive_type, dim=1)
        else:
            idx = primitive_type.long()
        
        target_goals = self.goals[idx]
        
        # 순차적 예측
        for t in range(steps):
            trajectories[:, t] = x_t
            
            # 상태에 따라 델타(dx) 생성
            if self.dynamical_system_order == 1:
                # 1차 시스템: 목표를 향한 방향
                dx = target_goals - x_t
                dx_normalized = self.normalize_velocity(dx)
            else:
                # 2차 시스템: 목표로 향하는 가속도 계산
                pos = x_t[:, :self.workspace_dim]
                vel = x_t[:, self.workspace_dim:]
                
                pos_goals = target_goals
                dir_to_goal = pos_goals - pos
                # 감쇠 효과 (종말 속도 0으로 수렴하도록)
                damping = 0.1
                acc = dir_to_goal - damping * self.denormalize_velocity(vel)
                
                # 가속도 정규화
                acc_normalized = self.normalize_acceleration(acc)
                # 정규화된 속도와 가속도를 연결
                dx_normalized = torch.cat([vel, acc_normalized], dim=1)
            
            # 다음 상태 계산
            x_t = self.forward(x_t, dx_normalized, primitive_type)
            
        return trajectories
    
    def set_normalization_params(
        self,
        vel_min: torch.Tensor,
        vel_max: torch.Tensor,
        acc_min: Optional[torch.Tensor] = None,
        acc_max: Optional[torch.Tensor] = None
    ):
        """
        정규화 파라미터 설정
        
        Args:
            vel_min: (1, workspace_dim, 1) 최소 속도
            vel_max: (1, workspace_dim, 1) 최대 속도
            acc_min: (1, workspace_dim, 1) 최소 가속도, 2차일 때만 필요
            acc_max: (1, workspace_dim, 1) 최대 가속도, 2차일 때만 필요
        """
        self.vel_min.copy_(vel_min)
        self.vel_max.copy_(vel_max)
        
        
        if self.dynamical_system_order == 2:
            assert acc_min is not None and acc_max is not None, "2차 시스템은 가속도 정규화 파라미터가 필요합니다."
            self.acc_min.copy_(acc_min)
            self.acc_max.copy_(acc_max)
            
    
    def set_goals(self, goals: torch.Tensor):
        """
        목표 위치 설정
        
        Args:
            goals: (n_primitives, workspace_dim) 목표점
        """
        assert goals.shape == self.goals.shape, \
            f"Expected shape {self.goals.shape}, got {goals.shape}"
        self.goals.copy_(goals)
        
    def compute_loss(self, x_t: torch.Tensor, x_tp1: torch.Tensor, 
                    primitive_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        다이나믹스 모델 손실 계산
        
        Args:
            x_t: (B, dim_state) 현재 상태
            x_tp1: (B, dim_state) 실제 다음 상태
            primitive_type: (B,) 프리미티브 인덱스
            
        Returns:
            손실 딕셔너리
        """
        # 타깃 프리미티브의 목표 가져오기
        if primitive_type.ndim == 2:
            idx = torch.argmax(primitive_type, dim=1)
        else:
            idx = primitive_type.long()
        
        target_goals = self.goals[idx]
        
        # 상태에 따라 델타(dx) 생성
        if self.dynamical_system_order == 1:
            # 1차 시스템: 목표를 향한 방향
            dx = target_goals - x_t
            dx_normalized = self.normalize_velocity(dx)
        else:
            # 2차 시스템: 목표로 향하는 가속도 계산
            pos = x_t[:, :self.workspace_dim]
            vel = x_t[:, self.workspace_dim:]
            
            pos_goals = target_goals
            dir_to_goal = pos_goals - pos
            # 감쇠 효과 (종말 속도 0으로 수렴하도록)
            damping = 0.1
            acc = dir_to_goal - damping * self.denormalize_velocity(vel)
            
            # 가속도 정규화
            acc_normalized = self.normalize_acceleration(acc)
            # 정규화된 속도와 가속도를 연결
            dx_normalized = torch.cat([vel, acc_normalized], dim=1)
        
        # 예측 다음 상태
        x_tp1_pred = self.forward(x_t, dx_normalized, primitive_type)
        
        # MSE 손실
        mse_loss = nn.functional.mse_loss(x_tp1_pred, x_tp1)
        
        # 위치와 속도(있는 경우) 분리
        pos_pred = x_tp1_pred[:, :self.workspace_dim]
        pos_true = x_tp1[:, :self.workspace_dim]
        pos_loss = nn.functional.mse_loss(pos_pred, pos_true)
        
        losses = {
            "total": mse_loss,
            "mse": mse_loss,
            "pos": pos_loss,
        }
        
        # 2차 시스템이면 속도 손실 추가
        if self.dynamical_system_order == 2:
            vel_pred = x_tp1_pred[:, self.workspace_dim:]
            vel_true = x_tp1[:, self.workspace_dim:]
            vel_loss = nn.functional.mse_loss(vel_pred, vel_true)
            losses["vel"] = vel_loss
        
        return losses
        
    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        사전 훈련된 가중치 불러오기
        
        Args:
            checkpoint_path: 체크포인트 경로
            strict: 엄격한 로딩 여부
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint, strict=strict)
        self.eval()
        print(f"Loaded state dynamics checkpoint from {checkpoint_path}")
