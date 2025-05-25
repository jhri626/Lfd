"""
잠재 공간 다이나믹스 모델링 클래스
"""
import torch
import torch.nn as nn
import os
import numpy as np
import submodule.BCSDM.utils.R2_functions as R2
from typing import Union, Optional, List, Dict, Tuple, Any


class LatentDynamics(nn.Module):
    """
    잠재 공간 다이나믹스 모델링
    
    * **입력**: 현재 잠재 상태 z_t (B, latent_dim), primitive_type ids or one-hot (B,) or (B, n_primitives)
    * **출력**: 다음 잠재 상태 z_{t+1} (B, latent_dim)
    
    CONDOR 스타일의 다이나믹스 시스템 구현:
    - 현재 잠재 상태와 목표 상태 간의 차이에 기반한 선형 다이나믹스
    - 어댑티브 게인(adaptive gain) 옵션 지원
    - 추가 비선형성을 위한 MLP 네트워크
    """

    def __init__(
        self,
        latent_dim: int,
        n_primitives: int = 1,
        multi_motion: bool = False,
        adaptive_gains: bool = True,
        latent_gain: float = 1.0,
        latent_gain_lower_limit: float = 0.1,
        latent_gain_upper_limit: float = 10.0,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        잠재 다이나믹스 모델 초기화
        
        Args:
            latent_dim (int): 잠재 공간 차원
            n_primitives (int): 프리미티브 수
            multi_motion (bool): 여러 프리미티브 지원 여부
            adaptive_gains (bool): 어댑티브 게인 사용 여부
            latent_gain (float): 기본 게인 값 (adaptive_gains=False일 때 사용)
            latent_gain_lower_limit (float): 게인 하한값 (adaptive_gains=True일 때 사용)
            latent_gain_upper_limit (float): 게인 상한값 (adaptive_gains=True일 때 사용)
            device (str): 계산 장치
        """
        super().__init__()        
        self.device = torch.device(device) if isinstance(device, str) else device
        self.n_primitives = n_primitives
        self.latent_dim = latent_dim
        self.multi_motion = multi_motion
        
        # 단순화: 고정된 파라미터 사용 (학습 없음)
        self.adaptive_gains = False  # 어댑티브 게인 제거
        self.delta_t = 1.0  # 고정된 시간 간격
        self.latent_gain = latent_gain
        self.latent_gain_lower_limit = latent_gain_lower_limit
        self.latent_gain_upper_limit = latent_gain_upper_limit

        # 프리미티브 인코딩 (multi-motion인 경우)
        if multi_motion:
            self.register_buffer(
                "primitive_encodings", torch.eye(n_primitives, device=self.device)
            )
        
        # 목표점 임베딩 저장 버퍼
        self.register_buffer(
            "goals_latent", torch.zeros(n_primitives, latent_dim, device=self.device)
        )
        
        self.to(device)
        
        # 학습용 파라미터
        self.delta_t = 1.0  # 시간 간격
          
        self.scale_factor = 4.0  # contraction factor for gvf_R2

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
    def forward(self, z: torch.Tensor, latent_traj: torch.Tensor, primitive_type: torch.Tensor) -> torch.Tensor:
        """
        순전파: 현재 잠재 상태로부터 다음 상태 예측
        
        단순 오일러 적분:
        z_{t+1} = z_t + delta_z * delta_t
        
        여기서 delta_z는 encoder의 출력으로 간주됨
        
        Args:
            z: (B, latent_dim) 현재 잠재 상태
            primitive_type: (B,) 또는 (B, n_primitives) 프리미티브 ID 또는 원핫
            
        Returns:
            (B, latent_dim) 다음 잠재 상태
        """
        z = z.to(self.device)
        primitive_type = primitive_type.to(self.device)
        
        # 수치 안정성을 위한 클램핑: 너무 큰 값이나 NaN 방지
        z = torch.clamp(z, min=-1e10, max=1e10)
        
        # 프리미티브 인덱스에 대한 목표 얻기
        if primitive_type.ndim == 2:
            # 원핫이면 인덱스로 변환
            idx = torch.argmax(primitive_type, dim=1)
        else:
            idx = primitive_type
        
        
        zdot_traj = latent_traj[:,1:]-latent_traj[:,:-1]
        zdot_traj = torch.cat([zdot_traj, zdot_traj.new_zeros(zdot_traj.size(0), 1, zdot_traj.size(2))], dim=1)
        
        zdot = R2.gvf_R2(z, self.scale_factor, latent_traj, zdot_traj)
        
        # 2. 오일러 적분 수행
        z_next = z + zdot * self.delta_t
        
        return z_next
    
    def multi_step_prediction(
        self, 
        z_init: torch.Tensor, 
        primitive_type: torch.Tensor, 
        steps: int = 100
    ) -> torch.Tensor:
        """
        여러 스텝의 상태 예측
        
        Args:
            z_init: (B, latent_dim) 초기 잠재 상태
            primitive_type: (B,) 프리미티브 인덱스
            steps: 예측할 스텝 수
            
        Returns:
            (B, steps, latent_dim) 예측된 잠재 상태 궤적
        """
        batch_size = z_init.shape[0]
        trajectories = torch.zeros(batch_size, steps, self.latent_dim, device=self.device)
        
        # 초기 상태
        z_t = z_init
        
        # 순차적 예측
        for t in range(steps):
            trajectories[:, t] = z_t
            z_t = self.forward(z_t, primitive_type)
            
        return trajectories
    def set_goals_latent(self, goals_latent: torch.Tensor):
        """
        잠재 공간 목표 설정
        
        Args:
            goals_latent: (n_primitives, latent_dim) 목표 임베딩
        """
        assert goals_latent.shape == self.goals_latent.shape, \
            f"Expected shape {self.goals_latent.shape}, got {goals_latent.shape}"
        self.goals_latent.copy_(goals_latent)
        
    def compute_loss(self, z_t: torch.Tensor, z_tp1: torch.Tensor, 
                    primitive_type: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        다이나믹스 모델 손실 계산
        
        Args:
            z_t: (B, latent_dim) 현재 잠재 상태
            z_tp1: (B, latent_dim) 실제 다음 잠재 상태
            primitive_type: (B,) 프리미티브 인덱스
            
        Returns:
            손실 딕셔너리 (총 손실, MSE 손실, 정규화 손실 등)
        """
        # 예측 다음 상태
        z_tp1_pred = self.forward(z_t, primitive_type)
        
        # MSE 손실
        mse_loss = nn.functional.mse_loss(z_tp1_pred, z_tp1)
        
        # 손실 딕셔너리 초기화
        losses = {
            "mse": mse_loss,
        }
        
        # 어댑티브 게인을 사용하는 경우 추가 정규화 손실
        if self.adaptive_gains:
            # 게인 값 얻기
            batch_size = z_t.shape[0]
            gain_scale = self.gain_network(z_t).squeeze(-1)
            gain = self.latent_gain_lower_limit + gain_scale * (self.latent_gain_upper_limit - self.latent_gain_lower_limit)
            
            # 게인 정규화: 최소 게인 이상을 유지하도록 L1 페널티 적용
            # 이는 게인이 지나치게 작아지는 것을 방지
            gain_reg_loss = torch.mean(torch.abs(gain - self.latent_gain_lower_limit))
            
            # 딕셔너리에 추가
            losses["gain_reg"] = 0.01 * gain_reg_loss  # 작은 가중치를 적용
        else:
            losses["gain_reg"] = torch.tensor(0.0, device=self.device)
          # 비선형 항(네트워크)이 제거되었으므로 해당 정규화도 제거
        losses["nl_reg"] = torch.tensor(0.0, device=self.device)
        
        # 총 손실 계산
        total_loss = losses["mse"] + losses["gain_reg"] + losses["nl_reg"]
        losses["total"] = total_loss
        
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
        print(f"Loaded latent dynamics checkpoint from {checkpoint_path}")
