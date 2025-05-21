"""
Decoder 모듈: 잠재 공간에서 상태 공간으로 매핑하는 디코더
"""
import torch
import torch.nn as nn
import os
import numpy as np
from typing import Union, Optional, List, Dict


class Decoder(nn.Module):
    """
    latent-space → task-space 디코더
    
    * **입력**: 잠재 벡터 z_t (B, latent_dim), primitive_type ids or one-hot (B,) or (B, n_primitives)
    * **출력**: 상태 x_t (B, dim_state)
    """

    def __init__(
        self,
        latent_dim: int,
        dim_state: int,
        n_primitives: int = 1,
        hidden_sizes: List[int] = [256, 256],
        multi_motion: bool = False,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        디코더 초기화
        
        Args:
            latent_dim (int): 잠재 공간 차원
            dim_state (int): 상태 공간 차원
            n_primitives (int): 프리미티브 수
            hidden_sizes (List[int]): 은닉층 크기 목록
            multi_motion (bool): 여러 프리미티브 지원 여부
            device (str): 계산 장치
        """
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dim_state = dim_state
        self.n_primitives = n_primitives
        self.latent_dim = latent_dim
        self.multi_motion = multi_motion
        self.hidden_sizes = hidden_sizes

        # 프리미티브 인코딩 (multi-motion인 경우)
        if multi_motion:
            self.register_buffer(
                "primitive_encodings", torch.eye(n_primitives, device=self.device)
            )

        # 입력 차원 계산: 잠재 벡터 + 프리미티브 one-hot (만약 multi_motion이면)
        input_dim = latent_dim + (n_primitives if multi_motion else 0)

        # MLP 구축
        layers = []
        in_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            in_dim = hidden_size
        
        # 출력층
        layers.append(nn.Linear(in_dim, dim_state))
        
        self.network = nn.Sequential(*layers)
        
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

    def forward(self, z: torch.Tensor, primitive_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        순전파: 잠재 벡터와 프리미티브를 입력받아 상태 계산
        
        Args:
            z: (B, latent_dim) 잠재 벡터
            primitive_type: (B,) 또는 (B, n_primitives) 프리미티브 ID 또는 원핫
            
        Returns:
            (B, dim_state) 상태 벡터
        """
        z = z.to(self.device)
        
        if primitive_type is not None and self.multi_motion:
            primitive_type = primitive_type.to(self.device)
            primitive_enc = self.encode_primitives(primitive_type)
            z = torch.cat([z, primitive_enc], dim=1)
            
        return self.network(z)
        
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
        print(f"Loaded decoder checkpoint from {checkpoint_path}")
