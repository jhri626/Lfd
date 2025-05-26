"""
Encoder 모듈: 상태 공간에서 잠재 공간으로 매핑하는 인코더
"""
import torch
import torch.nn as nn
import os
import numpy as np
from typing import Union, Optional, List


class Encoder(nn.Module):
    """
    Task-space → latent-space 인코더
    
    * **입력**: 상태 x_t (B, dim_state), primitive_type ids or one-hot (B,) or (B, n_primitives)
    * **출력**: 잠재 벡터 z_t (B, latent_dim)
    """

    def __init__(
        self,
        dim_state: int,
        latent_dim: int,
        n_primitives: int = 1,
        hidden_sizes: List[int] = [256, 256],
        multi_motion: bool = False,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        인코더 초기화
        
        Args:
            dim_state (int): 상태 차원 (위치+속도)
            latent_dim (int): 잠재 공간 차원
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
        
        # 목표점 임베딩 저장 버퍼
        self.register_buffer(
            "goals_latent", torch.zeros(n_primitives, latent_dim, device=self.device)
        )

        # 입력 차원 계산: 상태 + 프리미티브 one-hot (만약 multi_motion이면)
        input_dim = dim_state + (n_primitives if multi_motion else 0)

        # MLP 구축
        layers = []
        in_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            in_dim = hidden_size
        
        # 출력층
        layers.append(nn.Linear(in_dim, latent_dim))
        
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

    def forward(self, x: torch.Tensor, primitive_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        순전파: 상태와 프리미티브를 입력받아 잠재 벡터 계산
        
        Args:
            x: (B, dim_state) 상태 벡터
            primitive_type: (B,) 또는 (B, n_primitives) 프리미티브 ID 또는 원핫
            
        Returns:
            (B, latent_dim) 잠재 벡터
        """
        x = x.to(self.device)
        
        if primitive_type is not None and self.multi_motion:
            primitive_type = primitive_type.to(self.device)
            primitive_enc = self.encode_primitives(primitive_type)
            x = torch.cat([x, primitive_enc], dim=1)
            
        return self.network(x)    
    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        사전 훈련된 가중치 불러오기 (체크포인트와 현재 모델 구조 간 매핑 포함)
        
        Args:
            checkpoint_path: 체크포인트 경로
            strict: 엄격한 로딩 여부
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print("Checkpoint keys:", checkpoint.keys())
        print("Current model keys:", self.state_dict().keys())
        
        # 체크포인트와 현재 모델 구조 간 매핑
        mapped_state_dict = {}
        
        # 레이어 매핑 정의
        layer_mapping = {
            'layer1.0': 'network.0',  # 첫 번째 Linear
            'layer1.1': 'network.1',  # 첫 번째 LayerNorm
            'layer2.0': 'network.3',  # 두 번째 Linear
            'layer2.1': 'network.4',  # 두 번째 LayerNorm
            'layer5': 'network.6'     # 출력 Linear
        }
        
        # 버퍼 매핑
        if 'goals_latent_space' in checkpoint:
            mapped_state_dict['goals_latent'] = checkpoint['goals_latent_space']
        
        if 'primitives_encodings' in checkpoint and hasattr(self, 'primitive_encodings'):
            mapped_state_dict['primitive_encodings'] = checkpoint['primitives_encodings']
        
        # 레이어 가중치 및 바이어스 매핑
        for old_key, new_key in layer_mapping.items():
            if f'{old_key}.weight' in checkpoint:
                mapped_state_dict[f'{new_key}.weight'] = checkpoint[f'{old_key}.weight']
            if f'{old_key}.bias' in checkpoint:
                mapped_state_dict[f'{new_key}.bias'] = checkpoint[f'{old_key}.bias']
        
        # 매핑된 상태 사전 출력
        print("Mapped state dict keys:", mapped_state_dict.keys())
        
        # 매핑된 상태 사전 로드
        missing_keys, unexpected_keys = self.load_state_dict(mapped_state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")
        
        # 평가 모드로 설정
        self.eval()
        
        print(f"Loaded encoder checkpoint from {checkpoint_path}")
        
    def update_goals_latent(self, goals: torch.Tensor):
        """
        태스크 공간 목표를 잠재 공간으로 매핑하여 저장
        
        Args:
            goals: (n_primitives, dim_workspace) 정규화된 목표점
        """
        for i in range(self.n_primitives):
            # 프리미티브 ID
            if self.multi_motion:
                prim = torch.tensor([i], device=self.device)
            else:
                prim = None
                
            # 상태 벡터 구성 (위치만 있는 경우)
            inp = torch.zeros([1, self.dim_state], device=self.device)
            workspace_dim = goals.shape[1]  # 태스크 공간 차원 (위치만)
            inp[:, :workspace_dim] = goals[i].to(self.device).view(1, -1)
            
            # 인코딩하여 저장
            with torch.no_grad():
                self.goals_latent[i] = self.forward(inp, prim).squeeze(0)
                
    def get_goal_latent(self, primitive_type: torch.Tensor) -> torch.Tensor:
        """
        프리미티브 ID에 해당하는 잠재 목표 반환
        
        Args:
            primitive_type: (B,) 프리미티브 ID 인덱스
            
        Returns:
            (B, latent_dim) 잠재 목표 벡터
        """
        if primitive_type.ndim == 2:
            # 원핫 형태면 인덱스로 변환
            primitive_type = primitive_type.argmax(dim=1)
            
        return self.goals_latent[primitive_type.long()]
