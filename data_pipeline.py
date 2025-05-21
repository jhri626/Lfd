#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
데이터 파이프라인 모듈: CONDOR에서 영감을 받은 모션 프리미티브 모델을 위한 데이터 처리

이 모듈은 다음 기능을 포함합니다:
- 데이터 로딩 및 전처리
- 데이터셋과 데이터 로더 생성
- 학습/평가를 위한 데이터 준비
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import json

# CONDOR 모듈 경로 추가
base_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(base_dir, 'submodule', 'CONDOR', 'src')
sys.path.insert(0, src_path)

# CONDOR 모듈 임포트
from submodule.CONDOR.src.data_preprocessing.data_preprocessor import DataPreprocessor
from submodule.CONDOR.src.agent.utils.dynamical_system_operations import normalize_state


@dataclass
class DataPipelineParams:
    """데이터 파이프라인 파라미터"""
    # 기본 설정
    workspace_dimensions: int = 2                    # 작업 공간 차원
    dynamical_system_order: int = 2                  # 동역학 차수
    dataset_name: str = "LAIR"                       # 데이터셋 이름
    selected_primitives_ids: str = "0"               # 사용할 프리미티브 ID
    trajectories_resample_length: int = 2000         # 궤적 리샘플링 길이
    state_increment: float = 0.2                     # 상태 증분
    workspace_boundaries_type: str = "from data"     # 작업 공간 경계 유형
    workspace_boundaries: Tuple = ((-1, 1),) * 3     # 작업 공간 경계
    spline_sample_type: str = "evenly spaced"        # 스플라인 샘플링 유형
    evaluation_samples_length: int = 10              # 평가 샘플 길이
    imitation_window_size: int = 2                   # 모방 윈도우 크기
    
    # 학습 파라미터
    batch_size: int = 128                            # 배치 크기
    shuffle: bool = True                             # 셔플 여부
    num_workers: int = 4                             # 데이터 로더 워커 수
    pin_memory: bool = True                          # 메모리 고정 여부


class TrajectoryDataset(Dataset):
    """
    궤적 데이터셋 클래스
    
    궤적 데이터를 토치 데이터셋으로 변환하여 모델 학습에 사용
    """
    def __init__(
        self, 
        demos_np: np.ndarray, 
        prim_ids_np: np.ndarray, 
        min_vel: torch.Tensor, 
        max_vel: torch.Tensor, 
        order: int,
        min_acc: Optional[torch.Tensor] = None,
        max_acc: Optional[torch.Tensor] = None,
        delta_t: float = 1.0
    ):
        """
        데이터셋 초기화
        
        Args:
            demos_np: (n_traj, n_steps, dim_ws, window) 데모 궤적
            prim_ids_np: (n_traj,) 프리미티브 ID
            min_vel: (1, dim_ws) 최소 속도
            max_vel: (1, dim_ws) 최대 속도
            order: 동역학계 차수
            min_acc: (1, dim_ws) 최소 가속도 (2차 시스템)
            max_acc: (1, dim_ws) 최대 가속도 (2차 시스템)
            delta_t: 시간 간격
        """
        # 토치 텐서 변환
        self.demos = torch.from_numpy(demos_np).float()
        self.prim_ids = torch.tensor(prim_ids_np, dtype=torch.long)
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.min_acc = min_acc
        self.max_acc = max_acc
        self.order = order
        self.delta_t = delta_t
        
        # 데이터 크기 정보
        self.n_traj, self.n_steps, self.dim_ws, window = self.demos.shape
        assert window > order, "윈도우 크기는 시스템 차수보다 커야 합니다."
        
        print(f"데이터셋 생성: {self.n_traj} 궤적, 각 {self.n_steps} 스텝, 차원 {self.dim_ws}, 차수 {self.order}")

    def __getitem__(self, idx):
        # 인덱스에서 궤적 및 시간 스텝 추출
        traj = idx // self.n_steps
        t = idx % self.n_steps
        
        # 윈도우 추출
        window = self.demos[traj, t]  # (dim_ws, window)
        pos = window[:, 0]  # 현재 위치
        
        # 상태 벡터 구성 (차수에 따라)
        if self.order == 2:
            # 2차 시스템: 위치 + 속도
            next_pos = window[:, 1]
            raw_velocity = (next_pos - pos) / self.delta_t
            
            # 속도 정규화
            vel_norm = normalize_state(
                raw_velocity,
                x_min=self.min_vel,
                x_max=self.max_vel
            ).squeeze(0)
            
            x_t = torch.cat((pos, vel_norm), dim=0)  # (2*dim_ws,)
            
            # 다음 상태 계산 (t+1 시점)
            pos_tp1 = window[:, 1]  # t+1 위치
            next_pos2 = window[:, 2]  # t+2 위치
            raw_vel2 = (next_pos2 - pos_tp1) / self.delta_t
            
            vel_norm2 = normalize_state(
                raw_vel2,
                x_min=self.min_vel,
                x_max=self.max_vel
            ).squeeze(0)
            
            x_tp1 = torch.cat((pos_tp1, vel_norm2), dim=0)  # (2*dim_ws,)
        else:
            # 1차 시스템: 위치만
            x_t = pos
            x_tp1 = window[:, 1]  # t+1 위치
        
        # 프리미티브 ID 및 인덱스
        prim_id = self.prim_ids[traj]
        t_idx = torch.tensor(t, dtype=torch.long)
        traj_idx = torch.tensor(traj, dtype=torch.long)
        
        return x_t, x_tp1, prim_id, traj_idx, t_idx
    
    def __len__(self):
        return self.n_traj * self.n_steps


class DataPipeline:
    """
    데이터 파이프라인 클래스
    
    데이터 로딩, 전처리 및 데이터셋/로더 생성을 담당
    """
    def __init__(self, params: DataPipelineParams, verbose: bool = True):
        """
        파이프라인 초기화
        
        Args:
            params: 데이터 파이프라인 파라미터
            verbose: 상세 출력 여부
        """
        self.params = params
        self.verbose = verbose
        
        if verbose:
            print("데이터 파이프라인 초기화:")
            print(f"  데이터셋: {params.dataset_name}")
            print(f"  프리미티브 ID: {params.selected_primitives_ids}")
            print(f"  작업 공간 차원: {params.workspace_dimensions}")
            print(f"  동역학계 차수: {params.dynamical_system_order}")
        
    def load_and_preprocess(self):
        """
        데이터 로딩 및 전처리
        
        Returns:
            전처리된 데이터 딕셔너리
        """
        if self.verbose:
            print("데이터 로딩 및 전처리 시작...")
        
        # CONDOR의 DataPreprocessor 사용
        preprocessor = DataPreprocessor(params=self.params, verbose=self.verbose)
        data = preprocessor.run()
        
        if self.verbose:
            print("데이터 전처리 완료")
            
            # 데이터 정보 출력
            demos = data['demonstrations train']
            n_traj, n_steps, dim_ws, window = demos.shape
            print(f"데모 궤적: {n_traj}개, 각 {n_steps} 스텝")
            print(f"작업 공간 차원: {dim_ws}, 윈도우 크기: {window}")
            
            prim_ids = data['demonstrations primitive id']
            n_prims = len(np.unique(prim_ids))
            print(f"프리미티브 수: {n_prims}")
            
        return data
    
    def create_dataset(self, data: Dict[str, Any], delta_t: float = 1.0):
        """
        데이터셋 생성
        
        Args:
            data: 전처리된 데이터 딕셔너리
            delta_t: 시간 간격
            
        Returns:
            데이터셋 객체
        """
        print(data)
        demos = data['demonstrations train']
        prim_ids = data['demonstrations primitive id']
        
        # 속도/가속도 정규화 파라미터
        min_vel = torch.from_numpy(data['vel min train'].reshape(1, -1)).float()
        max_vel = torch.from_numpy(data['vel max train'].reshape(1, -1)).float()
        
        if self.params.dynamical_system_order == 2:
            # 2차 시스템인 경우 가속도 정규화 파라미터
            min_acc = torch.from_numpy(data['acc min train'].reshape(1, -1)).float() if 'acc min train' in data else None
            max_acc = torch.from_numpy(data['acc max train'].reshape(1, -1)).float() if 'acc max train' in data else None
            
            dataset = TrajectoryDataset(
                demos_np=demos,
                prim_ids_np=prim_ids,
                min_vel=min_vel,
                max_vel=max_vel,
                min_acc=min_acc,
                max_acc=max_acc,
                order=self.params.dynamical_system_order,
                delta_t=delta_t
            )
        else:
            # 1차 시스템
            dataset = TrajectoryDataset(
                demos_np=demos,
                prim_ids_np=prim_ids,
                min_vel=min_vel,
                max_vel=max_vel,
                order=self.params.dynamical_system_order,
                delta_t=delta_t
            )
            
        return dataset
    
    def create_data_loader(self, dataset: TrajectoryDataset, batch_size: Optional[int] = None):
        """
        데이터 로더 생성
        
        Args:
            dataset: 데이터셋 객체
            batch_size: 배치 크기 (None이면 params에서 가져옴)
            
        Returns:
            데이터 로더 객체
        """
        if batch_size is None:
            batch_size = self.params.batch_size
            
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.params.shuffle,
            drop_last=True,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory
        )
        
        if self.verbose:
            print(f"데이터 로더 생성: 배치 크기 {batch_size}, 배치 수 {len(loader)}")
            
        return loader
    
    def run(self):
        """
        전체 파이프라인 실행
        
        Returns:
            data: 전처리된 데이터 딕셔너리
            dataset: 데이터셋 객체
            loader: 데이터 로더 객체
        """
        # 데이터 로딩 및 전처리
        data = self.load_and_preprocess()
        
        # 데이터셋 생성
        dataset = self.create_dataset(data)
        
        # 데이터 로더 생성
        loader = self.create_data_loader(dataset)
        
        return data, dataset, loader


def normalize_velocity(velocity, min_vel, max_vel):
    """
    속도 정규화 유틸리티 함수
    
    Args:
        velocity: 정규화할 속도
        min_vel: 최소 속도
        max_vel: 최대 속도
        
    Returns:
        정규화된 속도
    """
    return normalize_state(velocity, x_min=min_vel, x_max=max_vel)


def normalize_acceleration(acceleration, min_acc, max_acc):
    """
    가속도 정규화 유틸리티 함수
    
    Args:
        acceleration: 정규화할 가속도
        min_acc: 최소 가속도
        max_acc: 최대 가속도
        
    Returns:
        정규화된 가속도
    """
    return normalize_state(acceleration, x_min=min_acc, x_max=max_acc)


def save_preprocessed_data(data: Dict[str, Any], save_path: str):
    """
    전처리된 데이터 저장
    
    Args:
        data: 저장할 데이터 딕셔너리
        save_path: 저장 경로
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 넘파이 배열 저장
    np.savez_compressed(
        save_path,
        **{k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    )
    
    print(f"전처리된 데이터가 {save_path}에 저장됨")


def load_preprocessed_data(load_path: str):
    """
    전처리된 데이터 로드
    
    Args:
        load_path: 로드할 파일 경로
        
    Returns:
        로드된 데이터 딕셔너리
    """
    loaded = np.load(load_path, allow_pickle=True)
    data = {k: loaded[k] for k in loaded.files}
    
    print(f"전처리된 데이터를 {load_path}에서 로드함")
    return data


# 예시 코드
if __name__ == "__main__":
    # 파라미터 초기화
    params = DataPipelineParams(
        workspace_dimensions=2,
        dynamical_system_order=2,
        dataset_name="LAIR",
        selected_primitives_ids="0",
        batch_size=128
    )
    
    # 파이프라인 실행
    pipeline = DataPipeline(params)
    data, dataset, loader = pipeline.run()
    
    # 데이터 저장 예시
    save_preprocessed_data(data, "results/preprocessed_data.npz")
    
    # 테스트: 첫 번째 배치 출력
    for x_t, x_tp1, prim_id, traj_idx, t_idx in loader:
        print("배치 크기:", x_t.shape)
        print("현재 상태:", x_t[0])
        print("다음 상태:", x_tp1[0])
        print("프리미티브 ID:", prim_id[0])
        print("궤적 인덱스:", traj_idx[0])
        print("시간 인덱스:", t_idx[0])
        break  # 첫 번째 배치만 출력
