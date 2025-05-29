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
    imitation_window_size: int = 5                   # 모방 윈도우 크기 (시간적 문맥을 위한 윈도우 크기)
    delta_t: float =0.1
    
    # 학습 파라미터
    batch_size: int = 128                            # 배치 크기
    shuffle: bool = True                             # 셔플 여부
    num_workers: int = 4                             # 데이터 로더 워커 수
    pin_memory: bool = True                          # 메모리 고정 여부
    
    verbose: bool = True                           # 상세 출력 여부


class TrajectoryDataset(Dataset):
    """
    궤적 데이터셋 클래스
    
    궤적 데이터를 토치 데이터셋으로 변환하여 모델 학습에 사용
    """
    def __init__(
        self, 
        demos_np: np.ndarray, 
        prim_ids_np: np.ndarray,
        vel_min_np: np.ndarray, 
        vel_max_np: np.ndarray, 
        order: int,
        acc_min_np: Optional[np.ndarray] = None,
        acc_max_np: Optional[np.ndarray] = None,
        delta_t: float = 1.0,
        use_per_traj_normalization: bool = True
    ):
        """
        데이터셋 초기화
        
        Args:
            demos_np: (n_traj, n_steps, dim_ws, window) 데모 궤적
            prim_ids_np: (n_traj,) 프리미티브 ID
            vel_min_np: 최소 속도 (use_per_traj_normalization=True이면 (n_traj, dim_ws) 아니면 (1, dim_ws))
            vel_max_np: 최대 속도 (use_per_traj_normalization=True이면 (n_traj, dim_ws) 아니면 (1, dim_ws))
            order: 동역학계 차수
            acc_min_np: 최소 가속도 (use_per_traj_normalization=True이면 (n_traj, dim_ws) 아니면 (1, dim_ws)) (2차 시스템용)
            acc_max_np: 최대 가속도 (use_per_traj_normalization=True이면 (n_traj, dim_ws) 아니면 (1, dim_ws)) (2차 시스템용)
            delta_t: 시간 간격
            use_per_traj_normalization: 궤적별 정규화 사용 여부
        """
        # 토치 텐서 변환
        self.demos     = torch.tensor(demos_np).float()       # (n_traj, n_steps, dim_ws, window)
        self.prim_ids  = torch.tensor(prim_ids_np).long()
        self.order = order
        self.delta_t = delta_t
        self.use_per_traj_normalization = use_per_traj_normalization
        
        # Dimensions
        self.n_traj, self.n_steps, self.dim_ws, window = self.demos.shape
        assert window > order, "Window size must exceed system order"

        # Precompute full-trajectory states (position (+velocity))
        # Position: demos[..., 0]
        positions = self.demos[..., 0]                             # (n_traj, n_steps, dim_ws)
        if self.order == 2:
            # Compute raw velocities
            raw_vel = (self.demos[..., 1] - positions) / self.delta_t  # (n_traj, n_steps, dim_ws)
            # Normalize velocities per trajectory or globally
            if self.use_per_traj_normalization:
                # expand min/max per traj to time steps
                min_vel = vel_min_np.unsqueeze(1).expand(-1, self.n_steps, -1)
                max_vel = vel_max_np.unsqueeze(1).expand(-1, self.n_steps, -1)
            else:
                min_vel = vel_min_np.repeat(self.n_traj, self.n_steps, 1)
                max_vel = vel_max_np.repeat(self.n_traj, self.n_steps, 1)
            vel_norm = normalize_state(raw_vel, x_min=min_vel, x_max=max_vel)
            # Concatenate position and normalized velocity
            self.full_trajectory = torch.cat((positions, vel_norm), dim=-1)  # (n_traj, n_steps, dim_ws*2)

            # Precompute windowed inputs by concatenating demo window and velocity window
            # original windowed positions: self.demos[..., :window]
            pos_windows = self.demos[..., :window]                    # (n_traj, n_steps, dim_ws, window)
            vel_windows = vel_norm.unsqueeze(-1).expand(-1, -1, -1, window)
            self.window_data = torch.cat((pos_windows, vel_windows), dim=2)
            # new window_data shape: (n_traj, n_steps, dim_ws*2, window)
        else:
            # First-order: only positions
            self.full_trajectory = positions                           # (n_traj, n_steps, dim_ws)
            self.window_data = self.demos                              # shape unchanged

        # Effective dataset length: exclude last time-step per trajectory
        self.effective_len = self.n_traj * (self.n_steps - 1)
        
        print(f"데이터셋 생성: {self.n_traj} 궤적, 각 {self.n_steps} 스텝, 차원 {self.dim_ws}, 차수 {self.order}")
        print(f"궤적별 정규화 사용: {use_per_traj_normalization}")    
        
    def __getitem__(self, idx):
        """
        Fast lookup using precomputed tensors
        """
        # Determine trajectory index and time-step
        traj_idx = idx // (self.n_steps - 1)
        step_idx = idx % (self.n_steps - 1)

        # Current window (pos + vel if 2nd order)
        window_current = self.window_data[traj_idx, step_idx]
        # Full trajectory for this sample
        full_traj = self.full_trajectory[traj_idx]
        # Primitive ID
        prim_id = self.prim_ids[traj_idx]

        # Return: windowed input, prim ID, full trajectory, trajectory index, time index
        return window_current, prim_id, full_traj, traj_idx, step_idx
    
    def __len__(self):
        # 마지막 스텝을 제외한 유효한 인덱스만 반환
        return self.effective_len


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
            print(f"  trajectories_resample_length: {params.trajectories_resample_length}")
            print(f"  imitation_window_size: {params.imitation_window_size}")
        
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
    def create_dataset(self, data: Dict[str, Any], delta_t: float = 1.0, use_per_traj_normalization: bool = True):
        """
        데이터셋 생성
        
        Args:
            data: 전처리된 데이터 딕셔너리
            delta_t: 시간 간격
            use_per_traj_normalization: 궤적별 정규화 사용 여부
            
        Returns:
            데이터셋 객체
        """
        demos = data['demonstrations train']
        prim_ids = data['demonstrations primitive id']
        n_traj = demos.shape[0]
        dim_ws = demos.shape[2]
        
        # 속도/가속도 정규화 파라미터
        if use_per_traj_normalization:
            # 궤적별 정규화를 위한 준비
            # 데이터에서 각 궤적별 최소/최대 속도 계산 또는 로드
            try:
                # 각 궤적별 min/max 속도 계산
                vel_min_per_traj = np.zeros((n_traj, dim_ws))
                vel_max_per_traj = np.zeros((n_traj, dim_ws))
                
                if self.verbose:
                    print("각 궤적별 정규화 파라미터 계산 중...")
                
                # 각 궤적에 대해 정규화 파라미터 계산
                for i in range(n_traj):
                    traj_data = demos[i]  # (n_steps, dim_ws, window)
                    for j in range(dim_ws):
                        # 각 차원별 최소/최대 속도 추출
                        vel_min_per_traj[i, j] = np.min(data['vel min train'][j])
                        vel_max_per_traj[i, j] = np.max(data['vel max train'][j])
                
                min_vel = torch.from_numpy(vel_min_per_traj).float()  # (n_traj, dim_ws)
                max_vel = torch.from_numpy(vel_max_per_traj).float()  # (n_traj, dim_ws)
                
                # 가속도 정규화 파라미터 (2차 시스템용)
                if self.params.dynamical_system_order == 2 and 'acc min train' in data and 'acc max train' in data:
                    acc_min_per_traj = np.zeros((n_traj, dim_ws))
                    acc_max_per_traj = np.zeros((n_traj, dim_ws))
                    
                    for i in range(n_traj):
                        for j in range(dim_ws):
                            # 각 차원별 최소/최대 가속도 추출
                            acc_min_per_traj[i, j] = np.min(data['acc min train'][j])
                            acc_max_per_traj[i, j] = np.max(data['acc max train'][j])
                    
                    min_acc = torch.from_numpy(acc_min_per_traj).float()  # (n_traj, dim_ws)
                    max_acc = torch.from_numpy(acc_max_per_traj).float()  # (n_traj, dim_ws)
                else:
                    min_acc = None
                    max_acc = None
                    
                if self.verbose:
                    print(f"궤적별 정규화 파라미터 계산 완료: {n_traj}개 궤적")
            except Exception as e:
                print(f"궤적별 정규화 파라미터 계산 중 오류 발생: {e}")
                print("전체 데이터셋 정규화로 대체합니다.")
                use_per_traj_normalization = False
        
        # 궤적별 정규화를 사용하지 않거나 오류 발생 시 전체 데이터셋 정규화
        if not use_per_traj_normalization:
            min_vel_global = torch.from_numpy(data['vel min train'].reshape(1, -1)).float()  # (1, dim_ws)
            max_vel_global = torch.from_numpy(data['vel max train'].reshape(1, -1)).float()  # (1, dim_ws)
            
            # 모든 궤적에 동일한 정규화 파라미터 적용
            min_vel = min_vel_global.repeat(n_traj, 1)  # (n_traj, dim_ws)
            max_vel = max_vel_global.repeat(n_traj, 1)  # (n_traj, dim_ws)
            
            # 가속도 정규화 파라미터 (2차 시스템용)
            if self.params.dynamical_system_order == 2 and 'acc min train' in data and 'acc max train' in data:
                min_acc_global = torch.from_numpy(data['acc min train'].reshape(1, -1)).float()  # (1, dim_ws)
                max_acc_global = torch.from_numpy(data['acc max train'].reshape(1, -1)).float()  # (1, dim_ws)
                
                min_acc = min_acc_global.repeat(n_traj, 1)  # (n_traj, dim_ws)
                max_acc = max_acc_global.repeat(n_traj, 1)  # (n_traj, dim_ws)
            else:
                min_acc = None
                max_acc = None
        
        # 데이터셋 생성
        if self.params.dynamical_system_order == 2:
            dataset = TrajectoryDataset(
                demos_np=demos,
                prim_ids_np=prim_ids,
                vel_min_np=min_vel,
                vel_max_np=max_vel,
                acc_min_np=min_acc,
                acc_max_np=max_acc,
                order=self.params.dynamical_system_order,
                delta_t=delta_t,
                use_per_traj_normalization=use_per_traj_normalization
            )
        else:
            # 1차 시스템
            dataset = TrajectoryDataset(
                demos_np=demos,
                prim_ids_np=prim_ids,
                vel_min_np=min_vel,
                vel_max_np=max_vel,
                order=self.params.dynamical_system_order,
                delta_t=delta_t,
                use_per_traj_normalization=use_per_traj_normalization
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
    def run(self, use_per_traj_normalization: bool = True):
        """
        전체 파이프라인 실행
        
        Args:
            use_per_traj_normalization: 궤적별 정규화 사용 여부
            
        Returns:
            data: 전처리된 데이터 딕셔너리
            dataset: 데이터셋 객체
            loader: 데이터 로더 객체
        """
        # 데이터 로딩 및 전처리
        data = self.load_and_preprocess()
        
        # 데이터셋 생성
        dataset = self.create_dataset(data, use_per_traj_normalization=use_per_traj_normalization)
        
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


def denormalize_velocity(velocity_norm, min_vel, max_vel):
    """
    정규화된 속도를 원래 스케일로 복원하는 함수
    
    Args:
        velocity_norm: 정규화된 속도 
        min_vel: 최소 속도
        max_vel: 최대 속도
        
    Returns:
        원래 스케일의 속도
    """
    # normalize_state는 (x - x_min)/(x_max - x_min)을 2배 하고 -1 연산
    # 역연산: (velocity_norm + 1) / 2 * (max_vel - min_vel) + min_vel
    return (velocity_norm + 1.0) * 0.5 * (max_vel - min_vel) + min_vel


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
    
    # 데이터를 타입별로 분류
    np_arrays = {}
    other_data = {}
    
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            np_arrays[k] = v
        else:
            # numpy가 아닌 데이터는 pickle로 직렬화
            other_data[k] = v
    
    # 메타데이터 생성
    metadata = {
        'keys': list(data.keys()),
        'shapes': {k: v.shape if isinstance(v, np.ndarray) else None for k, v in data.items()},
        'dtypes': {k: str(v.dtype) if isinstance(v, np.ndarray) else str(type(v)) for k, v in data.items()},
        'other_data': other_data  # numpy가 아닌 데이터를 메타데이터에 포함
    }
    
    # 데이터 저장
    save_dict = {
        'metadata': metadata,
        **np_arrays  # numpy 배열
    }
    
    np.savez_compressed(save_path, **save_dict)
    print(f"전처리된 데이터가 {save_path}에 저장됨")
    print(f"저장된 numpy 배열 키: {list(np_arrays.keys())}")
    if other_data:
        print(f"메타데이터에 저장된 기타 데이터 키: {list(other_data.keys())}")


def load_preprocessed_data(load_path: str) -> Dict[str, np.ndarray]:
    """
    전처리된 데이터 로드
    
    Args:
        load_path: 로드할 파일 경로
        
    Returns:
        로드된 데이터 딕셔너리
    """
    try:
        loaded = np.load(load_path, allow_pickle=True)
        
        if 'metadata' not in loaded:
            print("Warning: 이전 형식의 캐시 파일입니다.")
            return {k: loaded[k] for k in loaded.files}
            
        # 메타데이터 추출
        metadata = loaded['metadata'].item()
        
        # numpy 배열 추출
        data = {k: loaded[k] for k in loaded.files if k != 'metadata'}
        
        # 기타 데이터 복원 (있는 경우)
        if 'other_data' in metadata:
            data.update(metadata['other_data'])
        
        # 데이터 검증
        loaded_keys = set(data.keys())
        expected_keys = set(metadata['keys'])
        
        if loaded_keys != expected_keys:
            missing = expected_keys - loaded_keys
            extra = loaded_keys - expected_keys
            if missing:
                print(f"Warning: 누락된 키: {missing}")
            if extra:
                print(f"Warning: 추가된 키: {extra}")
                
        # numpy 배열 형상 검증
        for k, v in data.items():
            if isinstance(v, np.ndarray) and k in metadata['shapes']:
                expected_shape = metadata['shapes'][k]
                if v.shape != expected_shape:
                    print(f"Warning: '{k}'의 형상이 다릅니다. 예상: {expected_shape}, 실제: {v.shape}")
        
        print(f"데이터를 {load_path}에서 로드함")
        print(f"로드된 키: {list(data.keys())}")
        return data
        
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {str(e)}")
        raise


# 예시 코드
if __name__ == "__main__":
    # 파라미터 초기화
    params = DataPipelineParams(
        workspace_dimensions=2,
        dynamical_system_order=2,
        dataset_name="LAIR",
        selected_primitives_ids="0",
        batch_size=128,
        verbose=True
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
