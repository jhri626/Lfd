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
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import argparse
from dataclasses import asdict
import json
import yaml  # YAML 파일 처리를 위한 모듈 추가

# 분리된 모듈 가져오기
from models.motion_primitive_model import MotionPrimitiveModel
from utils.visualization import visualize_from_dataset, evaluate_and_visualize_trajectories
from utils.test_data import create_test_data

# 데이터 파이프라인 가져오기
from data_pipeline import DataPipeline, DataPipelineParams


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
        with open(args.config, 'r',encoding='utf-8') as file:
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
        batch_size=args.batch_size,
        verbose=True
    )
    
    # YAML 설정 파일에서 추가 데이터 파이프라인 파라미터 적용
    if args.config:
        with open(args.config, 'r',encoding='utf-8') as file:
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
    # 궤적별 정규화 기본값 설정
    use_traj_norm = True

    # 데이터 로딩 및 전처리
    if args.use_cached_data and os.path.exists(data_cache_path):
        print(f"캐시된 데이터를 로드합니다: {data_cache_path}")
        from data_pipeline import load_preprocessed_data
        preprocessed_data = load_preprocessed_data(data_cache_path)
        
        # 데이터셋 및 로더 생성
        pipeline = DataPipeline(data_params)
        print(f"궤적별 정규화 사용: {use_traj_norm}")
        dataset = pipeline.create_dataset(preprocessed_data, use_per_traj_normalization=use_traj_norm)
        loader = pipeline.create_data_loader(dataset)
    else:
        print("데이터 파이프라인 실행 중...")
        pipeline = DataPipeline(data_params)
        print(f"궤적별 정규화 사용: {use_traj_norm}")
        preprocessed_data, dataset, loader = pipeline.run(use_per_traj_normalization=use_traj_norm)
        
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
        with open(args.config, 'r',encoding='utf-8') as file:
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
        with open(args.config, 'r',encoding='utf-8') as file:
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
    model.configure_optimizers(**optimizer_params)    # 학습용 데이터셋 준비
    print("학습 데이터셋 준비 중...")
    # 데이터 로더에서 데이터 사용
    print(f"원본 데이터 형태: 배치 크기 {loader.batch_size}, 배치 수 {len(loader)}")
    
    # 오토인코더 학습 시 사용할 데이터 로더 전달
    # 윈도우 크기 잡아온 상태인 windows 텐서와 prim_ids 배열을 그대로 전달
    
    # 학습 에포크 설정
    training_epochs = {
            'autoencoder': args.epochs,
            'latent_dynamics': args.epochs,
            'state_dynamics': args.epochs
        }
    
    # YAML 설정 파일에서 개별 에포크 설정 적용
    if args.config:
        with open(args.config, 'r',encoding='utf-8') as file:
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
                training_epochs['state_dynamics'] = config['training']['epochs_state_dynamics']      # 오토인코더 학습 (인코더와 디코더만 학습)
    print(f"\n오토인코더 학습 시작... ({training_epochs['autoencoder']} 에포크)")
    ae_losses = model.train_autoencoder(
        epochs=training_epochs['autoencoder'],
        dataloader=loader  # 기존에 생성된 데이터로더 전달
    )
    
    
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
            'n_samples': 5,  # 각 궤적당 샘플 포인트 수
            'steps': 100,    # 궤적 생성 스텝 수
            'sample_radius': 0.1  # 시작점 주변 샘플링 반경
        }
        
        # YAML 설정 파일에서 시각화 설정 적용
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            if 'visualization' in config:
                viz_config = config['visualization']
                if 'dataset_visualization' in viz_config:
                    dataset_viz_config = viz_config['dataset_visualization']
                    viz_params.update(dataset_viz_config)
        
        # 데이터셋 기반 궤적 시각화
        if viz_params.get('enable', True):
            print("\n데이터셋 기반 궤적 시각화 중...")
            
            # 프리미티브와 궤적 ID 설정
            primitive_ids = viz_params.get('primitive_ids', [0])
            trajectory_ids = viz_params.get('trajectory_ids', [0])
            
            # 각 프리미티브/궤적 조합에 대해 시각화
            for prim_id in primitive_ids:
                for traj_id in trajectory_ids:
                    
                    print(f"프리미티브 {prim_id}, 궤적 {traj_id} 시각화 중...")
                    
                    # 데이터셋에서 궤적 가져와서 시각화
                    save_path = os.path.join(args.save_dir, f"dataset_traj_prim{prim_id}_traj{traj_id}.png")
                    visualize_from_dataset(
                        model=model,
                        dataset=preprocessed_data,
                        prim_id=prim_id,
                        traj_id=traj_id,
                        n_samples=viz_params['n_samples'],
                        steps=viz_params['steps'],
                        sample_radius=viz_params['sample_radius'],
                        save_path=save_path,
                        show_original=True
                    )
                    print(f"  - 파일 저장됨: {save_path}")
                
                    
                        
        print(f"궤적 시각화가 {args.save_dir}에 저장됨")
        
        # 일괄 평가 기능 - 여러 프리미티브와 궤적에 대한 평가
        if 'batch_evaluation' in config.get('visualization', {}):
            batch_eval_config = config['visualization']['batch_evaluation']
            if batch_eval_config.get('enable', False):
                print("\n일괄 궤적 평가 및 시각화 중...")
                
                # 평가 디렉토리 설정
                eval_dir = os.path.join(args.save_dir, 'trajectory_evaluation')
                os.makedirs(eval_dir, exist_ok=True)
                
                # 설정 파일에서 파라미터 가져오기
                prim_ids = batch_eval_config.get('primitive_ids', None)  # None이면 모든 프리미티브
                traj_per_prim = batch_eval_config.get('trajectories_per_primitive', 3)
                n_samples = batch_eval_config.get('n_samples', 5)
                steps = batch_eval_config.get('steps', 100)
                sample_radius = batch_eval_config.get('sample_radius', 0.1)
                seed = batch_eval_config.get('seed', None)
                
                # 일괄 평가 실행
                try:
                    evaluate_and_visualize_trajectories(
                        model=model,
                        dataset=preprocessed_data,
                        save_dir=eval_dir,
                        prim_ids=prim_ids,
                        traj_per_prim=traj_per_prim,
                        n_samples=n_samples,
                        steps=steps,
                        sample_radius=sample_radius,
                        seed=seed
                    )
                    print(f"일괄 평가 완료. 결과 저장 위치: {eval_dir}")
                except Exception as e:
                    print(f"일괄 평가 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
