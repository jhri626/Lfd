# apply_triplet_encoder_per_traj.py

import argparse
import os
import sys
import numpy as np
import torch

# 1) PYTHONPATH 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(base_dir, 'submodule', 'CONDOR', 'src')
sys.path.insert(0, src_path)

# 2) 필요한 모듈 import
from data_preprocessing.data_preprocessor import DataPreprocessor
from utils.neural_net import Encoder

def parse_args():
    p = argparse.ArgumentParser("Apply trained triplet-encoder per trajectory")
    p.add_argument('--dataset',     type=str, required=True,
                   help="예: LASA, LAIR 등")
    p.add_argument('--primitives',  type=str, required=True,
                   help="콤마 구분된 ID (예: '0,1')")
    p.add_argument('--checkpoint',  type=str, default='./checkpoints/triplet_encoder.pth',
                   help="학습된 encoder state_dict 경로")
    p.add_argument('--output-dir',  type=str, required=True,
                   help="각 궤적별 임베딩을 저장할 디렉토리")
    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1) DataPreprocessor 로 raw demonstrations 불러오기 ---
    params = argparse.Namespace(
        trajectories_resample_length=100,
        state_increment=0.1,
        workspace_dimensions=2,
        dynamical_system_order=2,
        workspace_boundaries_type='from data',
        workspace_boundaries=[[0,0],[0,0]],
        evaluation_samples_length=50,
        dataset_name=args.dataset,
        selected_primitives_ids=args.primitives,
        spline_sample_type='evenly spaced',
        imitation_window_size=10,
        verbose=False
    )
    pre = DataPreprocessor(params, verbose=False)
    out = pre.run()
    raw_demos = out['demonstrations raw']            # list of (D, T_i)
    prim_ids  = out['demonstrations primitive id']   # list length N

    # --- 2) Encoder 재생성 + 체크포인트 로드 ---
    encoder = Encoder(
        dim_state=raw_demos[0].shape[0],   # D
        latent_dim=4,
        hidden_layers=[128,64],
        n_primitives=len(set(prim_ids)),
        multi_motion=False
    ).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(sd)
    encoder.eval()

    # --- 3) 출력 디렉토리 준비 ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 4) 궤적별로 임베딩 계산 및 저장 ---
    for i, (traj, prim) in enumerate(zip(raw_demos, prim_ids)):
        # traj: (D, T_i) → 전치 → (T_i, D)
        X = torch.FloatTensor(traj.T).to(device)      # shape (T_i, D)
        P = torch.full((X.size(0),), prim, dtype=torch.long, device=device)

        with torch.no_grad():
            emb = encoder(X, P)                       # (T_i, latent_dim)
            emb = emb.cpu().numpy()

        # 파일명: output-dir/traj_{i}_prim{prim}.npz
        fname = f"traj_{i:03d}_prim{prim}.npz"
        out_path = os.path.join(args.output_dir, fname)
        np.savez_compressed(out_path,
                            embeddings=emb,
                            primitive_id=np.array(prim, dtype=int))
        print(f"Saved trajectory {i} (length {emb.shape[0]}) → {out_path}")

if __name__ == '__main__':
    main()
