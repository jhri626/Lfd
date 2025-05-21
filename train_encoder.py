import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import sys
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(base_dir, 'submodule', 'CONDOR', 'src')
sys.path.insert(0, src_path)
from dataclasses import dataclass, asdict
import json
from submodule.CONDOR.src.data_preprocessing.data_preprocessor import DataPreprocessor
from submodule.CONDOR.src.agent.utils.dynamical_system_operations import normalize_state
from encoder import Encoder
from sceduler import CosineAnnealingWarmUpRestarts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Enable cuDNN tuning

@dataclass
class TrainParams:
    # Pipeline params (unchanged)
    workspace_dimensions: int = 2
    dynamical_system_order: int = 2
    dataset_name: str = "LAIR"
    selected_primitives_ids: str ="0"
    trajectories_resample_length: int = 2000
    state_increment: float = 0.2
    workspace_boundaries_type: str = "from data"
    workspace_boundaries: tuple = ((-1, 1),)*3
    spline_sample_type: str = "evenly spaced"
    evaluation_samples_length: int = 10
    imitation_window_size: int = 2  # need at least 2 for triplet

    # Encoder architecture
    latent_space_dim: int = 4
    hidden_size: int = 300

    # Training hyperparameters
    batch_size: int = 250
    epochs: int = 25000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    cold_start : int = 5000
    triplet_margin: float = 1e-4
    linear_mapping_param : float = 0.05
    device: str = "cuda"  # or dynamically set to torch.device(...)
    results_path: str = "results/ver8"

class TripletDemoDataset(Dataset):
    def __init__(self, demos_np, prim_ids_np, min_vel, max_vel, order,delta_t=1.0):
        # demos_np: (n_traj, n_steps, dim_ws, window)
        self.demos     = torch.from_numpy(demos_np).float()
        self.prim_ids  = torch.tensor(prim_ids_np, dtype=torch.long)
        self.min_vel   = min_vel  # torch.Tensor shape (1, dim_ws)
        self.max_vel   = max_vel  # torch.Tensor shape (1, dim_ws)
        self.order     = order    # 1 or 2
        self.delta_t   = delta_t
        self.n_traj, self.n_steps, dim_ws, window = self.demos.shape
        assert window >= order, "window 크기는 시스템 차수 이상이어야 합니다."

    def __getitem__(self, idx):
        traj = idx // self.n_steps
        t    = idx % self.n_steps

        window = self.demos[traj, t]           # (dim_ws, window)
        pos    = window[:, 0]                 # 위치 x_t
        # 만약 2차라면 속도도 추가
        if self.order == 2:
            next_pos     = window[:, 1]
            raw_velocity = (next_pos - pos) / self.delta_t  # Δt=1 로 가정
            # print(raw_velocity)
            # normalize to [-1,1]
            vel_norm = normalize_state(
                raw_velocity,  # (1, dim_ws, 1)
                x_min=self.min_vel,                  # (1, dim_ws, 1)
                x_max=self.max_vel                   # (1, dim_ws, 1)
            ).squeeze(0)
            # 상태 벡터: [pos; vel_norm]
            # print(pos,vel_norm)
            # print(pos.shape,vel_norm.shape)
            x_t = torch.cat((pos, vel_norm), dim=0)    # (2*dim_ws,)
        else:
            x_t = pos

        # 긍정/부정 샘플 계산 시에도 동일하게 처리
        # 예: 다음 스텝을 positive로…
        pos_window = window[:, 1]
        if self.order == 2:
            next_pos2     = window[:, 2]  # x_{t+2}
            raw_vel2      = (next_pos2 - pos_window) / self.delta_t

            vel_norm2     = normalize_state(
                raw_vel2,
                x_min=self.min_vel,
                x_max=self.max_vel
            ).squeeze(0)
            
            x_tp1 = torch.cat((pos_window, vel_norm2), dim=0)
        else:
            x_tp1 = pos_window
            
        
        prim_id = self.prim_ids[traj]
        t_idx = torch.tensor(t, dtype=torch.long)
        traj_idx = torch.tensor(traj, dtype=torch.long)
        # print(traj_idx)
        return x_t, x_tp1, prim_id, traj_idx, t_idx
    
    def __len__(self):
        return self.n_traj * self.n_steps

def main():
    params = TrainParams()
    epoch_grads = []

    os.makedirs(params.results_path, exist_ok=True)
    params_dict = asdict(params)  # Convert dataclass to dict
    json_path = os.path.join(params.results_path, 'train_params.json')
    with open(json_path, 'w') as f:
        json.dump(params_dict, f, indent=4)  # Write with pretty formatting
    print(f"Training parameters saved to {json_path}")

    # 1. Preprocess (unchanged)
    data = DataPreprocessor(params=params, verbose=True).run()
    demos = data['demonstrations train']             # citeturn4file1
    prim_ids = data['demonstrations primitive id']   
    goals = data['goals training']                   # citeturn4file1

    # 2. Dataset & DataLoader
    device = params.device
    min_vel = torch.from_numpy(data['vel min train'].reshape(1, -1)) \
                  .float()
    max_vel = torch.from_numpy(data['vel max train'].reshape(1, -1)) \
                  .float()

    # 3) Dataset 생성 시 필수 인자 함께 전달
    dataset = TripletDemoDataset(
        demos_np = demos,
        prim_ids_np = prim_ids,
        min_vel = min_vel,
        max_vel = max_vel,
        order = params.dynamical_system_order
    )
    
    loader = DataLoader(dataset,
                        batch_size=params.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=4,    # Number of parallel data-loading workers
                        pin_memory=True)   # Enable faster host→GPU transfers)

    # 3. Model & Optimizer
    dim_state = params.workspace_dimensions * params.dynamical_system_order
    n_primitives = len(np.unique(prim_ids))
    device = params.device

    encoder = Encoder(
        dim_state=dim_state,
        n_primitives=n_primitives,
        latent_space_dim=params.latent_space_dim,
        hidden_size=params.hidden_size,
        device=device,
        multi_motion = False
    ).to(device)
    goals = torch.from_numpy(goals).float().to(device)
    print(goals)
    encoder.update_goals_latent_space(goals)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    loss_trp = torch.nn.TripletMarginLoss(margin=params.triplet_margin)
    loss_mse = torch.nn.MSELoss()
    
    scheduler = CosineAnnealingWarmUpRestarts(optimizer,
    T_0=2000, T_mult=2, eta_max=params.learning_rate,T_up=100,gamma=0.5)            


    epoch_losses = []
    # print(prim_ids)
    # print(n_primitives)
    n_traj    = dataset.n_traj               # 궤적 개수
    D         = params.latent_space_dim
    T = params.trajectories_resample_length - 1
    e         = 0.1                         # 노이즈 스케일 (조절 가능)
    device    = params.device

    # (1) 모든 궤적에 대해 동일한 base start (예: 0벡터)
    base_start = torch.ones(D, device=device)*2

# (2) 데모별로 살짝씩 다른 start_points 생성
    start_points = base_start.unsqueeze(0).repeat(n_traj, 1) \
             + (torch.randn(n_traj, D, device=device) * e) 
    print(start_points)
    # 4. Training loop
    for epoch in range(1, params.epochs + 1):
        total_loss = 0.0     
        grad_sum = 0.0
        grad_count = 0   
        for x_t, x_tp1, prim, traj_idx, t_idx  in loader:
            x_t     = x_t.to(device, non_blocking=True)
            x_tp1   = x_tp1.to(device, non_blocking=True)
            prim    = prim.to(device, non_blocking=True)
            traj_idx= traj_idx.to(device, non_blocking=True)
            t_idx   = t_idx.to(device, non_blocking=True)
            
            # Anchor: goal embedding
            pos    = encoder(x_tp1, prim)
            neg    = encoder(x_t, prim)
            
            # print(traj_idx)
            line_end = torch.zeros(params.latent_space_dim, device=device)           # 예: 원점
            line_start = start_points[traj_idx]
            anchor_vec = encoder.get_goals_latent_space_batch(prim)
            loss_anchor = torch.norm(anchor_vec, p=2, dim=1).pow(2).mean()

            alpha = (t_idx.float()/T).view(-1,1)   # [B,1]
            lin  = (1-alpha) * line_start + alpha * line_end
            loss_lin = loss_mse(neg, lin)
                
            loss = loss_lin + 1e-3 * loss_anchor   # 1e-3은 가중치 예시
            
            if epoch > params.cold_start:
                loss = np.exp(np.log(params.linear_mapping_param) * 1/(params.epochs -params.cold_start)* (epoch-params.cold_start))* loss
                loss += loss_trp(anchor_vec, pos, neg) 
                
            
            optimizer.zero_grad()
            loss.backward()
            for p in encoder.parameters():
                if p.grad is None:
                    continue
                grad_sum += p.grad.abs().mean().item()
                grad_count += 1
            
            optimizer.step()

            total_loss += loss.item()
            encoder.update_goals_latent_space(goals)
            scheduler.step()
            
            
        # print(anchor_vec)
        if epoch == params.epochs:
            print(encoder.get_goals_latent_space_batch(prim))
        if epoch == params.cold_start:
            torch.save(encoder.state_dict(), params.results_path + "/encoder_coldStart_ver8.pt")
        avg_epoch_grad = grad_sum / grad_count if grad_count > 0 else 0.0
        epoch_grads.append(avg_epoch_grad)
        epoch_losses.append(total_loss)
        print(f"  [Avg grad] {avg_epoch_grad:.6e}")
        print(f"Epoch {epoch}/{params.epochs} — Triplet Loss: {total_loss:.6f}")

    # 5. Save
    torch.save(encoder.state_dict(), params.results_path + "/encoder_triplet_ver8.pt")
    print("Encoder training complete. Weights saved to", params.results_path + "/encoder_triplet_ver8.pt")


    metrics_path = os.path.join(params.results_path, 'training_metrics.json')
    with open(metrics_path, 'w') as mf:
        json.dump({'epoch_grads': epoch_grads, 'epoch_losses': epoch_losses}, mf, indent=4)
    print(f"Training metrics saved to {metrics_path}")
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(epoch_grads) + 1), epoch_grads, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Gradient Magnitude')
    plt.title('Epoch-wise Average Gradient Magnitudes')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()
