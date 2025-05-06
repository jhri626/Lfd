# train_full.py

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(base_dir, 'submodule', 'CONDOR', 'src')
sys.path.insert(0, src_path)

# --- 1) Import modules ---
from submodule.CONDOR.src.data_preprocessing.data_preprocessor import DataPreprocessor
from utils.neural_net import Encoder
from submodule.CONDOR.src.agent.utils.ranking_losses import TripletLoss
# --- Dataset wrapper with padding handling ---
class PreprocessedDataset(Dataset):
    def __init__(self, demos_train, prim_ids):
        """
        Custom Dataset for preprocessed trajectories. Handles variable lengths
        by padding internally but provides access only to valid data pairs.

        demos_train: list of np.ndarray (dim, T_i) or np.ndarray (N, T, D)
        prim_ids:    list or array of length N
        """
        prim_ids = np.array(prim_ids, dtype=np.int64)

        # case 1: demos_train is a list of (dim, T_i) arrays (variable lengths)
        if isinstance(demos_train, list):
            n = len(demos_train)
            dims = [traj.shape[0] for traj in demos_train]
            if len(set(dims)) != 1:
                raise ValueError(f"All trajectories must have same state-dimension but got {set(dims)}")
            D = dims[0]

            # --- Start of Changes ---
            # Store the original length of each trajectory
            self._original_lengths = [traj.shape[1] for traj in demos_train]
            # Find the maximum length for padding
            T_max = max(self._original_lengths)

            # Create a padded numpy array to store all trajectories (for easy indexing)
            # Data access will still be limited to original lengths via _valid_indices
            data = np.zeros((n, T_max, D), dtype=float)
            for i, traj in enumerate(demos_train):
                ti = traj.shape[1]
                # Transpose traj from (D, T_i) to (T_i, D) before placing in data
                data[i, :ti, :] = traj.T # Copy original data, leave rest as zeros (padding)

            self.trajs = data        # shape (N, T_max, D) - padded array
            self.T = T_max           # Max (padded) length
            self.D = D               # State dimension
            self.N = n               # Number of trajectories

            # Pre-calculate all valid (trajectory_idx, time_step) pairs
            # A valid pair (x_t, x_{t+1}) exists for t0 from 0 up to original_length - 2
            self._valid_indices = []
            for traj_idx in range(self.N):
                original_length = self._original_lengths[traj_idx]
                 # Iterate up to original_length - 1 to get pairs (t0, t0+1)
                 # where t0+1 is the last actual data point index
                for t0 in range(original_length - 1):
                     self._valid_indices.append((traj_idx, t0))

            print(f"Created {len(self._valid_indices)} valid (state, next_state) pairs from {self.N} trajectories.")

            # --- End of Changes ---

        # case 2: demos_train is already a 3D np.ndarray (assumed uniform length or already padded externally)
        # We still generate valid indices based on the assumed length self.T
        elif isinstance(demos_train, np.ndarray) and demos_train.ndim == 3:
            self.trajs = demos_train  # assumed shape (N, T, D)
            self.N, self.T, self.D = demos_train.shape

            # --- Start of Changes: Generate valid indices assuming uniform length ---
            # Assume all trajectories in the 3D array have the length self.T
            self._original_lengths = [self.T] * self.N
            self._valid_indices = []
            for traj_idx in range(self.N):
                 original_length = self._original_lengths[traj_idx]
                 # Iterate up to original_length - 1 to get pairs (t0, t0+1)
                 # This will cover all steps if original_length == self.T
                 for t0 in range(original_length - 1):
                     self._valid_indices.append((traj_idx, t0))

            print(f"Created {len(self._valid_indices)} valid (state, next_state) pairs from {self.N} uniform length trajectories.")
            # --- End of Changes ---

        else:
            raise ValueError(
                "Unsupported demos_train type: "
                f"{type(demos_train)} with ndim={getattr(demos_train, 'ndim', None)}"
            )

        self.prim_ids = prim_ids
        assert self.N == len(self.prim_ids), \
            f"Number of prim_ids ({len(self.prim_ids)}) must match number of trajectories ({self.N})"

    # --- Start of Changes ---
    def __len__(self):
        """
        Returns the total number of valid (state, next_state) pairs across all trajectories.
        This is the effective size of the dataset for training.
        """
        return len(self._valid_indices)
    # --- End of Changes ---


    def __getitem__(self, idx):
        """
        Retrieves a single valid (state, next_state, primitive_id) tuple.
        The index 'idx' maps to a pre-calculated valid pair index.
        """
        # --- Start of Changes ---
        # Get the actual trajectory index and time step from the pre-calculated valid indices
        traj_idx, t0 = self._valid_indices[idx]
        # --- End of Changes ---

        # Retrieve data points from the padded array using the valid indices
        x_t = self.trajs[traj_idx, t0  ] # shape (D,) - current state
        x_tp1 = self.trajs[traj_idx, t0+1] # shape (D,) - next state
        prim = self.prim_ids[traj_idx]   # Primitive ID for this trajectory

        return (
            torch.FloatTensor(x_t),
            torch.FloatTensor(x_tp1),
            torch.tensor(prim, dtype=torch.long)
        )

def main():
    # --- 0) Params 설정 (예시) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = SimpleNamespace(
        # DataPreprocessor 용
        trajectories_resample_length=100,
        state_increment=0.1,
        workspace_dimensions=2,
        dynamical_system_order=2,
        workspace_boundaries_type='from data',
        workspace_boundaries=[[0,0],[0,0]],
        evaluation_samples_length=50,
        dataset_name='LAIR',
        selected_primitives_ids='0,1,2,3,4,5,6,7,8,9,10',
        spline_sample_type='evenly spaced',
        imitation_window_size=10,
        verbose=False,
        # Encoder 학습용
        batch_size=1,
        latent_dim=4,
        neurons_hidden_layers=[128,64],
        n_primitives=2,
        multi_motion=False,
        lr=1e-3,
        margin=1e-4,
        epochs=1,
        output_dir='./checkpoints'
    )

    # --- 1) 데이터 전처리 실행 ---
    pre = DataPreprocessor(params, verbose=False)
    out = pre.run()

    demos_train = out['demonstrations raw']              # (N, W, D)
    prim_ids    = out['demonstrations primitive id']       # (N,)
    print("Number of demonstrations:", len(demos_train))
    print("Shapes of each demo (dim, length):")
    for i, traj in enumerate(demos_train):
        print(f"  demo {i}: {traj.shape}")
    # 'goals training' 은 normalized goal 들: shape (n_primitives, D)
    goals = torch.FloatTensor(out['goals']).to(device)
    print(goals.shape)
    

    # --- 2) DataLoader 준비 ---
    dataset = PreprocessedDataset(demos_train, prim_ids)
    loader  = DataLoader(dataset, batch_size=params.batch_size,
                         shuffle=False, num_workers=4)

    # --- 3) 모델/손실/옵티마이저 초기화 ---
    encoder = Encoder(
        dim_state=dataset.D,
        latent_dim=params.latent_dim,
        hidden_layers=params.neurons_hidden_layers,
        n_primitives=params.n_primitives,
        multi_motion=params.multi_motion
    ).to(device)

    triplet_loss = TripletLoss(margin=params.margin, swap=True)
    optimizer    = optim.Adam(encoder.parameters(), lr=params.lr)

    # --- 4) 학습 루프 ---
    for epoch in range(1, params.epochs + 1):
        encoder.train()
        running_loss = 0.0

        for x_t, x_tp1, prim in loader:
            x_t   = x_t.to(device)
            x_tp1 = x_tp1.to(device)
            prim  = prim.to(device)
            # print("x_t: ", x_t)

            # 4.1) 임베딩
            emb_t    = encoder(x_t, prim)      # anchor 이전 상태
            emb_tp1  = encoder(x_tp1, prim)    # anchor 긍정(positive) 상태
            # goal 이 anchor 역할
            emb_goal   = torch.zeros(params.latent_dim)
            


            # 4.2) Triplet 손실: (anchor=goal, positive=emb_tp1, negative=emb_t)
            loss = triplet_loss(emb_goal, emb_tp1, emb_t)

            # 4.3) 역전파 및 갱신
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"[Epoch {epoch:03d}/{params.epochs}] Avg Triplet Loss: {avg_loss:.4f}")

    # --- 5) 모델 저장 ---
    os.makedirs(params.output_dir, exist_ok=True)
    ckpt = os.path.join(params.output_dir, 'triplet_encoder.pth')
    torch.save(encoder.state_dict(), ckpt)
    print(f"Saved encoder to {ckpt}")

if __name__ == '__main__':
    main()
