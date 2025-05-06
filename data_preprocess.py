# convert_dataset_to_npz.py

import argparse
import os
import sys
import numpy as np

# Add submodule src to Python path for imports
base_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(base_dir, 'submodule', 'CONDOR', 'src')
sys.path.insert(0, src_path)

from data_preprocessing.data_loader import load_demonstrations  # now resolvable

def parse_args():
    p = argparse.ArgumentParser(
        description="Load a DEMO dataset and save its contents to a .npz file"
    )
    p.add_argument("--dataset", "-d", type=str, required=True,
                   help="Name of the dataset (e.g. 'LASA')")
    p.add_argument("--primitives", "-p", type=str, required=True,
                   help="Comma-separated list of primitive IDs (e.g. '0,1,2')")
    p.add_argument("--outdir", "-o", type=str, default="./npz_data",
                   help="Directory in which to save the .npz file")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load demonstrations via loader
    loaded = load_demonstrations(
        dataset_name=args.dataset,
        selected_primitives_ids=args.primitives
    )
    # loaded keys: 'demonstrations raw', 'demonstrations primitive id', 'n primitives', 'delta t eval'

    # 2) Convert "demonstrations raw" list to an object array manually
    raw_list = loaded['demonstrations raw']  # list of np arrays, each shape (dim, T)
    n_demos = len(raw_list)
    demos_raw = np.empty(n_demos, dtype=object)
    for i, traj in enumerate(raw_list):
        demos_raw[i] = traj

    prim_ids = np.array(loaded['demonstrations primitive id'], dtype=np.int32)
    n_primitives = loaded['n primitives']
    delta_t_eval = np.array(loaded['delta t eval'], dtype=float)

    # 3) Save all to .npz
    out_fname = f"{args.dataset}_{args.primitives}.npz"
    out_path = os.path.join(args.outdir, out_fname)
    np.savez_compressed(
        out_path,
        **{
            'demonstrations raw': demos_raw,
            'demonstrations primitive id': prim_ids,
            'n primitives': n_primitives,
            'delta t eval': delta_t_eval
        }
    )

    # 4) Print confirmation and contents
    print(f"Saved dataset → {out_path}\nContents:")
    data = np.load(out_path, allow_pickle=True)
    for key in data.files:
        arr = data[key]
        print(f"  {key!r}: shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")


if __name__ == "__main__":
    main()
