#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization script for CONDOR-inspired motion primitives.

This standalone script extracts and consolidates the visualization logic from the
original training pipeline. Execute it after training to generate qualitative
plots of the learned motion primitives and their roll-outs.
"""

import os
import argparse
import yaml

# Torch is only required for loading tensors from the cached dataset
import torch  # noqa: F401  # Imported for side-effects (device placement)

from utils.visualization import (
    visualize_from_dataset,
    evaluate_and_visualize_trajectories,
)
from models.motion_primitive_model import MotionPrimitiveModel
from data_pipeline import load_preprocessed_data


def load_model(checkpoint_dir: str, device: str = "cpu") -> MotionPrimitiveModel:
    """Utility wrapper to restore a trained :class:`MotionPrimitiveModel`."""
    model = MotionPrimitiveModel.load_from_checkpoint(checkpoint_dir, map_location=device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize learned CONDOR motion primitives from a trained model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory that contains the trained model checkpoints",
    )
    parser.add_argument(
        "--data_cache",
        type=str,
        required=True,
        help="Path to the cached pre-processed dataset (.npz) generated during training",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/vis",
        help="Directory where visualizations will be written",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML file that overrides default visualization parameters",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Load model & pre-processed data
    # ---------------------------------------------------------------------
    model = load_model(args.model_dir)
    preprocessed_data = load_preprocessed_data(args.data_cache)

    # ---------------------------------------------------------------------
    # 2. Set normalization parameters for the model
    # ---------------------------------------------------------------------
    vel_min = torch.from_numpy(preprocessed_data['vel min train'].reshape(1, -1, 1)).float().to(model.device)
    vel_max = torch.from_numpy(preprocessed_data['vel max train'].reshape(1, -1, 1)).float().to(model.device)

    if 'acc min train' in preprocessed_data and 'acc max train' in preprocessed_data:
        acc_min = torch.from_numpy(preprocessed_data['acc min train'].reshape(1, -1, 1)).float().to(model.device)
        acc_max = torch.from_numpy(preprocessed_data['acc max train'].reshape(1, -1, 1)).float().to(model.device)
        model.set_normalization_params(vel_min, vel_max, acc_min, acc_max)
    else:
        model.set_normalization_params(vel_min, vel_max)

    # ---------------------------------------------------------------------
    # 3. Default visualization hyper-parameters
    # ---------------------------------------------------------------------
    viz_params = {
        "n_samples": 5,  # Samples per trajectory
        "steps": 100,  # Roll-out steps
        "sample_radius": 0.1,  # Radius around initial point
        "primitive_ids": [0],  # Which primitive indices to plot
        "trajectory_ids": [0],  # Which demonstration indices to plot
        "batch_eval": False,  # Whether to run aggregate evaluation
    }

    # ---------------------------------------------------------------------
    # 4. YAML overrides (if provided)
    # ---------------------------------------------------------------------
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "visualization" in cfg:
            viz_cfg = cfg["visualization"]
            # Per-trajectory overrides
            viz_params.update(viz_cfg.get("dataset_visualization", {}))
            # Batch evaluation toggle
            viz_params["batch_eval"] = viz_cfg.get("batch_evaluation", {}).get("enable", False)
            batch_cfg = viz_cfg.get("batch_evaluation", {})
        else:
            batch_cfg = {}
    else:
        batch_cfg = {}

    # ---------------------------------------------------------------------
    # 5. Individual trajectory visualizations
    # ---------------------------------------------------------------------
    for prim_id in viz_params["primitive_ids"]:
        for traj_id in viz_params["trajectory_ids"]:
            fig_path = os.path.join(
                args.save_dir, f"dataset_traj_prim{prim_id}_traj{traj_id}.png"
            )
            visualize_from_dataset(
                model=model,
                dataset=preprocessed_data,
                prim_id=prim_id,
                traj_id=traj_id,
                n_samples=viz_params["n_samples"],
                steps=viz_params["steps"],
                sample_radius=viz_params["sample_radius"],
                save_path=fig_path,
                show_original=True,
            )
            print(f"[INFO] Saved {fig_path}")

    # ---------------------------------------------------------------------
    # 6. Optional batch-level evaluation & visualization
    # ---------------------------------------------------------------------
    if viz_params["batch_eval"]:
        eval_dir = os.path.join(args.save_dir, "trajectory_evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        evaluate_and_visualize_trajectories(
            model=model,
            dataset=preprocessed_data,
            save_dir=eval_dir,
            prim_ids=batch_cfg.get("primitive_ids", None),
            traj_per_prim=batch_cfg.get("trajectories_per_primitive", 3),
            n_samples=batch_cfg.get("n_samples", 5),
            steps=batch_cfg.get("steps", 100),
            sample_radius=batch_cfg.get("sample_radius", 0.1),
            seed=batch_cfg.get("seed", None),
        )
        print(f"[INFO] Batch evaluation saved in {eval_dir}")


if __name__ == "__main__":
    main()
