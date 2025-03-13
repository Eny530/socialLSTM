"""Evaluation script for trajectory prediction models."""

import argparse
import os
import time
import torch
import numpy as np
import random
import logging
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import get_config
from models import SocialLSTM, VanillaLSTM
from utils import (
    data_loader, rel_to_abs, compute_metrics_per_sequence,
    compute_overall_metrics, visualize_trajectory
)


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path, model_type, args, device):
    """
    Load a saved model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint
        model_type: Type of model ('social_lstm' or 'vanilla_lstm')
        args: Configuration parameters
        device: Device to load the model on

    Returns:
        Loaded model and checkpoint data
    """
    try:
        if model_type == 'social_lstm':
            model = SocialLSTM(
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                embedding_dim=args.embedding_dim,
                h_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                neighborhood_size=args.neighborhood_size,
                grid_size=args.grid_size,
                device=device
            )
        elif model_type == 'vanilla_lstm':
            model = VanillaLSTM(
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                embedding_dim=args.embedding_dim,
                h_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                device=device
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return model, checkpoint
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def evaluate(args):
    """
    Evaluate the trajectory prediction model.

    Args:
        args: Configuration parameters including:
            - model: Model type ('social_lstm' or 'vanilla_lstm')
            - dataset: Dataset name (eth, hotel, univ, zara1, zara2)
            - checkpoint_path: Path to the model checkpoint
            - log_dir: Directory for storing evaluation logs
            - run_name: Unique name for this evaluation run
            - num_vis_samples: Number of prediction samples to visualize

    Returns:
        None. Results are saved to the specified log directory.
    """
    # Set up logging
    log_path = os.path.join(args.log_dir, args.run_name, 'eval')
    os.makedirs(log_path, exist_ok=True)

    log_file = os.path.join(log_path, 'eval.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Set random seed
    set_random_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Determine checkpoint path
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    try:
        model, checkpoint = load_model(checkpoint_path, args.model, args, device)
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return

    # Initialize data loader - use 'val' instead of 'test' since data_loader doesn't support 'test'
    logger.info(f"Initializing data loader for {args.dataset} dataset...")
    test_loader = data_loader(args, 'val')

    # Evaluation
    logger.info("Starting evaluation...")
    model.eval()

    all_metrics = []
    vis_count = 0
    vis_dir = os.path.join(log_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluation")):
            # Move batch to device
            processed_batch = {}
            for k, v in batch.items():
                if isinstance(v, list):
                    # Convert lists of tensors to a list on the device
                    processed_batch[k] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in v]
                elif isinstance(v, torch.Tensor):
                    # Move tensors to device
                    processed_batch[k] = v.to(device)
                else:
                    # Keep other types as is
                    processed_batch[k] = v
            batch = processed_batch

            # Forward pass
            # Ensure seq_start_end is a tensor, not a list
            seq_start_end = batch['seq_start_end']
            # if isinstance(seq_start_end, list):
            #     # Convert list to tensor if needed
            #     seq_start_end = torch.tensor(seq_start_end, device=device)

            pred_traj_rel = model(
                obs_traj_rel=batch['obs_traj_rel'],
                seq_start_end=seq_start_end,
                obs_traj=batch['obs_traj']
            )

            # Convert to absolute coordinates for visualization and metrics
            pred_traj = rel_to_abs(pred_traj_rel, batch['obs_traj'][-1])

            # Compute metrics
            batch_metrics = compute_metrics_per_sequence(
                pred_traj, batch['pred_traj'], batch['seq_start_end']
            )
            all_metrics.extend(batch_metrics)

            # Visualize some predictions
            if vis_count < args.num_vis_samples:
                vis_path = os.path.join(vis_dir, f'batch_{batch_idx}')

                visualize_trajectory(
                    obs_traj=batch['obs_traj'],
                    pred_traj=pred_traj,
                    gt_traj=batch['pred_traj'],
                    seq_start_end=batch['seq_start_end'],
                    save_path=vis_path
                )
                vis_count += 1

    # Compute overall metrics
    overall_metrics = compute_overall_metrics(all_metrics)

    # Log results
    logger.info(f"Evaluation Results:")
    logger.info(f"Average ADE: {overall_metrics['avg_ade']:.4f} m")
    logger.info(f"Average FDE: {overall_metrics['avg_fde']:.4f} m")
    logger.info(f"Weighted ADE: {overall_metrics['weighted_ade']:.4f} m")
    logger.info(f"Weighted FDE: {overall_metrics['weighted_fde']:.4f} m")
    logger.info(f"Total pedestrians: {overall_metrics['total_peds']}")

    # Save results to JSON
    results_path = os.path.join(log_path, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(overall_metrics, f, indent=4)

    logger.info(f"Saved results to {results_path}")

    # Plot distribution of errors
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    ade_values = [m['ade'] for m in all_metrics]
    plt.hist(ade_values, bins=20)
    plt.xlabel('ADE [m]')
    plt.ylabel('Count')
    plt.title('Distribution of ADE')

    plt.subplot(1, 2, 2)
    fde_values = [m['fde'] for m in all_metrics]
    plt.hist(fde_values, bins=20)
    plt.xlabel('FDE [m]')
    plt.ylabel('Count')
    plt.title('Distribution of FDE')

    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'error_distribution.png'), dpi=300)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Evaluate trajectory prediction model")

    # Add evaluation-specific arguments
    parser.add_argument('--model', type=str, default='social_lstm',
                      choices=['vanilla_lstm', 'social_lstm'],
                      help='Model to use')
    parser.add_argument('--dataset', type=str, default='eth',
                      help='Dataset name: eth, hotel, univ, zara1, zara2')
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='Directory containing dataset files')
    parser.add_argument('--checkpoint_path', type=str, default='',
                      help='Path to the checkpoint to evaluate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory containing checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                      help='Directory for logging')
    parser.add_argument('--num_vis_samples', type=int, default=10,
                      help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')

    # Parse command line arguments
    args = parser.parse_args()

    # If checkpoint_path not specified, use default path
    if not args.checkpoint_path:
        args.checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model}_{args.dataset}", 'best_model.pt')

    # Get other necessary configurations
    config_args = get_config()

    # Update args with other necessary parameters from config
    for key, value in vars(config_args).items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    # Create run name
    args.run_name = f"{args.model}_{args.dataset}_eval"

    evaluate(args)