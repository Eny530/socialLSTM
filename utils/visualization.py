"""Visualization utilities for trajectory prediction."""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


def visualize_trajectory(
    obs_traj: torch.Tensor,
    pred_traj: torch.Tensor,
    gt_traj: Optional[torch.Tensor] = None,
    seq_start_end: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    limit: int = 10,
):
    """
    Visualize predicted trajectories.
    
    Args:
        obs_traj: Observed trajectory [obs_len, batch, 2]
        pred_traj: Predicted trajectory [pred_len, batch, 2]
        gt_traj: Ground truth trajectory [pred_len, batch, 2]
        seq_start_end: Start and end indices for sequences [batch, 2]
        save_path: Path to save the visualization
        limit: Maximum number of trajectories to visualize
    """
    # Move to numpy for easier plotting
    obs_traj = obs_traj.cpu().numpy()
    pred_traj = pred_traj.cpu().numpy()
    if gt_traj is not None:
        gt_traj = gt_traj.cpu().numpy()
    
    # If seq_start_end is not provided, assume all trajectories are in one sequence
    if seq_start_end is None:
        seq_start_end = [(0, obs_traj.shape[1])]
    else:
        seq_start_end = seq_start_end.cpu().numpy()
    
    # Count number of sequences to plot
    num_seq = min(len(seq_start_end), limit)
    
    # Create figures for each sequence
    for i in range(num_seq):
        plt.figure(figsize=(10, 8))
        
        # Get sequence information
        (start, end) = seq_start_end[i]
        if hasattr(start, 'item'):  # Check if it's a tensor object
            start = start.item()
            end = end.item()
        
        # Limit the number of pedestrians to visualize
        num_peds = min(end - start, limit)
        
        # Plot each pedestrian's trajectory
        for ped_idx in range(start, start + num_peds):
            # Observed trajectory (blue)
            plt.plot(
                obs_traj[:, ped_idx, 0], obs_traj[:, ped_idx, 1],
                'b-', label='Observed' if ped_idx == start else None
            )
            
            # Predicted trajectory (red)
            plt.plot(
                pred_traj[:, ped_idx, 0], pred_traj[:, ped_idx, 1],
                'r-', label='Predicted' if ped_idx == start else None
            )
            
            # Ground truth trajectory (green) if available
            if gt_traj is not None:
                plt.plot(
                    gt_traj[:, ped_idx, 0], gt_traj[:, ped_idx, 1],
                    'g-', label='Ground Truth' if ped_idx == start else None
                )
            
            # Mark the last observed point (transition point)
            plt.plot(
                obs_traj[-1, ped_idx, 0], obs_traj[-1, ped_idx, 1],
                'ko', markersize=6
            )
        
        # Add legend and labels
        plt.legend()
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(f'Trajectory Visualization (Sequence {i+1})')
        plt.grid(True)
        
        # Save or show the figure
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(f"{save_path}_seq{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        metrics: Dictionary of metrics per epoch
        save_path: Path to save the visualization
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation losses
    epochs = range(1, len(train_losses) + 1)
    axs[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axs[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot ADE
    if 'avg_ade' in metrics:
        axs[0, 1].plot(epochs, metrics['avg_ade'], 'g-', label='Average ADE')
        axs[0, 1].set_title('Average Displacement Error (ADE)')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('ADE [m]')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
    
    # Plot FDE
    if 'avg_fde' in metrics:
        axs[1, 0].plot(epochs, metrics['avg_fde'], 'm-', label='Average FDE')
        axs[1, 0].set_title('Final Displacement Error (FDE)')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('FDE [m]')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
    
    # Plot additional metrics
    if 'weighted_ade' in metrics:
        axs[1, 1].plot(epochs, metrics['weighted_ade'], 'c-', label='Weighted ADE')
        if 'weighted_fde' in metrics:
            axs[1, 1].plot(epochs, metrics['weighted_fde'], 'y-', label='Weighted FDE')
        axs[1, 1].set_title('Weighted Metrics')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Error [m]')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
    
    # Adjust layout and save or show the figure
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
