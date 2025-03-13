"""Metrics for evaluating trajectory prediction models."""

import torch
import numpy as np
from typing import Dict, Tuple, List


def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> float:
    """
    Compute Average Displacement Error (ADE).
    
    Args:
        pred_traj: Predicted trajectory [pred_len, batch, 2]
        gt_traj: Ground truth trajectory [pred_len, batch, 2]
        
    Returns:
        Average displacement error
    """
    # Calculate L2 distance for each point in the trajectory
    diff = pred_traj - gt_traj
    dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
    
    # Average over batch and time
    ade = torch.mean(dist)
    
    return ade.item()


def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> float:
    """
    Compute Final Displacement Error (FDE).
    
    Args:
        pred_traj: Predicted trajectory [pred_len, batch, 2]
        gt_traj: Ground truth trajectory [pred_len, batch, 2]
        
    Returns:
        Final displacement error
    """
    # Calculate L2 distance for the final prediction
    diff = pred_traj[-1] - gt_traj[-1]
    dist = torch.sqrt(torch.sum(diff ** 2, dim=1))
    
    # Average over batch
    fde = torch.mean(dist)
    
    return fde.item()


def compute_metrics_per_sequence(
    pred_traj: torch.Tensor, 
    gt_traj: torch.Tensor,
    seq_start_end: torch.Tensor
) -> List[Dict[str, float]]:
    """
    Compute metrics for each sequence.
    
    Args:
        pred_traj: Predicted trajectory [pred_len, batch, 2]
        gt_traj: Ground truth trajectory [pred_len, batch, 2]
        seq_start_end: Start and end indices for sequences [batch, 2] or [2]

    Returns:
        List of metrics for each sequence
    """
    metrics_per_seq = []

    # Ensure both tensors are on CPU
    # But keep as tensors until actual computation
    pred_traj = pred_traj.detach().cpu()
    gt_traj = gt_traj.detach().cpu()
    seq_start_end = seq_start_end.detach().cpu()

    # Handle different seq_start_end formats
    if seq_start_end.dim() == 1:
        # Single sequence case with just start and end
        seq_start_end = seq_start_end.unsqueeze(0)

    # Compute metrics for each sequence
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()

        # Extract sequence data
        curr_pred_traj = pred_traj[:, start:end, :]
        curr_gt_traj = gt_traj[:, start:end, :]

        # Compute metrics
        ade = compute_ade(curr_pred_traj, curr_gt_traj)
        fde = compute_fde(curr_pred_traj, curr_gt_traj)

        metrics_per_seq.append({
            'ade': ade,
            'fde': fde,
            'num_peds': end - start
        })

    return metrics_per_seq


def compute_overall_metrics(metrics_per_seq: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute overall metrics from per-sequence metrics.

    Args:
        metrics_per_seq: List of metrics for each sequence

    Returns:
        Overall metrics
    """
    total_peds = sum(seq['num_peds'] for seq in metrics_per_seq)

    # Compute weighted average based on number of pedestrians
    weighted_ade = sum(seq['ade'] * seq['num_peds'] for seq in metrics_per_seq) / total_peds
    weighted_fde = sum(seq['fde'] * seq['num_peds'] for seq in metrics_per_seq) / total_peds

    # Compute simple average
    avg_ade = np.mean([seq['ade'] for seq in metrics_per_seq])
    avg_fde = np.mean([seq['fde'] for seq in metrics_per_seq])

    return {
        'weighted_ade': weighted_ade,
        'weighted_fde': weighted_fde,
        'avg_ade': avg_ade,
        'avg_fde': avg_fde,
        'total_peds': total_peds
    }