"""Loss functions for training trajectory prediction models."""

import torch
import numpy as np
from typing import Dict, Tuple


def l2_loss(pred_traj: torch.Tensor, pred_traj_gt: torch.Tensor, mode: str = 'sum') -> torch.Tensor:
    """
    L2 loss for trajectory prediction.
    
    Args:
        pred_traj: Predicted trajectory [pred_len, batch, 2]
        pred_traj_gt: Ground truth trajectory [pred_len, batch, 2]
        mode: Reduction mode ('sum' or 'mean' or 'none')
        
    Returns:
        L2 loss
    """
    seq_len, batch, _ = pred_traj.size()
    
    # Calculate squared error
    error = (pred_traj_gt - pred_traj) ** 2
    
    # Sum over coordinates
    error = torch.sum(error, dim=2)
    
    # Apply reduction
    if mode == 'sum':
        return torch.sum(error)
    elif mode == 'mean':
        return torch.mean(error)
    elif mode == 'none':
        return error
    else:
        raise ValueError("Mode must be 'sum', 'mean', or 'none'")


def displacement_error(pred_traj: torch.Tensor, pred_traj_gt: torch.Tensor, mode: str = 'mean') -> torch.Tensor:
    """
    Average displacement error (ADE) for trajectory prediction.
    
    Args:
        pred_traj: Predicted trajectory [pred_len, batch, 2]
        pred_traj_gt: Ground truth trajectory [pred_len, batch, 2]
        mode: Reduction mode ('sum' or 'mean' or 'none')
        
    Returns:
        Average displacement error
    """
    # Calculate L2 distance
    error = l2_loss(pred_traj, pred_traj_gt, mode='none')
    
    # Square root to get Euclidean distance
    error = torch.sqrt(error)
    
    # Apply reduction along time dimension
    if mode == 'sum':
        return torch.sum(error, dim=0)
    elif mode == 'mean':
        return torch.mean(error, dim=0)
    elif mode == 'none':
        return error
    else:
        raise ValueError("Mode must be 'sum', 'mean', or 'none'")


def final_displacement_error(pred_traj: torch.Tensor, pred_traj_gt: torch.Tensor, mode: str = 'mean') -> torch.Tensor:
    """
    Final displacement error (FDE) for trajectory prediction.
    
    Args:
        pred_traj: Predicted trajectory [pred_len, batch, 2]
        pred_traj_gt: Ground truth trajectory [pred_len, batch, 2]
        mode: Reduction mode ('sum' or 'mean' or 'none')
        
    Returns:
        Final displacement error
    """
    # Get the last prediction
    pred_pos_last = pred_traj[-1]
    gt_pos_last = pred_traj_gt[-1]
    
    # Calculate squared error
    error = (gt_pos_last - pred_pos_last) ** 2
    
    # Sum over coordinates
    error = torch.sum(error, dim=1)
    
    # Square root to get Euclidean distance
    error = torch.sqrt(error)
    
    # Apply reduction
    if mode == 'sum':
        return torch.sum(error)
    elif mode == 'mean':
        return torch.mean(error)
    elif mode == 'none':
        return error
    else:
        raise ValueError("Mode must be 'sum', 'mean', or 'none'")
