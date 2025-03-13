"""Utility functions for data processing."""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import DataLoader

from datasets.eth_ucy import ETHUCYDataset


def rel_to_abs(rel_traj: torch.Tensor, start_pos: torch.Tensor) -> torch.Tensor:
    """
    Convert relative coordinates to absolute coordinates.
    
    Args:
        rel_traj: Relative coordinates [seq_len, batch, 2]
        start_pos: Starting positions [batch, 2]
        
    Returns:
        Absolute coordinates [seq_len, batch, 2]
    """
    # Store original device before converting to numpy
    device = rel_traj.device

    # Convert to numpy for easier manipulation
    rel_traj_np = rel_traj.detach().cpu().numpy()
    start_pos_np = start_pos.detach().cpu().numpy()

    # Get dimensions
    seq_len, batch, _ = rel_traj_np.shape

    # Initialize absolute trajectory
    abs_traj = np.zeros((seq_len, batch, 2))
    abs_traj[0] = start_pos_np + rel_traj_np[0]

    # Compute cumulative sum
    for i in range(1, seq_len):
        abs_traj[i] = abs_traj[i-1] + rel_traj_np[i]

    # Convert back to tensor and move to original device
    return torch.from_numpy(abs_traj).to(device)


def get_dset_path(data_dir: str, dataset_name: str) -> str:
    """
    Get the path to the dataset.

    Args:
        data_dir: Directory containing dataset files
        dataset_name: Dataset name

    Returns:
        Path to the dataset
    """
    return os.path.join(data_dir, dataset_name)


def data_loader(args, phase: str) -> DataLoader:
    """
    Creates data loaders for training and testing.

    Args:
        args: Arguments
        phase: Phase (train or val)

    Returns:
        Data loader
    """
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False

    # Create dataset
    dset = ETHUCYDataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        min_ped=1
    )

    # Create loader
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=seq_collate
    )

    return loader


def seq_collate(data: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
    """
    Collate function for the dataloader that handles variable-sized sequences.
    Instead of stacking tensors with different dimensions, we keep them as a list.

    Args:
        data: List of dictionaries with data samples

    Returns:
        Batch of data as lists of tensors
    """
    # Return a single sample directly if batch size is 1
    if len(data) == 1:
        return data[0]

    # For multiple samples, we need a more complex approach
    # We'll treat each sequence as a separate entity

    # Extract keys from the first sample
    keys = data[0].keys()

    # Create empty dictionaries for each key
    batch = {k: [] for k in keys}

    # Fill the batch
    for item in data:
        for k in keys:
            batch[k].append(item[k])

    return batch


def random_rotate(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Randomly rotate trajectories for data augmentation.

    Args:
        data: Dictionary with trajectory data

    Returns:
        Rotated data
    """
    # Copy data
    rotated_data = {k: v.clone() for k, v in data.items()}

    # Generate random rotation angle
    theta = np.random.uniform(0, 2 * np.pi)

    # Create rotation matrix
    rot_matrix = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=torch.float32)

    # Apply rotation to trajectories
    for key in ['obs_traj', 'pred_traj', 'obs_traj_rel', 'pred_traj_rel']:
        if key in rotated_data:
            # Reshape for matrix multiplication
            orig_shape = rotated_data[key].shape
            reshaped = rotated_data[key].view(-1, 2)

            # Apply rotation
            rotated = torch.matmul(reshaped, rot_matrix)

            # Reshape back
            rotated_data[key] = rotated.view(orig_shape)

    return rotated_data