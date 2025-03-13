"""Base dataset class for trajectory datasets."""

from typing import Dict, List, Tuple, Optional, Union
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """Base class for trajectory datasets."""
    
    def __init__(
        self,
        data_dir: str,
        obs_len: int = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_ped: int = 1,
        delim: str = ' ',
        norm_lap_matr: bool = True
    ):
        """
        Initialize trajectory dataset.
        
        Args:
            data_dir: Directory containing dataset files
            obs_len: Observed trajectory length
            pred_len: Prediction trajectory length
            skip: Number of frames to skip
            min_ped: Minimum number of pedestrians in a sequence
            delim: Delimiter used in the dataset files
            norm_lap_matr: Whether to normalize Laplacian Matrix
        """
        super(TrajectoryDataset, self).__init__()
        
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.min_ped = min_ped
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        
        # Data structures
        self.sequences = []
        self.seq_start_end = []
        
    def _load_data(self, file_path: str) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Load and process data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Tuple of (sequences, sequence_start_end)
        """
        raise NotImplementedError("Dataset-specific loading not implemented")
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence.
        
        Args:
            idx: Sequence index
            
        Returns:
            Dictionary containing:
                - obs_traj: Observed trajectory [obs_len, num_peds, 2]
                - pred_traj: Ground truth future trajectory [pred_len, num_peds, 2]
                - obs_traj_rel: Observed velocity [obs_len, num_peds, 2]
                - pred_traj_rel: Ground truth future velocity [pred_len, num_peds, 2]
                - seq_start_end: Start and end indices for sequences
        """
        sequence = self.sequences[idx]
        seq_start, seq_end = self.seq_start_end[idx]
        
        # Extract observed and prediction trajectories
        obs_traj = sequence[:self.obs_len, :, :]
        pred_traj = sequence[self.obs_len:, :, :]
        
        # Convert to relative coordinates (velocities)
        obs_traj_rel = np.zeros_like(obs_traj)
        pred_traj_rel = np.zeros_like(pred_traj)
        
        obs_traj_rel[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]
        pred_traj_rel[0, :, :] = pred_traj[0, :, :] - obs_traj[-1, :, :]
        pred_traj_rel[1:, :, :] = pred_traj[1:, :, :] - pred_traj[:-1, :, :]
        
        # Convert to tensors
        obs_traj = torch.from_numpy(obs_traj).float()
        pred_traj = torch.from_numpy(pred_traj).float()
        obs_traj_rel = torch.from_numpy(obs_traj_rel).float()
        pred_traj_rel = torch.from_numpy(pred_traj_rel).float()
        seq_start_end = torch.tensor([seq_start, seq_end]).long()
        
        return {
            'obs_traj': obs_traj,
            'pred_traj': pred_traj,
            'obs_traj_rel': obs_traj_rel,
            'pred_traj_rel': pred_traj_rel,
            'seq_start_end': seq_start_end
        }
