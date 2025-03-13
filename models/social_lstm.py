"""
Social LSTM model implementation.
Based on the paper:
"Social LSTM: Human Trajectory Prediction in Crowded Spaces" by Alahi et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class SocialPooling(nn.Module):
    """Social Pooling Layer for the Social LSTM model."""
    
    def __init__(
        self,
        h_dim: int,
        embedding_dim: int,
        grid_size: int,
        neighborhood_size: float,
        device: torch.device
    ):
        """
        Initialize the social pooling layer.
        
        Args:
            h_dim: Hidden state dimension
            embedding_dim: Embedding dimension for the pooled social tensor
            grid_size: Grid size for discretizing the neighborhood
            neighborhood_size: Neighborhood size in meters
            device: Device to run the model on
        """
        super(SocialPooling, self).__init__()
        
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        self.device = device
        
        # Pooling layer that maps from NxN grid to a single vector
        self.pool_net = nn.Sequential(
            nn.Linear(h_dim * grid_size * grid_size, embedding_dim),
            nn.ReLU()
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_start_end: torch.Tensor,
        curr_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the social pooling layer.
        
        Args:
            hidden_states: Hidden states of all pedestrians [num_peds, h_dim]
            seq_start_end: Start and end indices for sequences [batch, 2]
            curr_pos: Current positions of all pedestrians [num_peds, 2]
            
        Returns:
            Pooled social context [num_peds, embedding_dim]
        """
        num_peds = hidden_states.size(0)
        
        # Initialize the pooled tensor
        pooled_tensor = torch.zeros(
            num_peds, self.grid_size, self.grid_size, self.h_dim,
            device=self.device
        )
        
        # For each sequence, compute social tensor
        for batch_idx, (start, end) in enumerate(seq_start_end):
            sequence_length = end - start
            
            # Skip if only one pedestrian
            if sequence_length <= 1:
                continue
                
            # Get positions and hidden states for the current sequence
            curr_hidden = hidden_states[start:end]
            curr_positions = curr_pos[start:end]
            
            # For each pedestrian, compute relative positions to others
            for ped_idx in range(sequence_length):
                # Get the current pedestrian's position and state
                ped_pos = curr_positions[ped_idx]
                ped_hidden = curr_hidden[ped_idx]
                
                # Compute relative positions of all other pedestrians
                rel_pos = curr_positions - ped_pos.unsqueeze(0)
                
                # Convert to grid coordinates
                grid_size_norm = self.neighborhood_size / (self.grid_size - 1)
                rel_pos = torch.clamp(
                    rel_pos / grid_size_norm,
                    min=-(self.grid_size // 2),
                    max=(self.grid_size // 2)
                )
                rel_grid_pos = (rel_pos + self.grid_size // 2).long()
                
                # For all other pedestrians, update the pooled tensor
                for other_idx in range(sequence_length):
                    if other_idx != ped_idx:
                        other_x, other_y = rel_grid_pos[other_idx]
                        other_hidden = curr_hidden[other_idx]
                        
                        # Skip if outside grid bounds
                        if (0 <= other_x < self.grid_size and 
                            0 <= other_y < self.grid_size):
                            # Add to pooled tensor
                            pooled_tensor[start + ped_idx, other_y, other_x] += other_hidden
        
        # Reshape and pass through pool_net
        pooled_tensor = pooled_tensor.view(num_peds, -1)
        pooled_output = self.pool_net(pooled_tensor)
        
        return pooled_output


class SocialLSTMCell(nn.Module):
    """LSTM cell with social pooling."""
    
    def __init__(
        self,
        input_dim: int,
        h_dim: int,
        social_dim: int,
        dropout: float = 0.0
    ):
        """
        Initialize the Social LSTM cell.
        
        Args:
            input_dim: Input dimension
            h_dim: Hidden state dimension
            social_dim: Social context dimension
            dropout: Dropout probability
        """
        super(SocialLSTMCell, self).__init__()
        
        # Dimension parameters
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.social_dim = social_dim
        
        # LSTM gates
        self.i_gate = nn.Linear(input_dim + h_dim + social_dim, h_dim)
        self.f_gate = nn.Linear(input_dim + h_dim + social_dim, h_dim)
        self.o_gate = nn.Linear(input_dim + h_dim + social_dim, h_dim)
        self.g_gate = nn.Linear(input_dim + h_dim + social_dim, h_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        social_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Social LSTM cell.
        
        Args:
            x: Input tensor [batch, input_dim]
            h_prev: Previous hidden state [batch, h_dim]
            c_prev: Previous cell state [batch, h_dim]
            social_tensor: Social tensor [batch, social_dim]
            
        Returns:
            Tuple of (hidden state, cell state)
        """
        # Concatenate inputs
        combined = torch.cat([x, h_prev, social_tensor], dim=1)
        
        # Compute gate values
        i = torch.sigmoid(self.i_gate(combined))
        f = torch.sigmoid(self.f_gate(combined))
        o = torch.sigmoid(self.o_gate(combined))
        g = torch.tanh(self.g_gate(combined))
        
        # Update cell and hidden states
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        
        h = self.dropout(h)
        
        return h, c


class SocialLSTM(nn.Module):
    """Social LSTM model for trajectory prediction."""
    
    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        embedding_dim: int = 64,
        h_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        neighborhood_size: float = 2.0,
        grid_size: int = 8,
        device: torch.device = None
    ):
        """
        Initialize the Social LSTM model.
        
        Args:
            obs_len: Observed trajectory length
            pred_len: Prediction trajectory length
            embedding_dim: Embedding dimension
            h_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            neighborhood_size: Neighborhood size in meters
            grid_size: Grid size for discretizing the neighborhood
            device: Device to run the model on
        """
        super(SocialLSTM, self).__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embedding layer for converting input to embedding
        self.input_embedding = nn.Linear(2, embedding_dim)
        
        # Social pooling layer
        self.social_pooling = SocialPooling(
            h_dim=h_dim,
            embedding_dim=embedding_dim,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size,
            device=self.device
        )
        
        # LSTM cells
        self.lstm_cells = nn.ModuleList([
            SocialLSTMCell(
                input_dim=embedding_dim,
                h_dim=h_dim,
                social_dim=embedding_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(h_dim, 2)
    
    def forward(
        self,
        obs_traj_rel: torch.Tensor,
        seq_start_end: torch.Tensor,
        obs_traj: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.0
    ) -> torch.Tensor:
        """
        Forward pass for the Social LSTM model.
        
        Args:
            obs_traj_rel: Observed trajectory velocity [obs_len, num_peds, 2]
            seq_start_end: Start and end indices for sequences
            obs_traj: Observed trajectory positions [obs_len, num_peds, 2]
            teacher_forcing_ratio: Ratio for teacher forcing

        Returns:
            Predicted trajectory [pred_len, num_peds, 2]
        """
        # Handle seq_start_end in different formats
        if seq_start_end.dim() == 1:
            # If it's a single tensor with just start and end
            seq_start_end = seq_start_end.unsqueeze(0)

        num_peds = obs_traj_rel.size(1)

        # Initialize hidden and cell states
        h = [torch.zeros(num_peds, self.h_dim, device=self.device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(num_peds, self.h_dim, device=self.device)
             for _ in range(self.num_layers)]

        # For storing predictions
        pred_traj_rel = []

        # Initialize current position for social pooling
        if obs_traj is None:
            # If absolute positions not provided, use zeros
            curr_pos = torch.zeros(num_peds, 2, device=self.device)
        else:
            # Use the last observed position
            curr_pos = obs_traj[-1]

        # Encode observed trajectory
        for obs_step in range(self.obs_len):
            curr_vel = obs_traj_rel[obs_step]

            # Embed current velocity
            curr_emb = self.input_embedding(curr_vel)

            # Update current positions for social pooling
            if obs_step > 0 and obs_traj is None:
                curr_pos = curr_pos + obs_traj_rel[obs_step]
            elif obs_traj is not None:
                curr_pos = obs_traj[obs_step]

            # Get social context for each pedestrian
            social_tensor = self.social_pooling(h[-1], seq_start_end, curr_pos)

            # Update states through LSTM layers
            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.lstm_cells[layer_idx](
                    curr_emb, h[layer_idx], c[layer_idx], social_tensor
                )

                if layer_idx < self.num_layers - 1:
                    curr_emb = h[layer_idx]

        # Predict future trajectory
        for pred_step in range(self.pred_len):
            # Get social context for each pedestrian
            social_tensor = self.social_pooling(h[-1], seq_start_end, curr_pos)

            # Generate output from last hidden state
            pred_vel = self.output_layer(h[-1])

            # Update current position
            curr_pos = curr_pos + pred_vel

            # Save prediction
            pred_traj_rel.append(pred_vel)

            # Embed current velocity
            curr_emb = self.input_embedding(pred_vel)

            # Update states through LSTM layers
            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.lstm_cells[layer_idx](
                    curr_emb, h[layer_idx], c[layer_idx], social_tensor
                )

                if layer_idx < self.num_layers - 1:
                    curr_emb = h[layer_idx]

        # Stack predictions
        pred_traj_rel = torch.stack(pred_traj_rel, dim=0)

        return pred_traj_rel