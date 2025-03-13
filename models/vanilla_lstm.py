"""
Vanilla LSTM model implementation as a baseline for trajectory prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union


class VanillaLSTM(nn.Module):
    """Vanilla LSTM model for trajectory prediction."""
    
    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        embedding_dim: int = 64,
        h_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        device: torch.device = None
    ):
        """
        Initialize the Vanilla LSTM model.
        
        Args:
            obs_len: Observed trajectory length
            pred_len: Prediction trajectory length
            embedding_dim: Embedding dimension
            h_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            device: Device to run the model on
        """
        super(VanillaLSTM, self).__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embedding layer for converting input to embedding
        self.input_embedding = nn.Linear(2, embedding_dim)
        
        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )
        
        # LSTM decoder
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )
        
        # Output layer
        self.output_layer = nn.Linear(h_dim, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        obs_traj_rel: torch.Tensor,
        seq_start_end: torch.Tensor = None,
        obs_traj: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.0
    ) -> torch.Tensor:
        """
        Forward pass for the Vanilla LSTM model.
        
        Args:
            obs_traj_rel: Observed trajectory velocity [obs_len, num_peds, 2]
            seq_start_end: Not used in this model, included for API compatibility
            obs_traj: Not used in this model, included for API compatibility
            teacher_forcing_ratio: Ratio for teacher forcing
            
        Returns:
            Predicted trajectory [pred_len, num_peds, 2]
        """
        # Extract dimensions
        seq_len, batch, input_dim = obs_traj_rel.size()
        
        # Embed observed trajectory
        obs_traj_embedding = self.input_embedding(obs_traj_rel.reshape(-1, input_dim))
        obs_traj_embedding = obs_traj_embedding.reshape(seq_len, batch, self.embedding_dim)
        
        # Encode observed trajectory
        _, (hidden, cell) = self.encoder(obs_traj_embedding)
        
        # For storing predictions
        pred_traj_rel = []
        
        # Initial decoder input (last observed velocity)
        decoder_input = obs_traj_embedding[-1].unsqueeze(0)
        
        # Predict future trajectory
        for _ in range(self.pred_len):
            # Get output from decoder
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            
            # Apply dropout
            output = self.dropout(output)
            
            # Get velocity prediction
            pred_vel = self.output_layer(output.squeeze(0))
            pred_traj_rel.append(pred_vel)
            
            # Prepare next input
            decoder_input = self.input_embedding(pred_vel).unsqueeze(0)
        
        # Stack predictions
        pred_traj_rel = torch.stack(pred_traj_rel, dim=0)
        
        return pred_traj_rel
