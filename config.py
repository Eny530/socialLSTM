"""Configuration parameters for Social LSTM model."""

import argparse
import os
from datetime import datetime

def get_config():
    parser = argparse.ArgumentParser(description="Social LSTM")
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='eth', 
                        help='Dataset name: eth, hotel, univ, zara1, zara2')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing dataset files')
    parser.add_argument('--obs_len', type=int, default=8,
                        help='Observed trajectory length')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='Prediction trajectory length')
    parser.add_argument('--skip', type=int, default=1,
                        help='Number of frames to skip')
    parser.add_argument('--delim', type=str, default=' ',
                        help='Delimiter used in the dataset files (space by default, tab will be auto-detected)')

    # Model parameters
    parser.add_argument('--model', type=str, default='social_lstm',
                        choices=['vanilla_lstm', 'social_lstm'],
                        help='Model to use')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden state dimension')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--neighborhood_size', type=float, default=2.0,
                        help='Neighborhood size (in meters)')
    parser.add_argument('--grid_size', type=int, default=8,
                        help='Grid size for social pooling')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--clip_grad', type=float, default=10.0,
                        help='Clip gradients value')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Logging and saving parameters
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory for tensorboard')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log stats every N iterations')
    parser.add_argument('--val_every', type=int, default=1,
                        help='Validate every N epochs')

    args = parser.parse_args()

    # Create unique run name based on time
    args.run_name = f"{args.model}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, args.run_name), exist_ok=True)

    return args