"""Training script for trajectory prediction models."""

import os
import time
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

from config import get_config
from models import SocialLSTM, VanillaLSTM
from utils import (
    data_loader, l2_loss, displacement_error, final_displacement_error,
    compute_metrics_per_sequence, compute_overall_metrics, rel_to_abs,
    plot_training_curves, visualize_trajectory
)


def init_weights(m):
    """Initialize model weights."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('LSTM') != -1:
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    """
    Train the model.
    
    Args:
        args: Configuration parameters
    """
    # Set up logging
    log_path = os.path.join(args.log_dir, args.run_name)
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, 'train.log')
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
    
    # Initialize data loaders
    logger.info("Initializing data loaders...")
    train_loader = data_loader(args, 'train')
    val_loader = data_loader(args, 'val')
    
    # Initialize model
    logger.info(f"Initializing {args.model} model...")
    if args.model == 'social_lstm':
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
    elif args.model == 'vanilla_lstm':
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
        raise ValueError(f"Unknown model: {args.model}")
    
    model.to(device)
    model.apply(init_weights)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_path)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    metrics_history = {'avg_ade': [], 'avg_fde': [], 'weighted_ade': [], 'weighted_fde': []}
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            # Process each sequence in the batch separately
            batch_loss = 0.0

            # Check if we have a list of tensors or a single tensor
            if isinstance(batch['obs_traj_rel'], list):
                # Process as a list of tensors
                for i in range(len(batch['obs_traj_rel'])):
                    # Get single sequence data
                    seq_data = {
                        'obs_traj_rel': batch['obs_traj_rel'][i].to(device),
                        'obs_traj': batch['obs_traj'][i].to(device),
                        'pred_traj_rel': batch['pred_traj_rel'][i].to(device),
                        'seq_start_end': batch['seq_start_end'][i].to(device),
                    }

                    # Forward pass for this sequence
                    pred_traj_rel = model(
                        obs_traj_rel=seq_data['obs_traj_rel'],
                        seq_start_end=seq_data['seq_start_end'],
                        obs_traj=seq_data['obs_traj']
                    )

                    # Loss calculation for this sequence
                    seq_loss = l2_loss(
                        pred_traj_rel,
                        seq_data['pred_traj_rel'],
                        mode='sum'
                    )

                    # Update total loss
                    batch_loss += seq_loss

                    # Backward pass for this sequence
                    seq_loss.backward()
            else:
                # Single tensor case (batch size 1)
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                pred_traj_rel = model(
                    obs_traj_rel=batch['obs_traj_rel'],
                    seq_start_end=batch['seq_start_end'],
                    obs_traj=batch['obs_traj']
                )

                # Loss calculation
                batch_loss = l2_loss(
                    pred_traj_rel,
                    batch['pred_traj_rel'],
                    mode='sum'
                )
                # batch_loss is already a tensor in this case

                # Backward pass
                batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            # Update statistics
            train_loss += batch_loss.item()
            num_batches += 1

            # Logging
            if batch_idx % args.log_every == 0:
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {batch_loss.item():.4f}"
                )
                writer.add_scalar('train/batch_loss', batch_loss.item(), epoch * len(train_loader) + batch_idx)

        # Compute average training loss
        train_loss /= num_batches
        train_losses.append(train_loss)
        writer.add_scalar('train/epoch_loss', train_loss, epoch)

        # Validation
        if (epoch + 1) % args.val_every == 0:
            model.eval()
            val_loss = 0
            val_ade = 0
            val_fde = 0
            num_val_batches = 0

            all_metrics = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    # Process each sequence in the batch separately
                    batch_loss = 0.0
                    batch_metrics = []

                    # Check if we have a list of tensors or a single tensor
                    if isinstance(batch['obs_traj_rel'], list):
                        # Process as a list of tensors
                        for i in range(len(batch['obs_traj_rel'])):
                            # Get single sequence data
                            seq_data = {
                                'obs_traj_rel': batch['obs_traj_rel'][i].to(device),
                                'obs_traj': batch['obs_traj'][i].to(device),
                                'pred_traj_rel': batch['pred_traj_rel'][i].to(device),
                                'pred_traj': batch['pred_traj'][i].to(device),
                                'seq_start_end': batch['seq_start_end'][i].to(device),
                            }

                            # Forward pass for this sequence
                            pred_traj_rel = model(
                                obs_traj_rel=seq_data['obs_traj_rel'],
                                seq_start_end=seq_data['seq_start_end'],
                                obs_traj=seq_data['obs_traj']
                            )

                            # Convert to absolute coordinates for visualization and metrics
                            pred_traj = rel_to_abs(pred_traj_rel, seq_data['obs_traj'][-1])

                            # Loss calculation for this sequence
                            seq_loss = l2_loss(
                                pred_traj_rel,
                                seq_data['pred_traj_rel'],
                                mode='sum'
                            )

                            # Accumulate loss
                            batch_loss += seq_loss.item()

                            # Compute metrics for this sequence
                            seq_metrics = compute_metrics_per_sequence(
                                pred_traj, seq_data['pred_traj'], seq_data['seq_start_end']
                            )
                            batch_metrics.extend(seq_metrics)
                    else:
                        # Single tensor case (batch size 1)
                        # Move batch to device
                        batch = {k: v.to(device) for k, v in batch.items()}

                        # Forward pass
                        pred_traj_rel = model(
                            obs_traj_rel=batch['obs_traj_rel'],
                            seq_start_end=batch['seq_start_end'],
                            obs_traj=batch['obs_traj']
                        )

                        # Convert to absolute coordinates for visualization and metrics
                        pred_traj = rel_to_abs(pred_traj_rel, batch['obs_traj'][-1])

                        # Loss calculation
                        batch_loss = l2_loss(
                            pred_traj_rel,
                            batch['pred_traj_rel'],
                            mode='sum'
                        ).item()

                        # Compute metrics
                        batch_metrics = compute_metrics_per_sequence(
                            pred_traj, batch['pred_traj'], batch['seq_start_end']
                        )

                    # Update statistics
                    val_loss += batch_loss
                    all_metrics.extend(batch_metrics)
                    num_val_batches += 1

            # Compute average validation loss
            val_loss /= num_val_batches
            val_losses.append(val_loss)

            # Compute overall metrics
            overall_metrics = compute_overall_metrics(all_metrics)

            # Update metrics history
            for key in metrics_history:
                if key in overall_metrics:
                    metrics_history[key].append(overall_metrics[key])

            # Log validation results
            logger.info(
                f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, "
                f"ADE: {overall_metrics['avg_ade']:.4f}, "
                f"FDE: {overall_metrics['avg_fde']:.4f}"
            )

            # Add to tensorboard
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/ade', overall_metrics['avg_ade'], epoch)
            writer.add_scalar('val/fde', overall_metrics['avg_fde'], epoch)

            # Checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.checkpoint_dir, args.run_name, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'metrics': overall_metrics
                }, checkpoint_path)
                logger.info(f"Saved best model checkpoint to {checkpoint_path}")

            # Adjust learning rate
            scheduler.step(val_loss)

        # Regular checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, args.run_name, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Log epoch time
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

    # Plot training curves
    curves_path = os.path.join(log_path, 'training_curves')
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        metrics=metrics_history,
        save_path=curves_path
    )

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, args.run_name, 'final_model.pt')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1] if val_losses else None,
        'metrics': {k: v[-1] for k, v in metrics_history.items() if v}
    }, final_path)
    logger.info(f"Saved final model to {final_path}")

    logger.info("Training completed!")

    return model


if __name__ == "__main__":
    import torch.nn as nn

    args = get_config()
    train(args)