#!/usr/bin/env python3
"""
Training script for all compared model architectures

This script sequentially :
1. Set up data loaders with appropriate augmentations.
2. Create the specified model architecture.
3. Configure optimizers and loss functions(Strongly relies on the model specific training_step and validation_step methods).
4. Optionally load from a checkpoint to resume training.
5. Run the training loop  (Strongly relies on the model specific training_step and validation_step methods).
6. Save checkpoints and log metrics/images to TensorBoard.


It is designed to adapt to all the architectures implemented in Networks.py. Even though they have very different training procedures.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import networks, losses, and data manager
from Networks import *
from Data_Manager import HypersimDataset, UnpairedImageDataset


def create_model(architecture):
    """Create model based on architecture choice"""
    if architecture == 'autoencoder':
        model = Autoencoder()
        print(f"Created Autoencoder")
    elif architecture == 'vae':
        model = VariationalAutoencoder()
        print(f"Created Variational Autoencoder")
    elif architecture == 'aegan':
        model = AEGAN()
        print(f"Created Autoencoder GAN")
    elif architecture == 'vaegan':
        model = VAEGAN()
        print(f"Created VAE-GAN")
    elif architecture == 'cycleae':
        model = CycleAE()
        print(f"Created Cycle Autoencoder")
    elif architecture == 'cyclevae':
        model = CycleVAE()
        print(f"Created Cycle Variational Autoencoder")
    elif architecture == 'cycleaegan':
        model = CycleAEGAN()
        print(f"Created Cycle Autoencoder GAN")
    elif architecture == 'cyclevaegan':
        model = CycleVAEGAN()
        print(f"Created Cycle VAE-GAN")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


def save_checkpoint(model, epoch, loss, args, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_states': model.save_optimizer_states(),
        'loss': loss,
        'args': vars(args)
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, filename, device):
    """Load model checkpoint, retrieves epoch and loss from metadata"""
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state(s)
        if 'optimizer_states' in checkpoint:
            model.load_optimizer_states(checkpoint['optimizer_states'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from {filename} (epoch {epoch}, loss {loss:.4f})")
        return epoch, loss
    else:
        raise FileNotFoundError(f"No checkpoint found at {filename}")


def train_epoch(model, dataloader, device, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {}
    last_output = None
    last_x = None
    last_y = None
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # Prepare batch (move to device)
        batch['x'] = batch['x'].to(device)
        batch['y'] = batch['y'].to(device)
        
        # Model handles : forward, loss, backward, optimizer step
        metrics = model.training_step(batch)
        
        # Track losses
        total_loss += metrics['G_loss']
        for key, value in metrics.items():
            if key not in loss_components:
                loss_components[key] = 0.0
            loss_components[key] += value
        
        # Update progress bar
        pbar.set_postfix({'loss': metrics['G_loss']})
        
        # Keep last batch for visualization
        last_x = batch['x']
        last_y = batch['y']
        with torch.no_grad():
            last_output = model(last_x, last_y)[0] # Get output only (first element)
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_loss_components, last_output, last_x, last_y


def validate(model, dataloader, device, args):
    """Run validation on the test set"""
    model.eval()
    total_loss = 0.0
    loss_components = {}
    last_output = None
    last_x = None
    last_y = None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Prepare batch (move to device)
            batch['x'] = batch['x'].to(device)
            batch['y'] = batch['y'].to(device)
            
            # Model handles validation
            metrics = model.validation_step(batch)
            
            # Extract output for visualization
            output = metrics.pop('output')
            
            # Track losses
            total_loss += metrics['G_loss']
            for key, value in metrics.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
            
            # Keep last batch for visualization
            last_output = output
            last_x = batch['x']
            last_y = batch['y']
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_loss_components, last_output, last_x, last_y


def create_dataloaders_paired(args):
    """
    Create train and test dataloaders for paired dataset with appropriate transforms
    
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing (or None if test_split=0)
    """
    # Define augmentation transform for general modalities (depth, semantic, normal)
    general_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        #transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=args.image_size, scale=(0.33, 1.0), ratio=(1,1), interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    
    # Define augmentation transform specifically for color images
    color_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    ])
    
    # Create dataset with transforms
    train_dataset = HypersimDataset(
        root_dir=args.data_dir + '/paired',
        modalities=[args.source_modality, args.target_modality],
        transform=general_transform,
        color_transform=color_transform, 
        return_scene_info=True
    )
    
    # Split dataset into train/test if needed
    if args.test_split > 0:
        train_size = int((1 - args.test_split) * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size]
        )
        print(f"Training samples: {train_size}, Testing samples: {test_size}")
    else:
        test_dataset = None
        print(f"Training samples: {len(train_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    return train_loader, test_loader


def create_dataloaders_unpaired(args):
    """
    Create train and test dataloaders for unpaired dataset using summer2winter dataset.
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing (or None if test_split=0)
    """
    unpaired_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    train_dataset = UnpairedImageDataset(
        root_dir=args.data_dir + "/unpaired",
        split="train",
        transform=unpaired_transform
    )

    test_dataset = UnpairedImageDataset(
        root_dir=args.data_dir + "/unpaired",
        split="test",
        transform=unpaired_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    return train_loader, test_loader
    


def main(args):

    # Checks of good practice
    if args.architecture in ['autoencoder', 'vae']:
        if args.source_modality != args.target_modality:
            raise ValueError("Source and target modalities should be the same for Autoencoder/VAE architectures.")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Determine output directory: if resuming, continue in the checkpoint directory
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        output_dir = checkpoint_path.parent
        print(f"Resuming run in directory: {output_dir}")
    else:
        # Create new output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / f"{args.architecture}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save arguments only for new runs
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"Output directory: {output_dir}")
    
    # Create TensorBoard writer
    tensorboard_dir = output_dir / 'tensorboard'
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Run 'tensorboard --logdir={tensorboard_dir}' to visualize")
    
    # Create dataloaders (paired or unpaired)
    if args.unpaired:
        train_loader, test_loader = create_dataloaders_unpaired(args)
        print("Using unpaired dataset (summer2winter)")
    else:
        train_loader, test_loader = create_dataloaders_paired(args)
        print("Using paired dataset (hypersim)")
    
    # Create model
    model = create_model(args.architecture).to(device)
    
    # Configure model optimizers and losses
    model.configure_optimizers(lr=args.lr) # Each model will create its own optimizers
    model.configure_loss(
        lambda_kl=args.lambda_kl,
        lambda_gan=args.lambda_gan,
        lambda_identity=args.lambda_identity
    ) # We give all the lambdas even if not used by the architecture, the model will pick what it needs
    
    print(f"Model configured with optimizers and loss functions")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_path = Path(args.resume)
        start_epoch, _ = load_checkpoint(model, checkpoint_path, device)
        start_epoch += 1
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_test_loss = float('inf')
    

    #### ACTUAL TRAINING LOOP ####

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_loss_components, train_output, train_x, train_y = train_epoch(
            model, train_loader, device, args
        )
        print(f"Train Loss: {train_loss:.4f}")
        for key, value in train_loss_components.items():
            print(f"  {key}: {value:.4f}")
        
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        for key, value in train_loss_components.items():
            writer.add_scalar(f'Loss_Components/train_{key}', value, epoch)
        
        
        # On test set
        if test_loader is not None:
            test_loss, test_loss_components, test_output, test_x, test_y = validate(
                model, test_loader, device, args
            )
            print(f"Test Loss: {test_loss:.4f}")
            for key, value in test_loss_components.items():
                print(f"  {key}: {value:.4f}")
            
            # Log test metrics to TensorBoard
            writer.add_scalar('Loss/test', test_loss, epoch)
            for key, value in test_loss_components.items():
                writer.add_scalar(f'Loss_Components/test_{key}', value, epoch)
            
            # Log test images to TensorBoard
            if epoch % args.log_image_freq == 0:
                test_x_vis = test_x[:4] * 0.5 + 0.5
                test_y_vis = test_y[:4] * 0.5 + 0.5
                test_output_vis = test_output[:4] * 0.5 + 0.5
                
                writer.add_images(f'{args.source_modality}/test_input', test_x_vis, epoch)
                writer.add_images(f'{args.target_modality}/test_target', test_y_vis, epoch)
                writer.add_images(f'{args.target_modality}/test_output', test_output_vis, epoch)
        else:
            test_loss = train_loss
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(model, epoch, train_loss, args, checkpoint_path)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_path = output_dir / "best_model.pth"
            save_checkpoint(model, epoch, test_loss, args, best_path)
            print(f"New best model saved (test_loss: {test_loss:.4f})")
    
    # Close TensorBoard writer
    writer.close()
    print(f"\nTraining completed.Models saved to {output_dir}")
    print(f"TensorBoard logs : tensorboard --logdir={tensorboard_dir}")



if __name__ == '__main__':

    ##### ARGUMENTS DEFINITION #####
    parser = argparse.ArgumentParser(description='Train VAE-CycleGAN models')
    
    # Architecture selection
    parser.add_argument('--architecture', type=str, default='autoencoder',
                        choices=['autoencoder', 'vae', 'aegan',
                                 'vaegan', 'cycleae', 'cyclevae',
                                 'cycleaegan', 'cyclevaegan'],
                        help='Network architecture to train')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='Path to dataset directory')
    parser.add_argument('--source_modality', type=str, default='depth',
                        help='Source modality (input)')
    parser.add_argument('--target_modality', type=str, default='normal',
                        help='Target modality (output)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--unpaired', action='store_true',
                        help='Use unpaired dataset (summer2winter) instead of paired hypersim dataset')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    
    # Loss hyperparameters
    parser.add_argument('--lambda_kl', type=float, default=1e-5,
                        help='KL divergence weight for VAE')
    parser.add_argument('--lambda_gan', type=float, default=1.0,
                        help='GAN loss weight for AEGAN')
    parser.add_argument('--lambda_identity', type=float, default=5.0,
                        help='Identity loss weight for AEGAN')
    
    
    # Checkpointing and output
    parser.add_argument('--output_dir', type=str, default='runs',
                        help='Directory to save models and logs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_image_freq', type=int, default=5,
                        help='Log images to TensorBoard every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loading workers')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    

    #### START TRAINING ####
    main(args)