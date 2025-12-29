#!/usr/bin/env python3
"""
Training script for all compared model architectures

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
from Losses import *
from Data_Manager import HypersimDataset


def create_model(args):
    """Create model based on architecture choice"""
    if args.architecture == 'autoencoder':
        model = Autoencoder()
        print(f"Created Autoencoder")
    elif args.architecture == 'vae':
        model = VariationalAutoencoder()
        print(f"Created Variational Autoencoder")
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    return model


def create_loss_function(args):
    """Create loss function based on architecture choice"""
    if args.architecture == 'ae':
        criterion = AELoss()
        print("Using AE Loss (L1 Reconstruction)")
    elif args.architecture == 'vae':
        criterion = VAELoss()
        print(f"Using VAE Loss (L1 + KL)")
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    return criterion


def save_checkpoint(model, optimizer, epoch, loss, args, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args)
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename, device):
    """Load model checkpoint, retrives epoch and loss from metadata"""
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from {filename} (epoch {epoch}, loss {loss:.4f})")
        return epoch, loss
    else:
        raise FileNotFoundError(f"No checkpoint found at {filename}")


def train_epoch(model, dataloader, criterion, optimizer, device, args, writer=None, epoch=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {}
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # Get input and target modalities
        x = batch[args.source_modality].to(device)
        y = batch[args.target_modality].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        if args.architecture == 'autoencoder':
            output = model(x)
            loss, losses_dict = criterion(output, x, y)
        elif args.architecture == 'vae':
            output, mu, logvar = model(x)
            loss, losses_dict = criterion((output, mu, logvar), x, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        for key, value in losses_dict.items():
            if key not in loss_components:
                loss_components[key] = 0.0
            loss_components[key] += value
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_loss_components, output, x, y


def validate(model, dataloader, criterion, device, args):
    """Run validation on the test set"""
    model.eval()
    total_loss = 0.0
    loss_components = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Get input and target modalities
            x = batch[args.source_modality].to(device)
            y = batch[args.target_modality].to(device)
            
            # Forward pass
            if args.architecture == 'autoencoder':
                output = model(x)
                loss, losses_dict = criterion(output, x, y)
            elif args.architecture == 'vae':
                output, mu, logvar = model(x)
                loss, losses_dict = criterion((output, mu, logvar), x, y)
            
            # Track losses
            total_loss += loss.item()
            for key, value in losses_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_loss_components, output, x, y


def main(args):

    # Checks of good practice
    if args.architecture == 'autoencoder' or args.architecture == 'vae' :
        if args.source_modality != args.target_modality:
            raise ValueError("Source and target modalities should be the same for Autoencoder/VAE architectures.")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.architecture}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create TensorBoard writer
    tensorboard_dir = output_dir / 'tensorboard'
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Run 'tensorboard --logdir={tensorboard_dir}' to visualize")
    
    
    # Create datasets
        ## Define augmentation transform for general modalities (depth, semantic, normal)
    general_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        #transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=args.image_size, scale=(0.33, 1.0), ratio=(1,1), interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    
        ## Define augmentation transform specifically for color images
    color_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    ])
    
        ## Create dataset with transforms
    train_dataset = HypersimDataset(
        root_dir=args.data_dir,
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
    
    # Create model
    model = create_model(args).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create loss function
    criterion = create_loss_function(args)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path, device)
            start_epoch += 1
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_test_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_loss_components, train_output, train_x, train_y = train_epoch(
            model, train_loader, criterion, optimizer, device, args
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
                model, test_loader, criterion, device, args
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
            save_checkpoint(model, optimizer, epoch, train_loss, args, checkpoint_path)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_path = output_dir / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, test_loss, args, best_path)
            print(f"New best model saved (test_loss: {test_loss:.4f})")
    
    # Close TensorBoard writer
    writer.close()
    print(f"\nTraining completed.Models saved to {output_dir}")
    print(f"TensorBoard logs : tensorboard --logdir={tensorboard_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE-CycleGAN models')
    
    # Architecture selection
    parser.add_argument('--architecture', type=str, default='ae',
                        choices=['ae', 'vae'], ## rest of architectures can be added here
                        help='Network architecture to train')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--source_modality', type=str, default='depth',
                        help='Source modality (input)')
    parser.add_argument('--target_modality', type=str, default='normal',
                        help='Target modality (output)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    
    
    # Checkpointing and output
    parser.add_argument('--output_dir', type=str, default='output',
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
    
    main(args)