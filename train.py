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
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import networks, losses, data manager, and utilities
from Networks import *
from Data_Manager import HypersimDataset, SatelliteMapDataset
from utils import (
    save_checkpoint,
    load_checkpoint,
    load_pretrained_doubleae_to_cycleae,
    load_pretrained_doublevae_to_cyclevae,
    truncate_tensorboard_events
)


def create_model(architecture, paired=True):
    """Create model based on architecture choice"""
    if architecture == 'autoencoder':
        model = Autoencoder()
        print(f"Created Autoencoder")
    elif architecture == 'doubleae':
        model = DoubleAutoencoder()
        print(f"Created Double Autoencoder (shared encoder, two decoders)")
    elif architecture == 'doublevae':
        model = DoubleVariationalAutoencoder()
        print(f"Created Double Variational Autoencoder (shared encoder, separate VAE blocks, two decoders)")
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
        model = CycleAE(paired=paired)
        print(f"Created Cycle Autoencoder ({'paired' if paired else 'unpaired'} mode)")
    elif architecture == 'cyclevae':
        model = CycleVAE(paired=paired)
        print(f"Created Cycle Variational Autoencoder ({'paired' if paired else 'unpaired'} mode)")
    elif architecture == 'cycleaegan':
        model = CycleAEGAN(paired=paired)
        print(f"Created Cycle Autoencoder GAN ({'paired' if paired else 'unpaired'} mode)")
    elif architecture == 'cyclevaegan':
        model = CycleVAEGAN(paired=paired)
        print(f"Created Cycle VAE-GAN ({'paired' if paired else 'unpaired'} mode)")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


def train_epoch(model, dataloader, device, args, writer=None, epoch=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {}
    last_output = None
    last_x = None
    last_y = None

    pbar = tqdm(dataloader, desc='Training')
    nan_count = 0
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
            # Autoencoder and VAE only need x, others need both x and y
            if isinstance(model, (Autoencoder, VariationalAutoencoder)):
                last_output = model(last_x)[0]
            else:
                last_output = model(last_x, last_y)[0]

    # Average losses
    valid_batches = len(dataloader)
    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
        avg_loss_components = {k: v / valid_batches for k, v in loss_components.items()}
    else:
        avg_loss = float('nan')
        avg_loss_components = {k: float('nan') for k in loss_components.keys()}

    return avg_loss, avg_loss_components, last_output, last_x, last_y


def validate(model, dataloader, device, args):
    """Run validation on the test set"""
    model.eval()
    total_loss = 0.0
    loss_components = {}
    last_Gx = None
    last_Fy = None
    last_x = None
    last_y = None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Prepare batch (move to device)
            batch['x'] = batch['x'].to(device)
            batch['y'] = batch['y'].to(device)
            
            # Model handles validation
            metrics = model.validation_step(batch)
            
            # Extract outputs for visualization (Gx and optionally Fy)
            Gx = metrics.pop('Gx')
            Fy = metrics.pop('Fy', None)  # Optional, only for Cycle/Double models
            
            # Track losses
            total_loss += metrics['G_loss']
            for key, value in metrics.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
            
            # Keep last batch for visualization
            last_Gx = Gx
            last_Fy = Fy
            last_x = batch['x']
            last_y = batch['y']
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_loss_components, last_Gx, last_Fy, last_x, last_y


def create_dataloaders_hypersim(args):
    """
    Create train and test dataloaders for Hypersim dataset with appropriate transforms
    
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing (or None if test_split=0)
    """
    # Define augmentation transform for general modalities (depth, semantic, normal)
    # ToTensor() converts [0,255] to [0,1], Normalize converts [0,1] to [-1,1] to match Tanh output
    general_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        #transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=args.image_size, scale=(0.33, 1.0), ratio=(1,1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),  # [0,255] -> [0,1]
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [0,1] -> [-1,1]
    ])
    
    # Define augmentation transform specifically for color images
    color_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    ])
    
    # Create dataset with transforms
    train_dataset = HypersimDataset(
        root_dir=args.data_dir,
        modalities=[args.source_modality, args.target_modality],
        transform=general_transform,
        color_transform=color_transform, 
        return_scene_info=True,
        paired_mode=args.paired  # Use paired or unpaired mode based on args
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


def create_dataloaders_maps(args):
    """
    Create train and test dataloaders for satellite-to-map dataset.
    Images are in [0,1] range (ToTensor only, no Normalize).

    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
    """
    maps_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(
            size=args.image_size,
            scale=(0.33, 1.0),
            ratio=(1, 1),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),  # [0,255] -> [0,1]
    ])

    maps_test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = SatelliteMapDataset(
        root_dir=args.data_dir + "/maps",
        split="train",
        transform=maps_transform
    )

    test_dataset = SatelliteMapDataset(
        root_dir=args.data_dir + "/maps",
        split="val",
        transform=maps_test_transform
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

    # Set default modality names based on dataset
    dataset_modality_defaults = {
        'hypersim': ('depth', 'normal'),
        'maps':     ('satellite', 'map'),
    }
    # Support legacy 'paired'/'unpaired' dataset names (both use Hypersim)
    if args.dataset in ['paired', 'unpaired']:
        dataset_modality_defaults[args.dataset] = ('depth', 'normal')
    default_source, default_target = dataset_modality_defaults[args.dataset]
    if args.source_modality is None:
        args.source_modality = default_source
    if args.target_modality is None:
        args.target_modality = default_target

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: You are using CPU for training. This will be very slow. Consider using a GPU if available.")
        input("Press Enter to continue or Ctrl+C to abort...")

    # Enable anomaly detection to find NaN source (disable for production - slows training)
    if False:  # Set to True to enable
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection enabled - will show exact operation if NaN occurs")
    
    # Determine output directory: if resuming, continue in the checkpoint directory
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        output_dir = checkpoint_path.parent
        print(f"Resuming run in directory: {output_dir}")
    else:
        # Create new output directory with timestamp
        type_data = args.dataset
        timestamp = datetime.now().strftime('%m%d_%H%M')
        output_dir = Path(args.output_dir) / f"{args.architecture}_{timestamp}_{args.source_modality}_to_{args.target_modality}_{type_data}"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save arguments only for new runs
        with open(output_dir / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"Output directory: {output_dir}")
    
    # If resuming, truncate TensorBoard events to the checkpoint epoch
    tensorboard_dir = output_dir / 'tensorboard'
    resume_epoch = None
    if args.resume:
        # Peek at checkpoint to get epoch before truncating TensorBoard
        checkpoint = torch.load(args.resume, map_location='cpu')
        resume_epoch = checkpoint['epoch']
        truncate_tensorboard_events(tensorboard_dir, resume_epoch)

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Run 'tensorboard --logdir={tensorboard_dir}' to visualize")

    # Create dataloaders
    if args.dataset == 'maps':
        train_loader, test_loader = create_dataloaders_maps(args)
        print("Using maps dataset (satellite-to-map)")
    else:  # 'hypersim', 'paired', or 'unpaired' all use Hypersim dataset
        train_loader, test_loader = create_dataloaders_hypersim(args)
        print(f"Using Hypersim dataset in {'paired' if args.paired else 'unpaired'} mode")
    
    # Create model
    model = create_model(args.architecture, paired=args.paired).to(device)

    # Validate pretrained args mutual exclusivity
    if args.pretrained_doubleae is not None and args.pretrained_doublevae is not None:
        raise ValueError("Cannot specify both --pretrained_doubleae and --pretrained_doublevae")

    # Load pretrained DoubleAutoencoder weights into CycleAE if specified
    if args.pretrained_doubleae is not None:
        if args.architecture not in ['cycleae', 'cyclevae', 'cycleaegan', 'cyclevaegan']:
            raise ValueError(f"--pretrained_doubleae can only be used with Cycle architectures, not {args.architecture}")
        print(f"\nInitializing {args.architecture} from pretrained DoubleAutoencoder...")
        load_pretrained_doubleae_to_cycleae(model, args.pretrained_doubleae, device)
        print("Pretraining loaded successfully\n")

    # Load pretrained DoubleVariationalAutoencoder weights into CycleVAE/CycleVAEGAN if specified
    if args.pretrained_doublevae is not None:
        if args.architecture not in ['cyclevae', 'cyclevaegan']:
            raise ValueError(f"--pretrained_doublevae can only be used with CycleVAE or CycleVAEGAN architectures, not {args.architecture}")
        print(f"\nInitializing {args.architecture} from pretrained DoubleVariationalAutoencoder...")
        load_pretrained_doublevae_to_cyclevae(model, args.pretrained_doublevae, device)
        print("Pretraining loaded successfully\n")

    # Configure model optimizers and losses
    model.configure_optimizers(lr=args.lr) # Each model will create its own optimizers
    model.configure_loss(
        lambda_kl=args.lambda_kl,
        lambda_gan=args.lambda_gan,
        lambda_identity=args.lambda_identity,
        lambda_cycle=args.lambda_cycle,
        lambda_recon=args.lambda_recon)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_path = Path(args.resume)
        start_epoch, _ = load_checkpoint(model, checkpoint_path, device)
        start_epoch += 1

    
    # We give all the lambdas even if not used by the architecture, the model will pick what it needs
    print(f"Model configured with optimizers and loss functions")
    
    # Initial validation (epoch -1) before training starts
    print(f"\n{'='*80}")
    print("INITIAL VALIDATION (Before Training)")
    print(f"{'='*80}")
    if test_loader is not None:
        initial_test_loss, initial_test_loss_components, initial_test_Gx, initial_test_Fy, initial_test_x, initial_test_y = validate(
            model, test_loader, device, args
        )
        print(f"Initial Test Loss: {initial_test_loss:.4f}")
        for key, value in initial_test_loss_components.items():
            try:
                print(f"  {key}: {value:.6f}")
            except Exception as e:
                print(f"  {key}: {value} (could not format as float: {e})")
        
        # Skip TensorBoard logging for initial validation (out of scale values)
        # The validation is still run and printed to console for reference

        print(f"{'='*80}\n")
    
    # Free up VRAM after initial validation
    del initial_test_x, initial_test_y, initial_test_Gx
    if initial_test_Fy is not None:
        del initial_test_Fy
    torch.cuda.empty_cache()
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_test_loss = float('inf')
    

    #### ACTUAL TRAINING LOOP ####

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_loss_components, train_output, train_x, train_y = train_epoch(
            model, train_loader, device, args, writer=writer, epoch=epoch
        )
        print(f"Train Loss: {train_loss:.4f}")
        for key, value in train_loss_components.items():
            print(f"  {key}: {value:.6f}")
        
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        for key, value in train_loss_components.items():
            writer.add_scalar(f'Loss_Components_train/{key}', value, epoch)
        
        
        # On test set
        if test_loader is not None and epoch % args.log_image_freq == 0 :
            test_loss, test_loss_components, test_Gx, test_Fy, test_x, test_y = validate(
                model, test_loader, device, args
            )
            print(f"Test Loss: {test_loss:.4f}")
            for key, value in test_loss_components.items():
                try:
                    print(f"  {key}: {value:.6f}")
                except Exception as e:
                    print(f"  {key}: {value} (could not format as float: {e})")
            
            # Log test metrics to TensorBoard
            writer.add_scalar('Loss/test', test_loss, epoch)
            for key, value in test_loss_components.items():
                writer.add_scalar(f'Loss_Components_test/{key}', value, epoch)
            
            # Log test images to TensorBoard
            # All datasets are in [0,1] range
            test_x_vis = test_x[:4].clamp(0, 1)
            test_y_vis = test_y[:4].clamp(0, 1)
            test_Gx_vis = test_Gx[:4].clamp(0, 1)

            writer.add_images(f'{args.source_modality}/test_x', test_x_vis, epoch)
            writer.add_images(f'{args.target_modality}/test_y', test_y_vis, epoch)
            writer.add_images(f'{args.target_modality}/test_Gx', test_Gx_vis, epoch)

            # Log Fy if available (for Cycle/Double architectures)
            if test_Fy is not None:
                test_Fy_vis = test_Fy[:4].clamp(0, 1)
                writer.add_images(f'{args.source_modality}/test_Fy', test_Fy_vis, epoch)
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_path = output_dir / "best_model.pth"
                save_checkpoint(model, epoch, test_loss, args, best_path)
                print(f"New best model saved (test_loss: {test_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(model, epoch, train_loss, args, checkpoint_path)    
    
    # Close TensorBoard writer
    writer.close()
    print(f"\nTraining completed.Models saved to {output_dir}")
    print(f"TensorBoard logs : tensorboard --logdir={tensorboard_dir}")



if __name__ == '__main__':
    

    ##### ARGUMENTS DEFINITION #####
    parser = argparse.ArgumentParser(description='Train VAE-CycleGAN models')
    
    # Architecture selection
    parser.add_argument('--architecture', type=str, default='autoencoder',
                        choices=['autoencoder', 'doubleae', 'doublevae', 'vae', 'aegan',
                                 'vaegan', 'cycleae', 'cyclevae',
                                 'cycleaegan', 'cyclevaegan'],
                        help='Network architecture to train')
    parser.add_argument('--paired', action='store_true', default=False,
                        help='Use paired training mode (with translation/identity loss). Default is unpaired (cycle loss only).')
    parser.add_argument('--unpaired', dest='paired', action='store_false',
                        help='Use unpaired training mode (cycle loss only). This is the default.')
    
    # Transfer Learning & Pretraining parameters
    parser.add_argument('--pretrained_doubleae', type=str, default=None,
                        help='Path to a pretrained DoubleAutoencoder checkpoint to initialize CycleAE (both G and F)')
    parser.add_argument('--pretrained_doublevae', type=str, default=None,
                        help='Path to a pretrained DoubleVariationalAutoencoder checkpoint to initialize CycleVAE/CycleVAEGAN (both G and F)')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='Path to dataset directory')
    parser.add_argument('--source_modality', type=str, default=None,
                        help='Source modality (input). Defaults: hypersim=depth, maps=satellite')
    parser.add_argument('--target_modality', type=str, default=None,
                        help='Target modality (output). Defaults: hypersim=normal, maps=map')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--dataset', type=str, default='hypersim',
                        choices=['hypersim', 'paired', 'unpaired', 'maps'],
                        help='Dataset to use: hypersim (depth/normal/etc), maps (satellite-to-map). Note: paired/unpaired are legacy names for hypersim')
    
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
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                        help='Cycle consistency loss weight for Cycle architectures')
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                        help='Reconstruction/translation loss weight (set to 0 to disable)')
    
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

    main(args)