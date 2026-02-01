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

# Import networks, losses, and data manager
from Networks import *
from Data_Manager import HypersimDataset, UnpairedImageDataset


def create_model(architecture):
    """Create model based on architecture choice"""
    if architecture == 'autoencoder':
        model = Autoencoder()
        print(f"Created Autoencoder")
    elif architecture == 'doubleae':
        model = DoubleAutoencoder()
        print(f"Created Double Autoencoder (shared encoder, two decoders)")
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

        # Ensure optimizer is configured before loading its state
        if not hasattr(model, 'optimizer') or model.optimizer is None:
            # Try to configure optimizer with default args (will be replaced if needed)
            try:
                model.configure_optimizers()
            except Exception:
                pass

        # Load optimizer state(s)
        if 'optimizer_states' in checkpoint:
            model.load_optimizer_states(checkpoint['optimizer_states'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from {filename} (epoch {epoch}, loss {loss:.4f})")
        return epoch, loss
    else:
        raise FileNotFoundError(f"No checkpoint found at {filename}")


def load_pretrained_doubleae_to_cycleae(cycleae_model, doubleae_checkpoint_path, device):
    """
    Load pretrained DoubleAutoencoder weights into a CycleAE model.
    
    The DoubleAutoencoder has:
    - encoder (shared)
    - decoder_A (reconstructs source modality)
    - decoder_B (reconstructs target modality)
    
    The CycleAE has:
    - G.encoder + G.decoder (translates source -> target)
    - F.encoder + F.decoder (translates target -> source)
    
    Mapping strategy:
    - G.encoder <- encoder (shared encoder for A->B translation)
    - G.decoder <- decoder_B (translates to target modality)
    - F.encoder <- encoder (shared encoder for B->A translation)
    - F.decoder <- decoder_A (translates to source modality)
    
    Args:
        cycleae_model: CycleAE model to initialize
        doubleae_checkpoint_path: Path to DoubleAutoencoder checkpoint
        device: Device to load the checkpoint on
    """
    if not os.path.exists(doubleae_checkpoint_path):
        raise FileNotFoundError(f"No DoubleAutoencoder checkpoint found at {doubleae_checkpoint_path}")
    
    # Load checkpoint
    print(f"Loading DoubleAutoencoder weights from {doubleae_checkpoint_path}")
    checkpoint = torch.load(doubleae_checkpoint_path, map_location=device)
    doubleae_state_dict = checkpoint['model_state_dict']
    
    # Extract DoubleAutoencoder components
    encoder_state = {}
    decoder_A_state = {}
    decoder_B_state = {}
    
    for key, value in doubleae_state_dict.items():
        if key.startswith('encoder.'):
            # Remove 'encoder.' prefix
            new_key = key[8:]
            encoder_state[new_key] = value
        elif key.startswith('decoder_A.'):
            # Remove 'decoder_A.' prefix
            new_key = key[10:]
            decoder_A_state[new_key] = value
        elif key.startswith('decoder_B.'):
            # Remove 'decoder_B.' prefix
            new_key = key[10:]
            decoder_B_state[new_key] = value
    
    # Load into CycleAE
    # G translates A->B, so use encoder + decoder_B
    print("Loading encoder + decoder_B into G (A->B translation)")
    cycleae_model.G.encoder.load_state_dict(encoder_state)
    cycleae_model.G.decoder.load_state_dict(decoder_B_state)
    
    # F translates B->A, so use encoder + decoder_A
    print("Loading encoder + decoder_A into F (B->A translation)")
    cycleae_model.F.encoder.load_state_dict(encoder_state)
    cycleae_model.F.decoder.load_state_dict(decoder_A_state)
    
    print("Successfully loaded DoubleAutoencoder weights into CycleAE")
    print("  G (A->B): encoder (shared) + decoder_B (target modality)")
    print("  F (B->A): encoder (shared) + decoder_A (source modality)")


def truncate_tensorboard_events(tensorboard_dir, max_epoch):
    """
    Truncate TensorBoard events to keep only events up to max_epoch.
    This allows resuming training from a checkpoint without duplicate/divergent curves.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tensorboard_dir = Path(tensorboard_dir)
    event_files = sorted(glob.glob(str(tensorboard_dir / "events.out.tfevents.*")))

    if not event_files:
        print("No TensorBoard event files found, nothing to truncate")
        return

    # Load all events using EventAccumulator
    ea = EventAccumulator(str(tensorboard_dir))
    ea.Reload()

    # Collect all scalar data up to max_epoch
    scalars_to_keep = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        scalars_to_keep[tag] = [(e.step, e.value) for e in events if e.step <= max_epoch]

    # Collect all image data up to max_epoch
    images_to_keep = {}
    for tag in ea.Tags().get('images', []):
        events = ea.Images(tag)
        images_to_keep[tag] = [(e.step, e.encoded_image_string, e.width, e.height) for e in events if e.step <= max_epoch]

    # Remove old event files
    for event_file in event_files:
        os.remove(event_file)
        print(f"Removed old TensorBoard event file: {event_file}")

    # Rewrite events using a new SummaryWriter
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # Rewrite scalars
    for tag, events in scalars_to_keep.items():
        for step, value in events:
            writer.add_scalar(tag, value, step)

    # Rewrite images
    for tag, events in images_to_keep.items():
        for step, encoded_image, width, height in events:
            # Decode and add image
            import io
            from PIL import Image
            import numpy as np
            img = Image.open(io.BytesIO(encoded_image))
            img_array = np.array(img)
            # TensorBoard expects (H, W, C) for add_image with dataformats='HWC'
            writer.add_image(tag, img_array, step, dataformats='HWC')

    writer.close()

    kept_scalars = sum(len(v) for v in scalars_to_keep.values())
    kept_images = sum(len(v) for v in images_to_keep.values())
    print(f"Truncated TensorBoard logs to epoch {max_epoch}: kept {kept_scalars} scalar events and {kept_images} image events")


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


def create_dataloaders_paired(args):
    """
    Create train and test dataloaders for paired dataset with appropriate transforms
    
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
    
def load_component_checkpoint(component, filename, device):
    """
    Load weights from an Autoencoder checkpoint into a specific component (G or F)
    """
    if os.path.exists(filename):
        print(f"Loading component weights from {filename}...")
        checkpoint = torch.load(filename, map_location=device)
        
        # Le checkpoint contient 'model_state_dict', on le charge dans le composant
        # Le composant (ex: model.G) est une instance d'Autoencoder, donc les clés correspondent
        component.load_state_dict(checkpoint['model_state_dict'])
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"No checkpoint found at {filename}")

def main(args):

    # Checks of good practice
    if args.architecture in ['autoencoder', 'vae']:
        if args.source_modality != args.target_modality:
            raise ValueError("Source and target modalities should be the same for Autoencoder/VAE architectures.")
    
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
        type_data = "unpaired" if args.unpaired else "paired"
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

    # Create dataloaders (paired or unpaired)
    if args.unpaired:
        train_loader, test_loader = create_dataloaders_unpaired(args)
        print("Using unpaired dataset (summer2winter)")
    else:
        train_loader, test_loader = create_dataloaders_paired(args)
        print("Using paired dataset (hypersim)")
    
    # Create model
    model = create_model(args.architecture).to(device)

    # Load pretrained DoubleAutoencoder weights into CycleAE if specified
    if args.pretrained_doubleae is not None:
        if args.architecture not in ['cycleae', 'cyclevae', 'cycleaegan', 'cyclevaegan']:
            raise ValueError(f"--pretrained_doubleae can only be used with Cycle architectures, not {args.architecture}")
        print(f"\nInitializing {args.architecture} from pretrained DoubleAutoencoder...")
        load_pretrained_doubleae_to_cycleae(model, args.pretrained_doubleae, device)
        print("Pretraining loaded successfully\n")

    # Configure model optimizers and losses
    model.configure_optimizers(lr=args.lr) # Each model will create its own optimizers
    model.configure_loss(
        lambda_kl=args.lambda_kl,
        lambda_gan=args.lambda_gan,
        lambda_identity=args.lambda_identity,
        lambda_cycle=args.lambda_cycle)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_path = Path(args.resume)
        start_epoch, _ = load_checkpoint(model, checkpoint_path, device)
        start_epoch += 1

    
    # We give all the lambdas even if not used by the architecture, the model will pick what it needs
    print(f"Model configured with optimizers and loss functions")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
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
            test_x_vis = test_x[:4] * 0.5 + 0.5
            test_y_vis = test_y[:4] * 0.5 + 0.5
            test_Gx_vis = test_Gx[:4] * 0.5 + 0.5
            
            writer.add_images(f'{args.source_modality}/test_x', test_x_vis, epoch)
            writer.add_images(f'{args.target_modality}/test_y', test_y_vis, epoch)
            writer.add_images(f'{args.target_modality}/test_Gx', test_Gx_vis, epoch)
            
            # Log Fy if available (for Cycle/Double architectures)
            if test_Fy is not None:
                test_Fy_vis = test_Fy[:4] * 0.5 + 0.5
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
                        choices=['autoencoder', 'doubleae', 'vae', 'aegan',
                                 'vaegan', 'cycleae', 'cyclevae',
                                 'cycleaegan', 'cyclevaegan'],
                        help='Network architecture to train')
    
    # Transfer Learning & Pretraining parameters
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights and train decoder only (Autoencoder only)')
    parser.add_argument('--pretrained_G', type=str, default=None,
                        help='Path to a pretrained Autoencoder checkpoint to initialize G (CycleAE only)')
    parser.add_argument('--pretrained_F', type=str, default=None,
                        help='Path to a pretrained Autoencoder checkpoint to initialize F (CycleAE only)')
    parser.add_argument('--pretrained_doubleae', type=str, default=None,
                        help='Path to a pretrained DoubleAutoencoder checkpoint to initialize CycleAE (both G and F)')
    
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
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                        help='Cycle consistency loss weight for Cycle architectures')
    
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