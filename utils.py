#!/usr/bin/env python3
"""
Utility functions for training and testing VAE-CycleGAN models.

This module contains helper functions for:
- Checkpoint saving and loading
- Pretrained model loading (DoubleAE/DoubleVAE to Cycle models)
- TensorBoard event file management
"""

import os
import glob
from pathlib import Path
import torch


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


def load_pretrained_doublevae_to_cyclevae(cycle_model, doublevae_checkpoint_path, device):
    """
    Load pretrained DoubleVariationalAutoencoder weights into a CycleVAE or CycleVAEGAN model.

    The DoubleVariationalAutoencoder has:
    - encoder (shared)
    - vae_encoder_block_A, vae_decoder_block_A (VAE blocks for source modality)
    - vae_encoder_block_B, vae_decoder_block_B (VAE blocks for target modality)
    - decoder_A (reconstructs source modality)
    - decoder_B (reconstructs target modality)

    The CycleVAE / CycleVAEGAN has:
    - G = VariationalAutoencoder (encoder + variational_encoder_block + variational_decoder_block + decoder)
    - F = VariationalAutoencoder (encoder + variational_encoder_block + variational_decoder_block + decoder)

    Mapping strategy:
    - G.encoder <- encoder, G.variational_encoder_block <- vae_encoder_block_B,
      G.variational_decoder_block <- vae_decoder_block_B, G.decoder <- decoder_B
    - F.encoder <- encoder, F.variational_encoder_block <- vae_encoder_block_A,
      F.variational_decoder_block <- vae_decoder_block_A, F.decoder <- decoder_A

    Args:
        cycle_model: CycleVAE or CycleVAEGAN model to initialize
        doublevae_checkpoint_path: Path to DoubleVariationalAutoencoder checkpoint
        device: Device to load the checkpoint on
    """
    if not os.path.exists(doublevae_checkpoint_path):
        raise FileNotFoundError(f"No DoubleVariationalAutoencoder checkpoint found at {doublevae_checkpoint_path}")

    # Load checkpoint
    print(f"Loading DoubleVariationalAutoencoder weights from {doublevae_checkpoint_path}")
    checkpoint = torch.load(doublevae_checkpoint_path, map_location=device)
    doublevae_state_dict = checkpoint['model_state_dict']

    # Extract components from DoubleVariationalAutoencoder state dict
    encoder_state = {}
    vae_encoder_block_A_state = {}
    vae_encoder_block_B_state = {}
    vae_decoder_block_A_state = {}
    vae_decoder_block_B_state = {}
    decoder_A_state = {}
    decoder_B_state = {}

    for key, value in doublevae_state_dict.items():
        if key.startswith('encoder.'):
            new_key = key[len('encoder.'):]
            encoder_state[new_key] = value
        elif key.startswith('vae_encoder_block_A.'):
            new_key = key[len('vae_encoder_block_A.'):]
            vae_encoder_block_A_state[new_key] = value
        elif key.startswith('vae_encoder_block_B.'):
            new_key = key[len('vae_encoder_block_B.'):]
            vae_encoder_block_B_state[new_key] = value
        elif key.startswith('vae_decoder_block_A.'):
            new_key = key[len('vae_decoder_block_A.'):]
            vae_decoder_block_A_state[new_key] = value
        elif key.startswith('vae_decoder_block_B.'):
            new_key = key[len('vae_decoder_block_B.'):]
            vae_decoder_block_B_state[new_key] = value
        elif key.startswith('decoder_A.'):
            new_key = key[len('decoder_A.'):]
            decoder_A_state[new_key] = value
        elif key.startswith('decoder_B.'):
            new_key = key[len('decoder_B.'):]
            decoder_B_state[new_key] = value

    # Load into CycleVAE / CycleVAEGAN
    # G translates A->B, so use encoder + vae_block_B + decoder_B
    print("Loading encoder + vae_block_B + decoder_B into G (A->B translation)")
    cycle_model.G.encoder.load_state_dict(encoder_state)
    cycle_model.G.variational_encoder_block.load_state_dict(vae_encoder_block_B_state)
    cycle_model.G.variational_decoder_block.load_state_dict(vae_decoder_block_B_state)
    cycle_model.G.decoder.load_state_dict(decoder_B_state)

    # F translates B->A, so use encoder + vae_block_A + decoder_A
    print("Loading encoder + vae_block_A + decoder_A into F (B->A translation)")
    cycle_model.F.encoder.load_state_dict(encoder_state)
    cycle_model.F.variational_encoder_block.load_state_dict(vae_encoder_block_A_state)
    cycle_model.F.variational_decoder_block.load_state_dict(vae_decoder_block_A_state)
    cycle_model.F.decoder.load_state_dict(decoder_A_state)

    # Sanity check: verify G and F aren't swapped
    print("Running sanity check on weight transfer...")
    for (name_cycle, param_cycle), (name_src, param_src) in zip(
        cycle_model.G.decoder.state_dict().items(),
        decoder_B_state.items()
    ):
        assert torch.equal(param_cycle, param_src), \
            f"G.decoder mismatch at {name_cycle} — G and F may be swapped!"

    for (name_cycle, param_cycle), (name_src, param_src) in zip(
        cycle_model.F.decoder.state_dict().items(),
        decoder_A_state.items()
    ):
        assert torch.equal(param_cycle, param_src), \
            f"F.decoder mismatch at {name_cycle} — G and F may be swapped!"

    for (name_cycle, param_cycle), (name_src, param_src) in zip(
        cycle_model.G.variational_decoder_block.state_dict().items(),
        vae_decoder_block_B_state.items()
    ):
        assert torch.equal(param_cycle, param_src), \
            f"G.variational_decoder_block mismatch at {name_cycle} — G and F may be swapped!"

    for (name_cycle, param_cycle), (name_src, param_src) in zip(
        cycle_model.F.variational_decoder_block.state_dict().items(),
        vae_decoder_block_A_state.items()
    ):
        assert torch.equal(param_cycle, param_src), \
            f"F.variational_decoder_block mismatch at {name_cycle} — G and F may be swapped!"

    print("Sanity check passed: G uses B components, F uses A components")

    print("Successfully loaded DoubleVariationalAutoencoder weights")
    print("  G (A->B): encoder (shared) + vae_block_B + decoder_B (target modality)")
    print("  F (B->A): encoder (shared) + vae_block_A + decoder_A (source modality)")


def truncate_tensorboard_events(tensorboard_dir, max_epoch):
    """
    Truncate TensorBoard events to keep only events up to max_epoch.
    This allows resuming training from a checkpoint without duplicate/divergent curves.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

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
