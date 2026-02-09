#!/usr/bin/env python3
"""
Test script for evaluating and comparing trained model architectures.

This script:
1. Auto-discovers all trained models in the runs/ directory
2. Loads best models with their saved configurations
3. Creates appropriate test dataloaders (paired or unpaired)
4. Runs inference on test samples
5. Generates comparison figures with proper labeling
6. Saves outputs to test_results/ directory
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import networks and data manager
from Networks import *
from Data_Manager import HypersimDataset, UnpairedImageDataset, SatelliteMapDataset


def discover_runs(runs_dir='runs'):
    """
    Scan the runs directory and discover all trained models.

    Args:
        runs_dir: Path to the runs directory

    Returns:
        List of dicts with keys: 'run_dir', 'run_name', 'architecture', 'args', 'best_model_path'
    """
    runs = []
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        print(f"Warning: runs directory '{runs_dir}' not found")
        return runs

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir():
            continue

        args_path = run_dir / 'args.json'
        best_model_path = run_dir / 'best_model.pth'

        if not args_path.exists() or not best_model_path.exists():
            print(f"Skipping {run_dir.name}: missing args.json or best_model.pth")
            continue

        with open(args_path, 'r') as f:
            args = json.load(f)

        runs.append({
            'run_dir': run_dir,
            'run_name': run_dir.name,
            'architecture': args['architecture'],
            'args': args,
            'best_model_path': best_model_path
        })

    return runs


def create_model(architecture):
    """Create model based on architecture choice - same as train.py"""
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
    elif architecture == 'doubleae':
        model = DoubleAutoencoder()
        print(f"Created Double Autoencoder")
    elif architecture == 'doublevae':
        model = DoubleVariationalAutoencoder()
        print(f"Created Double Variational Autoencoder")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


def load_model_for_inference(architecture, checkpoint_path, device):
    """
    Load a model for inference (no optimizer needed).

    Args:
        architecture: Model architecture name
        checkpoint_path: Path to checkpoint file
        device: torch device

    Returns:
        Loaded model in eval mode
    """
    model = create_model(architecture).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    if isinstance(loss, float):
        print(f"  Loaded {architecture} from epoch {epoch} (loss: {loss:.4f})")
    else:
        print(f"  Loaded {architecture} from epoch {epoch}")

    return model


def create_test_transform_paired(image_size=256):
    """
    Create deterministic test transform for paired dataset.
    No random augmentations - only resize and tensor conversion.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def create_test_transform_unpaired(image_size=256):
    """
    Create deterministic test transform for unpaired dataset.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def create_test_dataloader_paired(args, num_samples=None):
    """
    Create test dataloader for paired dataset (HypersimDataset).

    Args:
        args: dict with 'data_dir', 'source_modality', 'target_modality', 'image_size'
        num_samples: Limit number of samples (None for all)

    Returns:
        DataLoader for testing
    """
    test_transform = create_test_transform_paired(args.get('image_size', 256))

    dataset = HypersimDataset(
        root_dir=args['data_dir'] + '/paired',
        modalities=[args['source_modality'], args['target_modality']],
        transform=test_transform,
        color_transform=None,
        return_scene_info=True
    )

    # Use same split logic as training but only take test portion
    test_split = args.get('test_split', 0.1)
    if test_split > 0:
        train_size = int((1 - test_split) * len(dataset))
        test_size = len(dataset) - train_size
        _, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        test_dataset = dataset

    # Optionally limit samples
    if num_samples is not None and num_samples < len(test_dataset):
        indices = list(range(num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    return DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.get('num_workers', 4),
        pin_memory=True
    )


def create_test_dataloader_unpaired(args, num_samples=None):
    """
    Create test dataloader for unpaired dataset.
    """
    test_transform = create_test_transform_unpaired(args.get('image_size', 256))

    test_dataset = UnpairedImageDataset(
        root_dir=args['data_dir'] + "/unpaired",
        split="test",
        transform=test_transform
    )

    # Optionally limit samples
    if num_samples is not None and num_samples < len(test_dataset):
        indices = list(range(num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    return DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.get('num_workers', 4),
        pin_memory=True
    )


def create_test_dataloader_maps(args, num_samples=None):
    """
    Create test dataloader for satellite-to-map dataset.

    Args:
        args: dict with 'data_dir', 'image_size'
        num_samples: Limit number of samples (None for all)

    Returns:
        DataLoader for testing
    """
    test_transform = transforms.Compose([
        transforms.Resize((args.get('image_size', 256), args.get('image_size', 256))),
        transforms.ToTensor(),
    ])

    test_dataset = SatelliteMapDataset(
        root_dir=args['data_dir'] + "/maps",
        split="val",
        transform=test_transform
    )

    if num_samples is not None and num_samples < len(test_dataset):
        indices = list(range(num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    return DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.get('num_workers', 4),
        pin_memory=True
    )


def run_inference(model, batch, architecture, device, unpaired=False):
    """
    Run inference on a batch, handling different model architectures.

    Args:
        model: The loaded model
        batch: Dict with 'x', 'y' (or legacy 'A', 'B' for unpaired)
        architecture: Architecture name string
        device: torch device
        unpaired: Whether this is unpaired data

    Returns:
        output tensor, x tensor, y tensor
    """
    model.eval()
    with torch.no_grad():
        # Handle legacy unpaired batch keys ('A', 'B' -> 'x', 'y')
        if unpaired or 'A' in batch:
            x = batch['A'].to(device)
            y = batch['B'].to(device)
        else:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

        # Models that need both x and y for forward pass
        if architecture.startswith('cycle') or architecture.startswith('double'):
            output = model(x, y)[0]
        else:
            output = model(x)[0]

        return output, x, y


def denormalize_for_display(tensor, unpaired=False):
    """
    Convert tensor to displayable range [0, 1].

    For paired data: images are already in [0,1]
    For unpaired data: images are already in [0,1]
    """
    return tensor.clamp(0, 1)


def tensor_to_numpy_image(tensor):
    """Convert a tensor to numpy image for matplotlib."""
    # tensor shape: (C, H, W) or (B, C, H, W)
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Move to CPU and convert to numpy
    img = tensor.cpu().numpy()

    # Transpose from (C, H, W) to (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    # Clip to valid range
    img = np.clip(img, 0, 1)

    return img


def create_comparison_figure(results, sample_idx, output_path, unpaired=False):
    """
    Create a comparison figure showing input, target, and outputs from all models.

    Args:
        results: List of dicts with 'model_name', 'architecture', 'input', 'target', 'output'
        sample_idx: Index of the current sample
        output_path: Path to save the figure
        unpaired: Whether data is from unpaired dataset
    """
    n_models = len(results)

    # Create figure: 3 columns (input, target, output) x n_models rows
    fig, axes = plt.subplots(n_models, 3, figsize=(12, 4 * n_models))

    if n_models == 1:
        axes = axes.reshape(1, -1)

    # Column titles
    col_titles = ['Input (x)', 'Target (y)', 'Output (G(x))']

    for row, result in enumerate(results):
        # Denormalize images
        input_img = denormalize_for_display(result['input'], unpaired)
        target_img = denormalize_for_display(result['target'], unpaired)
        output_img = denormalize_for_display(result['output'], unpaired)

        # Convert to numpy
        input_np = tensor_to_numpy_image(input_img)
        target_np = tensor_to_numpy_image(target_img)
        output_np = tensor_to_numpy_image(output_img)

        # Plot images
        axes[row, 0].imshow(input_np)
        axes[row, 1].imshow(target_np)
        axes[row, 2].imshow(output_np)

        # Row label (model name)
        axes[row, 0].set_ylabel(result['model_name'], fontsize=10, fontweight='bold')

        # Column titles (only on first row)
        if row == 0:
            for col, title in enumerate(col_titles):
                axes[row, col].set_title(title, fontsize=12)

        # Remove axis ticks
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.suptitle(f'Sample {sample_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved comparison figure: {output_path.name}")


def create_single_model_grid(model_name, samples, output_path, unpaired=False, max_samples=8):
    """
    Create a grid figure for a single model showing multiple samples.

    Args:
        model_name: Name of the model
        samples: List of dicts with 'input', 'target', 'output'
        output_path: Path to save the figure
        unpaired: Whether data is from unpaired dataset
        max_samples: Maximum number of samples to show
    """
    n_samples = min(len(samples), max_samples)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Model: {model_name}', fontsize=14, fontweight='bold')

    col_titles = ['Input', 'Target', 'Output']

    for row in range(n_samples):
        sample = samples[row]

        input_img = denormalize_for_display(sample['input'], unpaired)
        target_img = denormalize_for_display(sample['target'], unpaired)
        output_img = denormalize_for_display(sample['output'], unpaired)

        input_np = tensor_to_numpy_image(input_img)
        target_np = tensor_to_numpy_image(target_img)
        output_np = tensor_to_numpy_image(output_img)

        axes[row, 0].imshow(input_np)
        axes[row, 1].imshow(target_np)
        axes[row, 2].imshow(output_np)

        axes[row, 0].set_ylabel(f'Sample {row + 1}', fontsize=10)

        if row == 0:
            for col, title in enumerate(col_titles):
                axes[row, col].set_title(title, fontsize=12)

        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved grid figure: {output_path.name}")


def get_modality_key(args):
    """Get a unique key for the modality configuration."""
    return f"{args['source_modality']}_to_{args['target_modality']}"


def evaluate_model_group(runs, device, output_dir, num_samples, num_comparison_figures, unpaired=False):
    """
    Evaluate a group of models (all paired or all unpaired).
    Groups models by their modality configuration for proper comparison.

    Args:
        runs: List of run dicts
        device: torch device
        output_dir: Path to save outputs
        num_samples: Number of samples to evaluate
        num_comparison_figures: Number of comparison figures to generate
        unpaired: Whether this is unpaired data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group runs by modality configuration
    modality_groups = {}
    for run in runs:
        key = get_modality_key(run['args'])
        if key not in modality_groups:
            modality_groups[key] = []
        modality_groups[key].append(run)

    print(f"\nFound {len(modality_groups)} modality configuration(s):")
    for key, group_runs in modality_groups.items():
        print(f"  - {key}: {len(group_runs)} model(s)")

    # Process each modality group separately
    for modality_key, group_runs in modality_groups.items():
        print(f"\n{'-'*60}")
        print(f"Processing modality: {modality_key}")
        print(f"{'-'*60}")

        # Create subdirectory for this modality group
        group_output_dir = output_dir / modality_key
        group_output_dir.mkdir(parents=True, exist_ok=True)

        # Load all models in this group
        print("\nLoading models...")
        models = []
        for run in group_runs:
            try:
                model = load_model_for_inference(
                    run['architecture'],
                    run['best_model_path'],
                    device
                )
                models.append({
                    'model': model,
                    'run': run
                })
            except Exception as e:
                print(f"Error loading {run['run_name']}: {e}")
                continue

        if not models:
            print("No models loaded successfully for this group!")
            continue

        # Create dataloader using the first model's args as reference
        ref_args = models[0]['run']['args']

        print("\nCreating test dataloader...")
        dataset_type = ref_args.get('dataset', 'unpaired' if ref_args.get('unpaired', False) else 'paired')
        if unpaired:
            dataloader = create_test_dataloader_unpaired(ref_args, num_samples)
        elif dataset_type == 'maps':
            dataloader = create_test_dataloader_maps(ref_args, num_samples)
        else:
            dataloader = create_test_dataloader_paired(ref_args, num_samples)

        print(f"Test dataset size: {len(dataloader.dataset)}")

        # Run inference and collect results
        all_results = {m['run']['run_name']: [] for m in models}

        print("\nRunning inference...")
        for sample_idx, batch in enumerate(tqdm(dataloader, desc='Testing')):
            sample_results = []

            for model_info in models:
                model = model_info['model']
                run = model_info['run']

                output, x, y = run_inference(
                    model, batch, run['architecture'], device, unpaired
                )

                result = {
                    'model_name': run['run_name'],
                    'architecture': run['architecture'],
                    'input': x.cpu(),
                    'target': y.cpu(),
                    'output': output.cpu()
                }

                sample_results.append(result)
                all_results[run['run_name']].append(result)

            # Create per-sample comparison figure (all models side by side)
            if sample_idx < num_comparison_figures:
                fig_path = group_output_dir / f'comparison_sample_{sample_idx:04d}.png'
                create_comparison_figure(sample_results, sample_idx, fig_path, unpaired)

        # Create per-model grid figures
        print("\nGenerating per-model grids...")
        for model_info in models:
            run_name = model_info['run']['run_name']
            samples = all_results[run_name]
            fig_path = group_output_dir / f'grid_{run_name}.png'
            create_single_model_grid(run_name, samples, fig_path, unpaired)

        # Save summary JSON for this group
        summary = {
            'modality': modality_key,
            'source_modality': ref_args['source_modality'],
            'target_modality': ref_args['target_modality'],
            'num_models': len(models),
            'num_samples': len(dataloader.dataset),
            'unpaired': unpaired,
            'models': [
                {
                    'name': m['run']['run_name'],
                    'architecture': m['run']['architecture'],
                    'checkpoint': str(m['run']['best_model_path']),
                    'training_args': m['run']['args']
                }
                for m in models
            ]
        }

        with open(group_output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved summary to: {group_output_dir / 'summary.json'}")


def evaluate_models(args):
    """
    Main evaluation function.

    Args:
        args: Namespace with test configuration
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"test_results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Discover runs
    runs = discover_runs(args.runs_dir)

    if not runs:
        print("No trained models found!")
        return

    print(f"\nDiscovered {len(runs)} trained models:")
    for run in runs:
        print(f"  - {run['run_name']} ({run['architecture']})")

    # Filter by architecture if specified
    if args.architectures:
        runs = [r for r in runs if r['architecture'] in args.architectures]
        print(f"\nFiltered to {len(runs)} models matching architectures: {args.architectures}")

    # Determine dataset type for each run (handles legacy 'unpaired' flag and new 'dataset' key)
    def get_dataset_type(run_args):
        if 'dataset' in run_args:
            return run_args['dataset']
        # Legacy format: convert 'unpaired' boolean to dataset string
        return 'unpaired' if run_args.get('unpaired', False) else 'paired'

    # Group runs by dataset type
    paired_runs = [r for r in runs if get_dataset_type(r['args']) == 'paired']
    unpaired_runs = [r for r in runs if get_dataset_type(r['args']) == 'unpaired']
    maps_runs = [r for r in runs if get_dataset_type(r['args']) == 'maps']

    dataset_filter = getattr(args, 'dataset_filter', None)

    # Process paired models
    if paired_runs and dataset_filter in (None, 'paired'):
        print(f"\n{'='*60}")
        print(f"Evaluating {len(paired_runs)} paired dataset models")
        print(f"{'='*60}")

        evaluate_model_group(
            paired_runs,
            device,
            output_dir / 'paired',
            args.num_samples,
            args.num_comparison_figures,
            unpaired=False
        )

    # Process unpaired models
    if unpaired_runs and dataset_filter in (None, 'unpaired'):
        print(f"\n{'='*60}")
        print(f"Evaluating {len(unpaired_runs)} unpaired dataset models")
        print(f"{'='*60}")

        evaluate_model_group(
            unpaired_runs,
            device,
            output_dir / 'unpaired',
            args.num_samples,
            args.num_comparison_figures,
            unpaired=True
        )

    # Process maps models
    if maps_runs and dataset_filter in (None, 'maps'):
        print(f"\n{'='*60}")
        print(f"Evaluating {len(maps_runs)} maps dataset models")
        print(f"{'='*60}")

        evaluate_model_group(
            maps_runs,
            device,
            output_dir / 'maps',
            args.num_samples,
            args.num_comparison_figures,
            unpaired=False
        )

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test and compare trained VAE-CycleGAN models')

    # Run discovery
    parser.add_argument('--runs_dir', type=str, default='runs',
                        help='Directory containing trained model runs')

    # Filter options
    parser.add_argument('--architectures', type=str, nargs='+', default=None,
                        help='Filter to specific architectures (e.g., autoencoder vae aegan)')
    parser.add_argument('--dataset_filter', type=str, default=None,
                        choices=['paired', 'unpaired', 'maps'],
                        help='Only evaluate models trained on this dataset')

    # Test configuration
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of test samples to evaluate')
    parser.add_argument('--num_comparison_figures', type=int, default=10,
                        help='Number of side-by-side comparison figures to generate')

    # Output
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save test results')

    # Device
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()

    evaluate_models(args)
