#!/usr/bin/env python3
"""
Script to visualize all available modalities from the Hypersim dataset.
Saves example images showing the same scene with all different modalities.
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# All available modalities in Hypersim dataset
ALL_MODALITIES = [
    'color',
    'depth',
    'normal',
    'normal_bump',
    'normal_world',
    'position',
    'render_entity_id',
    'semantic',
    'semantic_instance'
]

MODALITY_DESCRIPTIONS = {
    'color': 'RGB Color Image',
    'depth': 'Depth Map (distance from camera)',
    'normal': 'Surface Normals (camera space)',
    'normal_bump': 'Bump-mapped Normals',
    'normal_world': 'Surface Normals (world space)',
    'position': '3D Position Map',
    'render_entity_id': 'Entity/Object ID',
    'semantic': 'Semantic Segmentation',
    'semantic_instance': 'Instance Segmentation'
}


def find_sample_frame(dataset_dir: str, scene_name: str = None, frame_id: str = "0000"):
    """Find a sample frame with all modalities available."""
    dataset_path = Path(dataset_dir)

    # Get list of scenes
    scenes = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    if not scenes:
        raise ValueError(f"No scenes found in {dataset_dir}")

    # Use specified scene or first available
    if scene_name:
        scene_dir = dataset_path / scene_name
        if not scene_dir.exists():
            raise ValueError(f"Scene {scene_name} not found")
    else:
        scene_dir = scenes[0]

    # Find camera directory
    cam_dirs = sorted(scene_dir.glob('cam_*'))
    if not cam_dirs:
        raise ValueError(f"No camera directories found in {scene_dir}")

    cam_dir = cam_dirs[0]

    return scene_dir.name, cam_dir.name, frame_id, cam_dir


def load_all_modalities(cam_dir: Path, frame_id: str):
    """Load all modalities for a given frame."""
    modalities = {}

    for modality in ALL_MODALITIES:
        img_path = cam_dir / f"frame_{frame_id}_{modality}.png"
        if img_path.exists():
            img = Image.open(img_path)
            modalities[modality] = np.array(img)
        else:
            print(f"  Warning: {modality} not found at {img_path}")
            modalities[modality] = None

    return modalities


def save_modality_grid(modalities: dict, output_dir: str, scene_name: str, frame_id: str):
    """Save a grid visualization of all modalities."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter out None modalities
    available = {k: v for k, v in modalities.items() if v is not None}
    n_modalities = len(available)

    # Calculate grid size
    n_cols = 3
    n_rows = (n_modalities + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_modalities > 1 else [axes]

    for idx, (modality, img) in enumerate(available.items()):
        ax = axes[idx]
        ax.imshow(img)
        ax.set_title(f"{modality}\n({MODALITY_DESCRIPTIONS[modality]})", fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_modalities, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Hypersim Dataset Modalities\nScene: {scene_name}, Frame: {frame_id}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    grid_path = os.path.join(output_dir, f"modalities_grid_{scene_name}_{frame_id}.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved grid: {grid_path}")

    return grid_path


def save_individual_modalities(modalities: dict, output_dir: str, scene_name: str, frame_id: str):
    """Save each modality as an individual image."""
    individual_dir = os.path.join(output_dir, f"{scene_name}_{frame_id}")
    os.makedirs(individual_dir, exist_ok=True)

    saved_files = []
    for modality, img in modalities.items():
        if img is not None:
            # Save original
            img_pil = Image.fromarray(img)
            save_path = os.path.join(individual_dir, f"{modality}.png")
            img_pil.save(save_path)
            saved_files.append(save_path)

    print(f"  Saved {len(saved_files)} individual modality images to {individual_dir}")
    return saved_files


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Hypersim dataset modalities")
    parser.add_argument("--dataset", type=str, default="dataset/hypersim",
                        help="Path to the Hypersim dataset")
    parser.add_argument("--scene", type=str, default=None,
                        help="Specific scene name (default: first available)")
    parser.add_argument("--frame", type=str, default="0000",
                        help="Frame ID to visualize (default: 0000)")
    parser.add_argument("--output", type=str, default="modalities_examples",
                        help="Output directory for saved images")

    args = parser.parse_args()

    print("=" * 60)
    print("HYPERSIM DATASET MODALITIES VIEWER")
    print("=" * 60)
    print(f"\nAvailable modalities ({len(ALL_MODALITIES)}):")
    for i, mod in enumerate(ALL_MODALITIES, 1):
        print(f"  {i}. {mod:20s} - {MODALITY_DESCRIPTIONS[mod]}")

    print(f"\nDataset path: {args.dataset}")
    print(f"Output directory: {args.output}")

    # Find sample frame
    print("\nFinding sample frame...")
    scene_name, cam_name, frame_id, cam_dir = find_sample_frame(
        args.dataset, args.scene, args.frame
    )
    print(f"  Scene: {scene_name}")
    print(f"  Camera: {cam_name}")
    print(f"  Frame: {frame_id}")

    # Load all modalities
    print("\nLoading modalities...")
    modalities = load_all_modalities(cam_dir, frame_id)
    available_count = sum(1 for v in modalities.values() if v is not None)
    print(f"  Loaded {available_count}/{len(ALL_MODALITIES)} modalities")

    # Save visualization
    print("\nSaving visualizations...")
    save_modality_grid(modalities, args.output, scene_name, frame_id)
    save_individual_modalities(modalities, args.output, scene_name, frame_id)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
