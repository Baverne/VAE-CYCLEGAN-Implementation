#!/usr/bin/env python3
"""
Data Manager for Hypersim Dataset
Provides PyTorch DataLoader with support for multiple modalities and data augmentation.
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class HypersimDataset(Dataset):
    """
    Dataset structure:
        datasets/
            ai_001_001_unknown/
                cam_00/
                    frame_0000_depth.png
                    frame_0000_semantic.png
                    frame_0000_normal.png
                    frame_0000_color.png (optional)
    
    Args:
        root_dir: Root directory containing scene folders
        modalities: List of modalities to load (e.g., ['depth', 'semantic', 'normal', 'color'])
        transform: Optional transform to apply to all modalities
        color_transform: Optional transform specifically for color/color images
        return_scene_info: Whether to return scene and camera number and scene type in the batch
    """
    
    def __init__(
        self,
        root_dir: str,
        modalities: List[str] = ['color','depth','normal_world','normal','semantic','semantic_instance','normal'],
        transform: Optional[Callable] = None,
        color_transform: Optional[Callable] = None,
        return_scene_info: bool = True,
        paired_mode: bool = True # in this mode, x and y are returned as 'x' and 'y' keys
    ):
        self.root_dir = Path(root_dir)
        self.modalities = modalities
        self.transform = transform
        self.color_transform = color_transform
        self.return_scene_info = return_scene_info
        self.paired_mode = paired_mode
        
        # Check paired_mode requirements
        if self.paired_mode:
            if len(self.modalities) not in [1, 2]:
                raise ValueError(f"paired_mode requires 1 or 2 modalities, got {len(self.modalities)}")
        
        # Scan dataset and build file index
        self.samples = self._scan_dataset()
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {root_dir}")
        
        print(f"  Loaded dataset with {len(self.samples)} samples")
        print(f"  Modalities: {', '.join(self.modalities)}")
        print(f"  Scenes: {len(self.get_unique_scenes())}")
    
    def _scan_dataset(self) -> List[Dict]:
        """
        Scan the dataset directory and build an index of all samples.
        
        Returns:
            List of dictionaries containing sample information
        """
        samples = []
        
        # Iterate through scene directories
        for scene_dir in sorted(self.root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            # Parse scene name and type
            scene_name_parts = scene_dir.name.split('_')
            if len(scene_name_parts) >= 4:
                # Format: ai_001_001_unknown -> scene_num=ai_001_001, scene_type=unknown
                scene_num = '_'.join(scene_name_parts[:3])
                scene_type = '_'.join(scene_name_parts[3:])
            else:
                scene_num = scene_dir.name
                scene_type = 'unknown'
            
            # Look for camera directories
            cam_dirs = list(scene_dir.glob('cam_*'))
            
            for cam_dir in cam_dirs:
                if not cam_dir.is_dir():
                    continue
                
                camera_name = cam_dir.name
                
                # Find all frame files for the first modality to get frame list
                first_modality = self.modalities[0]
                frame_files = sorted(cam_dir.glob(f'frame_*_{first_modality}.png'))
                
                for frame_file in frame_files:
                    # Extract frame number from filename
                    # Format: frame_0000_depth.png -> frame_id=0000
                    frame_id = frame_file.stem.split('_')[1]
                    
                    # Build paths for all modalities
                    modality_paths = {}
                    all_exist = True
                    
                    for modality in self.modalities:
                        modality_path = cam_dir / f'frame_{frame_id}_{modality}.png'
                        if modality_path.exists():
                            modality_paths[modality] = modality_path
                        else:
                            # If modality doesn't exist, skip this sample
                            all_exist = False
                            break
                    
                    if all_exist:
                        # Extract camera number from camera_name (e.g., cam_00 -> 00)
                        cam_num = camera_name.replace('cam_', '')
                        
                        samples.append({
                            'scene_dir': scene_dir,
                            'scene_num': scene_num,
                            'scene_type': scene_type,
                            'camera': camera_name,
                            'cam_num': cam_num,
                            'frame_id': frame_id,
                            'modality_paths': modality_paths
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a sample from the dataset.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Dictionary containing:
                - Each modality as a tensor
                - 'scene_num': Scene number (if return_scene_info=True)
                - 'scene_type': Scene type (if return_scene_info=True)
                - 'cam_num': Camera number (if return_scene_info=True)
                - 'frame_id': Frame identifier
        """
        sample_info = self.samples[idx]
        
        result = {}
        
        # First, load all modalities as PIL images
        pil_images = {}
        for modality, path in sample_info['modality_paths'].items():
            pil_images[modality] = Image.open(path).convert('RGB')
        
        # Apply general transform to all modalities with the same random state
        if self.transform is not None:
            # Set random seed based on index to ensure reproducibility if needed
            # But allow random transforms to be the same across modalities
            random_state = torch.get_rng_state()
            
            for modality, img in pil_images.items():
                # Reset random state for each modality to ensure same transform
                torch.set_rng_state(random_state)
                
                # Apply color-specific transform BEFORE general transform for color modality
                # This ensures ColorJitter is applied on PIL image before ToTensor/Normalize
                if modality == 'color' and self.color_transform is not None:
                    img = self.color_transform(img)
                
                # Apply general transform (spatial transforms + ToTensor + Normalize)
                transformed_img = self.transform(img)
                
                # Ensure final result is a tensor (in case transform doesn't include ToTensor)
                if not isinstance(transformed_img, torch.Tensor):
                    transformed_img = transforms.ToTensor()(transformed_img)
                
                result[modality] = transformed_img
        else:
            # No transform provided - apply color transform to color if available, then convert to tensor
            for modality, img in pil_images.items():
                if modality == 'color' and self.color_transform is not None:
                    img = self.color_transform(img)
                
                # Convert to tensor if not already
                if not isinstance(img, torch.Tensor):
                    img = transforms.ToTensor()(img)
                
                result[modality] = img
        
        # Add scene information if requested
        if self.return_scene_info:
            result['scene_num'] = sample_info['scene_num']
            result['scene_type'] = sample_info['scene_type']
            result['cam_num'] = sample_info['cam_num']
        
        # Add frame identifier
        result['frame_id'] = sample_info['frame_id']
        
        # Apply paired_mode
        if self.paired_mode:
            paired_result = {}
            if len(self.modalities) == 1:
                # Single modality: x = y (autoencoder mode)
                modality = self.modalities[0]
                paired_result['x'] = result[modality]
                paired_result['y'] = result[modality]
            elif len(self.modalities) == 2:
                # Two modalities: x = first (input), y = second (target)
                paired_result['x'] = result[self.modalities[0]]
                paired_result['y'] = result[self.modalities[1]]
            
            # Keep scene info if requested
            if self.return_scene_info:
                paired_result['scene_num'] = result['scene_num']
                paired_result['scene_type'] = result['scene_type']
                paired_result['cam_num'] = result['cam_num']
            paired_result['frame_id'] = result['frame_id']
            
            return paired_result
        else:
            # Unpaired mode: shuffle y by picking a random sample
            unpaired_result = {}
            if len(self.modalities) == 2:
                # x from current index
                unpaired_result['x'] = result[self.modalities[0]]
                
                # y from random index (unpaired)
                random_idx = random.randint(0, len(self.samples) - 1)
                random_sample = self.samples[random_idx]
                random_y_path = random_sample['modality_paths'][self.modalities[1]]
                random_y_img = Image.open(random_y_path).convert('RGB')
                
                # Apply transforms to random y
                if self.transform is not None:
                    if self.modalities[1] == 'color' and self.color_transform is not None:
                        random_y_img = self.color_transform(random_y_img)
                    random_y_img = self.transform(random_y_img)
                    if not isinstance(random_y_img, torch.Tensor):
                        random_y_img = transforms.ToTensor()(random_y_img)
                else:
                    if self.modalities[1] == 'color' and self.color_transform is not None:
                        random_y_img = self.color_transform(random_y_img)
                    if not isinstance(random_y_img, torch.Tensor):
                        random_y_img = transforms.ToTensor()(random_y_img)
                
                unpaired_result['y'] = random_y_img
                
                # Keep scene info from x
                if self.return_scene_info:
                    unpaired_result['scene_num'] = result['scene_num']
                    unpaired_result['scene_type'] = result['scene_type']
                    unpaired_result['cam_num'] = result['cam_num']
                unpaired_result['frame_id'] = result['frame_id']
                
                return unpaired_result
            else:
                raise ValueError("Unpaired mode requires exactly 2 modalities")
        
        return result
    
    def get_unique_scenes(self) -> List[str]:
        """Return list of unique scene numbers in the dataset."""
        return sorted(list(set(sample['scene_num'] for sample in self.samples)))
    
    def get_unique_scene_types(self) -> List[str]:
        """Return list of unique scene types in the dataset."""
        return sorted(list(set(sample['scene_type'] for sample in self.samples)))
    
    def filter_by_scene(self, scene_nums: List[str]) -> 'HypersimDataset':
        """
        Args:
            scene_nums: List of scene numbers to include
            
        Returns:
            New HypersimDataset with filtered samples
        """
        filtered_dataset = HypersimDataset.__new__(HypersimDataset)
        filtered_dataset.root_dir = self.root_dir
        filtered_dataset.modalities = self.modalities
        filtered_dataset.transform = self.transform
        filtered_dataset.color_transform = self.color_transform
        filtered_dataset.return_scene_info = self.return_scene_info
        filtered_dataset.samples = [
            s for s in self.samples if s['scene_num'] in scene_nums
        ]
        return filtered_dataset
    
    def filter_by_scene_type(self, scene_types: List[str]) -> 'HypersimDataset':
        """ 
        Args:
            scene_types: List of scene types to include
            
        Returns:
            New HypersimDataset with filtered samples
        """
        filtered_dataset = HypersimDataset.__new__(HypersimDataset)
        filtered_dataset.root_dir = self.root_dir
        filtered_dataset.modalities = self.modalities
        filtered_dataset.transform = self.transform
        filtered_dataset.color_transform = self.color_transform
        filtered_dataset.return_scene_info = self.return_scene_info
        filtered_dataset.samples = [
            s for s in self.samples if s['scene_type'] in scene_types
        ]
        return filtered_dataset


class SatelliteMapDataset(Dataset):
    """
    Paired satellite-to-map dataset (pix2pix maps format).

    Each image is 1200x600: left half is satellite photo, right half is map.

    Directory structure:
        root_dir/
            train/
                1.jpg, 2.jpg, ...
            val/
                1.jpg, 2.jpg, ...

    Returns:
        dict with 'x' (satellite, left half) and 'y' (map, right half)
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.image_dir):
            raise ValueError(f"Directory not found: {self.image_dir}")

        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"  Loaded {split} split with {len(self.images)} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')

        # Split image: left half = satellite (x), right half = map (y)
        w, h = img.size  # 1200, 600
        half_w = w // 2

        satellite = img.crop((0, 0, half_w, h))      # left half
        map_img = img.crop((half_w, 0, w, h))         # right half

        # Apply transforms with same random state for both halves
        if self.transform is not None:
            random_state = torch.get_rng_state()
            satellite = self.transform(satellite)
            torch.set_rng_state(random_state)
            map_img = self.transform(map_img)
        else:
            satellite = transforms.ToTensor()(satellite)
            map_img = transforms.ToTensor()(map_img)

        return {'x': satellite, 'y': map_img}


if __name__ == "__main__":
    import torchvision.utils as vutils

    # Create output directory for input examples
    output_dir = Path("input_examples")
    output_dir.mkdir(exist_ok=True)

    # Define transform with augmentation
    general_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=256, scale=(0.33, 1.0), ratio=(1,1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    # Define color-specific transform
    color_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    ])

    # Create dataset in paired mode (x=depth, y=normal)
    dataset = HypersimDataset(
        root_dir='datasets',
        modalities=['depth', 'normal'],
        transform=general_transform,
        color_transform=color_transform,
        return_scene_info=True,
        paired_mode=True
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(dataloader))

    # Show x and y shapes
    print(f"\n{'='*50}")
    print(f"Paired Mode - X and Y Shapes")
    print(f"{'='*50}")
    print(f"X (input) shape:  {batch['x'].shape}")
    print(f"Y (target) shape: {batch['y'].shape}")
    print(f"X value range: [{batch['x'].min():.3f}, {batch['x'].max():.3f}]")
    print(f"Y value range: [{batch['y'].min():.3f}, {batch['y'].max():.3f}]")
    print(f"Scene: {batch['scene_num']}")
    print(f"Frame: {batch['frame_id']}")

    # Save x and y images as PNG
    # Save first sample from batch
    x_img = batch['x'][0]  # Shape: [C, H, W]
    y_img = batch['y'][0]

    # Save individual images
    vutils.save_image(x_img, output_dir / "x_sample.png")
    vutils.save_image(y_img, output_dir / "y_sample.png")

    # Save grid of all batch samples
    vutils.save_image(batch['x'], output_dir / "x_batch_grid.png", nrow=2, normalize=True)
    vutils.save_image(batch['y'], output_dir / "y_batch_grid.png", nrow=2, normalize=True)

    # Save side-by-side comparison (x | y)
    comparison = torch.cat([batch['x'], batch['y']], dim=3)  # Concatenate along width
    vutils.save_image(comparison, output_dir / "xy_comparison.png", nrow=2, normalize=True)

    print(f"\n{'='*50}")
    print(f"Saved PNG images to '{output_dir}/':")
    print(f"  - x_sample.png       (single X sample)")
    print(f"  - y_sample.png       (single Y sample)")
    print(f"  - x_batch_grid.png   (all X in batch)")
    print(f"  - y_batch_grid.png   (all Y in batch)")
    print(f"  - xy_comparison.png  (X|Y side-by-side)")
    print(f"{'='*50}")
    