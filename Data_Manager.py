#!/usr/bin/env python3
"""
Data Manager for Hypersim Dataset
Provides PyTorch DataLoader with support for multiple modalities and data augmentation.
"""

import os
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
        return_scene_info: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.modalities = modalities
        self.transform = transform
        self.color_transform = color_transform
        self.return_scene_info = return_scene_info
        
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
                
                # Apply general transform (spatial transforms will be identical)
                transformed_img = self.transform(img)
                
                # Apply color-specific transform for color modality (in addition to general transform)
                if modality == 'color' and self.color_transform is not None:
                    # If transform returns tensor, convert back to PIL for color transform
                    if isinstance(transformed_img, torch.Tensor):
                        # Convert tensor back to PIL for color transform
                        transformed_img = transforms.ToPILImage()(transformed_img)
                        transformed_img = self.color_transform(transformed_img)
                        # ToTensor will be applied below
                    else:
                        transformed_img = self.color_transform(transformed_img)
                
                # Ensure final result is a tensor
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


if __name__ == "__main__":
    """
    # Basic usage without transforms

    dataset = HypersimDataset(
        root_dir='datasets',
        modalities=['color','depth','normal_world','normal','semantic','semantic_instance','normal'],
        return_scene_info=True
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        print(f"depth shape: {batch['depth'].shape}")
        print(f"semantic shape: {batch['semantic'].shape}")
        print(f"normal shape: {batch['normal'].shape}")
        print(f"scene_num: {batch['scene_num']}")
        print(f"scene_type: {batch['scene_type']}")
        print(f"cam_num: {batch['cam_num']}")
        print(f"frame_id: {batch['frame_id']}")
        print(f"color max min : {batch['color'].min()} , {batch['color'].max()}")
        print(f"depth max min : {batch['depth'].min()} , {batch['depth'].max()}")
        print(f"semantic max min : {batch['semantic'].min()} , {batch['semantic'].max()}")
        break
    """
    # Example 2: With data augmentation transforms
    
    # Define augmentation transform for general modalities (depth, semantic, normal)
    general_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=256, scale=(0.33, 1.0), ratio=(1,1), interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    
    # Define augmentation transform specifically for color/color images
    color_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    ])
    
    # Create dataset with transforms
    dataset_augmented = HypersimDataset(
        root_dir='datasets',
        modalities=['depth', 'semantic', 'normal'],
        transform=general_transform,
        color_transform=color_transform, 
        return_scene_info=True
    )
    
    dataloader_augmented = DataLoader(dataset_augmented, batch_size=4, shuffle=True)
    
    batch = next(iter(dataloader_augmented))
    print(f"Augmented batch - depth range: [{batch['depth'].min():.3f}, {batch['depth'].max():.3f}]")
    print(f"Augmented batch - semantic range: [{batch['semantic'].min():.3f}, {batch['semantic'].max():.3f}]")
    print(f"Shape: {batch['depth'].shape}")
    
    # Example 3: Minimal transform (just normalization)
    
    # Minimal transform without augmentation
    simple_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset_simple = HypersimDataset(
        root_dir='datasets',
        modalities=['depth', 'semantic', 'normal'],
        transform=simple_transform,
        return_scene_info=True
    )
    
    dataloader_simple = DataLoader(dataset_simple, batch_size=2, shuffle=False)
    batch = next(iter(dataloader_simple))
    print(f"Simple transform - batch shape: {batch['depth'].shape}")
    print(f"Value range (normalized): [{batch['depth'].min():.3f}, {batch['depth'].max():.3f}]")
    