#!/usr/bin/env python3
"""
Script pour télécharger un échantillon du dataset Hypersim avec diversité maximale.

Utilisation:
    cloner le repo ml-hypersim depuis https://github.com/apple/ml-hypersim : git clone https://github.com/apple/ml-hypersim
    installer les dépendances requises : pip install -r requirements.txt
    
    puis exécuter ce script.
    Exemples:
    # Télécharger 100 images avec depth, semantic et normal
    python download_dataset_sample.py --num_images 100 --modalities depth semantic normal --repo_path /path/to/ml-hypersim --output_dir my_dataset --seed 123
    
    # Télécharger 50 images avec toutes les modalités
    python download_dataset_sample.py --num_images 50 --modalities all_modalities --repo_path /path/to/ml-hypersim --output_dir my_dataset --seed 123


 
"""

import argparse
import os
import sys
import zipfile
import requests
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from collections import defaultdict
import shutil
from tqdm import tqdm

# Augmenter la vitesse de téléchargement
zipfile.ZipExtFile.MIN_READ_SIZE = 2 ** 20

# Base URL et liste des scènes disponibles
BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/"

# Définition des modalités disponibles
# Format: (nom_modalité, nom_fichier_source, is_hdf5, répertoire_source)
MODALITIES_CONFIG = [
    ('color', 'tonemap.jpg', False, 'final_preview'),
    ('depth', 'depth_meters.hdf5', True, 'geometry_hdf5'),
    ('semantic', 'semantic.hdf5', True, 'geometry_hdf5'),
    ('semantic_instance', 'semantic_instance.hdf5', True, 'geometry_hdf5'),
    ('normal', 'normal_cam.hdf5', True, 'geometry_hdf5'),
    ('normal_world', 'normal_world.hdf5', True, 'geometry_hdf5'),
    ('normal_bump', 'normal_bump_cam.hdf5', True, 'geometry_hdf5'),
    ('position', 'position.hdf5', True, 'geometry_hdf5'),
    ('render_entity_id', 'render_entity_id.hdf5', True, 'geometry_hdf5'),
]

# Construction du mapping des modalités
MODALITY_MAPPINGS = {}
for modality_name, source_file, is_hdf5, source_dir in MODALITIES_CONFIG:
    pattern = f'scene_cam_{{cam}}_{source_dir}/frame.{{frame:04d}}.{source_file}'
    MODALITY_MAPPINGS[modality_name] = {
        'pattern': pattern,
        'is_hdf5': is_hdf5,
        'output_name': f'{modality_name}.png'
    }


class WebFile:
    """Fichier web avec support de lecture partielle."""
    def __init__(self, url, session):
        with session.head(url) as response:
            size = int(response.headers["content-length"])
        
        self.url = url
        self.session = session
        self.offset = 0
        self.size = size
    
    def seekable(self):
        return True
    
    def tell(self):
        return self.offset
    
    def available(self):
        return self.size - self.offset
    
    def seek(self, offset, whence=0):
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset = min(self.offset + offset, self.size)
        elif whence == 2:
            self.offset = max(0, self.size + offset)
    
    def read(self, n=None):
        if n is None:
            n = self.available()
        else:
            n = min(n, self.available())
        
        end_inclusive = self.offset + n - 1
        
        headers = {
            "Range": f"bytes={self.offset}-{end_inclusive}",
        }
        
        with self.session.get(self.url, headers=headers) as response:
            data = response.content
        
        self.offset += len(data)
        
        return data


def normalize_for_display(data):
    """Normalise les données pour affichage."""
    data = np.array(data, dtype=np.float32)
    
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data)
    
    data_min = np.min(data[valid_mask])
    data_max = np.max(data[valid_mask])
    
    if data_max - data_min < 1e-10:
        return np.zeros_like(data)
    
    normalized = (data - data_min) / (data_max - data_min)
    normalized[~valid_mask] = 0
    
    return normalized


def convert_hdf5_to_png(hdf5_data, modality_name, output_path):
    """Convertit les données HDF5 en PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Traitement selon la modalité
    if 'semantic' in modality_name or 'render_entity_id' in modality_name:
        # Segmentation - utiliser colormap
        if len(hdf5_data.shape) == 2:
            normalized = normalize_for_display(hdf5_data)
            cmap = plt.get_cmap('tab20')
            img = cmap(normalized)[:, :, :3]
        else:
            img = normalize_for_display(hdf5_data)
    
    elif 'normal' in modality_name:
        # Normal maps - convertir de [-1, 1] à [0, 1]
        img = (hdf5_data + 1.0) / 2.0
        img = np.clip(img, 0, 1)
    
    elif 'depth' in modality_name or 'position' in modality_name:
        # Depth/position - normaliser avec colormap
        if len(hdf5_data.shape) == 2:
            normalized = normalize_for_display(hdf5_data)
            cmap = plt.get_cmap('plasma')
            img = cmap(normalized)[:, :, :3]
        else:
            img = normalize_for_display(hdf5_data)
    
    else:
        # Générique
        img = normalize_for_display(hdf5_data)
    
    plt.imsave(output_path, img)
    return output_path


def load_scene_metadata(repo_path=None):
    """Charge les métadonnées des scènes."""
    if repo_path is None:
        # Chemin relatif par défaut (script dans contrib/99991)
        metadata_path = Path('../../evermotion_dataset/analysis/metadata_camera_trajectories.csv')
    else:
        metadata_path = Path(repo_path) / 'evermotion_dataset' / 'analysis' / 'metadata_camera_trajectories.csv'
    
    if not metadata_path.exists():
        print(f"⚠️  Métadonnées non trouvées: {metadata_path}")
        return {}
    
    df = pd.read_csv(metadata_path)
    
    # Créer un mapping scene_name -> scene_type
    scene_types = {}
    for _, row in df.iterrows():
        animation = row['Animation']
        scene_name = '_'.join(animation.split('_')[:3])  # ai_001_001
        scene_type = row['Scene type']
        
        if scene_name not in scene_types:
            scene_types[scene_name] = scene_type
    
    return scene_types


def get_scene_name_with_type(scene_name, scene_types):
    """Retourne le nom de scène avec son type."""
    scene_type = scene_types.get(scene_name, 'unknown')
    # Nettoyer le type pour le nom de fichier
    scene_type_clean = scene_type.lower().replace(' ', '_').replace('(', '').replace(')', '')
    return f"{scene_name}_{scene_type_clean}"


def plan_download(num_images, seed=42, repo_path=None):
    """
    Planifie quelles images télécharger en maximisant la diversité de scènes.
    
    Retourne une liste déterministe de (scene_name, camera_name, frame_id).
    """
    # Fixer le seed pour reproductibilité
    np.random.seed(seed)
    
    # Charger les métadonnées
    if repo_path is None:
        # Chemin relatif par défaut (script dans contrib/99991)
        metadata_path = Path('../../evermotion_dataset/analysis/metadata_images.csv')
    else:
        metadata_path = Path(repo_path) / 'evermotion_dataset' / 'analysis' / 'metadata_images.csv'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"⚠️  Métadonnées non trouvées, utilisation d'un plan par défaut")
    print(f"metadata_path : {metadata_path}")
    df = pd.read_csv(metadata_path)
    print("Dataframe des métadonnées non filtrées:")
    print(df)
    # Ne garder que les images publiquement disponibles
    df = df[df['included_in_public_release'] == True]
    print("Dataframe des métadonnées filtrées:")
    print(df)

    
    # Obtenir toutes les scènes uniques
    scenes = df['scene_name'].unique()
    scenes.sort()  # Tri pour déterminisme
    
    # Calculer combien d'images par scène (distribution uniforme)
    images_per_scene = max(1, num_images // len(scenes))
    
    plan = []
    scene_idx = 0
    
    while len(plan) < num_images:
        for scene in scenes:
            if len(plan) >= num_images:
                break
            
            # Obtenir les caméras et frames pour cette scène
            scene_data = df[df['scene_name'] == scene]
            
            if len(scene_data) == 0:
                continue
            
            # Prendre une caméra (la première par ordre alphabétique pour déterminisme)
            cameras = sorted(scene_data['camera_name'].unique())
            camera = cameras[0]
            
            # Prendre des frames espacées uniformément
            scene_camera_data = scene_data[scene_data['camera_name'] == camera]
            frames = sorted(scene_camera_data['frame_id'].unique())
            
            if len(frames) == 0:
                continue
            
            # Sélectionner une frame (espacée uniformément)
            frame_idx = (len(plan) // len(scenes)) % len(frames)
            frame = frames[min(frame_idx, len(frames) - 1)]
            
            plan.append((scene, camera, frame))
    
    return plan[:num_images]


def download_and_convert(session, url, scene_name, camera_name, frame_id, modalities, output_dir, scene_types, temp_dir, verbose=True):
    """Télécharge et convertit en png les modalités pour une image donnée."""
    
    scene_name_with_type = get_scene_name_with_type(scene_name, scene_types)
    
    # Créer le répertoire de sortie qi il n'existe pas
    output_scene_dir = output_dir / scene_name_with_type / camera_name
    if not output_scene_dir.exists():
        output_scene_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\n Téléchargement: {scene_name_with_type}/{camera_name}/frame_{frame_id:04d}")
    
    try:
        # Ouvrir le fichier ZIP distant
        f = WebFile(url, session)
        z = zipfile.ZipFile(f)
        
        downloaded_count = 0
        
        for modality in modalities:
            if modality not in MODALITY_MAPPINGS:
                print(f"  ⚠️  Modalité inconnue: {modality}")
                continue
            
            mapping = MODALITY_MAPPINGS[modality]
            
            # Construire le chemin du fichier dans le ZIP
            cam_num = camera_name.replace('cam_', '')
            file_pattern = mapping['pattern'].format(cam=cam_num, frame=frame_id)
            file_path_in_zip = f"{scene_name}/images/{file_pattern}"
            
            try:
                # Vérifier si le fichier existe dans le ZIP
                if file_path_in_zip not in z.namelist():
                    print(f"  ⚠️  Fichier non trouvé: {file_pattern}")
                    continue
                
                output_filename = f"frame_{frame_id:04d}_{mapping['output_name']}"
                output_path = output_scene_dir / output_filename
                
                # Vérifier si le fichier existe déjà
                if output_path.exists():
                    if verbose:
                        print(f"  ⏭️  {modality}: déjà téléchargé")
                    downloaded_count += 1
                    continue
                
                if mapping['is_hdf5']:
                    # Extraire temporairement le HDF5
                    temp_hdf5 = temp_dir / f"temp_{modality}.hdf5"
                    with z.open(file_path_in_zip) as zf:
                        with open(temp_hdf5, 'wb') as tf:
                            tf.write(zf.read())
                    
                    # Lire et convertir
                    with h5py.File(temp_hdf5, 'r') as hf:
                        data = hf['dataset'][:]
                    
                    convert_hdf5_to_png(data, modality, output_path)
                    
                    # Supprimer le fichier temporaire
                    temp_hdf5.unlink()
                else:
                    # Fichier JPG/PNG - copier directement
                    with z.open(file_path_in_zip) as zf:
                        # Si c'est un JPG, le convertir en PNG
                        if file_path_in_zip.endswith('.jpg'):
                            import PIL.Image
                            img = PIL.Image.open(zf)
                            img.save(output_path)
                        else:
                            with open(output_path, 'wb') as of:
                                of.write(zf.read())
                if verbose:
                    print(f"  ✓ {modality}: {output_filename}")
                downloaded_count += 1
                
            except Exception as e:
                print(f"  ✗ Erreur {modality}: {e}")
                continue
        
        return downloaded_count > 0
        
    except Exception as e:
        print(f"  ✗ Erreur lors du téléchargement: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Télécharge un échantillon diversifié du dataset Hypersim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    # Télécharger 100 images avec depth, semantic et normal
    python download_dataset_sample.py --num_images 100 --modalities depth semantic normal --repo_path /path/to/ml-hypersim --output_dir my_dataset --seed 123
    
    # Télécharger 50 images avec toutes les modalités
    python download_dataset_sample.py --num_images 50 --modalities all_modalities --repo_path /path/to/ml-hypersim --output_dir my_dataset --seed 123
    
"""
    )
    
    parser.add_argument('--num_images', type=int, required=True,
                        help='Nombre d\'images à télécharger')
    parser.add_argument('--modalities', nargs='+', required=True,
                        help='Liste des modalités à télécharger ou "all_modalities"')
    parser.add_argument('--output_dir', type=str, default='hypersim_sample',
                        help='Répertoire de sortie (défaut: hypersim_sample)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed pour reproductibilité (défaut: 42)')
    parser.add_argument('--repo_path', type=str, default=None,
                        help='Chemin vers le repo ml-hypersim (défaut: chemin relatif depuis contrib/99991)')
    
    args = parser.parse_args()
    
    # Traiter les modalités
    if 'all_modalities' in args.modalities:
        modalities = list(MODALITY_MAPPINGS.keys())
    else:
        modalities = args.modalities
        # Vérifier que toutes les modalités sont valides
        invalid = [m for m in modalities if m not in MODALITY_MAPPINGS]
        if invalid:
            print(f"❌ Modalités invalides: {invalid}")
            print(f"Modalités disponibles: {list(MODALITY_MAPPINGS.keys())}")
            return 1
    
    output_dir = Path(args.output_dir)
    temp_dir = output_dir / '_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TÉLÉCHARGEMENT D'UN ÉCHANTILLON DU DATASET HYPERSIM")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"  Nombre d'images: {args.num_images}")
    print(f"  Modalités: {', '.join(modalities)}")
    print(f"  Répertoire de sortie: {output_dir}")
    print(f"  Seed: {args.seed}")
    
    # Charger les métadonnées des scènes
    print(f"\n📚 Chargement des métadonnées...")
    scene_types = load_scene_metadata(repo_path=args.repo_path)
    print(f"  Types de scènes chargés: {len(scene_types)}")
    
    # Planifier les téléchargements
    print(f"\n📋 Planification des téléchargements...")
    plan = plan_download(args.num_images, seed=args.seed, repo_path=args.repo_path)
    print(f"  Images planifiées: {len(plan)}")
    
    if len(plan) < 20:
        print("plan :")
        print(plan)
    # Grouper par scène pour optimiser les téléchargements
    scenes_to_download = defaultdict(list)
    for scene_name, camera_name, frame_id in plan:
        scenes_to_download[scene_name].append((camera_name, frame_id))
    
    print(f"  Scènes différentes: {len(scenes_to_download)}")
    
    # Créer une session pour réutiliser les connexions
    session = requests.session()
    
    # Télécharger
    total_downloaded = 0
    total_failed = 0
    
    total_frames = sum(len(frames) for frames in scenes_to_download.values())
    with tqdm(total=total_frames, desc="Téléchargement des images") as pbar:
        for scene_name, frames in scenes_to_download.items():
            # Trouver l'URL correspondante
            url = f"{BASE_URL}{scene_name}.zip"
            
            # Télécharger toutes les frames de cette scène
            for camera_name, frame_id in frames:
                success = download_and_convert(
                    session, url, scene_name, camera_name, frame_id,
                    modalities, output_dir, scene_types, temp_dir, verbose=False
                )
                if success:
                    total_downloaded += 1
                else:
                    total_failed += 1
                pbar.update(1)
    
    # Nettoyer le répertoire temporaire
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Résumé
    print("\n" + "="*70)
    print("TÉLÉCHARGEMENT TERMINÉ")
    print("="*70)
    print(f"\n✓ Images téléchargées avec succès: {total_downloaded}/{args.num_images}")
    if total_failed > 0:
        print(f"✗ Échecs: {total_failed}")
    print(f"\n📁 Répertoire de sortie: {output_dir.absolute()}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
