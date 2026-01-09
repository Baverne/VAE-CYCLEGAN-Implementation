# VAE-CYCLEGAN-Implementation

## Development Steps

| Step | Task | Status |
|------|------|--------|
| 0 |**Paper Study** | |
| 0.1 | Download the [paper](https://openreview.net/pdf?id=fu0NN8GRQ7) |✅|
| 0.2 | Determine Networks Architectures | ✅ |
| 1 | **Implement Networks** | |
| 1.1 | AE (Autoencoder) | ✅ |
| 1.2 | AE - GAN (Generative Adversarial Network) | ✅ |
| 1.3 | Cycle - AE | ✅ |
| 1.4 | AE - CycleGAN | ✅ |
| 1.5 | VAE (Variational Autoencoder) | ✅ |
| 1.6 | VAE-GAN | ✅ |
| 1.7 | Cycle - VAE | ✅ |
| 1.8 | VAE-CycleGAN | ✅ |
| 2 | **Constitute the Datasets** | |
| 2.1 | Constitute a paired dataset (work for all architectures) | ✅ |
| 2.2 | Constitute an unpaired dataset (face to face, day to night, rainy to sunny ... ) | ✅ |
| 3 | **Training** | |
| 3.1 | Train AE | ⬜ |
| 3.2 | Train AE - GAN | ⬜ |
| 3.3 | Train Cycle - AE | ⬜ |
| 3.4 | Train AE - CycleGAN | ⬜ |
| 3.5 | Train VAE | ⬜ |
| 3.6 | Train VAE-GAN | ⬜ |
| 3.7 | Train Cycle - VAE | ⬜ |
| 3.8 | Train VAE-CycleGAN | ⬜ |
| 4 | **Create the latex summary** | ⬜ |
| 4.1 | Introduction (Paper, Motivation, problem statement, related work...) | ⬜ |
| 4.2 | Methodology (Architectures, Loss functions, Datasets, Training details, Decisions we made ...) | ⬜ |
| 4.3 | Experiments and Results (Quantitative and Qualitative results, Analysis ...) | ⬜ |
| 4.4 | Conclusion and Future Work | ⬜ |
| 5 | **Extensions** | |
| 5.1 | Domain Adaptation : Since it's unpaired, can we use real life X with synthetic Y? | ⬜ |
| 5.2 | Transfer Learning on New Translation Task ? | ⬜ |

## Paired Dataset

We use the **Hypersim Dataset**, a synthetic dataset that provides photorealistic images with extra modalities for various environments.

### Downloading the Dataset

The full Hypersim dataset is quite large (~1.9 TB). For practical purposes, we provide a script to download a smaller, diverse sample of the dataset.
The dataset can be downloaded using the provided script that samples diverse images from the full Hypersim dataset:

```powershell
python Dataset\download_dataset_sample.py --num_images 5000 --modalities all_modalities --repo_path path\to\ml-hypersim --output_dir datasets\paired --seed 42
```

**Prerequisites:**
1. Clone the ml-hypersim repository:
   ```powershell
   git clone https://github.com/apple/ml-hypersim
   ```

2. Install required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```



## Training


### Training the Auto-Encoder

The training script supports multiple network architectures. For now, it only features the Auto-Encoder (ae) and Variational Auto-Encoder (vae) and AE-GAN (aegan).

**Basic usage:**
```powershell
python train.py --architecture ae --source_modality depth --target_modality depth --epochs 10 --save_freq 5 --log_image_freq 2
```

**Main Training options:**

- `--architecture`: Choose the network architecture (`ae`, `vae`,`vae_gan`, etc.)
- `--source_modality`: Input modality (e.g., `depth`, `normal`, `semantic`, `color`)
- `--target_modality`: Target modality (e.g., `normal`, `depth`, `semantic`, `color`)
- `--data_dir`: Path to the dataset directory (default: `dataset`)
- `--batch_size`: Batch size for training (default: `4`)
- `--epochs`: Number of training epochs (default: `100`)
- `--lr`: Learning rate (default: `0.0002`)
- `--image_size`: Image size for training (default: `256`)
- `--val_split`: Validation split ratio (default: `0.1`)
- `--output_dir`: Directory to save models and logs (default: `output`)
- `--save_freq`: Save checkpoint every N epochs (default: `10`)
- `--log_image_freq`: Log images to TensorBoard every N epochs (default: `5`)
- `--resume`: Path to checkpoint to resume training

**Example commands:**

Train Auto-Encoder (depth → normal):
```powershell
python train.py --architecture autoencoder --source_modality depth --target_modality normal --batch_size 8 --epochs 100
```

Train VAE (depth → normal):
```powershell
python train.py --architecture vae --source_modality depth --target_modality normal --latent_dim 1024 --lambda_kl 1e-5 --epochs 100
```

Resume training from checkpoint:
```powershell
python train.py --architecture autoencoder --resume output/autoencoder_20231228_120000/checkpoint_epoch_50.pth
```

### Output Structure

Training outputs are saved in the `output` directory with the following structure:
```
output/
  autoencoder_20231228_120000/
    args.json                 # Training arguments
    checkpoint_epoch_10.pth   # Periodic checkpoints
    checkpoint_epoch_20.pth
    best_model.pth           # Best model based on validation loss
    tensorboard/             # TensorBoard logs
```

## Aknowledgements

### Dataset Source

The Hypersim Dataset is publicly available at: https://github.com/apple/ml-hypersim

### Citation

```bibtex
@inproceedings{roberts:2021,
    author    = {Mike Roberts AND Jason Ramapuram AND Anurag Ranjan AND Atulit Kumar AND
                 Miguel Angel Bautista AND Nathan Paczan AND Russ Webb AND Joshua M. Susskind},
    title     = {{Hypersim}: {A} Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding},
    booktitle = {International Conference on Computer Vision (ICCV) 2021},
    year      = {2021}
}

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```

### Accessing the Unpaired Dataset (Summer2Winter Yosemite)

For unpaired image-to-image translation tasks (e.g., season transfer), we use [Summer2Winter Yosemite dataset](https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite?resource=download) available on Kaggle.

**Steps to access and use the dataset:**
1. Go to the [Summer2Winter Yosemite Kaggle page](https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite?resource=download).
2. Download the dataset (requires a Kaggle account).
3. Extract the contents and place them in the `datasets/unpaired/` directory:
   ```bash
   unzip summer2winter-yosemite.zip -d datasets/unpaired/
   ```
4. The dataset contains two folders: `summer` and `winter`, each with unpaired images for the respective seasons.
5. Use these folders for unpaired translation experiments (e.g., training CycleGAN).

**Note:**
- If you use Kaggle's API, you can download directly with:
  ```bash
  kaggle datasets download -d balraj98/summer2winter-yosemite -p datasets/unpaired/
  unzip datasets/unpaired/summer2winter-yosemite.zip -d datasets/unpaired/
  ```
- Make sure you have the [Kaggle API](https://github.com/Kaggle/kaggle-api) installed and configured.

For more details, see the dataset page on Kaggle.