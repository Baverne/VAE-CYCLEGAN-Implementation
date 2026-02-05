"""
All necessary networks for VAE-CygleGAN implementation

The Networks.py is separated in three sections:
1. Atomic Network Components: Basic building blocks like convolutional layers, residual blocks, upsampling/downsampling blocks, etc.
2. Molecular Networks: Compositions of atomic components to form encoders, decoders, discriminators, etc.
3. Network Composites: Higher-level architectures combining molecular networks for specific tasks like Autoencoder, Variational Autoencoder, AEGAN, VAEGAN, CycleAE, CycleVAE, CycleAEGAN, and CycleVAEGAN.

All networkds in this third are designed like this :
class Network_name (nn.Module):
    def __init__(self):
        super(Network_name, self).__init__()

    def forward(self, x):
        Forward pass of the network
        All Networks expect Gx as their first output
    
    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        Configure optimizer for training

    def save_optimizer_states(self):
        Save optimizer states for checkpointing
    
    def load_optimizer_states(self, states):
        Load optimizer states from checkpoint
    
    def configure_loss(self, **kwargs):
        Configure loss functions (ignores unused kwargs)
    
    def training_step(self, batch):
        Training step for Autoencoder
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
    
    def validation_step(self, batch):
        Validation step for Autoencoder
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output

Theses extra methods allow each network to have its own training and validation procedures, making the training script more modular and adaptable to different architectures.
"""

import numpy as np
import torch
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from Losses import *

### ATOMIC NETWORK COMPONENTS ###

class CaSb (nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=3, activation="ReLU", use_norm = True):
        super(CaSb, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect')
        self.norm = nn.InstanceNorm2d(out_channels)
        if activation == "ReLU":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "Identity":
            self.activation = nn.Identity()
        else :
            raise NotImplementedError("Activation not implemented")
        self.use_norm = use_norm

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.activation(x)
        return x

class D (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(D, self).__init__()
        self.PixelUnshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.conv = nn.Conv2d(in_channels*4, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.PixelUnshuffle(x)
        x = self.conv(x)
        x = self.activation(x)  # ReLU BEFORE InstanceNorm
        x = self.norm(x)
        return x

class R (nn.Module):
    def __init__(self, out_channels):
        super(R, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.activation1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.norm2 = nn.InstanceNorm2d(out_channels)
        #self.activation2 = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activation1(x)  # ReLU BEFORE InstanceNorm
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return x

class U (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U, self).__init__()
        self.PixelShuffle = nn.PixelShuffle(upscale_factor=2)
        self.conv = nn.Conv2d(in_channels//4, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.PixelShuffle(x)
        x = self.conv(x)
        x = self.activation(x)  # ReLU BEFORE InstanceNorm
        x = self.norm(x)
        return x

class S (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(S, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = self.conv(x)
        return x
    
class L (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = self.conv(x)
        return x


### MOLECULAR NETWORKS ###

class Encoder (nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        layers = []
        layers.append(CaSb(3, 64, kernel_size=7, stride=1))
        layers.append(D(64, 128))
        layers.append(D(128, 256))
        layers.append(D(256, 512))
        layers.append(D(512, 1024))
        layers.append(R(1024))
        self.model = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization for ReLU networks"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.model(x)

class Decoder (nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        layers = []
        layers.append(R(1024))
        layers.append(U(1024, 512))
        layers.append(U(512, 256))
        layers.append(U(256, 128))
        layers.append(U(128, 64))
        layers.append(CaSb(64, 3, kernel_size=7, stride=1, activation="Identity", use_norm=False)) # No norm, Identity activation
        self.model = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def forward(self, x):
        out = self.model(x)
        return out
    
    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization for ReLU networks"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class VariationalEncoderBlock (nn.Module):
    def __init__(self, in_channels, latent_dim=64):
        super(VariationalEncoderBlock, self).__init__()
        self.muConv = L(in_channels, latent_dim)
        self.logvarConv = nn.Sequential(S(in_channels, latent_dim), S(latent_dim, latent_dim))
    def forward(self, x):
        mu = self.muConv(x)
        logvar = self.logvarConv(x)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class VariationalDecoderBlock (nn.Module):
    def __init__(self, latent_dim=64, out_channels=1024):
        super(VariationalDecoderBlock, self).__init__()
        self.conv = S(latent_dim, out_channels)

    def forward(self, z):
        x_recon = self.conv(z)
        return x_recon


class Discriminator (nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(CaSb(3, 64, kernel_size=4, stride=2, padding=1, activation="LeakyReLU", use_norm=False))
        layers.append(CaSb(64, 128, kernel_size=4, stride=2, padding=1, activation="LeakyReLU"))
        layers.append(CaSb(128, 256, kernel_size=4, stride=2, padding=1, activation="LeakyReLU"))
        layers.append(CaSb(256, 512, kernel_size=4, stride=2, padding=1, activation="LeakyReLU"))
        layers.append(spectral_norm(nn.Conv2d(512, 1, kernel_size=16, stride=1, padding=0)))  # On your figure you put 2 as output channels whereas here it's 1 ?

        self.model = nn.Sequential(*layers)

        # Apply proper weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization for LeakyReLU networks"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        #shape (B, 1, 1, 1) to (B,1)
        return self.model(x).view(-1, 1).squeeze(1) # Then we should not squeeze and have shape (B x 1 x 1 x 2) ?
    # i guess we can squeeze to have (B x 2) for real/fake classification but also keep 1 channel that will be 0 for fake and 1 for real ?
    # Don't know what's the best option 
    
### NETWORKS COMPOSITES ###
### Paired Dataset Networks ###

class Autoencoder (nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # architecture related attributes
        self.encoder = Encoder()
        self.decoder = Decoder()

        # loss related attributes to be set in "configure_loss"
        self.optimizer = None
        self.loss_fn = None

        # Apply proper weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization for ReLU networks"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        z = self.encoder(x)
        Gx = self.decoder(z)
        return Gx # All Network expect Gx as their first output
    
    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999), decoder_only=False):
        """Configure optimizer for training. If decoder_only, only optimize decoder."""
        if decoder_only:
            self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr, betas=betas)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return self.optimizer
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        return {'optimizer': self.optimizer.state_dict()}
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])
        else :
            raise KeyError("optimizer state not found in states")

    def configure_loss(self, **kwargs):
        """Configure loss functions (ignores unused kwargs)"""
        self.loss_fn = TranslationLoss()

    def training_step(self, batch):
        """
        Training step for Autoencoder
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if self.loss_fn is None:
            raise ValueError("Loss function has not been configured yet.")
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")

        x = batch['x']
        y = batch['y']

        # Forward pass
        output = self(x)

        # Compute loss
        loss_trans = self.loss_fn(output, y)

        # Check for NaN/Inf before backward
        if torch.isnan(loss_trans) or torch.isinf(loss_trans):
            print("NaN or Inf detected in loss during training step. Printing the actual weight of the network for debugging:")
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(f"Parameter: {name}, Value: {param.data}")
            print("printing all the computation graph values for debugging:")
            self.optimizer.zero_grad()
            for p in self.parameters():
                if p.grad is not None:
                    print(p.grad)
            return {
                'nan_detected': True,
                'G_loss': float('nan'),
                'loss_trans': float('nan'),
                'total_loss': float('nan')
            }

        # Backward pass
        self.optimizer.zero_grad()
        loss_trans.backward()
        self.optimizer.step()

        # Return metrics
        return {
            'G_loss': loss_trans.item(),
            'loss_trans': loss_trans.item(),
            'total_loss': loss_trans.item()
        }

    def validation_step(self, batch):
        """
        Validation step for Autoencoder
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if self.loss_fn is None:
            raise ValueError("Loss function has not been configured yet.")

        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            output = self(x)
            
            # Compute loss
            loss_trans = self.loss_fn(output, y)
            
            # Return metrics
            return {
                'G_loss': loss_trans.item(),
                'total_loss': loss_trans.item(),
                'loss_trans': loss_trans.item(),
                'Gx': output  # Gx = G(x) for visualization
            }

class DoubleAutoencoder(nn.Module):
    """
    Double Autoencoder with shared encoder and two separate decoders.
    
    This architecture is designed for pretraining before CycleAE:
    - One shared encoder learns a common latent representation
    - Decoder A reconstructs modality A (source)
    - Decoder B reconstructs modality B (target)
    
    After training, the decoders can be swapped to create translation networks:
    - G (A→B) = Encoder + Decoder_B
    - F (B→A) = Encoder + Decoder_A
    
    The batch should contain:
    - 'x': images from modality A (source)
    - 'y': images from modality B (target)
    
    Both are encoded with the shared encoder and reconstructed with their respective decoders.
    """
    def __init__(self):
        super(DoubleAutoencoder, self).__init__()
        # Shared encoder
        self.encoder = Encoder()
        # Two separate decoders for each modality
        self.decoder_A = Decoder()  # Reconstructs source modality
        self.decoder_B = Decoder()  # Reconstructs target modality

        # loss related attributes to be set in "configure_loss"
        self.optimizer = None
        self.loss_fn = None

    def forward(self, x, y):
        """
        Forward pass for both modalities.
        
        Args:
            x: Input from modality A (source)
            y: Input from modality B (target)
        
        Returns:
            Gx: Reconstruction of x (x -> encoder -> decoder_A -> Gx)
            Gy: Reconstruction of y (y -> encoder -> decoder_B -> Gy)
        """
        # Encode both modalities with shared encoder
        z_x = self.encoder(x)
        z_y = self.encoder(y)
        
        # Decode with respective decoders
        Gx = self.decoder_A(z_x)  # Reconstruct source
        Gy = self.decoder_B(z_y)  # Reconstruct target
        
        return Gx, Gy  # First output is Gx for compatibility
    
    def translate_A_to_B(self, x):
        """Translate from modality A to B (uses decoder_B)"""
        z = self.encoder(x)
        return self.decoder_B(z)
    
    def translate_B_to_A(self, y):
        """Translate from modality B to A (uses decoder_A)"""
        z = self.encoder(y)
        return self.decoder_A(z)
    
    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizer for training"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return self.optimizer
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        return {'optimizer': self.optimizer.state_dict()}
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])
        else:
            raise KeyError("optimizer state not found in states")
    
    def configure_loss(self, **kwargs):
        """Configure loss functions (ignores unused kwargs)"""
        self.loss_fn = TranslationLoss()
    
    def training_step(self, batch):
        """
        Training step for DoubleAutoencoder.
        
        Args:
            batch: dict with 'x' (source modality) and 'y' (target modality)
        
        Returns:
            dict with loss metrics
        """
        if self.loss_fn is None:
            raise ValueError("Loss function has not been configured yet.")
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")

        x = batch['x']  # Source modality
        y = batch['y']  # Target modality
        
        # Forward pass - reconstruct both modalities
        Gx, Gy = self(x, y)

        # Compute reconstruction losses for both modalities
        loss_recon_A = self.loss_fn(Gx, x)  # Reconstruction of source
        loss_recon_B = self.loss_fn(Gy, y)  # Reconstruction of target
        
        # Total loss is sum of both reconstruction losses
        total_loss = loss_recon_A + loss_recon_B

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return metrics
        return {
            'G_loss': total_loss.item(),
            'loss_recon_A': loss_recon_A.item(),
            'loss_recon_B': loss_recon_B.item(),
            'total_loss': total_loss.item()
        }

    def validation_step(self, batch):
        """
        Validation step for DoubleAutoencoder.
        
        Args:
            batch: dict with 'x' (source modality) and 'y' (target modality)
        
        Returns:
            dict with loss metrics and output
        """
        if self.loss_fn is None:
            raise ValueError("Loss function has not been configured yet.")

        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            Gx, Gy = self(x, y)
            
            # Compute losses
            loss_recon_A = self.loss_fn(Gx, x)
            loss_recon_B = self.loss_fn(Gy, y)
            total_loss = loss_recon_A + loss_recon_B
            
            # Return metrics
            return {
                'G_loss': total_loss.item(),
                'total_loss': total_loss.item(),
                'loss_recon_A': loss_recon_A.item(),
                'loss_recon_B': loss_recon_B.item(),
                'Gx': Gx,  # Reconstruction of source modality
                'Fy': Gy   # Reconstruction of target modality
            }
    
    def create_cycle_ae(self):
        """
        Create a CycleAE from this pretrained DoubleAutoencoder.
        
        The resulting CycleAE will have:
        - G (A→B): shared encoder + decoder_B
        - F (B→A): shared encoder + decoder_A
        
        Note: Both G and F share the same encoder weights.
        
        Returns:
            CycleAE: A new CycleAE with weights initialized from this model
        """
        cycle_ae = CycleAE()
        
        # G translates A→B: encoder + decoder_B
        cycle_ae.G.encoder.load_state_dict(self.encoder.state_dict())
        cycle_ae.G.decoder.load_state_dict(self.decoder_B.state_dict())
        
        # F translates B→A: encoder + decoder_A  
        cycle_ae.F.encoder.load_state_dict(self.encoder.state_dict())
        cycle_ae.F.decoder.load_state_dict(self.decoder_A.state_dict())
        
        return cycle_ae

class VariationalAutoencoder (nn.Module):
    def __init__(self, latent_dim=64):
        super(VariationalAutoencoder, self).__init__()
        # architecture related attributes
        self.encoder = Encoder()
        self.variational_encoder_block = VariationalEncoderBlock(in_channels=1024, latent_dim=latent_dim)
        self.variational_decoder_block = VariationalDecoderBlock(latent_dim=latent_dim, out_channels=1024)
        self.decoder = Decoder()

        # loss related attributes to be set in "configure_loss"
        self.optimizer = None
        self.loss_trans_fn = None
        self.loss_kl_fn = None
        self.lambda_kl = 0

        # Apply proper weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization for ReLU networks"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        z, mu, logvar = self.variational_encoder_block(encoded)
        decoded_latent = self.variational_decoder_block(z)
        Gx = self.decoder(decoded_latent)
        return Gx, mu, logvar # All Netework expect Gx as their first output

    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizer for training"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return self.optimizer

    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        return {'optimizer': self.optimizer.state_dict()}

    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])
        else :
            raise KeyError("optimizer state not found in states")

    def configure_loss(self, **kwargs):
        """Configure loss functions"""
        self.loss_trans_fn = TranslationLoss()
        self.loss_kl_fn = KLDivergenceLoss()
        self.lambda_kl = kwargs.get('lambda_kl', 1e-5)

    def training_step(self, batch):
        """
        Training step for VAE
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if self.loss_trans_fn is None:
            raise ValueError("Translation loss function has not been configured yet.")
        if self.loss_kl_fn is None:
            raise ValueError("KL divergence loss function has not been configured yet.")
        x = batch['x']
        y = batch['y']

        # Forward pass
        output, mu, logvar = self(x)

        # Compute losses
        loss_trans = self.loss_trans_fn(output, y)
        loss_kl = self.loss_kl_fn(mu, logvar)
        G_loss = loss_trans + self.lambda_kl * loss_kl

        # Backward pass
        self.optimizer.zero_grad()
        G_loss.backward()
        self.optimizer.step()

        # Return metrics
        return {
            'G_loss': G_loss.item(),
            'loss_trans': loss_trans.item(),
            'loss_kl': loss_kl.item()
        }

    def validation_step(self, batch):
        """
        Validation step for VAE
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if self.loss_trans_fn is None:
            raise ValueError("Translation loss function has not been configured yet.")
        if self.loss_kl_fn is None:
            raise ValueError("KL divergence loss function has not been configured yet.")

        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            output, mu, logvar = self(x)
            
            # Compute losses
            loss_trans = self.loss_trans_fn(output, y)
            loss_kl = self.loss_kl_fn(mu, logvar)
            total_loss = loss_trans + self.lambda_kl * loss_kl
            
            # Return metrics
            return {
                'G_loss': total_loss.item(),
                'loss_trans': loss_trans.item(),
                'loss_kl': loss_kl.item(),
                'Gx': output  # Gx = G(x) for visualization
            }

    
class AEGAN (nn.Module):
    def __init__(self):
        super(AEGAN, self).__init__()
        # architecture related attributes
        self.G = Autoencoder()
        self.D = Discriminator()

        # Apply weight initialization (DCGAN style)
        self.apply(self._init_weights_)

        # loss related attributes to be set in "configure_loss"
        self.optimizer_G = None
        self.optimizer_D = None
        self.loss_trans_fn = None
        self.loss_gan_gen_fn = None
        self.loss_gan_disc_fn = None
        self.loss_identity_fn = None
        self.lambda_gan = 0
        self.lambda_identity = 0

    def _init_weights_(self, module):
        """Initialize weights using Kaiming initialization for ReLU networks"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, y):
        Gx = self.G(x)
        Gy = self.G(y)
        DGx = self.D(Gx)
        Dy = self.D(y)
        return Gx, Gy, DGx, Dy

    def configure_optimizers(self, lr=2e-4, betas=(0.5, 0.999)):
        """Configure optimizers for Generator and Discriminator"""
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=betas)
        return self.optimizer_G, self.optimizer_D
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")
        return {
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict()
        }
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")
        if 'optimizer_G' in states:
            self.optimizer_G.load_state_dict(states['optimizer_G'])
        else :
            raise KeyError("optimizer_G state not found in states")
        if 'optimizer_D' in states:
            self.optimizer_D.load_state_dict(states['optimizer_D'])
        else :
            raise KeyError("optimizer_D state not found in states")

    
    def configure_loss(self, **kwargs):
        """Configure loss functions"""
        self.loss_trans_fn = TranslationLoss()
        self.loss_gan_gen_fn = GANLossGenerator()
        self.loss_gan_disc_fn = GANLossDiscriminator()
        self.loss_identity_fn = nn.L1Loss()  # Identity loss as L1 loss between Gy and y
        self.lambda_gan = kwargs.get('lambda_gan', 1.0)
        self.lambda_identity = kwargs.get('lambda_identity', 5.0)

    def training_step(self, batch):
        """
        Training step for AEGAN (trains both G and D)
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")
        if self.loss_trans_fn is None:
            raise ValueError("Translation loss function has not been configured yet.")
        if self.loss_gan_gen_fn is None:
            raise ValueError("GAN generator loss function has not been configured yet.")
        if self.loss_gan_disc_fn is None:
            raise ValueError("GAN discriminator loss function has not been configured yet.")
        if self.loss_identity_fn is None:
            raise ValueError("Identity loss function has not been configured yet.")

        x = batch['x']
        y = batch['y']

        ### Train Generator
        self.optimizer_G.zero_grad()
        
        # Forward pass for generator training
        Gx, Gy, DGx, Dy = self.forward(x, y)
        
        # Compute generator losses
        loss_trans = self.loss_trans_fn(Gx, y)
        loss_gan_g, loss_gan_g_real, loss_gan_g_fake = self.loss_gan_gen_fn(Dy, DGx)
        loss_id = self.loss_identity_fn(Gy, y)
        G_loss = loss_trans + self.lambda_gan * loss_gan_g + self.lambda_identity * loss_id

        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        self.optimizer_G.step()

        ### Train Discriminator
        self.optimizer_D.zero_grad()
        
        # Forward pass for discriminator training (detach generator outputs)
        Gx_detached = Gx.detach()  # Detach the generator output
        DGx_detached = self.D(Gx_detached)
        Dy_detached = self.D(y)
        
        # Compute discriminator loss
        D_loss, D_loss_real, D_loss_fake = self.loss_gan_disc_fn(Dy_detached, DGx_detached)

        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
        self.optimizer_D.step()
        
        # Discriminator statistics (computed after training, detached)
        with torch.no_grad():
            d_y_mean = Dy_detached.mean().item()
            d_gx_mean = DGx_detached.mean().item()

        return {
            'G_loss': G_loss.item(),
            'D_loss': D_loss.item(),
            'D_loss_real': D_loss_real.item(),
            'D_loss_fake': D_loss_fake.item(),
            'loss_trans': loss_trans.item(),
            'loss_gan_g': loss_gan_g.item(),
            'loss_identity': loss_id.item(),
            'd_y_mean': d_y_mean,
            'd_gx_mean': d_gx_mean
        }

    def validation_step(self, batch):
        """
        Validation step for AEGAN
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")
        if self.loss_trans_fn is None:
            raise ValueError("Translation loss function has not been configured yet.")
        if self.loss_gan_gen_fn is None:
            raise ValueError("GAN generator loss function has not been configured yet.")
        if self.loss_gan_disc_fn is None:
            raise ValueError("GAN discriminator loss function has not been configured yet.")
        if self.loss_identity_fn is None:
            raise ValueError("Identity loss function has not been configured yet.")

        with torch.no_grad():
            x = batch['x']
            y = batch['y']
            
            # Forward pass
            Gx, Gy, DGx, Dy = self.forward(x, y)
            
            # Compute losses
            loss_trans = self.loss_trans_fn(Gx, y)
            loss_gan_g, loss_gan_g_real, loss_gan_g_fake = self.loss_gan_gen_fn(Dy, DGx)  # GAN generator loss

            loss_id = self.loss_identity_fn(Gy, y)

            # We only take the loss_gan_g_fake for G_loss
            G_loss = loss_trans + self.lambda_gan * loss_gan_g + self.lambda_identity * loss_id
            D_loss, D_loss_real, D_loss_fake = self.loss_gan_disc_fn(Dy, DGx)

            # Return metrics
            return {
                'total_loss': (G_loss.item() + D_loss.item()),
                'G_loss': G_loss.item(),
                'D_loss': D_loss.item(),
                'D_loss_real': D_loss_real.item(),
                'D_loss_fake': D_loss_fake.item(),
                'loss_trans': loss_trans.item(),
                'loss_gan_g': loss_gan_g.item(),
                'loss_gan_g_real': loss_gan_g_real.item(),
                'loss_gan_g_fake': loss_gan_g_fake.item(),
                'loss_identity': loss_id.item(),
                'Gx': Gx  # Gx = G(x) for visualization
            }


class VAEGAN (nn.Module):

    def __init__(self, latent_dim=64):
        super(VAEGAN, self).__init__()
        self.G = VariationalAutoencoder(latent_dim)
        self.D = Discriminator()
        self.latent_dim = latent_dim
        
        # Debug mode for detailed logging
        self.debug_mode = False
        self.debug_info = {}


    def forward(self, x, y):
        Gx, mu, logvar = self.G(x)  # output of the VAE
        DGx = self.D(Gx)  # Discriminator output for generated data
        Dy = self.D(y)  # Discriminator output for real data
        return Gx, mu, logvar, DGx, Dy # All Netework expect Gx as their first output
    
    def configure_optimizers(self, lr=2e-4, betas=(0.5, 0.999)):
        """Configure optimizers for Generator and Discriminator"""
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=betas)
        return self.optimizer_G, self.optimizer_D
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")
        return {
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict()
        }
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")
        if 'optimizer_G' in states:
            self.optimizer_G.load_state_dict(states['optimizer_G'])
        else :
            raise KeyError("optimizer_G state not found in states")
        if 'optimizer_D' in states:
            self.optimizer_D.load_state_dict(states['optimizer_D'])
        else :
            raise KeyError("optimizer_D state not found in states")
    
    def configure_loss(self, **kwargs):
        """Configure loss functions"""
        self.translation_loss = TranslationLoss()
        self.gan_loss_gen = GANLossGenerator()
        self.gan_loss_disc = GANLossDiscriminator()
        self.identity_loss = IdentityLoss()
        self.kl_loss = KLDivergenceLoss()
        self.lambda_gan = kwargs.get('lambda_gan', 1.0)
        self.lambda_identity = kwargs.get('lambda_identity', 5.0)
        self.lambda_kl = kwargs.get('lambda_kl', 1e-5)

    def enable_debug_mode(self, enabled=True):
        """Enable/disable debug mode for detailed TensorBoard logging"""
        self.debug_mode = enabled

    def training_step(self, batch):
        """
        Training step for VAE-GAN (trains both G and D)
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")

        x = batch['x']
        y = batch['y']

        # Forward pass
        Gx, mu, logvar, DGx, Dy = self(x, y)

        # Generator losses
        loss_trans = self.translation_loss(Gx, y)
        total_loss_gan, loss_gan_real, loss_gan_fake = self.gan_loss_gen(Dy, DGx)
        loss_id = self.identity_loss(x, y, Gx, y)
        loss_kl = self.kl_loss(mu, logvar)
        G_loss = (loss_trans + self.lambda_gan * total_loss_gan +
                   self.lambda_identity * loss_id + self.lambda_kl * loss_kl)

        # Discriminator loss
        total_loss_gan_disc, loss_gan_real_disc, loss_gan_fake_disc = self.gan_loss_disc(Dy, DGx.detach())
        ### Train Generator
        self.optimizer_G.zero_grad()
        G_loss.backward(retain_graph=True)
        self.optimizer_G.step()

        ### Train Discriminator
        self.optimizer_D.zero_grad()
        total_loss_gan_disc.backward()
        self.optimizer_D.step()

        # Return metrics
        metrics = {
            'G_loss': G_loss.item(),
            'D_loss': total_loss_gan_disc.item(),
            'loss_gan_disc_real': loss_gan_real_disc.item(),
            'loss_gan_disc_fake': loss_gan_fake_disc.item(),
            'loss_trans': loss_trans.item(),
            'loss_gan_real': loss_gan_real.item(),
            'loss_gan_fake': loss_gan_fake.item(),
            'loss_identity': loss_id.item(),
            'loss_kl': loss_kl.item(),
        }
        
        # Only include debug_info if debug mode is enabled
        if self.debug_mode:
            metrics['debug_info'] = self.debug_info
            
        return metrics

    def validation_step(self, batch):
        """
        Validation step for VAE-GAN
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")

        with torch.no_grad():
            x = batch['x']
            y = batch['y']
            
            # Forward pass
            Gx, mu, logvar, DGx, Dx = self(x, y)
            
            # Compute losses
            loss_trans = self.translation_loss(Gx, y)
            total_loss_gan, loss_gan_real, loss_gan_fake = self.gan_loss_gen(Dx, DGx)
            loss_id = self.identity_loss(x, y, Gx, y)
            loss_kl = self.kl_loss(mu, logvar)
            G_loss = (loss_trans + self.lambda_gan * total_loss_gan +
                    self.lambda_identity * loss_id + self.lambda_kl * loss_kl)
            total_loss_gan_disc, loss_gan_real_disc, loss_gan_fake_disc = self.gan_loss_disc(Dx, DGx)
            # Return metrics
            return {
                'total_loss': (G_loss.item() + total_loss_gan_disc.item()),
                'G_loss': G_loss.item(),
                'D_loss': total_loss_gan_disc.item(),
                'loss_trans': loss_trans.item(),
                'loss_gan_real': loss_gan_real.item(),
                'loss_gan_fake': loss_gan_fake.item(),
                'loss_identity': loss_id.item(),
                'loss_kl': loss_kl.item(),
                'Gx': Gx  # Gx = G(x) for visualization
            }


class CycleAE (nn.Module):
    def __init__(self):
        super(CycleAE, self).__init__()
        self.F = Autoencoder()
        self.G = Autoencoder()
        
        # loss related attributes to be set in "configure_loss"
        self.optimizer = None
        self.loss_cycle = None
        self.loss_trans = None
        self.lambda_cycle = 0 
    
    def forward(self, x, y):
        Gx = self.G(x)
        FGx = self.F(Gx)
        Fy = self.F(y)
        GFy = self.G(Fy)
        return Gx, FGx, Fy, GFy # All Netework expect Gx as their first output
    
    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizer for training"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return self.optimizer
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        return {'optimizer': self.optimizer.state_dict()}
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])
        else :
            raise KeyError("optimizer state not found in states")
    
    def configure_loss(self, **kwargs):
        """Configure loss functions (ignores unused kwargs)"""
        self.loss_cycle = CycleConsistencyLoss()
        self.loss_trans = TranslationLoss()
        self.lambda_cycle = kwargs.get('lambda_cycle', 10.0)

    def training_step(self, batch):
        """
        Training step for CycleAE
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if self.loss_cycle is None or self.loss_trans is None:
            raise ValueError("Loss functions have not been configured yet.")
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        x = batch['x']
        y = batch['y']

        # Forward pass
        Gx, FGx, Fy, GFy = self(x, y)

        # Compute losses (FIXED: correct argument order for loss_cycle)
        loss_cycle = self.loss_cycle(x, y, FGx, GFy)
        loss_trans = self.loss_trans(Gx, y) + self.loss_trans(Fy, x)
        total_loss = self.lambda_cycle * loss_cycle + loss_trans

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_trans': loss_trans.item(),
            'G_loss': total_loss.item()
        }

    def validation_step(self, batch):
        """
        Validation step for CycleAE
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if self.loss_cycle is None or self.loss_trans is None:
            raise ValueError("Loss functions have not been configured yet.")
        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            Gx, FGx, Fy, GFy = self(x, y)

            # Compute losses (FIXED: correct argument order for loss_cycle)
            loss_cycle = self.loss_cycle(x, y, FGx, GFy)
            loss_trans = self.loss_trans(Gx, y) + self.loss_trans(Fy, x)
            # Use lambda_cycle for consistency with training
            total_loss = self.lambda_cycle * loss_cycle + loss_trans

            # Return metrics
            return {
                'total_loss': total_loss.item(),
                'loss_cycle': loss_cycle.item(),
                'loss_trans': loss_trans.item(),
                'G_loss': total_loss.item(),
                'Gx': Gx.detach(),  # Gx = G(x) translation A->B
                'Fy': Fy.detach()   # Fy = F(y) translation B->A
            }


class CycleVAE (nn.Module):
    def __init__(self, latent_dim=64):
        super(CycleVAE, self).__init__()
        self.F = VariationalAutoencoder(latent_dim)
        self.G = VariationalAutoencoder(latent_dim)

    def forward(self, x, y):
        Gx, mu_x, logvar_x = self.G(x)
        FGx, mu_FGx, logvar_FGx = self.F(Gx)
        Fy, mu_y, logvar_y = self.F(y)
        GFy, mu_GFy, logvar_GFy = self.G(Fy)
        return Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy # All Netework expect Gx as their first output

    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizer for training"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return self.optimizer
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        return {'optimizer': self.optimizer.state_dict()}
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])
        else :
            raise KeyError("optimizer state not found in states")
    
    def configure_loss(self, **kwargs):
        """Configure loss functions"""
        self.loss_cycle = CycleConsistencyLoss()
        self.loss_trans = TranslationLoss()
        self.loss_kl = KLDivergenceLoss()
        self.lambda_kl = kwargs.get('lambda_kl', 1e-5)

    def training_step(self, batch):
        """
        Training step for CycleVAE
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if self.loss_cycle is None or self.loss_trans is None or self.loss_kl is None:
            raise ValueError("Loss functions have not been configured yet.")
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")

        x = batch['x']
        y = batch['y']

        # Forward pass
        Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy = self(x, y)
        # Compute losses (FIXED: correct argument order for loss_cycle)
        loss_cycle = self.loss_cycle(x, y, FGx, GFy)
        loss_trans = self.loss_trans(Gx, y) + self.loss_trans(Fy, x)
        loss_kl = (self.loss_kl(mu_x, logvar_x) + self.loss_kl(mu_FGx, logvar_FGx) +
                    self.loss_kl(mu_y, logvar_y) + self.loss_kl(mu_GFy, logvar_GFy))

        total_loss = loss_cycle + loss_trans + self.lambda_kl * loss_kl

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_trans': loss_trans.item(),
            'loss_kl': loss_kl.item(),
            'G_loss': total_loss.item()
        }

    def validation_step(self, batch):
        """
        Validation step for CycleVAE
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if self.loss_cycle is None or self.loss_trans is None or self.loss_kl is None:
            raise ValueError("Loss functions have not been configured yet.")
        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy = self(x, y)

            # Compute losses (FIXED: correct argument order for loss_cycle)
            loss_cycle = self.loss_cycle(x, y, FGx, GFy)
            loss_trans = self.loss_trans(Gx, y) + self.loss_trans(Fy, x)
            loss_kl = (self.loss_kl(mu_x, logvar_x) + self.loss_kl(mu_FGx, logvar_FGx) +
                        self.loss_kl(mu_y, logvar_y) + self.loss_kl(mu_GFy, logvar_GFy))
            total_loss = loss_cycle + loss_trans + self.lambda_kl * loss_kl

            # Return metrics
            return {
                'total_loss': total_loss.item(),
                'loss_cycle': loss_cycle.item(),
                'loss_trans': loss_trans.item(),
                'loss_kl': loss_kl.item(),
                'G_loss': total_loss.item(),
                'Gx': Gx.detach(),  # Gx = G(x) translation A->B
                'Fy': Fy.detach()   # Fy = F(y) translation B->A
            }


class CycleAEGAN (nn.Module):
    def __init__(self):
        super(CycleAEGAN, self).__init__()
        self.F = Autoencoder()
        self.G = Autoencoder()
        self.DX = Discriminator()
        self.DY = Discriminator()

        # Apply weight initialization
        self.apply(self._init_weights)

        # Debug mode for detailed logging
        self.debug_mode = False
        self.debug_info = {}

        # Optimizer attributes
        self.optimizer_G = None
        self.optimizer_D = None

    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization for ReLU networks"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def enable_debug_mode(self, enabled=True):
        """Enable/disable debug mode for detailed TensorBoard logging"""
        self.debug_mode = enabled

    def forward(self, x, y):
        Gx = self.G(x)
        FGx = self.F(Gx)
        Fy = self.F(y)
        GFy = self.G(Fy)
        DYGx = self.DY(Gx)
        DXFy = self.DX(Fy)
        DXx = self.DX(x)
        DYy = self.DY(y)
        return Gx, FGx, Fy, GFy, DYGx, DXFy, DXx, DYy # All Netework expect Gx as their first output (added DXx, DYy for debug)
    
    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizers for Generators and Discriminators"""
        self.optimizer_G = torch.optim.Adam(
            list(self.F.parameters()) + list(self.G.parameters()),
            lr=lr, betas=betas
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.DX.parameters()) + list(self.DY.parameters()),
            lr=lr, betas=betas
        )
        return self.optimizer_G, self.optimizer_D

    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")
        return {
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict()
        }

    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")
        if 'optimizer_G' in states:
            self.optimizer_G.load_state_dict(states['optimizer_G'])
        else:
            raise KeyError("optimizer_G state not found in states")
        if 'optimizer_D' in states:
            self.optimizer_D.load_state_dict(states['optimizer_D'])
        else:
            raise KeyError("optimizer_D state not found in states")

    def configure_loss(self, **kwargs):
        """Configure loss functions"""
        self.loss_cycle = CycleConsistencyLoss()
        self.loss_gan_gen = GANLossGenerator()
        self.loss_gan_disc = GANLossDiscriminator()
        self.loss_identity = IdentityLoss()
        self.lambda_gan = kwargs.get('lambda_gan', 1.0)
        self.lambda_identity = kwargs.get('lambda_identity', 5.0)
        self.lambda_cycle = kwargs.get('lambda_cycle', 10.0)

    def training_step(self, batch):
        """
        Training step for CycleAEGAN (trains both G and D with alternating updates)
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if (self.loss_cycle is None or self.loss_gan_gen is None or
            self.loss_gan_disc is None or self.loss_identity is None):
            raise ValueError("Loss functions have not been configured yet.")
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")

        x = batch['x']
        y = batch['y']

        ### Train Generators (F and G)
        self.optimizer_G.zero_grad()

        # Forward pass
        Gx, FGx, Fy, GFy, DYGx, DXFy, DXx, DYy = self(x, y)

        # Compute generator losses
        loss_cycle = self.loss_cycle(x, y, FGx, GFy)
        loss_gan_g_x, loss_gan_g_x_real, loss_gan_g_x_fake = self.loss_gan_gen(DXx, DXFy)
        loss_gan_g_y, loss_gan_g_y_real, loss_gan_g_y_fake = self.loss_gan_gen(DYy, DYGx)
        loss_gan_g = loss_gan_g_x + loss_gan_g_y
        loss_identity = self.loss_identity(x, y, FGx, GFy) + self.loss_identity(y, x, GFy, FGx)

        G_loss = (self.lambda_cycle * loss_cycle +
                  self.lambda_gan * loss_gan_g +
                  self.lambda_identity * loss_identity)

        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.F.parameters()) + list(self.G.parameters()), max_norm=1.0
        )
        self.optimizer_G.step()

        ### Train Discriminators (DX and DY)
        self.optimizer_D.zero_grad()

        # Detach generator outputs to prevent gradients flowing to generators
        Gx_detached = Gx.detach()
        Fy_detached = Fy.detach()

        # Re-compute discriminator outputs on detached inputs
        DYGx_detached = self.DY(Gx_detached)
        DXFy_detached = self.DX(Fy_detached)
        DXx_detached = self.DX(x)
        DYy_detached = self.DY(y)

        # Discriminator loss: real should be ~0.9, fake should be ~0.1
        loss_gan_d_x, D_loss_x_real, D_loss_x_fake = self.loss_gan_disc(DXx_detached, DXFy_detached)
        loss_gan_d_y, D_loss_y_real, D_loss_y_fake = self.loss_gan_disc(DYy_detached, DYGx_detached)
        D_loss = loss_gan_d_x + loss_gan_d_y

        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.DX.parameters()) + list(self.DY.parameters()), max_norm=1.0
        )
        self.optimizer_D.step()

        # Discriminator statistics
        with torch.no_grad():
            d_x_real_mean = DXx_detached.mean().item()
            d_x_fake_mean = DXFy_detached.mean().item()
            d_y_real_mean = DYy_detached.mean().item()
            d_y_fake_mean = DYGx_detached.mean().item()

        # Return metrics
        return {
            'total_loss': (G_loss.item() + D_loss.item()),
            'G_loss': G_loss.item(),
            'D_loss': D_loss.item(),
            'D_loss_x_real': D_loss_x_real.item(),
            'D_loss_x_fake': D_loss_x_fake.item(),
            'D_loss_y_real': D_loss_y_real.item(),
            'D_loss_y_fake': D_loss_y_fake.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_gan_g': loss_gan_g.item(),
            'loss_gan_g_x_real': loss_gan_g_x_real.item(),
            'loss_gan_g_x_fake': loss_gan_g_x_fake.item(),
            'loss_gan_g_y_real': loss_gan_g_y_real.item(),
            'loss_gan_g_y_fake': loss_gan_g_y_fake.item(),
            'loss_identity': loss_identity.item(),
            'd_x_real_mean': d_x_real_mean,
            'd_x_fake_mean': d_x_fake_mean,
            'd_y_real_mean': d_y_real_mean,
            'd_y_fake_mean': d_y_fake_mean
        }

    def validation_step(self, batch):
        """
        Validation step for CycleAEGAN
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if (self.loss_cycle is None or self.loss_gan_gen is None or
            self.loss_gan_disc is None or self.loss_identity is None):
            raise ValueError("Loss functions have not been configured yet.")
        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            Gx, FGx, Fy, GFy, DYGx, DXFy, DXx, DYy = self(x, y)

            # Compute generator losses
            loss_cycle = self.loss_cycle(x, y, FGx, GFy)
            loss_gan_g_x, loss_gan_g_x_real, loss_gan_g_x_fake = self.loss_gan_gen(DXx, DXFy)
            loss_gan_g_y, loss_gan_g_y_real, loss_gan_g_y_fake = self.loss_gan_gen(DYy, DYGx)
            loss_gan_g = loss_gan_g_x + loss_gan_g_y
            loss_identity = self.loss_identity(x, y, FGx, GFy) + self.loss_identity(y, x, GFy, FGx)

            G_loss = (self.lambda_cycle * loss_cycle +
                      self.lambda_gan * loss_gan_g +
                      self.lambda_identity * loss_identity)

            # Compute discriminator losses
            loss_gan_d_x, D_loss_x_real, D_loss_x_fake = self.loss_gan_disc(DXx, DXFy)
            loss_gan_d_y, D_loss_y_real, D_loss_y_fake = self.loss_gan_disc(DYy, DYGx)
            D_loss = loss_gan_d_x + loss_gan_d_y

            # Return metrics
            return {
                'total_loss': (G_loss.item() + D_loss.item()),
                'G_loss': G_loss.item(),
                'D_loss': D_loss.item(),
                'D_loss_x_real': D_loss_x_real.item(),
                'D_loss_x_fake': D_loss_x_fake.item(),
                'D_loss_y_real': D_loss_y_real.item(),
                'D_loss_y_fake': D_loss_y_fake.item(),
                'loss_cycle': loss_cycle.item(),
                'loss_gan_g': loss_gan_g.item(),
                'loss_gan_g_x_real': loss_gan_g_x_real.item(),
                'loss_gan_g_x_fake': loss_gan_g_x_fake.item(),
                'loss_gan_g_y_real': loss_gan_g_y_real.item(),
                'loss_gan_g_y_fake': loss_gan_g_y_fake.item(),
                'loss_identity': loss_identity.item(),
                'Gx': Gx.detach(),  # Gx = G(x) translation A->B
                'Fy': Fy.detach()   # Fy = F(y) translation B->A
            }


class CycleVAEGAN (nn.Module):

    def __init__(self, latent_dim=64):
        super(CycleVAEGAN, self).__init__()
        self.F = VariationalAutoencoder(latent_dim)
        self.G = VariationalAutoencoder(latent_dim)
        self.DX = Discriminator()
        self.DY = Discriminator()

        # Apply weight initialization
        self.apply(self._init_weights)

        # Debug mode for detailed logging
        self.debug_mode = False
        self.debug_info = {}

        # Optimizer attributes
        self.optimizer_G = None
        self.optimizer_D = None

    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization for ReLU networks"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def enable_debug_mode(self, enabled=True):
        """Enable/disable debug mode for detailed TensorBoard logging"""
        self.debug_mode = enabled

    def forward(self, x, y):
        Gx, mu_x, logvar_x = self.G(x)
        FGx, mu_FGx, logvar_FGx = self.F(Gx)
        Fy, mu_y, logvar_y = self.F(y)
        GFy, mu_GFy, logvar_GFy = self.G(Fy)
        DYGx = self.DY(Gx)
        DXFy = self.DX(Fy)
        DXx = self.DX(x)
        DYy = self.DY(y)
        return Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy, DYGx, DXFy, DXx, DYy # All Netework expect Gx as their first output
    
    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizers for Generators and Discriminators"""
        self.optimizer_G = torch.optim.Adam(
            list(self.F.parameters()) + list(self.G.parameters()),
            lr=lr, betas=betas
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.DX.parameters()) + list(self.DY.parameters()),
            lr=lr, betas=betas
        )
        return self.optimizer_G, self.optimizer_D

    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")
        return {
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict()
        }

    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")
        if 'optimizer_G' in states:
            self.optimizer_G.load_state_dict(states['optimizer_G'])
        else:
            raise KeyError("optimizer_G state not found in states")
        if 'optimizer_D' in states:
            self.optimizer_D.load_state_dict(states['optimizer_D'])
        else:
            raise KeyError("optimizer_D state not found in states")

    def configure_loss(self, **kwargs):
        """Configure loss functions"""
        self.loss_cycle = CycleConsistencyLoss()
        self.loss_gan_gen = GANLossGenerator()
        self.loss_gan_disc = GANLossDiscriminator()
        self.loss_identity = IdentityLoss()
        self.loss_kl = KLDivergenceLoss()
        self.lambda_gan = kwargs.get('lambda_gan', 1.0)
        self.lambda_identity = kwargs.get('lambda_identity', 5.0)
        self.lambda_cycle = kwargs.get('lambda_cycle', 10.0)
        self.lambda_kl = kwargs.get('lambda_kl', 1e-5)

    def training_step(self, batch):
        """
        Training step for CycleVAEGAN (trains both G and D with alternating updates)
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if (self.loss_cycle is None or self.loss_gan_gen is None or
            self.loss_gan_disc is None or self.loss_identity is None or
            self.loss_kl is None):
            raise ValueError("Loss functions have not been configured yet.")
        if self.optimizer_G is None or self.optimizer_D is None:
            raise ValueError("Optimizers have not been configured yet.")

        x = batch['x']
        y = batch['y']

        ### Train Generators (F and G)
        self.optimizer_G.zero_grad()

        # Forward pass
        (Gx, FGx, Fy, GFy,
         mu_x, logvar_x, mu_FGx, logvar_FGx,
         mu_y, logvar_y, mu_GFy, logvar_GFy,
         DYGx, DXFy, DXx, DYy) = self(x, y)

        # Compute generator losses
        loss_cycle = self.loss_cycle(x, y, FGx, GFy)
        loss_gan_g_x, loss_gan_g_x_real, loss_gan_g_x_fake = self.loss_gan_gen(DXx, DXFy)
        loss_gan_g_y, loss_gan_g_y_real, loss_gan_g_y_fake = self.loss_gan_gen(DYy, DYGx)
        loss_gan_g = loss_gan_g_x + loss_gan_g_y
        loss_identity = (self.loss_identity(x, y, FGx, GFy) +
                         self.loss_identity(y, x, GFy, FGx))
        loss_kl = (self.loss_kl(mu_x, logvar_x) + self.loss_kl(mu_FGx, logvar_FGx) +
                   self.loss_kl(mu_y, logvar_y) + self.loss_kl(mu_GFy, logvar_GFy))

        G_loss = (self.lambda_cycle * loss_cycle +
                  self.lambda_gan * loss_gan_g +
                  self.lambda_identity * loss_identity +
                  self.lambda_kl * loss_kl)

        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.F.parameters()) + list(self.G.parameters()), max_norm=1.0
        )
        self.optimizer_G.step()

        ### Train Discriminators (DX and DY)
        self.optimizer_D.zero_grad()

        # Detach generator outputs to prevent gradients flowing to generators
        Gx_detached = Gx.detach()
        Fy_detached = Fy.detach()

        # Re-compute discriminator outputs on detached inputs
        DYGx_detached = self.DY(Gx_detached)
        DXFy_detached = self.DX(Fy_detached)
        DXx_detached = self.DX(x)
        DYy_detached = self.DY(y)

        # Discriminator loss: real should be ~0.9, fake should be ~0.1
        loss_gan_d_x, D_loss_x_real, D_loss_x_fake = self.loss_gan_disc(DXx_detached, DXFy_detached)
        loss_gan_d_y, D_loss_y_real, D_loss_y_fake = self.loss_gan_disc(DYy_detached, DYGx_detached)
        D_loss = loss_gan_d_x + loss_gan_d_y

        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.DX.parameters()) + list(self.DY.parameters()), max_norm=1.0
        )
        self.optimizer_D.step()

        # Discriminator statistics
        with torch.no_grad():
            d_x_real_mean = DXx_detached.mean().item()
            d_x_fake_mean = DXFy_detached.mean().item()
            d_y_real_mean = DYy_detached.mean().item()
            d_y_fake_mean = DYGx_detached.mean().item()

        # Return metrics
        return {
            'total_loss': (G_loss.item() + D_loss.item()),
            'G_loss': G_loss.item(),
            'D_loss': D_loss.item(),
            'D_loss_x_real': D_loss_x_real.item(),
            'D_loss_x_fake': D_loss_x_fake.item(),
            'D_loss_y_real': D_loss_y_real.item(),
            'D_loss_y_fake': D_loss_y_fake.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_gan_g': loss_gan_g.item(),
            'loss_gan_g_x_real': loss_gan_g_x_real.item(),
            'loss_gan_g_x_fake': loss_gan_g_x_fake.item(),
            'loss_gan_g_y_real': loss_gan_g_y_real.item(),
            'loss_gan_g_y_fake': loss_gan_g_y_fake.item(),
            'loss_identity': loss_identity.item(),
            'loss_kl': loss_kl.item(),
            'd_x_real_mean': d_x_real_mean,
            'd_x_fake_mean': d_x_fake_mean,
            'd_y_real_mean': d_y_real_mean,
            'd_y_fake_mean': d_y_fake_mean
        }

    def validation_step(self, batch):
        """
        Validation step for CycleVAEGAN
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if (self.loss_cycle is None or self.loss_gan_gen is None or
            self.loss_gan_disc is None or self.loss_identity is None or
            self.loss_kl is None):
            raise ValueError("Loss functions have not been configured yet.")
        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            (Gx, FGx, Fy, GFy,
             mu_x, logvar_x, mu_FGx, logvar_FGx,
             mu_y, logvar_y, mu_GFy, logvar_GFy,
             DYGx, DXFy, DXx, DYy) = self(x, y)

            # Compute generator losses
            loss_cycle = self.loss_cycle(x, y, FGx, GFy)
            loss_gan_g_x, loss_gan_g_x_real, loss_gan_g_x_fake = self.loss_gan_gen(DXx, DXFy)
            loss_gan_g_y, loss_gan_g_y_real, loss_gan_g_y_fake = self.loss_gan_gen(DYy, DYGx)
            loss_gan_g = loss_gan_g_x + loss_gan_g_y
            loss_identity = (self.loss_identity(x, y, FGx, GFy) +
                             self.loss_identity(y, x, GFy, FGx))
            loss_kl = (self.loss_kl(mu_x, logvar_x) + self.loss_kl(mu_FGx, logvar_FGx) +
                       self.loss_kl(mu_y, logvar_y) + self.loss_kl(mu_GFy, logvar_GFy))

            G_loss = (self.lambda_cycle * loss_cycle +
                      self.lambda_gan * loss_gan_g +
                      self.lambda_identity * loss_identity +
                      self.lambda_kl * loss_kl)

            # Compute discriminator losses
            loss_gan_d_x, D_loss_x_real, D_loss_x_fake = self.loss_gan_disc(DXx, DXFy)
            loss_gan_d_y, D_loss_y_real, D_loss_y_fake = self.loss_gan_disc(DYy, DYGx)
            D_loss = loss_gan_d_x + loss_gan_d_y

            # Return metrics
            return {
                'total_loss': (G_loss.item() + D_loss.item()),
                'G_loss': G_loss.item(),
                'D_loss': D_loss.item(),
                'D_loss_x_real': D_loss_x_real.item(),
                'D_loss_x_fake': D_loss_x_fake.item(),
                'D_loss_y_real': D_loss_y_real.item(),
                'D_loss_y_fake': D_loss_y_fake.item(),
                'loss_cycle': loss_cycle.item(),
                'loss_gan_g': loss_gan_g.item(),
                'loss_gan_g_x_real': loss_gan_g_x_real.item(),
                'loss_gan_g_x_fake': loss_gan_g_x_fake.item(),
                'loss_gan_g_y_real': loss_gan_g_y_real.item(),
                'loss_gan_g_y_fake': loss_gan_g_y_fake.item(),
                'loss_identity': loss_identity.item(),
                'loss_kl': loss_kl.item(),
                'Gx': Gx.detach(),  # Gx = G(x) translation A->B
                'Fy': Fy.detach()   # Fy = F(y) translation B->A
            }


### Unpaired Datasets Networks ###

class CycleAE_unpaired(nn.Module):
    def __init__(self):
        super(CycleAE_unpaired, self).__init__()
        self.F = Autoencoder()
        self.G = Autoencoder()

    def forward(self, x, y):
        Gx = self.G(x)
        FGx = self.F(Gx)
        Fy = self.F(y)
        GFy = self.G(Fy)
        return Gx, FGx, Fy, GFy # All Netework expect Gx as their first output

    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizer for training"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return self.optimizer
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        return {'optimizer': self.optimizer.state_dict()}
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])
        else :
            raise KeyError("optimizer state not found in states")
    
    def configure_loss(self, **kwargs):
        """Configure loss functions (ignores unused kwargs)"""
        self.loss_cycle = CycleConsistencyLoss()
        self.lambda_cycle = kwargs.get('lambda_cycle', 10.0)

    def training_step(self, batch):
        """
        Training step for CycleAE unpaired
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if self.loss_cycle is None:
            raise ValueError("Loss functions have not been configured yet.")
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        x = batch['x']
        y = batch['y']

        # Forward pass
        Gx, FGx, Fy, GFy = self(x, y)

        # Compute losses
        loss_cycle = self.loss_cycle(x, y, FGx, GFy)
        total_loss = self.lambda_cycle * loss_cycle

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'loss_cycle': loss_cycle.item(),
        }

    def validation_step(self, batch):
        """
        Validation step for CycleAE unpaired
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if self.loss_cycle is None:
            raise ValueError("Loss functions have not been configured yet.")
        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            Gx, FGx, Fy, GFy = self(x, y)

            # Compute losses
            loss_cycle = self.loss_cycle(x, y, FGx, GFy)
            total_loss = self.lambda_cycle * loss_cycle

            # Return metrics
            return {
                'total_loss': total_loss.item(),
                'loss_cycle': loss_cycle.item(),
                'output' : Gx.detach()  # For compatibility, return Gx as output ???????
            }
    
class CycleVAE_unpaired (nn.Module):
    def __init__(self, latent_dim=64):
        super(CycleVAE_unpaired, self).__init__()
        self.F = VariationalAutoencoder(latent_dim)
        self.G = VariationalAutoencoder(latent_dim)

    def forward(self, x, y):
        Gx, mu_x, logvar_x = self.G(x)
        FGx, mu_FGx, logvar_FGx = self.F(Gx)
        Fy, mu_y, logvar_y = self.F(y)
        GFy, mu_GFy, logvar_GFy = self.G(Fy)
        return Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy # All Netework expect Gx as their first output
    
    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizer for training"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return self.optimizer
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        return {'optimizer': self.optimizer.state_dict()}
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])
        else :
            raise KeyError("optimizer state not found in states")
    
    def configure_loss(self, **kwargs):
        """Configure loss functions"""
        self.loss_cycle = CycleConsistencyLoss()
        self.loss_kl = KLDivergenceLoss()
        self.lambda_cycle = kwargs.get('lambda_cycle', 10.0)
        self.lambda_kl = kwargs.get('lambda_kl', 1e-5)

    def training_step(self, batch):
        """
        Training step for CycleVAE unpaired
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if self.loss_cycle is None or self.loss_kl is None:
            raise ValueError("Loss functions have not been configured yet.")
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        
        x = batch['x']
        y = batch['y']

        # Forward pass
        Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy = self(x, y)
        # Compute losses
        loss_cycle = self.loss_cycle(x, y, FGx, GFy)
        loss_kl = (self.loss_kl(mu_x, logvar_x) + self.loss_kl(mu_FGx, logvar_FGx) +
                    self.loss_kl(mu_y, logvar_y) + self.loss_kl(mu_GFy, logvar_GFy))

        total_loss = self.lambda_cycle * loss_cycle + self.lambda_kl * loss_kl

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_kl': loss_kl.item(),
        }

    def validation_step(self, batch):
        """
        Validation step for CycleVAE unpaired
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if self.loss_cycle is None or self.loss_kl is None:
            raise ValueError("Loss functions have not been configured yet.")
        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy = self(x, y)

            # Compute losses
            loss_cycle = self.loss_cycle(x, y, FGx, GFy)
            loss_kl = (self.loss_kl(mu_x, logvar_x) + self.loss_kl(mu_FGx, logvar_FGx) +
                        self.loss_kl(mu_y, logvar_y) + self.loss_kl(mu_GFy, logvar_GFy))
            total_loss = self.lambda_cycle * loss_cycle + self.lambda_kl * loss_kl

            # Return metrics
            return {
                'total_loss': total_loss.item(),
                'loss_cycle': loss_cycle.item(),
                'loss_kl': loss_kl.item(),
                'output' : Gx.detach()  # For compatibility, return Gx as output ???????
            }


class CycleAEGAN_unpaired (nn.Module):
    def __init__(self):
        super(CycleAEGAN_unpaired, self).__init__()
        self.F = Autoencoder()
        self.G = Autoencoder()
        self.DX = Discriminator()
        self.DY = Discriminator()

    def forward(self, x, y):
        Gx = self.G(x)
        FGx = self.F(Gx)
        Fy = self.F(y)
        GFy = self.G(Fy)
        DYGx = self.DY(Gx)
        DXFy = self.DX(Fy)
        DXx = self.DX(x)
        DYy = self.DY(y)
        return Gx, FGx, Fy, GFy, DYGx, DXFy # All Netework expect Gx as their first output
    
    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizer for training"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return self.optimizer
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        return {'optimizer': self.optimizer.state_dict()}
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])
        else :
            raise KeyError("optimizer state not found in states")
    
    def configure_loss(self, **kwargs):
        """Configure loss functions (ignores unused kwargs)"""
        self.loss_cycle = CycleConsistencyLoss()
        self.loss_gan_gen = GANLossGenerator()
        self.lambda_gan = kwargs.get('lambda_gan', 1.0)
        self.lambda_cycle = kwargs.get('lambda_cycle', 10.0)

    def training_step(self, batch):
        """
        Training step for CycleAEGAN unpaired
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if (self.loss_cycle is None or self.loss_gan_gen is None):
            raise ValueError("Loss functions have not been configured yet.")
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        
        x = batch['x']
        y = batch['y']

        # Forward pass
        Gx, FGx, Fy, GFy, DYGx, DXFy = self(x, y)

        # Compute losses
        loss_cycle = self.loss_cycle(x, y, FGx, GFy)
        loss_gan_g_x, loss_gan_g_x_real, loss_gan_g_x_fake = self.loss_gan_gen(self.DX(x), DXFy)
        loss_gan_g_y, loss_gan_g_y_real, loss_gan_g_y_fake = self.loss_gan_gen(self.DY(y), DYGx)
        loss_gan_g = loss_gan_g_x + loss_gan_g_y

        total_loss = (self.lambda_cycle * loss_cycle +
                      self.lambda_gan * loss_gan_g)
        
         # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_gan_g': loss_gan_g.item(),
            'loss_gan_g_x_real': loss_gan_g_x_real.item(),
            'loss_gan_g_x_fake': loss_gan_g_x_fake.item(),
            'loss_gan_g_y_real': loss_gan_g_y_real.item(),
            'loss_gan_g_y_fake': loss_gan_g_y_fake.item(),
        }

    def validation_step(self, batch):
        """
        Validation step for CycleAEGAN unpaired
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if (self.loss_cycle is None or self.loss_gan_gen is None):
            raise ValueError("Loss functions have not been configured yet.")
        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            Gx, FGx, Fy, GFy, DYGx, DXFy = self(x, y)

            # Compute losses
            loss_cycle = self.loss_cycle(x, y, FGx, GFy)
            loss_gan_g_x, loss_gan_g_x_real, loss_gan_g_x_fake = self.loss_gan_gen(self.DX(x), DXFy)
            loss_gan_g_y, loss_gan_g_y_real, loss_gan_g_y_fake = self.loss_gan_gen(self.DY(y), DYGx)
            loss_gan_g = loss_gan_g_x + loss_gan_g_y
            total_loss = (self.lambda_cycle * loss_cycle +
                          self.lambda_gan * loss_gan_g)

            # Return metrics
            return {
                'total_loss': total_loss.item(),
                'loss_cycle': loss_cycle.item(),
                'loss_gan_g': loss_gan_g.item(),
                'loss_gan_g_x_real': loss_gan_g_x_real.item(),
                'loss_gan_g_x_fake': loss_gan_g_x_fake.item(),
                'loss_gan_g_y_real': loss_gan_g_y_real.item(),
                'loss_gan_g_y_fake': loss_gan_g_y_fake.item(),
                'output' : Gx.detach()
            }


class CycleVAEGAN_unpaired (nn.Module):
    def __init__(self, latent_dim=64):
        super(CycleVAEGAN_unpaired, self).__init__()
        self.F = VariationalAutoencoder(latent_dim)
        self.G = VariationalAutoencoder(latent_dim)
        self.DX = Discriminator()
        self.DY = Discriminator()

    def forward(self, x, y):
        Gx, mu_x, logvar_x = self.G(x)
        FGx, mu_FGx, logvar_FGx = self.F(Gx)
        Fy, mu_y, logvar_y = self.F(y)
        GFy, mu_GFy, logvar_GFy = self.G(Fy)
        DYGx = self.DY(Gx)
        DXFy = self.DX(Fy)
        DXx = self.DX(x)
        DYy = self.DY(y)
        return Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy, DYGx, DXFy, DXx, DYy # All Netework expect Gx as their first output

    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizer for training"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return self.optimizer

    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        return {'optimizer': self.optimizer.state_dict()}

    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        if 'optimizer' in states:
            self.optimizer.load_state_dict(states['optimizer'])
        else :
            raise KeyError("optimizer state not found in states")
    
    def configure_loss(self, **kwargs):
        """Configure loss functions"""
        self.loss_cycle = CycleConsistencyLoss()
        self.loss_gan_gen = GANLossGenerator()
        self.loss_kl = KLDivergenceLoss()
        self.lambda_gan = kwargs.get('lambda_gan', 1.0)
        self.lambda_cycle = kwargs.get('lambda_cycle', 10.0)
        self.lambda_kl = kwargs.get('lambda_kl', 1e-5)

    def training_step(self, batch):
        """
        Training step for CycleVAEGAN unpaired
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics
        """
        if (self.loss_cycle is None or self.loss_gan_gen is None or
            self.loss_kl is None):
            raise ValueError("Loss functions have not been configured yet.")
        if self.optimizer is None:
            raise ValueError("Optimizer has not been configured yet.")
        
        x = batch['x']
        y = batch['y']

        # Forward pass
        (Gx, FGx, Fy, GFy,
         mu_x, logvar_x, mu_FGx, logvar_FGx,
         mu_y, logvar_y, mu_GFy, logvar_GFy,
         DYGx, DXFy, DXx, DYy) = self(x, y)

        # Compute losses
        loss_cycle = self.loss_cycle(x, y, FGx, GFy)
        loss_gan_g_x, loss_gan_g_x_real, loss_gan_g_x_fake = self.loss_gan_gen(DXx, DXFy)
        loss_gan_g_y, loss_gan_g_y_real, loss_gan_g_y_fake = self.loss_gan_gen(DYy, DYGx)
        loss_gan_g = loss_gan_g_x + loss_gan_g_y
        loss_kl = (self.loss_kl(mu_x, logvar_x) + self.loss_kl(mu_FGx, logvar_FGx) +
                    self.loss_kl(mu_y, logvar_y) + self.loss_kl(mu_GFy, logvar_GFy))

        total_loss = (self.lambda_cycle * loss_cycle +
                      self.lambda_gan * loss_gan_g +
                      self.lambda_kl * loss_kl)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_gan_g': loss_gan_g.item(),
            'loss_gan_g_x_real': loss_gan_g_x_real.item(),
            'loss_gan_g_x_fake': loss_gan_g_x_fake.item(),
            'loss_gan_g_y_real': loss_gan_g_y_real.item(),
            'loss_gan_g_y_fake': loss_gan_g_y_fake.item(),
            'loss_kl': loss_kl.item(),
        }

    def validation_step(self, batch):
        """
        Validation step for CycleVAEGAN unpaired
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        if (self.loss_cycle is None or self.loss_gan_gen is None or
            self.loss_kl is None):
            raise ValueError("Loss functions have not been configured yet.")
        with torch.no_grad():
            x = batch['x']
            y = batch['y']

            # Forward pass
            (Gx, FGx, Fy, GFy,
             mu_x, logvar_x, mu_FGx, logvar_FGx,
             mu_y, logvar_y, mu_GFy, logvar_GFy,
             DYGx, DXFy, DXx, DYy) = self(x, y)
            
            # Compute losses
            loss_cycle = self.loss_cycle(x, y, FGx, GFy)
            loss_gan_g_x, loss_gan_g_x_real, loss_gan_g_x_fake = self.loss_gan_gen(DXx, DXFy)
            loss_gan_g_y, loss_gan_g_y_real, loss_gan_g_y_fake = self.loss_gan_gen(DYy, DYGx)
            loss_gan_g = loss_gan_g_x + loss_gan_g_y
            loss_kl = (self.loss_kl(mu_x, logvar_x) + self.loss_kl(mu_FGx, logvar_FGx) +
                        self.loss_kl(mu_y, logvar_y) + self.loss_kl(mu_GFy, logvar_GFy))
            total_loss = (self.lambda_cycle * loss_cycle +
                          self.lambda_gan * loss_gan_g +
                          self.lambda_kl * loss_kl)

            # Return metrics
            return {
                'total_loss': total_loss.item(),
                'loss_cycle': loss_cycle.item(),
                'loss_gan_g': loss_gan_g.item(),
                'loss_gan_g_x_real': loss_gan_g_x_real.item(),
                'loss_gan_g_x_fake': loss_gan_g_x_fake.item(),
                'loss_gan_g_y_real': loss_gan_g_y_real.item(),
                'loss_gan_g_y_fake': loss_gan_g_y_fake.item(),
                'loss_kl': loss_kl.item(),
                'output' : Gx.detach()
            }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # The commented shapes are the output shapes with input of shape (10, 3, 256, 256) -> Why ? Not coherent with the figures on Github
    x = torch.randn((10, 3, 256, 256)).to(device)
    
    # Test the Networks molecular components
    encoder = Encoder().to(device)
    encoded = encoder(x)
    print("Encoded shape:", encoded.shape)  # Encoded shape: torch.Size([10, 1024, 16, 16])

    # release memory
    del encoder
    torch.cuda.empty_cache()

    decoder = Decoder().to(device)
    decoded = decoder(encoded)
    print("Decoded shape:", decoded.shape)  # Decoded shape: torch.Size([10, 3, 256, 256])

    # release memory
    del decoder
    torch.cuda.empty_cache()

    variational_encoder_block = VariationalEncoderBlock(in_channels=1024, latent_dim=64).to(device)
    z, mu, logvar = variational_encoder_block(encoded)
    print("Variational Encoder Block output shape:", z.shape)  # Variational Encoder Block output shape: torch.Size([10, 64, 16, 16])

    # release memory
    del variational_encoder_block
    torch.cuda.empty_cache()

    variational_decoder_block = VariationalDecoderBlock(latent_dim=64, out_channels=1024).to(device)
    decoded_latent = variational_decoder_block(z)
    print("Variational Decoder Block output shape:", decoded_latent.shape)  # Variational Decoder Block output shape: torch.Size([10, 1024, 16, 16])
    
    # release memory
    del variational_decoder_block
    torch.cuda.empty_cache()

    discriminator = Discriminator().to(device)
    validity = discriminator(x)
    print("Discriminator output shape:", validity.shape)    # Discriminator output shape: torch.Size([10]) -> see comment in Discriminator forward method

    # release memory
    del discriminator
    torch.cuda.empty_cache()

    # Test the networks composites
    Autoencoder_instance = Autoencoder().to(device)
    Gx = Autoencoder_instance(x)
    print("Autoencoder output shape:", Gx.shape)  # Autoencoder output shape: torch.Size([10, 3, 256, 256])

    # release memory
    del Autoencoder_instance
    torch.cuda.empty_cache()

    vae = VariationalAutoencoder(latent_dim=64).to(device)
    x_recon, mu, logvar = vae(x)
    print("VAE Reconstructed shape:", x_recon.shape)    # VAE Reconstructed shape: torch.Size([10, 3, 256, 256])
    print("VAE Mu shape:", mu.shape)    # VAE Mu shape: torch.Size([10, 64, 16, 16])

    # release memory
    del vae
    torch.cuda.empty_cache()

    aegan = AEGAN().to(device) # There's no latent dim for AEGAN
    y = torch.randn((10, 3, 256, 256)).to(device)
    Gx, DGx, Dy = aegan(x, y)
    print("AEGAN Reconstructed shape:", Gx.shape)    # AEGAN Reconstructed shape: torch.Size([10, 3, 256, 256])
    print("AEGAN Discriminator output shape:", DGx.shape)    # AEGAN Discriminator output shape: torch.Size([10])

    # release memory
    del aegan
    torch.cuda.empty_cache()

    vaegan = VAEGAN(latent_dim=64).to(device)
    Gx, mu, logvar, DGx, Dy = vaegan(x, y)
    print("VAEGAN Reconstructed shape:", Gx.shape)    # VAEGAN Reconstructed shape: torch.Size([10, 3, 256, 256])
    print("VAEGAN Discriminator output shape:", DGx.shape)    # VAEGAN Discriminator output shape: torch.Size([10]) -> see comment in Discriminator forward method

    # release memory
    del vaegan
    torch.cuda.empty_cache()

    cycle_aegan = CycleAEGAN().to(device) # There's no latent dim for CycleAEGAN
    y = torch.randn((10, 3, 256, 256)).to(device)
    Gx, FGx, Fy, GFy, DYGx, DXFy = cycle_aegan(x, y)
    print("CycleAE Gx shape:", Gx.shape)   # CycleAE Gx shape: torch.Size([10, 3, 256, 256])
    print("CycleAE FGx shape:", FGx.shape)  # CycleAE FGx shape: torch.Size([10, 3, 256, 256])
    print("CycleAE Fy shape:", Fy.shape)  # CycleAE Fy shape: torch.Size([10, 3, 256, 256])
    print("CycleAE GFy shape:", GFy.shape) # CycleAE GFy shape: torch.Size([10, 3, 256, 256])


    # The last model requires too much memory to be tested with the others, 
    # so it is commented out. You can uncomment and test it separately if needed.
    # # release memory
    # del cycle_aegan
    # torch.cuda.empty_cache()

    # y = torch.randn((10, 3, 256, 256)).to(device)
    # cycle_vaegan = CycleVAEGAN(latent_dim=64).to(device)
    # (Gx, FGx, Fy, GFy,
    #  mu_x, logvar_x, mu_FGx, logvar_FGx,
    #  mu_y, logvar_y, mu_GFy, logvar_GFy,
    #  DYGx, DXFy, DXx, DYy) = cycle_vaegan(x, y)
    # print("CycleVAEGAN Gx shape:", Gx.shape)   # CycleVAEGAN Gx shape: torch.Size([10, 3, 256, 256])
    # print("CycleVAEGAN FGx shape:", FGx.shape)  # CycleVAEGAN FGx shape: torch.Size([10, 3, 256, 256])
    # print("CycleVAEGAN Fy shape:", Fy.shape)  # CycleVAEGAN Fy shape: torch.Size([10, 3, 256, 256])
    # print("CycleVAEGAN GFy shape:", GFy.shape) # CycleVAEGAN GFy shape: torch.Size([10, 3, 256, 256])

    # # release memory
    # del cycle_vaegan

    print("All tests passed successfully.")
