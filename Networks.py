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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from losses import *

### ATOMIC NETWORK COMPONENTS ###

class CaSb (nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=3, activation="ReLU"):
        super(CaSb, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect')
        self.norm = nn.InstanceNorm2d(out_channels)
        if activation == "ReLU":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        else :
            raise NotImplementedError("Activation not implemented")

    def forward(self, x):
        x = self.conv(x)
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
        x = self.norm(x)
        x = self.activation(x)
        return x

class R (nn.Module):
    def __init__(self, out_channels):
        super(R, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.activation1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.activation2 = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
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
        x = self.norm(x)
        x = self.activation(x)
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

    def forward(self, x):
        return self.model(x)
    
class VariationalEncoderBlock (nn.Module):
    def __init__(self, in_channels, latent_dim=64):
        super(VariationalEncoderBlock, self).__init__()
        self.muConv = L(in_channels, latent_dim)
        self.logvarConv = nn.Sequential(S(in_channels, latent_dim), S(latent_dim, latent_dim))

    def forward(self, x):
        mu = self.muConv(x)
        logvar = self.logvarConv(x)
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
    
class Decoder (nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        layers = []
        layers.append(R(1024))
        layers.append(U(1024, 512))
        layers.append(U(512, 256))
        layers.append(U(256, 128))
        layers.append(U(128, 64))
        layers.append(CaSb(64, 3, kernel_size=7, stride=1, activation="Tanh"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class Discriminator (nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(CaSb(3, 64, kernel_size=4, stride=2, padding=1, activation="LeakyReLU"))
        layers.append(CaSb(64, 128, kernel_size=4, stride=2, padding=1, activation="LeakyReLU"))
        layers.append(CaSb(128, 256, kernel_size=4, stride=2, padding=1, activation="LeakyReLU"))
        layers.append(CaSb(256, 512, kernel_size=4, stride=2, padding=1, activation="LeakyReLU"))
        layers.append(nn.Conv2d(512, 1, kernel_size=16, stride=1, padding=0))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        #shape (B, 1, 1, 1) to (B,1)
        return self.model(x).view(-1, 1).squeeze(1)
    
### NETWORKS COMPOSITES ###
    
class Autoencoder (nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # architecture related attributes
        self.encoder = Encoder()
        self.decoder = Decoder()

        # loss related attributes to be set in "configure_loss"
        self.optimizer = None
        self.loss_fn = None

    def forward(self, x):
        z = self.encoder(x)
        Gx = self.decoder(z)
        return Gx # All Netework expect Gx as their first output
    
    def configure_optimizers(self, lr=1e-4, betas=(0.5, 0.999)):
        """Configure optimizer for training"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return self.optimizer
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        return {'optimizer': self.optimizer.state_dict()}
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
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
        x = batch['x']
        y = batch['y']
        
        # Forward pass
        output = self(x)
        
        # Compute loss
        loss_trans = self.loss_fn(output, y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_trans.backward()
        self.optimizer.step()
        
        # Return metrics
        return {
            'G_loss': loss_trans.item(),
        }
    
    def validation_step(self, batch):
        """
        Validation step for Autoencoder
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        x = batch['x']
        y = batch['y']
        
        # Forward pass
        output = self(x)
        
        # Compute loss
        loss_trans = self.loss_fn(output, y)
        
        # Return metrics
        return {
            'total_loss': loss_trans.item(),
            'loss_trans': loss_trans.item(),
            'output': output
        }

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
        return {'optimizer': self.optimizer.state_dict()}
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
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
            'output': output
        }
    

    
class AEGAN (nn.Module):
    def __init__(self):
        super(AEGAN, self).__init__()
        # architecture related attributes
        self.G = Autoencoder()
        self.D = Discriminator()

        # loss related attributes to be set in "configure_loss"
        self.optimizer_G = None
        self.optimizer_D = None
        self.loss_trans_fn = None
        self.loss_gan_gen_fn = None
        self.loss_gan_disc_fn = None
        self.loss_identity_fn = None
        self.lambda_gan = 0
        self.lambda_identity = 0
    def forward(self, x):
        Gx= self.G(x)
        DGx = self.D(Gx)
        Dx = self.D(x)
        return Gx, DGx, Dx # All Netework expect Gx as their first output
    
    def configure_optimizers(self, lr=2e-4, betas=(0.5, 0.999)):
        """Configure optimizers for Generator and Discriminator"""
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=betas)
        return self.optimizer_G, self.optimizer_D
    
    def save_optimizer_states(self):
        """Save optimizer states for checkpointing"""
        return {
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict()
        }
    
    def load_optimizer_states(self, states):
        """Load optimizer states from checkpoint"""
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
        self.loss_identity_fn = IdentityLoss()
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
        x = batch['x']
        y = batch['y']
        
        ### Train Generator
        self.optimizer_G.zero_grad()
        
        # Forward pass
        Gx, DGx, Dx = self(x)
        
        # Generator losses
        loss_trans = self.loss_trans_fn(Gx, y)
        loss_gan_g = self.loss_gan_gen_fn(Dx, DGx)
        loss_id = self.loss_identity_fn(x, y, Gx, y)  # Simplified identity
        
        G_loss = loss_trans + self.lambda_gan * loss_gan_g + self.lambda_identity * loss_id
        
        # Backward and optimize
        G_loss.backward(retain_graph=True)
        self.optimizer_G.step()
        
        ### Train Discriminator
        self.optimizer_D.zero_grad()
        
        # Discriminator loss (reuse forward outputs)
        D_loss = self.loss_gan_disc_fn(Dx, DGx.detach())
        
        # Backward and optimize
        D_loss.backward()
        self.optimizer_D.step()
        
        # Return metrics
        return {
            'G_loss': G_loss.item(),
            'D_loss': D_loss.item(),
            'loss_trans': loss_trans.item(),
            'loss_gan_g': loss_gan_g.item(),
            'loss_identity': loss_id.item()
        }
    
    def validation_step(self, batch):
        """
        Validation step for AEGAN
        Args:
            batch: dict with 'x' (input) and 'y' (target)
        Returns:
            dict with loss metrics and output
        """
        x = batch['x']
        y = batch['y']
        
        # Forward pass
        Gx, DGx, Dx = self(x)
        
        # Compute losses
        loss_trans = self.loss_trans_fn(Gx, y)
        loss_gan_g = self.loss_gan_gen_fn(Dx, DGx)
        loss_id = self.loss_identity_fn(x, y, Gx, y)
        G_loss = loss_trans + self.lambda_gan * loss_gan_g + self.lambda_identity * loss_id
        D_loss = self.loss_gan_disc_fn(Dx, DGx)
        
        # Return metrics
        return {
            'total_loss': (G_loss.item() + D_loss.item()),
            'G_loss': G_loss.item(),
            'D_loss': D_loss.item(),
            'loss_trans': loss_trans.item(),
            'loss_gan_g': loss_gan_g.item(),
            'loss_identity': loss_id.item(),
        }


class VAEGAN (nn.Module):
    def __init__(self, latent_dim=64):
        super(VAEGAN, self).__init__()
        self.G = VariationalAutoencoder(latent_dim)
        self.D = Discriminator()
        self.latent_dim = latent_dim

    def forward(self, x):
        Gx, mu, logvar = self.G(x)
        DGx = self.D(Gx)
        Dx = self.D(x)
        return Gx, mu, logvar, DGx, Dx # All Netework expect Gx as their first output

class CycleAE (nn.Module):
    def __init__(self):
        super(CycleAE, self).__init__()
        self.F = Autoencoder()
        self.G = Autoencoder()
    def forward(self, x, y):
        Gx = self.G(x)
        FGx = self.F(Gx)
        Fy = self.F(y)
        GFy = self.G(Fy)
        return Gx, FGx, Fy, GFy # All Netework expect Gx as their first output
    
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

class CycleAEGAN (nn.Module):
    def __init__(self):
        super(CycleAE, self).__init__()
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
    
    
class CycleVAEGAN (nn.Module):
    def __init__(self, latent_dim=64):
        super(CycleVAEGAN, self).__init__()
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
    

if __name__ == "__main__":
    # Test the networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((10, 3, 256, 256)).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    z = encoder(x)
    print("Encoded shape:", z.shape)
    x_recon = decoder(z)
    print("Reconstructed shape:", x_recon.shape)
    vae = VariationalAutoencoder(latent_dim=64).to(device)
    x_recon, mu, logvar = vae(x)
    print("VAE Reconstructed shape:", x_recon.shape)
    print("VAE Mu shape:", mu.shape)

    discriminator = Discriminator().to(device)
    validity = discriminator(x)
    print("Discriminator output shape:", validity.shape)

    aegan = AEGAN(latent_dim=64).to(device)
    Gx, DGx, Dx = aegan(x)
    print("AEGAN Reconstructed shape:", Gx.shape)
    print("AEGAN Discriminator output shape:", DGx.shape)
    vaegan = VAEGAN(latent_dim=64).to(device)
    Gx, mu, logvar,DGx, Dx, = vaegan(x)
    print("VAEGAN Reconstructed shape:", Gx.shape)
    print("VAEGAN Discriminator output shape:", DGx.shape)
    cycle_ae = CycleAE().to(device)
    y = torch.randn((10, 3, 256, 256)).to(device)
    Gx, FGx, Fy, GFy = cycle_ae(x, y)
    print("CycleAE Gx shape:", Gx.shape)
    print("CycleAE FGx shape:", FGx.shape)
    print("CycleAE Fy shape:", Fy.shape)
    print("CycleAE GFy shape:", GFy.shape)

