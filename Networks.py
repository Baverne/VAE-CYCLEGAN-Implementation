"""
All necessary networks for VAE-CygleGAN implementation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

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
    def __init__(self, in_channels, latent_dim):
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
    def __init__(self, latent_dim, out_channels):
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
    
class Autoencoder (nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        Gx = self.decoder(z)
        return Gx

class VariationalAutoencoder (nn.Module):
    def __init__(self, latent_dim=64):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.variational_encoder_block = VariationalEncoderBlock(in_channels=1024, latent_dim=latent_dim)
        self.variational_decoder_block = VariationalDecoderBlock(latent_dim=latent_dim, out_channels=1024)
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        z, mu, logvar = self.variational_encoder_block(encoded)
        decoded_latent = self.variational_decoder_block(z)
        Gx = self.decoder(decoded_latent)
        return Gx, mu, logvar
    
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
    
class AEGAN (nn.Module):
    def __init__(self, latent_dim):
        super(AEGAN, self).__init__()
        self.G = Autoencoder()
        self.D = Discriminator()


    def forward(self, x):
        Gx= self.G(x)
        DGx = self.D(Gx)
        Dx = self.D(x)
        return Gx, DGx, Dx


class VAEGAN (nn.Module):
    def __init__(self, latent_dim):
        super(VAEGAN, self).__init__()
        self.G = VariationalAutoencoder(latent_dim)
        self.D = Discriminator()
        self.latent_dim = latent_dim

    def forward(self, x):
        Gx, mu, logvar = self.G(x)
        DGx = self.D(Gx)
        Dx = self.D(x)
        return Gx, mu, logvar, DGx, Dx

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
        return Gx, FGx, Fy, GFy
    
class CycleVAE (nn.Module):
    def __init__(self, latent_dim):
        super(CycleVAE, self).__init__()
        self.F = VariationalAutoencoder(latent_dim)
        self.G = VariationalAutoencoder(latent_dim)
    def forward(self, x, y):
        Gx, mu_x, logvar_x = self.G(x)
        FGx, mu_FGx, logvar_FGx = self.F(Gx)
        Fy, mu_y, logvar_y = self.F(y)
        GFy, mu_GFy, logvar_GFy = self.G(Fy)
        return Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy

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
        return Gx, FGx, Fy, GFy, DYGx, DXFy
    
    
class CycleVAEGAN (nn.Module):
    def __init__(self, latent_dim):
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
        return Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy, DYGx, DXFy, DXx, DYy
    






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

