"""
Loss functions for VAE-CycleGAN implementation

It defines the atomic loss functions and composite loss functions for different architectures.

It originally was meant to include the composite loss functions as well, but they are planned to be removed and implemented directly in the network classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

### ATOMIC LOSS FUNCTIONS ###
class TranslationLoss(nn.Module):
    """
    Translation Loss (L1 reconstruction)
    L_trans(x,y) = ||G(x) - y||_1
    """
    def __init__(self):
        super(TranslationLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, generated, target):
        return self.l1_loss(generated, target)


class CycleConsistencyLoss(nn.Module):
    """
    Cycle Consistency Loss
    L_cycle = ||F(G(x)) - x||_1 + ||G(F(y)) - y||_1
    """
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, x, y, FGx, GFy):
        loss_x = self.l1_loss(FGx, x)  # ||F(G(x)) - x||_1
        loss_y = self.l1_loss(GFy, y)  # ||G(F(y)) - y||_1
        return loss_x + loss_y


class IdentityLoss(nn.Module):
    """
    Identity Loss
    L_id = ||G(x) - x||_1 + ||F(y) - y||_1
    """
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, x, y, Gx, Fy):
        loss_G = self.l1_loss(Gx, x)  # ||G(x) - x||_1
        loss_F = self.l1_loss(Fy, y)  # ||F(y) - y||_1
        return loss_G + loss_F

class GANLossGenerator(nn.Module):
    """
    L_GAN^{X->Y} = D_Y(y)^2 + (1 - D_Y(G(x)))^2
    L_GAN^{Y->X} = D_X(x)^2 + (1 - D_X(F(y)))^2
    """
    def __init__(self):
        super(GANLossGenerator, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, D_real, D_fake):
        # D(y)^2 + (1 - D(G(x)))^2
        real_loss = self.mse_loss(D_real, torch.zeros_like(D_real))
        fake_loss = self.mse_loss(D_fake, torch.ones_like(D_fake))
        return real_loss + fake_loss


class GANLossDiscriminator(nn.Module):
    """
    Adversarial Loss for Discriminators (LSGAN)
    
    L_GAN^{D_X} = (1 - D_X(x))^2 + D_X(F(y))^2
    L_GAN^{D_Y} = (1 - D_Y(y))^2 + D_Y(G(x))^2
    """
    def __init__(self):
        super(GANLossDiscriminator, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, D_real, D_fake):
        # (1 - D(x))^2 + D(G(x))^2
        real_loss = self.mse_loss(D_real, torch.ones_like(D_real))
        fake_loss = self.mse_loss(D_fake, torch.zeros_like(D_fake))
        return real_loss + fake_loss


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence Loss for VAE
    
    L_KL = D_KL(q(z|x) || N(0,I))
         = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
    
    def forward(self, mu, logvar):
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalize by batch size and dimensions
        return kl_loss / (mu.size(0) * mu.numel() / mu.size(0))

### COMPOSITE LOSS FUNCTIONS (Soon removed to be implemented in the networks themselves)###

class CycleAELossPaired(nn.Module):
    """
    Loss for Cycle AE (paired)
    L = L_trans + λ_cycle * L_cycle
    """
    def __init__(self, lambda_cycle=10.0):
        super(CycleAELossPaired, self).__init__()
        self.translation_loss = TranslationLoss()
        self.cycle_loss = CycleConsistencyLoss()
        self.lambda_cycle = lambda_cycle
    
    def forward(self, model_output, x, y):
        Gx, FGx, Fy, GFy = model_output
        
        loss_trans = self.translation_loss(Gx, y)
        loss_cycle = self.cycle_loss(x, y, FGx, GFy)
        
        total_loss = loss_trans + self.lambda_cycle * loss_cycle
        
        losses_dict = {
            'loss_total': total_loss.item(),
            'loss_trans': loss_trans.item(),
            'loss_cycle': loss_cycle.item()
        }
        
        return total_loss, losses_dict

class CycleVAELossPaired(nn.Module):
    """
    Loss for Cycle-VAE (paired)
    L = L_trans + λ_cycle * L_cycle + λ_kl * L_KL
    """
    def __init__(self, lambda_cycle=10.0, lambda_kl=1e-5):
        super(CycleVAELossPaired, self).__init__()
        self.translation_loss = TranslationLoss()
        self.cycle_loss = CycleConsistencyLoss()
        self.kl_loss = KLDivergenceLoss()
        self.lambda_cycle = lambda_cycle
        self.lambda_kl = lambda_kl
    
    def forward(self, model_output, x, y):
        Gx, FGx, Fy, GFy, mu_x, logvar_x, mu_FGx, logvar_FGx, mu_y, logvar_y, mu_GFy, logvar_GFy = model_output
        
        loss_trans = self.translation_loss(Gx, y)
        loss_cycle = self.cycle_loss(x, y, FGx, GFy)
        
        # KL losses for both domains
        loss_kl_x = self.kl_loss(mu_x, logvar_x)
        loss_kl_y = self.kl_loss(mu_y, logvar_y)
        loss_kl = loss_kl_x + loss_kl_y
        
        total_loss = loss_trans + self.lambda_cycle * loss_cycle + self.lambda_kl * loss_kl
        
        losses_dict = {
            'loss_total': total_loss.item(),
            'loss_trans': loss_trans.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_kl': loss_kl.item(),
            'loss_kl_x': loss_kl_x.item(),
            'loss_kl_y': loss_kl_y.item()
        }
        
        return total_loss, losses_dict


class VAEGANLoss(nn.Module):
    """
    Loss for VAE-GAN (paired)
    L = L_trans + λ_GAN * L_GAN + λ_identity * L_id + λ_kl * L_KL
    """
    def __init__(self, lambda_gan=1.0, lambda_identity=5.0, lambda_kl=1e-5):
        super(VAEGANLoss, self).__init__()
        self.translation_loss = TranslationLoss()
        self.gan_loss_gen = GANLossGenerator()
        self.identity_loss = IdentityLoss()
        self.kl_loss = KLDivergenceLoss()
        self.lambda_gan = lambda_gan
        self.lambda_identity = lambda_identity
        self.lambda_kl = lambda_kl
    
    def forward(self, model_output, x, y):
        Gx, mu, logvar, DGx, Dx = model_output
        
        loss_trans = self.translation_loss(Gx, y)
        loss_gan = self.gan_loss_gen(Dx, DGx)
        loss_id = self.identity_loss(x, y, Gx, y)  # Simplified
        loss_kl = self.kl_loss(mu, logvar)
        
        total_loss = (loss_trans + self.lambda_gan * loss_gan + 
                     self.lambda_identity * loss_id + self.lambda_kl * loss_kl)
        
        losses_dict = {
            'loss_total': total_loss.item(),
            'loss_trans': loss_trans.item(),
            'loss_gan': loss_gan.item(),
            'loss_identity': loss_id.item(),
            'loss_kl': loss_kl.item()
        }
        
        return total_loss, losses_dict


# UNPAIRED MODELS

class CycleAELossUnpaired(nn.Module):
    """
    Loss for Cycle AE (unpaired)
    L = λ_cycle * L_cycle
    """


class AECycleGANLoss(nn.Module):
    """
    Loss for AE-CycleGAN (unpaired)
    L = λ_GAN * L_GAN + λ_identity * L_id + λ_cycle * L_cycle
    """


class CycleVAELossUnpaired(nn.Module):
    """
    Loss for Cycle-VAE (unpaired)
    L = λ_cycle * L_cycle + λ_kl * L_KL
    """



class VAECycleGANLoss(nn.Module):
    """
    Loss for VAE-CycleGAN (unpaired)
    L = λ_GAN * L_GAN + λ_identity * L_id + λ_cycle * L_cycle + λ_kl * L_KL
    """
