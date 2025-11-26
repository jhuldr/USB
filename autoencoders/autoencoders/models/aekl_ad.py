import torch
import hydra

from ..base.constants import MODEL_AEKLAD
from ..base.base_model import BaseModelVAEAD
from ..architectures.cnn import NLayerDiscriminator, weights_init
from ..architectures.medicalperceptual import PerceptualLoss
import torch.nn as nn
import torch.nn.functional as F


torch.cuda.empty_cache()

class AutoencoderKLAD(BaseModelVAEAD):
    """
    Autoencoder model with KL-regularized latent space based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        num_channels: sequence of block output channels.
        attention_levels: sequence of levels to add attention.
        latent_channels: latent embedding dimension.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_AEKLAD,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)
        #check if self.conv exists
        if hasattr(self, 'conv'):
            self.quant_conv_mu = hydra.utils.instantiate(self.conv,
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
                stride=1,
                kernel_size=1,
                padding=0,
            )
            self.quant_conv_log_sigma = hydra.utils.instantiate(self.conv,
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
                stride=1,
                kernel_size=1,
                padding=0,
            )
            self.post_quant_conv = hydra.utils.instantiate(self.conv,
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
                stride=1,
                kernel_size=1,
                padding=0,
            )
        else:
            self.quant_conv_mu = torch.nn.Conv2d(
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
                stride=1,
                kernel_size=1,
                padding=0,
            )
            self.quant_conv_log_sigma = torch.nn.Conv2d(
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
                stride=1,
                kernel_size=1,
                padding=0,
            )
            self.post_quant_conv = torch.nn.Conv2d(
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
                stride=1,
                kernel_size=1,
                padding=0,
            )
        self.discriminator = NLayerDiscriminator(input_nc=self.cfg.n_discriminator.disc_in_channels,
                                                 n_layers=self.cfg.n_discriminator.disc_num_layers, 
                                                 spatial_dim=self.spatial_dims,
                                                 ).apply(weights_init)
        self.adv_lossfn = nn.MSELoss(reduction='mean')

        self.perceptual_loss = PerceptualLoss(spatial_dims=self.spatial_dims, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.25)

    def encode(self, x):
        r"""Forward pass through encoder networks.
        """
  
        h = self.encoders[0](x[0])
        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        qz_x = hydra.utils.instantiate(
            self.cfg.encoder.default.enc_dist, loc=z_mu, scale=z_sigma
        )
        return [qz_x]

    def decode(self, qz_x):
        r"""Forward pass of joint latent dimensions through decoder networks.
        """
        #if qz_x[0] is a tensor then dont need to sample
        #if qz_x[0] is a distribution then need to sample
        if isinstance(qz_x[0], torch.Tensor):
            z = self.post_quant_conv(qz_x[0])
        else:
            z = self.post_quant_conv(qz_x[0]._sample(training=self._training))
        px_z = self.decoders[0](z)
        return [px_z]

    def forward_recon(self, x):
        r"""Apply encode and decode methods to input data to generate the joint latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        qz_x = self.encode(x)
        px_z = self.decode(qz_x)
        fwd_rtn = {"px_z": px_z, "qz_x": qz_x}
        return fwd_rtn

    def calc_kl(self, qz_x):
        r"""Calculate KL-divergence loss.

        Args:
            qz_xs (list): Single element list containing joint encoding distribution.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        kl = qz_x[0].kl_divergence(self.prior).mean(0).sum()
        return self.beta * kl

    def calc_ll(self, x, px_z):
        r"""Calculate log-likelihood loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            px_zs (list): list of decoding distributions.

        Returns:
            ll (torch.Tensor): Log-likelihood loss.
        """
        ll = px_z[0].log_likelihood(x[0]).mean()
        return ll                          

    def calc_perceptual(self, x, px_z):
        p_loss = torch.mean(self.perceptual_loss(px_z[0]._sample(), x[0])) 
        return p_loss*self.perceptual_weight
     
    def discriminator_loss(self, x, px_z):
        logits_fake = self.discriminator(px_z[0]._sample().contiguous().detach()) #not sure if i need contiguous part
        logits_real = self.discriminator(x[0].contiguous().detach())
        d_loss = self.d_loss(logits_real, logits_fake)
        return d_loss*self.disc_weight
    
    def generator_loss(self, px_z): 
        logits_fake = self.discriminator(px_z[0]._sample().contiguous().float())
        target_label = torch.ones_like(logits_fake)
        #g_loss = -torch.mean(logits_fake) 
        g_loss = self.adv_lossfn(logits_fake, target_label)
        return g_loss*self.disc_weight

    def d_loss(self, logits_real, logits_fake):
        target_label_real = torch.ones_like(logits_real)
        target_label_fake = torch.zeros_like(logits_fake)
        loss_real = self.adv_lossfn(logits_real, target_label_real)
        loss_fake = self.adv_lossfn(logits_fake, target_label_fake)
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def recon_loss(self, x, fwd_rtn):
        r"""Calculate AutoencoderKLAD loss
        """
        px_z = fwd_rtn["px_z"]
        qz_x = fwd_rtn["qz_x"]

        kl = self.calc_kl(qz_x)
        ll = self.calc_ll(x, px_z)
        p_loss = self.calc_perceptual(x, px_z)
        losses = {"kl": kl, "ll": ll, "p_loss": p_loss}
        return losses
