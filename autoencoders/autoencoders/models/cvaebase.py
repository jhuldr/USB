import torch
import hydra

from ..base.constants import MODEL_CVAEBASE
from ..base.base_model import BaseModelVAEBASE
from ..base.distributions import Normal

class cvaebase(BaseModelVAEBASE):
    r"""
    cVAE baseline model based on: https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI/blob/master/mains/main_VAE.py
    and: https://github.com/alawryaguila/multi-view-AE
    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_CVAEBASE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)


    def encode(self, x):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): Single element list of joint encoding distribution.
        """

        z_mu, z_log_var, reverse_intermediate_layer, dropout_layer, dec_dense, reshape = self.encoders[0](x[0])

        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        qz_x = hydra.utils.instantiate(
            self.cfg.encoder.default.enc_dist, loc=z_mu, scale=z_sigma
        )
        return [[qz_x], reverse_intermediate_layer, reshape, dropout_layer, dec_dense]

    def decode(self, qz_x, reverse_intermediate_layer, reshape, dropout_layer, dec_dense):
        r"""Forward pass of joint latent dimensions through decoder networks.
        """  

        if isinstance(qz_x[0], torch.Tensor):
            z = qz_x[0]
        else:
            z = qz_x[0]._sample(training=self._training)

        c = self.labels
        c = c.squeeze()
        if len(c.size()) == 1:
            c = c.unsqueeze(0)
        z = torch.hstack((z, c))
        z = dec_dense(z).view(z.size(0), -1, *reshape)
        z = dropout_layer(z)
        z = reverse_intermediate_layer(z)

        px_z = self.decoders[0](z)
        return [px_z]

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate the joint latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        fwd_rtn = self.encode(x)
        qz_x, reverse_intermediate_layer, reshape, dropout_layer, dec_dense = fwd_rtn
        px_z = self.decode(qz_x, reverse_intermediate_layer, reshape, dropout_layer, dec_dense)
        fwd_rtn = {"px_z": px_z, "qz_x": qz_x}
        return fwd_rtn

    def calc_kl(self, qz_x):
        r"""Calculate KL-divergence loss.

        Args:
            qz_xs (list): Single element list containing joint encoding distribution.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        if self.sparse:
            kl = qz_x[0].sparse_kl_divergence().mean(0).sum()
        else:
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

        ll = px_z[0].log_likelihood(x[0]).mean(0).sum() #first index is latent, second index is view
        return ll

    def loss_function(self, x, fwd_rtn):
        r"""Calculate Multimodal VAE loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the MVAE loss.
        """
        px_z = fwd_rtn["px_z"]
        qz_x = fwd_rtn["qz_x"]

        kl = self.calc_kl(qz_x)
        ll = self.calc_ll(x, px_z)

        total = kl - ll
        losses = {"loss": total, "kl": kl, "ll": ll}
        return losses