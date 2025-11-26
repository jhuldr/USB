#adapted from: https://github.com/Project-MONAI/GenerativeModels
# 
import ast
import torch
import hydra
import math

import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union
from torch.nn import Parameter
import functools #install?

class Encoder(nn.Module):
    """
    Configurable convolutional encoder.

    Args:
        input_dim (list): Dimensionality of the input data.
        num_filters (list[int]): Number of filters for each convolutional layer.
        input_shape (list[int]): Input shape to first conv layer.
        kernel_size (list[int]): kernel_size
        stride (list[int])
        padding (list[int])
        padding_mode (str)

    """
    def __init__(
        self,
        input_dim,
        z_dim,
        non_linear,
        bias,
        enc_dist,
        **kwargs
    ):
        super().__init__()

        self.input_size = input_dim
        self.z_dim = z_dim

        self.bias = bias
        self.non_linear = non_linear
        self.enc_dist = enc_dist

        self.layer_sizes = []

        conv_params = {k:v for (k,v) in kwargs.items() if k.startswith("layer")}

        layers = []

        num_layers = len(conv_params)
        for k,v in conv_params.items():
            l = v["layer"]
            v.pop('layer', None)
            layers.append(eval(f"nn.{l}(**v)"))

            out_size = [v1 for k1,v1 in v.items() if 'out_' in k1]
            if len(out_size) > 0:
                self.layer_sizes.append(out_size[0])

        # z_dim layer
        layers.append(nn.Linear(in_features=self.layer_sizes[-1], out_features=z_dim, bias=self.bias))
        self.layer_sizes.append(z_dim)

        self.encoder_layers = nn.Sequential(*layers)

    def forward(self, x):
        h1 = x
        for it_layer, layer in enumerate(self.encoder_layers[0:-1]):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.relu(h1)
        h1 = self.encoder_layers[-1](h1)
        return h1

class VariationalEncoder(Encoder):
    def __init__(
        self,
        input_dim,
        z_dim,
        non_linear,
        bias,
        sparse,
        log_alpha,
        enc_dist,
        **kwargs
    ):
        super().__init__(input_dim=input_dim,
                        z_dim=z_dim,
                        bias=bias,
                        non_linear=non_linear,
                        enc_dist=enc_dist,
                        **kwargs)

        self.sparse = sparse
        self.non_linear = non_linear
        self.log_alpha = log_alpha

        self.encoder_layers = self.encoder_layers[:-1]
        self.enc_mean_layer = nn.Linear(
            self.layer_sizes[-2],
            self.layer_sizes[-1],
            bias=self.bias,
        )

        if not self.sparse:
            self.enc_logvar_layer = nn.Linear(
                self.layer_sizes[-2],
                self.layer_sizes[-1],
                bias=self.bias,
            )

    def forward(self, x):
        h1 = x
        for it_layer, layer in enumerate(self.encoder_layers):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.relu(h1)

        if not self.sparse:
            mu = self.enc_mean_layer(h1)
            logvar = self.enc_logvar_layer(h1)
        else:
            mu = self.enc_mean_layer(h1)
            logvar = self.log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        z_dim,
        non_linear,
        bias,
        dec_dist,
        init_logvar=None,
        **kwargs
    ):
        super().__init__()

        self.input_size = input_dim
        self.z_dim = z_dim

        self.bias = bias
        self.dec_dist = dec_dist
        self.non_linear = non_linear

        self.layer_sizes = []
        self.layer_sizes.append(z_dim)

        conv_params = {k:v for (k,v) in kwargs.items() if k.startswith("layer")}

        layers = []

        # z_dim layer
        first_layer = list(conv_params)[0]
        out_dim = conv_params[first_layer]["out_features"]
        layers.append(nn.Linear(in_features=self.z_dim, out_features=out_dim, bias=self.bias))
        self.layer_sizes.append(out_dim)

        conv_params.pop(first_layer, None)
        num_layers = len(conv_params)
        for k,v in conv_params.items():
            l = v["layer"]
            v.pop('layer', None)
            layers.append(eval(f"nn.{l}(**v)")) 

            out_size = [v1 for k1,v1 in v.items() if 'out_' in k1]
            if len(out_size) > 0:
                self.layer_sizes.append(out_size[0])

        self.decoder_layers = nn.Sequential(*layers)

    def forward(self, z):
        x_rec = z
        for it_layer, layer in enumerate(self.decoder_layers[:-1]):
            x_rec = layer(x_rec)
            if self.non_linear:
                x_rec = F.relu(x_rec)

        x_rec = self.decoder_layers[-1](x_rec)
        x_rec = hydra.utils.instantiate(self.dec_dist, x=x_rec)

        return x_rec
    

def weights_init(m):
    """
    borrowed from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
        
        Borrowed from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
    """
    def __init__(self, input_nc=3, ndf=96, n_layers=3,spatial_dim=2): #ndf was 64
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if spatial_dim == 2:
            norm = nn.BatchNorm2d
            conv = nn.Conv2d
        else:
            norm = nn.BatchNorm3d
            conv = nn.Conv3d
        norm_layer = norm

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != norm
        else:
            use_bias = norm_layer != norm

        kw = 4
        padw = 1
        sequence = [conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            conv(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            conv(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)