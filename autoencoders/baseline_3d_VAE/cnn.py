#cVAE and VAE autoencoders based on:
#https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI
from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import numpy as np
# To install xformers, use pip install xformers==0.0.16rc401
if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops

    has_xformers = True
else:
    xformers = None
    has_xformers = False

def output_size(input_size, kernel_size, stride, padding):
    # Broadcast padding and kernel_size to all dimensions
    if not isinstance(kernel_size, Sequence):
        kernel_size = [kernel_size] * len(input_size)
    if not isinstance(padding, Sequence):
        padding = [padding] * len(input_size)

    # Calculate output size for each dimension
    output_size = tuple(
        (sz + 2 * pad - ksz) // stride + 1
        for sz, ksz, pad in zip(input_size, kernel_size, padding)
    )

    return output_size

class VariationalEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 in_channels=1,
                 intermediate_resolutions=[8,8], 
                 use_batchnorm=True, 
                 stride=2, 
                 dropout=0.1,
                 **kwargs):
        
        super().__init__()

        self.input_size = input_dim
        self.z_dim = z_dim
        self.stride = stride
        self.encoder = nn.ModuleList()
        num_pooling = int(math.log2(self.input_size[1]) - math.log2(float(intermediate_resolutions[0])))


        padding_needed = max(0, math.ceil((self.input_size[1] - 1) / self.stride) * self.stride + 1 - self.input_size[1])
        padding = padding_needed // 2  # Calculate padding on both sides
        output_shape = []
        curr_output_shape = self.input_size #first dimension is number of channels
        for i in range(num_pooling):
            filters = int(min(128, 32 * (2 ** i)))
            if i == 0:
                self.encoder.append(nn.Conv3d(in_channels, filters, kernel_size=5, stride=self.stride, padding=padding))
            else:
                filters_prev = int(min(128, 32 * (2 ** (i - 1))))
                self.encoder.append(nn.Conv3d(filters_prev, filters, kernel_size=5, stride=self.stride, padding=padding))
            if use_batchnorm:
                self.encoder.append(nn.BatchNorm3d(filters))
            curr_output_shape = output_size(curr_output_shape, 5, self.stride, padding)
            output_shape.append(curr_output_shape)
            self.encoder.append(nn.LeakyReLU(0.2, inplace=True))


        self.intermediate_conv = nn.Conv3d(filters, filters // 8, kernel_size=1, stride=1, padding=padding)

        curr_output_shape = output_size(curr_output_shape, 1, 1, 0)

        output_shape.append(curr_output_shape)
        self.intermediate_conv_reverse = nn.Conv3d(filters // 8, filters, kernel_size=1, stride=1, padding=padding)
        self.dropout = nn.Dropout(dropout)
        
        flatten_output = np.prod(output_shape[-1])*(filters // 8)
        self.reshape = output_shape[-1]

        self.mu = nn.Linear(flatten_output, self.z_dim)
        self.logvar = nn.Linear(flatten_output, self.z_dim)
        self.dec_dense = nn.Linear(self.z_dim, flatten_output)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = self.intermediate_conv(x)
        x = self.dropout(x)
        #flatten all dimensions other than batch
        x = x.view(x.size(0), -1)
        mu = self.dropout(self.mu(x))
        logvar = self.dropout(self.logvar(x))
        return mu, logvar, self.intermediate_conv_reverse, self.dropout, self.dec_dense, self.reshape

class ConditionalVariationalEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 in_channels=1,
                 intermediate_resolutions=[8,8], 
                 use_batchnorm=True, 
                 stride=2, 
                 dropout=0.1,
                 num_cat=2,
                 one_hot=False,
                 **kwargs):
        
        super().__init__()

        self.input_size = input_dim
        self.z_dim = z_dim
        self.stride = stride
        self.encoder = nn.ModuleList()
        self.one_hot = one_hot
        num_pooling = int(math.log2(self.input_size[1]) - math.log2(float(intermediate_resolutions[0])))

        in_channels += num_cat

        padding_needed = max(0, math.ceil((self.input_size[1] - 1) / self.stride) * self.stride + 1 - self.input_size[1])
        padding = padding_needed // 2  # Calculate padding on both sides
        output_shape = []
        curr_output_shape = self.input_size #first dimension is number of channels
        for i in range(num_pooling):
            filters = int(min(128, 32 * (2 ** i)))
            if i == 0:
                self.encoder.append(nn.Conv3d(in_channels, filters, kernel_size=5, stride=self.stride, padding=padding))
            else:
                filters_prev = int(min(128, 32 * (2 ** (i - 1))))
                self.encoder.append(nn.Conv3d(filters_prev, filters, kernel_size=5, stride=self.stride, padding=padding))
            if use_batchnorm:
                self.encoder.append(nn.BatchNorm3d(filters))
            curr_output_shape = output_size(curr_output_shape, 5, self.stride, padding)
            output_shape.append(curr_output_shape)
            self.encoder.append(nn.LeakyReLU(0.2, inplace=True))


        self.intermediate_conv = nn.Conv3d(filters, filters // 8, kernel_size=1, stride=1, padding=padding)

        curr_output_shape = output_size(curr_output_shape, 1, 1, 0)

        output_shape.append(curr_output_shape)
        self.intermediate_conv_reverse = nn.Conv3d(filters // 8, filters, kernel_size=1, stride=1, padding=padding)
        self.dropout = nn.Dropout(dropout)
        
        flatten_output = np.prod(output_shape[-1])*(filters // 8)
        self.reshape = output_shape[-1]

        self.mu = nn.Linear(flatten_output, self.z_dim)
        self.logvar = nn.Linear(flatten_output, self.z_dim)
        self.dec_dense = nn.Linear(self.z_dim + num_cat, flatten_output) #TODO CHECKKK! 



    def set_labels(self, labels):
        self.labels = labels 

    def forward(self, x):
        if self.one_hot:
            c = F.one_hot(self.labels.long(), self.num_cat)
        else:
            c = self.labels
        c = c.squeeze()
        #check c is the right shape 
        #if batch size is 1, c will be 1D, so add extra dimension
        if len(c.shape) == 1:
            c = c.unsqueeze(0)
        #project each label to the same size as the input
        c = c.view(c.shape[0], c.shape[1], 1, 1, 1).expand(c.shape[0], -1, 128, 128, 128)
        
        #concat the labels to the input (ie add extra channels)
        x = torch.cat([x, c], dim=1)
        
        for layer in self.encoder:
            x = layer(x)
        x = self.intermediate_conv(x)
        x = self.dropout(x)
        #flatten all dimensions other than batch
        x = x.view(x.size(0), -1)
        mu = self.dropout(self.mu(x))
        logvar = self.dropout(self.logvar(x))
        return mu, logvar, self.intermediate_conv_reverse, self.dropout, self.dec_dense, self.reshape
      
class Decoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 out_channels,
                 dec_dist,
                 intermediate_resolutions=[8,8], 
                 use_batchnorm=True, 
                 stride=2, 
                 dropout=0.1,
                 **kwargs):

        super().__init__()

        self.input_size = input_dim
        self.z_dim = z_dim
        self.dec_dist = dec_dist
        self.stride = stride
        num_pooling = int(math.log2(self.input_size[1]) - math.log2(float(intermediate_resolutions[0])))

        padding_needed = max(0, math.ceil((self.input_size[1] - 1) / self.stride) * self.stride + 1 - self.input_size[1])
        padding = padding_needed // 2  # Calculate padding on both sides

        #calculate number of filters for the encoder process
        prev_filters = int(min(128, 32 * (2 ** num_pooling)))

        self.decoder = nn.ModuleList()
        self.decoder.append(nn.BatchNorm3d(prev_filters))
        self.decoder.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(num_pooling):
            filters = int(max(32, 128 / (2 ** i)))
            self.decoder.append(nn.ConvTranspose3d(prev_filters, filters, kernel_size=5, stride=self.stride, padding=padding))
            self.decoder.append(nn.BatchNorm3d(filters))
            self.decoder.append(nn.LeakyReLU(0.2, inplace=True))
            prev_filters = filters
        #pad images to original size
        self.decoder.append(nn.Conv3d(filters, out_channels, kernel_size=1, stride=1, padding=padding))

    def forward(self, x):
        for layer in self.decoder:
            
            #if its the last layer, interpolate x back to original size
            if isinstance(layer, nn.Conv3d):
                x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=False)
            x = layer(x)
        x = hydra.utils.instantiate(self.dec_dist, x=x)
        
        return x


class ConditionalDecoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 z_dim, 
                 out_channels,
                 dec_dist,
                 intermediate_resolutions=[8,8], 
                 use_batchnorm=True, 
                 stride=2, 
                 dropout=0.1,
                 num_cat=2,
                 one_hot=False,
                 **kwargs):

        super().__init__()

        self.input_size = input_dim
        self.z_dim = z_dim
        self.dec_dist = dec_dist
        self.stride = stride
        self.one_hot = one_hot
        num_pooling = int(math.log2(self.input_size[1]) - math.log2(float(intermediate_resolutions[0])))

        padding_needed = max(0, math.ceil((self.input_size[1] - 1) / self.stride) * self.stride + 1 - self.input_size[1])
        padding = padding_needed // 2  # Calculate padding on both sides

        #calculate number of filters for the encoder process
        prev_filters = int(min(128, 32 * (2 ** num_pooling)))

        self.decoder = nn.ModuleList()
        self.decoder.append(nn.BatchNorm3d(prev_filters))
        self.decoder.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(num_pooling):
            filters = int(max(32, 128 / (2 ** i)))
            self.decoder.append(nn.ConvTranspose3d(prev_filters, filters, kernel_size=5, stride=self.stride, padding=padding))
            self.decoder.append(nn.BatchNorm3d(filters))
            self.decoder.append(nn.LeakyReLU(0.2, inplace=True))
            prev_filters = filters
        #pad images to original size
        self.decoder.append(nn.Conv3d(filters, out_channels, kernel_size=1, stride=1, padding=padding))

    def set_labels(self, labels):
        self.labels = labels

    def forward(self, x):
        for layer in self.decoder:
            
            #if its the last layer, interpolate x back to original size
            if isinstance(layer, nn.Conv3d):
                x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=False)
            x = layer(x)
        x = hydra.utils.instantiate(self.dec_dist, x=x)
        
        return x
