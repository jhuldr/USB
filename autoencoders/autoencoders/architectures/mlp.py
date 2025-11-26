
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class Discriminator(nn.Module):
    """MLP Discriminator
    
    Args:
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of output dimensions.
        hidden_layer_dim (list): Number of nodes per hidden layer.
        non_linear (bool): Whether to include a ReLU() function between layers.
        bias (bool): Whether to include a bias term in hidden layers.
        dropout_threshold (float): Dropout threshold of layers.
        is_wasserstein (bool): Whether model employs a wasserstein loss.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dim,
        non_linear,
        bias,
        dropout_threshold,
        is_wasserstein
    ):
        super().__init__()
        self.bias = bias
        self.non_linear = non_linear
        self.dropout_threshold = dropout_threshold
        self.is_wasserstein = is_wasserstein

        self.layer_sizes = [input_dim] + hidden_layer_dim + [output_dim]

        lin_layers = [
            nn.Linear(dim0, dim1, bias=self.bias)
            for dim0, dim1 in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]
        self.linear_layers = nn.Sequential(*lin_layers)

    def forward(self, x):
        for it_layer, layer in enumerate(self.linear_layers):
            x = F.dropout(layer(x), self.dropout_threshold, training=self.training)
            if it_layer < len(self.linear_layers) - 1:
                if self.non_linear:
                    x = F.relu(x)
            else:
                if self.is_wasserstein:
                    return x
                elif self.layer_sizes[-1] > 1:
                    x = nn.Softmax(dim=-1)(x)
                else:
                    x = torch.sigmoid(x)
        return x

