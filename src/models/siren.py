# SineLayer adapted from WIRE (https://github.com/vishwa91/wire, modules/siren.py),
# Copyright (c) 2022 Vishwanath Saragadam, which follows SIREN
# (https://github.com/vsitzmann/siren), Copyright (c) 2020 Vincent Sitzmann.
# Both MIT License; see THIRD_PARTY_LICENSES.md.
import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30,
        sigma0: float = 10.0,
        trainable: bool = True,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.batchnorm = batchnorm
        self.trainable = trainable

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Control whether weights are trainable
        self.linear.weight.requires_grad = trainable
        if bias:
            self.linear.bias.requires_grad = trainable

        self.init_weights()

        if self.batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(out_features)

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input: torch.Tensor):
        result = torch.sin(self.omega_0 * self.linear(input))
        if self.batchnorm:
            result = self.batchnorm_layer(result)
        return result

    def forward_with_intermediate(self, input: torch.Tensor):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
