from time import sleep
import torch

from math import ceil
import math
from typing import Iterable, List, Optional, Tuple, Union

import torch
from pyro.nn import DenseNN
from torch import nn
from typing import List, Optional, Tuple, Union


class AdditiveAffineNN(torch.nn.Module):
    """Provides a dense NN that computes loc and log_scale parameter for an affine transform that is purely additive, i.e. the log_scale component
    always returns the 0 vector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        nonlinearity: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()

        self.loc_fnc = DenseNN(
            input_dim, hidden_dims, [output_dim], nonlinearity=nonlinearity
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loc = self.loc_fnc(x)
        log_scale = torch.zeros_like(loc)
        return [loc, log_scale]


class LayerNormChannels(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        """
        This module applies layer norm across channels in an image.
        Args:
            c_in: Number of channels of the input
            eps: Small constant to stabilize std
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        y = y * self.gamma + self.beta
        return y


class GatedConv(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        kernel_size=3,
        padding=1,
        stride=1,
        nonlinearity: callable = nn.ReLU(),
        dilation=1,
    ):
        """
        This module applies a two-layer convolutional ResNet block with input gate
        Args:
            c_in: Number of channels of the input
            c_hidden: Number of hidden dimensions we want to model (usually similar to c_in)
        """
        super().__init__()

        assert stride == 1, "Stride > 1 cannot be used to skip connection."

        self.net = nn.Sequential(
            nonlinearity,
            nn.Conv2d(
                c_in,
                c_hidden,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
            ),
            nonlinearity,
            # The kernel size below is set to 1 to reduce the number of parameters.
            nn.Conv2d(
                c_hidden,
                2 * c_in,
                kernel_size=1,
                padding=padding,
                stride=stride,
                dilation=dilation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards method

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: network output.
        """
        out = self.net(x)
        # Split the output into filter and gate components.
        val, gate = out.chunk(2, dim=1)
        # Apply the gated residual connection after activation of the gate.
        ret = x + val * torch.sigmoid(gate)

        assert ret.shape == x.shape, f"Shape mismatch: {ret.shape} != {x.shape}"

        return ret


class ConvNet2D(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int = 3,
        rescale_hidden: int = 2,
        c_out: int = -1,
        num_layers: int = 3,
        nonlinearity: any = nn.ReLU(),
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        normalize_layers: bool = True,
        gating: bool = True,
    ):
        """
        Module that summarizes the previous blocks to a full convolutional
        neural network.

        Args:
            c_in: Number of input channels
            c_hidden: Number of hidden dimensions to use within the network
            rescale_hidden: Factor by which to rescale hight and width the
                hidden before and after the hidden layers.
            c_out: Number of output channels. If -1, the numberinput channels
                are used (affine coupling)
            num_layers: Number of gated ResNet blocks to apply
            nonlinearity: Nonlinearity to use within the network. ReLU
                allows to maintain piece-wise affinity.
            kernel_size: Size of the convolutional kernel.
            padding: Padding to apply to the convolutional layers. If None, the
                padding is set to half the kernel size.
        """
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.nonlinearity = nonlinearity
        c_out = c_out if c_out > 0 else c_in
        layers = []
        layers += [
            nn.Conv2d(
                c_in,
                c_hidden,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
            ),
        ]
        if rescale_hidden != 1:
            layers += [nn.MaxPool2d(rescale_hidden)]

        for layer_index in range(num_layers):
            if gating:
                layers += [
                    GatedConv(
                        c_hidden,
                        c_hidden,
                        kernel_size=kernel_size,
                        padding=padding,
                        stride=stride,
                        dilation=dilation,
                    ),
                    # nn.Conv2d(c_hidden, c_hidden, kernel_size=kernel_size, padding=padding),
                    nonlinearity,
                ]
            else:
                layers += [
                    nn.Conv2d(
                        c_hidden,
                        c_hidden,
                        kernel_size=kernel_size,
                        padding=padding,
                        stride=stride,
                        dilation=dilation,
                    ),
                    nonlinearity,
                ]
            if normalize_layers:
                layers += [
                    LayerNormChannels(c_hidden),
                ]

        # compute padding and output padding for rescaling via transposed convolutions
        if rescale_hidden != 1:
            diff = rescale_hidden - kernel_size
            if diff < 0:
                outpad = diff % 2
                pad = ceil(abs(diff) / 2.0)
            else:
                outpad = diff
                pad = 0

            layers += [
                nn.ConvTranspose2d(
                    c_hidden,
                    c_hidden,
                    kernel_size=kernel_size,
                    stride=stride,
                    output_padding=outpad,
                    padding=pad,
                ),
                nonlinearity,
            ]

        layers += [
            nn.Conv2d(
                c_hidden,
                c_out,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
            )
        ]
        self.nn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Forwards method

        Args:
            x: Input tensor.

        Returns:
            Network output.
        """
        return self.nn(x)


class CondConvNet2D(ConvNet2D):
    def __init__(
        self,
        c_in: int,
        c_hidden: int = 3,
        rescale_hidden: int = 2,
        c_out: int = -1,
        num_layers: int = 3,
        nonlinearity: any = nn.ReLU(),
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: int = None,
        **kwargs,  # Collect additional keyword arguments
    ):
        """
        Module that summarizes the previous blocks to a full convolutional
        neural network.

        Args:
            c_in: Number of input channels
            c_hidden: Number of hidden dimensions to use within the network
            rescale_hidden: Factor by which to rescale hight and width the
                hidden before and after the hidden layers.
            c_out: Number of output channels. If -1, the numberinput channels
                are used (affine coupling)
            num_layers: Number of gated ResNet blocks to apply
            nonlinearity: Nonlinearity to use within the network. ReLU
                allows to maintain piece-wise affinity.
            kernel_size: Size of the convolutional kernel.
            padding: Padding to apply to the convolutional layers. If None, the
                padding is set to half the kernel size.
        """
        # For c_out < 0, the parent class will set c_out to c_in. As we increase
        # c_in by one below, we need to set c_out explicitly.
        if c_out < 0:
            c_out = c_in

        super().__init__(
            c_in=c_in + 1,
            c_hidden=c_hidden,
            rescale_hidden=rescale_hidden,
            c_out=c_out,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            **kwargs,  # Pass additional keyword arguments to the parent class
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward method for conditional convolutional network.

        Args:
            x: Input tensor.
            context: Context tensor.

        Returns:
            Network output.
        """
        size_in = x.shape
        # Make sure to create a new obj. to avoid inplace operations.
        if context is None:
            context = torch.Tensor([0]).to(x.device)
        else:
            if not isinstance(context, torch.Tensor):
                context = torch.tensor(context).to(x.device)
            n_context_dims = len(context.shape)
            n_input_dims = len(x.shape)
            n_dims = n_input_dims - n_context_dims
            if n_dims > 0:
                shape = tuple(context.shape) + (1,) * n_dims
                context = context.reshape(*shape)

        height, width = x.shape[-2:]
        # Expand the context to the size of the input image.
        # Batch, Channel, Height, Width
        context = context.expand(x.shape[0], 1, height, width)
        x = torch.cat([x, context], dim=1)

        size_target = torch.Size([size_in[0], size_in[1] + 1, size_in[2], size_in[3]])
        assert x.shape == size_target, f"Shape mismatch: {x.shape} != {size_target}"
        return self.nn(x)


class ConditionalDenseNN(torch.nn.Module):
    """
    *NOTE*: This class is derived from pyro's ConditionalDenseNN.
    An implementation of a simple dense feedforward network taking a context variable, for use in, e.g.,
    some conditional flows such as :class:`pyro.distributions.transforms.ConditionalAffineCoupling`.

    Example usage:

    >>> input_dim = 10
    >>> context_dim = 5
    >>> x = torch.rand(100, input_dim)
    >>> z = torch.rand(100, context_dim)
    >>> nn = ConditionalDenseNN(input_dim, context_dim, [50], param_dims=[1, input_dim, input_dim])
    >>> a, b, c = nn(x, context=z)  # parameters of size (100, 1), (100, 10), (100, 10)

    :param input_dim: the dimensionality of the input
    :type input_dim: int
    :param context_dim: the dimensionality of the context variable
    :type context_dim: int
    :param hidden_dims: the dimensionality of the hidden units per layer
    :type hidden_dims: list[int]
    :param param_dims: shape the output into parameters of dimension (p_n,) for p_n in param_dims
        when p_n > 1 and dimension () when p_n == 1. The default is [1, 1], i.e. output two parameters of dimension ().
    :type param_dims: list[int]
    :param nonlinearity: The nonlinearity to use in the feedforward network such as torch.nn.ReLU(). Note that no
        nonlinearity is applied to the final network output, so the output is an unbounded real number.
    :type nonlinearity: torch.nn.Module

    """

    def __init__(
        self,
        input_dim,
        context_dim,
        hidden_dims,
        out_dim,
        nonlinearity=torch.nn.ReLU(),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim

        # Create masked layers
        layers = [
            torch.nn.Linear(input_dim, hidden_dims[0]),
            torch.nn.Linear(context_dim, hidden_dims[0]),
        ]
        for i in range(1, len(hidden_dims)):
            layers.append(torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        layers.append(torch.nn.Linear(hidden_dims[-1], out_dim))
        self.layers = torch.nn.ModuleList(layers)

        # Save the nonlinearity
        self.f = nonlinearity

    def forward(self, x, context=None):

        h = self.layers[0](x)
        if context is not None:
            h = h + self.layers[1](context)

        h = self.f(h)

        for layer in self.layers[2:-1]:
            h = self.f(layer(h))
        h = self.layers[-1](h)

        return h


class BottleneckConv(nn.Module):
    def __init__(
        self,
        c_in: Iterable[int],
        c_hidden_in: Iterable[int],
        c_hidden_out: Iterable[int],
        in_dims: Iterable[int],
        c_hidden: int = 3,
        nonlinearity: any = nn.ReLU(),
        kernel_size: int = 3,
    ):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Args:
            c_in: Number of input channels
            c_hidden: Number of hidden dimensions to use within the network
            rescale_hidden: Factor by which to rescale hight and width the hidden before and after the hidden layers.
            c_out: Number of output channels. If -1, the numberinput channels are used (affine coupling)
            num_layers: Number of gated ResNet blocks to apply
        """
        super().__init__()

        self.in_dims = in_dims
        self.n_pixels = math.prod(in_dims[1:])

        in_convolutions = []
        in_convolutions += [
            nn.Conv2d(c_in, c_hidden, kernel_size=kernel_size, padding="same"),
            nn.Conv2d(c_hidden, 1, kernel_size=kernel_size, padding="same"),
        ]
        self.in_convolutions = nn.ModuleList(in_convolutions)

        linear_layers = []
        linear_layers += [
            nn.Linear(self.n_pixels, self.n_pixels),
            nn.Linear(self.n_pixels, self.n_pixels),
        ]
        self.linear_layers = nn.ModuleList(linear_layers)

        out_convolutions = []
        out_convolutions += [
            nn.Conv2d(1, c_hidden, kernel_size=kernel_size, padding="same"),
            nn.Conv2d(c_hidden, c_in, kernel_size=kernel_size, padding="same"),
        ]
        self.out_convolutions = nn.ModuleList(out_convolutions)

        self.nonlinearity = nonlinearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards method

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: network output.
        """
        for conv in self.in_convolutions:
            x = conv(x)
            x = self.nonlinearity(x)

        x = x.view(x.shape[0], -1)
        for layer in self.linear_layers:
            x = layer(x)
            x = self.nonlinearity(x)

        x = x.view(x.shape[0], 1, *self.in_dims[1:])
        for conv in self.out_convolutions:
            x = conv(x)
            x = self.nonlinearity(x)
        return x
