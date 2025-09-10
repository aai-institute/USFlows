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


class LayerNormChannelsND(nn.Module):
    """Channel-wise LayerNorm for N-D spatial tensors (batch, channel, *spatial)
    Creates gamma and beta parameters shaped (1, C, 1, ..., 1) with `num_spatial_dims` trailing ones.
    """

    def __init__(self, c_in, num_spatial_dims: int = 2, eps=1e-5):
        super().__init__()
        shape = (1, c_in) + (1,) * num_spatial_dims
        self.gamma = nn.Parameter(torch.ones(*shape))
        self.beta = nn.Parameter(torch.zeros(*shape))
        self.eps = eps

    def forward(self, x):
        # mean/var over channels
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        y = y * self.gamma + self.beta
        return y


class GatedConvND(nn.Module):
    """Gated residual block that adapts to 1/2/3-D convolutions.
    The residual is automatically projected if channel counts differ.
    """

    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        nonlinearity: callable = nn.ReLU(),
        input_rank: int = 2,
    ):
        super().__init__()
        assert stride == 1, "Stride > 1 cannot be used to skip connection."

        conv_map = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        if input_rank not in conv_map:
            raise ValueError(f"Unsupported input rank {input_rank}")
        Conv = conv_map[input_rank]

        # first conv reduces/increases to intermediate channels (we choose c_out)
        self.net = nn.Sequential(
            nonlinearity,
            Conv(
                c_in,
                c_out,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
            ),
            nonlinearity,
            # final 1x1 conv produces 2 * c_out channels to split into val/gate
            Conv(
                c_out,
                2 * c_out,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
        )

        # projection for residual if channel counts differ
        if c_in != c_out:
            self.proj = Conv(c_in, c_out, kernel_size=1, padding=0)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        res = val * torch.sigmoid(gate)
        if self.proj is not None:
            x = self.proj(x)
        return x + res


class LayerNormVector(nn.Module):
    """LayerNorm for vector inputs shaped (batch, features)."""
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.layernorm = nn.LayerNorm(features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (batch, features) or (batch, features, 1) or (1, batch, features)
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.view(x.shape[0], x.shape[1])
        if x.dim() == 3 and x.shape[0] == 1 and x.shape[2] != 1:
            # shape (1, B, C) -> (B, C)
            x = x.permute(1, 2, 0).contiguous().view(x.shape[1], x.shape[2])
        return self.layernorm(x)


class GatedMLP(nn.Module):
    """Gated residual MLP block analogous to GatedConvND for vector inputs."""
    def __init__(self, in_features: int, out_features: int, nonlinearity: callable = nn.ReLU()):
        super().__init__()
        self.net1 = nn.Sequential(
            nonlinearity,
            nn.Linear(in_features, out_features),
            nonlinearity,
            nn.Linear(out_features, 2 * out_features),
        )
        # projection for residual if dims differ
        if in_features != out_features:
            self.proj = nn.Linear(in_features, out_features)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        out = self.net1(x)
        val, gate = out.chunk(2, dim=1)
        res = val * torch.sigmoid(gate)
        if self.proj is not None:
            x = self.proj(x)
        return x + res


class ConvNet(nn.Module):
    """Generic ConvNet that adapts to input topology (vector/1D/2D/3D) using `in_dims`.

    - Vector case (len(in_dims) == 1): builds an MLP path using GatedMLP blocks,
      LayerNormVector for normalization and Linear layers for in/out projections.
      Conv-specific kwargs are ignored for this case.
    - Spatial case (len(in_dims) > 1): preserves existing ConvND behavior using
      GatedConvND, LayerNormChannelsND etc.

    The forward normalizes a few common input layouts for backward compatibility:
      - vector path accepts (batch, features), (batch, features, 1), (1, batch, features)
      - conv path accepts (batch, channels, width/height/...) and also (L, batch, channels) or (1, batch, channels)
    """

    def __init__(
        self,
        in_dims: Iterable[int],
        c_hidden: List[int],
        c_out: int = -1,
        nonlinearity: any = nn.ReLU(),
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
        normalize_layers: bool = True,
        gating: bool = True,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        if not isinstance(in_dims, Iterable):
            raise ValueError("in_dims must be an iterable like [C, H, W] or [C] for vector")
        in_dims = list(in_dims)
        c_in = int(in_dims[0])
        is_vector = len(in_dims) == 1
        c_out = c_out if c_out > 0 else c_in

        if is_vector:
            # Build MLP / vector path
            assert len(c_hidden) > 0 and all([h > 0 for h in c_hidden]), "c_hidden must be non-empty list of positive ints"
            layers = []
            # initial linear projection
            first_hidden = int(c_hidden[0])
            layers.append(nn.Linear(c_in, first_hidden))
            # hidden blocks
            for i in range(len(c_hidden)):
                in_ch = int(c_hidden[i - 1]) if i > 0 else first_hidden
                out_ch = int(c_hidden[i])
                if gating:
                    layers.append(GatedMLP(in_ch, out_ch, nonlinearity=nonlinearity))
                else:
                    layers.append(nn.Sequential(nonlinearity, nn.Linear(in_ch, out_ch)))
                if normalize_layers:
                    layers.append(LayerNormVector(out_ch))
            # final linear to c_out
            layers.append(nn.Linear(int(c_hidden[-1]), c_out))
            self.nn = nn.Sequential(*layers)
            self.is_vector = True
            self._vector_in_features = c_in
        else:
            # Spatial / conv path: preserve existing ConvND behavior
            input_rank = max(1, len(in_dims) - 1)
            conv_map = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
            if input_rank not in conv_map:
                raise ValueError(f"Unsupported input rank {input_rank}")
            Conv = conv_map[input_rank]

            assert len(c_hidden) > 0 and all([h > 0 for h in c_hidden]), "c_hidden must be non-empty list of positive ints"
            first_hidden = c_hidden[0]
            layers = []
            layers.append(
                Conv(
                    c_in,
                    first_hidden,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                )
            )

            for i in range(len(c_hidden)):
                in_ch = c_hidden[i - 1] if i > 0 else first_hidden
                out_ch = c_hidden[i]
                if gating:
                    layers.append(
                        GatedConvND(
                            in_ch,
                            out_ch,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                            nonlinearity=nonlinearity,
                            input_rank=input_rank,
                        )
                    )
                    layers.append(nonlinearity)
                else:
                    layers.append(
                        Conv(
                            in_ch,
                            out_ch,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                        )
                    )
                    layers.append(nonlinearity)

                if normalize_layers:
                    layers.append(LayerNormChannelsND(out_ch, num_spatial_dims=input_rank))

            # final conv from last hidden -> c_out
            layers.append(
                Conv(
                    c_hidden[-1],
                    c_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                )
            )
            self.nn = nn.Sequential(*layers)
            self.is_vector = False
            self._spatial_rank = input_rank

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if self.is_vector:
            # Accept (batch, features) or (batch, features, 1) or (1, batch, features)
            if x.dim() == 3 and x.shape[-1] == 1:
                x = x.view(x.shape[0], x.shape[1])
            if x.dim() == 3 and x.shape[0] == 1 and x.shape[2] != 1:
                x = x.permute(1, 2, 0).contiguous().view(x.shape[1], x.shape[2])
            # final ensure shape (batch, features)
            if x.dim() != 2:
                x = x.view(x.shape[0], -1)
            return self.nn(x)
        else:
            # conv path: normalize shapes to (batch, channels, width/...)
            if x.dim() == 3:
                # detect (L, B, C) style where last dim matches channel count
                if x.shape[2] == int(getattr(self, "_spatial_rank", 1) and self.nn[0].in_channels) and x.shape[0] != x.shape[1]:
                    try:
                        x = x.permute(1, 2, 0).contiguous()
                    except Exception:
                        pass
                # also handle (1, B, C)
                elif x.shape[0] == 1 and x.shape[2] == self.nn[0].in_channels:
                    x = x.permute(1, 2, 0).contiguous()
            return self.nn(x)


class ConvNet2D(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int = 3,
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


class CondConvNet(ConvNet):
    """Conditional ConvNet that appends a context channel to the input.

    Mirrors `ConvNet` but increases the input channel count by one and expands
    the supplied `context` tensor to match the spatial topology of `x` before
    concatenation.
    """

    def __init__(
        self,
        in_dims: Iterable[int],
        c_hidden: List[int],
        c_out: int = -1,
        nonlinearity: any = nn.ReLU(),
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
        normalize_layers: bool = True,
        gating: bool = True,
        **kwargs,
    ):
        # if c_out < 0 we'll let parent compute it, but ConvNet expects c_out>0 or uses c_in
        # increase input channels by 1
        if not isinstance(in_dims, Iterable):
            raise ValueError("in_dims must be an iterable like [C, H, W]")
        in_dims = list(in_dims)
        in_dims_with_ctx = [in_dims[0] + 1] + in_dims[1:]

        super().__init__(
            in_dims=in_dims_with_ctx,
            c_hidden=c_hidden,
            c_out=c_out,
            nonlinearity=nonlinearity,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            normalize_layers=normalize_layers,
            gating=gating,
            **kwargs,
        )

        # Keep original in_dims for forward-time spatial handling
        self._orig_in_dims = in_dims

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Append expanded context as an extra channel and forward through network.

        context can be scalar, vector, or tensor with batch dim. It will be
        reshaped/expanded to (batch,1,*spatial) to match `x` before concatenation.
        """
        size_in = x.shape
        # default context
        if context is None:
            context = torch.tensor([0.0], device=x.device)
        else:
            if not isinstance(context, torch.Tensor):
                context = torch.tensor(context, device=x.device)

        # reshape context to have trailing singleton spatial dims to match x
        n_context_dims = len(context.shape)
        n_input_dims = len(x.shape)
        n_dims = n_input_dims - n_context_dims
        if n_dims > 0:
            context = context.reshape(*context.shape, *([1] * n_dims))

        # spatial dims to expand to
        spatial = x.shape[2:]

        # now expand to (batch, 1, *spatial). If context already has batch dim, expand will keep it.
        try:
            context = context.expand(x.shape[0], 1, *spatial)
        except Exception:
            # final fallback: create zeros
            context = torch.zeros((x.shape[0], 1, *spatial), device=x.device)

        x = torch.cat([x, context], dim=1)
        return self.nn(x)


class CondConvNet2D(ConvNet2D):
    def __init__(
        self,
        c_in: int,
        c_hidden: int = 3,
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
