import math
import numpy as np
import pyro 
from pyro import distributions as dist
from pyro.nn import DenseNN
from pyro.distributions.transforms import AffineCoupling, Permute, LowerCholeskyAffine
from pyro.infer import SVI
from typing import List
import torch

from sklearn.datasets import load_digits
from tqdm import tqdm

import torch
from torch.distributions.transforms import Transform
from torch.distributions.utils import lazy_property
from torch.functional import F
from torch import linalg as LA
from torch.nn import init

from pyro.distributions import constraints



class ScaleTransform(dist.TransformModule):
    """Implementation of a bijective scale transform. Applies a transform $y = \mathrm{diag}(\mathbf{scale})x$, where scale is a learnable parameter of dimension $\mathbf{dim}$
    
    *Note:* The implementation does not enforce the non-zero constraint of the diagonal elements of $\mathbf{U}$ during training.
    See :func:`add_jitter` and :func:`is_feasible` for a way to ensure that the transformation is invertible.
    """
    def __init__(self, dim: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.scale = torch.nn.Parameter(torch.empty(dim))
        self.init_params()
        
        self.bijective = True
        self.domain = dist.constraints.real_vector
        self.codomain = dist.constraints.real_vector
    
    def init_params(self):
        """initialization of the parameters"""
        fan_in = self.dim
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.scale, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.exp(self.log_scale)

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.exp(-self.log_scale)
    
    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def _inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.backward(x)
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return self.scale.sum().abs().log()

    def sign(self) -> int:
        return self.log_scale.sum().sign() 
    
    def is_feasible(self) -> bool:
        """Checks if the layer is feasible, i.e. if the diagonal elements of $\mathbf{U}$ are all positive"""
        return (self.scale > 0).all()
    
    def add_jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to the diagonal elements of $\mathbf{U}$. This is useful to ensure that the transformation is invertible."""
        perturbation = torch.randn(self.dim, device=self.U_raw.device) * jitter
        self.U_raw = self.scale + perturbation 

class Permute(pyro.distributions.TransformModule):
    """Permutation transform."""
    bijective = True
    volume_preserving = True

    def __init__(self, permutation, *, dim=-1, cache_size=1):
        super().__init__(cache_size=cache_size)

        if dim >= 0:
            raise ValueError("'dim' keyword argument must be negative")

        self.permutation = permutation
        self.dim = dim

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(constraints.real, -self.dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(constraints.real, -self.dim)

    @lazy_property
    def inv_permutation(self):
        result = torch.empty_like(self.permutation, dtype=torch.long)
        result[self.permutation] = torch.arange(
            self.permutation.size(0), dtype=torch.long, device=self.permutation.device
        )
        return result.to(self.permutation.device)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        return x.index_select(self.dim, self.permutation)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        return y.index_select(self.dim, self.inv_permutation)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not autoregressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        return torch.zeros(
            x.size()[: -self.event_dim], dtype=x.dtype, layout=x.layout, device=x.device
        )


    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return Permute(self.permutation, cache_size=cache_size)
    
    
class LUTransform(dist.TransformModule):
    """Implementation of a linear bijection transform. Applies a transform $y = \mathbf{L}\mathbf{U}x$, where $\mathbf{L}$ is a 
    lower triangular matrix with unit diagonal and $\mathbf{U}$ is an upper triangular matrix. Bijectivity is guaranteed by
    requiring that the diagonal elements of $\mathbf{U}$ are positive and the diagonal elements of  $\mathbf{L}$ are all $1$.
    
    *Note:* The implementation does not enforce the non-zero constraint of the diagonal elements of $\mathbf{U}$ during training.
    See :func:`add_jitter` and :func:`is_feasible` for a way to ensure that the transformation is invertible.
    """
    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.L_raw = torch.nn.Parameter(torch.empty(dim, dim))
        self.U_raw = torch.nn.Parameter(torch.empty(dim, dim))
        self.bias = torch.nn.Parameter(torch.empty(dim))
        self.dim = dim
        
        self.init_params()
        
        
        self.input_shape = dim
        self.bijective = True
        self.domain = dist.constraints.real_vector
        self.codomain = dist.constraints.real_vector
        
        self.L_mask = torch.tril(torch.ones(dim, dim), diagonal=1)
        self.U_mask = torch.triu(torch.ones(dim, dim), diagonal=0)
        
        self.L_raw.register_hook(lambda grad: grad * self.L_mask)
        self.U_raw.register_hook(lambda grad: grad * self.U_mask)
        
    def init_params(self):
        # Adopted from pytorch's Linear layer parameter initialization.

        init.kaiming_uniform_(self.L_raw, a=math.sqrt(self.dim))
        with torch.no_grad():
            self.L_raw.copy_(self.L_raw.tril().fill_diagonal_(1))
        init.kaiming_uniform_(self.U_raw, a=math.sqrt(self.dim))
        with torch.no_grad():
            self.U_raw.copy_(self.U_raw.triu())
        if self.bias is not None:
            fan_in = self.dim
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the affine transform $(LU)x + \mathrm{bias}$
        
        :param x: input tensor
        :type x: torch.Tensor
        :return: transformed tensor $(LU)x + \mathrm{bias}$
        """
        M_inv = LA.inv(self.weight)
        return torch.functional.F.linear(x, M_inv, self.bias)

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """Computes the inverse transform $(LU)^{-1}(y - \mathrm{bias})$ 
        
        :param y: input tensor
        :type y: torch.Tensor
        :return: transformed tensor $(LU)^{-1}(y - \mathrm{bias})$"""
        return  F.linear(y - self.bias, self.weight)
    
    @property
    def L(self):
        """The lower triangular matrix $\mathbf{L}$ of the layers LU decomposition"""
        return self.L_raw.tril().fill_diagonal_(1)
    
    @property
    def U(self):
        """The upper triangular matrix $\mathbf{U}$ of the layers LU decomposition"""
        return self.U_raw.triu()
    
    @property
    def weight(self):
        """Weight matrix of the affine transform"""
        return LA.matmul(self.L, self.U)
    
    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.backward(y)
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return LA.slogdet(self.weight)[1]
    
    def sign(self) -> int:
        return -1 * LA.slogdet(self.weight)[0]
    
    def to(self, device) -> None:
        self.L_raw = self.L_raw.to(device)
        self.U_raw = self.U_raw.to(device)
        self.bias = self.bias.to(device)
        return super().to(device)
    
    def is_feasible(self) -> bool:
        """Checks if the layer is feasible, i.e. if the diagonal elements of $\mathbf{U}$ are all positive"""
        return (self.U_raw.diag() > 0).all()
    
    def add_jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to the diagonal elements of $\mathbf{U}$. This is useful to ensure that the transformation is invertible."""
        perturbation = torch.randn(self.dim, device=self.U_raw.device) * jitter
        with torch.no_grad():
            self.U_raw.copy_(self.U_raw + perturbation * torch.eye(self.dim, device=self.U_raw.device)) 

class MaskedCoupling(dist.TransformModule):
    """Implementation of a masked coupling layer. The layer is defined by a mask that specifies which dimensions are passed through unchanged and which are transformed. 
    The layer is defined by a bijective function $y = \mathrm{mask} \odot x + (1 - \mathrm{mask}) \odot (x + \mathrm{transform}(x))$, where $\mathrm{mask}$ is a binary mask, 
    $\mathrm{transform}$ is a bijective function, and $\odot$ denotes element-wise multiplication.
    """
    def __init__(self, mask: torch.Tensor, conditioner: torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask
        self.conditioner = conditioner
        self.input_shape = mask.shape
        self.bijective = True
        self.domain = dist.constraints.real_vector
        self.codomain = dist.constraints.real_vector
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_masked = x * self.mask
        x_transformed = x + (1 - self.mask) * self.conditioner(x_masked)
        return x_transformed

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        y_masked = y * self.mask
        y_transformed = y - (1 - self.mask) * self.conditioner(y_masked)
        return  y_transformed
    
    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.backward(y)
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x_masked = x * self.mask
        return 0.
    
    def sign(self) -> int:
        return 0.
    
    def to(self, device):
        self.mask = self.mask.to(device)
        return super().to(device)