import math
from abc import ABC, abstractmethod
from time import sleep
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pyro
import torch
from torch import nn
from pyro import distributions as dist
from pyro.distributions import constraints
from pyro.distributions.transforms import Permute
from pyro.infer import SVI
from pyro.nn import DenseNN
from sklearn.datasets import load_digits
from torch import linalg as LA
from torch.distributions.transforms import Transform
from torch.distributions.utils import lazy_property
from torch.functional import F
from torch.nn import init
from tqdm import tqdm

from src.veriflow.linalg import solve_triangular

class BaseTransform(dist.TransformModule):
    """Base class for transforms. Implemented as a thin layer on top of pyro's TransformModule. The baseTransform
    provides additional methods for checking and constraints on the parameters of the transform.
    """

    def __init__(self, *args, **kwargs):
        super(dist.TransformModule, self).__init__(*args, **kwargs)

    @abstractmethod
    def is_feasible(self) -> bool:
        """Checks if the layer is feasible."""
        return True

    @abstractmethod
    def jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to the layer. This is useful to ensure that the transformation is invertible."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()
    
    @abstractmethod
    def backward(self, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()
    
    @abstractmethod
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform.
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor
            
        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        raise NotImplementedError()
    
    def log_prior(self) -> torch.Tensor:
        """Defines a uniform (pseudo-)prior."""
        return 0.0
        


class ScaleTransform(BaseTransform):
    """Implementation of a bijective scale transform. Applies a transform $y = \mathrm{diag}(\mathbf{scale})x$, where scale is a learnable parameter of dimension $\mathbf{dim}$

    *Note:* The implementation does not enforce the non-zero constraint of the diagonal elements of $\mathbf{U}$ during training.
    See :func:`add_jitter` and :func:`is_feasible` for a way to ensure that the transformation is invertible.
    """

    def __init__(
        self,
        in_dims: torch.Tensor,
        prior_scale: float = 1.0,
        *args,
        **kwargs
    ) -> None:
        """ Initializes the scale transform."""
        super().__init__(*args, **kwargs)
        self.in_dims = in_dims
        self.prior_scale = prior_scale
        self.dim = math.prod(in_dims) if isinstance(in_dims, Iterable) else in_dims
        self.scale = torch.nn.Parameter(torch.empty(in_dims))
        self.init_params()

        self.bijective = True
        self.domain = dist.constraints.real_vector
        self.codomain = dist.constraints.real_vector

    def init_params(self):
        """initialization of the parameters"""
        dim = self.dim
        bound = 1 / math.sqrt(dim) if dim > 0 else 0
        init.uniform_(self.scale, -bound, bound)

    def forward(self, x: torch.Tensor, context = None) -> torch.Tensor:
        """ Computes the affine transform $\mathbf{scale}x$
        
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: transformed tensor $\mathbf{scale}x$
        """
        return x * self.scale

    def backward(self, x: torch.Tensor, context = None) -> torch.Tensor:
        """ Computes the inverse transform $\mathbf{scale}^{-1}x$
        
        
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: transformed tensor $\mathbf{scale}^{-1}x$
        """
        return x / self.scale

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`forward`"""
        return self.forward(x)

    def _inverse(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`backward`"""
        return self.backward(x)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform $\mathbf{scale}x$.
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            float: log absolute determinant of the Jacobian of the transform $\mathbf{scale}x$
        """
        return self.scale.abs().log().sum()

    def sign(self) -> int:
        """ Computes the sign of the determinant of the Jacobian of the transform $\mathbf{scale}x$."""
        return 1 if (self.scale < 0).int().sum() % 2 == 0 else -1

    def is_feasible(self) -> bool:
        """Checks if the layer is feasible, i.e. if the diagonal elements of $\mathbf{U}$ are all positive"""
        return (self.scale != 0).all()

    def add_jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to the diagonal elements of $\mathbf{U}$."""
        perturbation = torch.randn(self.in_dims, device=self.U_raw.device) * jitter
        self.U_raw = self.scale + perturbation
        
    def log_prior(self) -> torch.Tensor:
        """Defines a log-normal prior on the diagonal elements of U Matrix,
        implicitply defining a log-normal prior on the absolute determinat
        of the transform."""
        d = self.dim

        # log-density of Normal in log-space
        x = self.scale.abs().log()
        log_prior = -(x * x).sum() / (self.prior_scale**2 / (d))
        # Change of variables to input space
        log_prior += -x.sum()
        return log_prior


class Permute(BaseTransform):
    """Permutation transform."""

    bijective = True
    volume_preserving = True

    def __init__(self, permutation: torch.tensor, *, dim: int = -1, cache_size: int = 1) -> None:
        """ Initializes the permutation transform.
        
        Args:
            permutation (torch.Tensor): permutation vector
        """
        super().__init__(cache_size=cache_size)

        if dim >= 0:
            raise ValueError("'dim' keyword argument must be negative")

        self.permutation = permutation
        self.dim = dim

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        """ Returns the domain of the transform."""
        return constraints.independent(constraints.real, -self.dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        """ Returns the codomain of the transform."""
        return constraints.independent(constraints.real, -self.dim)

    @lazy_property
    def inv_permutation(self):
        """ Returns the inverse permutation."""
        result = torch.empty_like(self.permutation, dtype=torch.long)
        result[self.permutation] = torch.arange(
            self.permutation.size(0), dtype=torch.long, device=self.permutation.device
        )
        return result.to(self.permutation.device)

    def _call(self, x: torch.Tensor):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        return x.index_select(self.dim, self.permutation)

    def _inverse(self, y: torch.Tensor):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        return y.index_select(self.dim, self.inv_permutation)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None):
        """
        Calculates the element-wise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not auto-regressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        return torch.zeros(
            x.size()[: -self.event_dim], dtype=x.dtype, layout=x.layout, device=x.device
        )

    def with_cache(self, cache_size: int = 1):
        """ Returns a new :class:`Permute` instance with a given cache size."""
        if self._cache_size == cache_size:
            return self
        return Permute(self.permutation, cache_size=cache_size)


class BijectiveLinearTransform(BaseTransform):
    """Simple implementation of a bijective linear transform. Applies a transform $y = \mathbf{W}x + \mathbf{b}$, where $\mathbf{W}$ is a
    learnable parameter matrix and $\mathbf{b}$ is a learnable bias vector.
    Note: This is a dummy implementation that does not enforce bijectivity nor is it intended to be trained.
    It acts as a simplification of the LU transform for verification purposes.
    """
    
    bijective = True
    volume_preserving = False
    domain = dist.constraints.real_vector
    codomain = dist.constraints.real_vector
    
    def __init__(self, dim: int, m: torch.Tensor, bias: torch.Tensor, m_inv: torch.Tensor = None, *args, **kwargs):
        """ Initializes the linear transform.
        Args:
            dim (int): dimension of the input and output
            m: weight matrix
            bias: bias vector
            m_inv: inverse weight matrix
        """
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.bias = bias
        
        self.forth = torch.nn.Linear(dim, dim, bias=True)
        self.forth.weight = torch.nn.Parameter(m)
        self.forth.bias = torch.nn.Parameter(bias)
        
        self.back = torch.nn.Linear(dim, dim, bias=True)
        self.back.weight = torch.nn.Parameter(m_inv)
        self.back.bias = torch.nn.Parameter(-torch.matmul(m_inv, bias))
        
        self.m_inv = m_inv
        with torch.no_grad():
            self.ladj = torch.linalg.slogdet(m)[1]
        
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor
        
        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        return self.ladj
    
    def forward(self, x: torch.Tensor, context = None) -> torch.Tensor:
        """ Computes the affine transform $y = \mathbf{W}x + \mathbf{b}$.
        
        Args:
            x (torch.Tensor): input tensor
            context (torch.Tensor): context tensor (ignored)
        """
        
        return self.forth(x)
    
    def backward(self, y: torch.Tensor, context = None) -> torch.Tensor:
        """ Computes the inverse transform $y = \mathbf{W}^{-1}x + \mathbf{b}$.
        
        Args:
            y (torch.Tensor): input tensor
            context (torch.Tensor): context tensor (ignored)
        """
        return self.back(y)            

class MaskedCoupling(BaseTransform):
    """Implementation of a masked coupling layer. The layer is defined by a mask that specifies which dimensions are passed through unchanged and which are transformed.
    The layer is defined by a bijective function $y = \mathrm{mask} \odot x + (1 - \mathrm{mask}) \odot (x + \mathrm{transform}(x))$, where $\mathrm{mask}$ is a binary mask,
    $\mathrm{transform}$ is a bijective function, and $\odot$ denotes element-wise multiplication.
    """

    def __init__(
        self, mask: torch.Tensor, conditioner: torch.nn.Module, *args, **kwargs
    ) -> None:
        """ Initializes the masked coupling layer.
        
        Args:
            mask (torch.Tensor): binary mask
            conditioner (torch.nn.Module): NN with same (input/output) shape as mask$
        """
        super().__init__(*args, **kwargs)
        self.mask = mask
        self.conditioner = conditioner
        self.input_shape = mask.shape
        self.bijective = True
        self.domain = dist.constraints.real_vector
        self.codomain = dist.constraints.real_vector

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Computes the affine transform 
        $\mathrm{mask} \odot x + (1 - \mathrm{mask}) \odot (x + \mathrm{transform}(x))$
        
        Args:
            x (torch.Tensor): input tensor
        """

        x_masked = x * self.mask
        if context is None:
            x_transformed = x + (1 - self.mask) * self.conditioner(x_masked)
        else:
            x_transformed = x + (1 - self.mask) * self.conditioner(x_masked, context)
        return x_transformed

    def backward(self, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Computes the inverse transform
        
        Args:
            y (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: transformed tensor
        """
        y_masked = y * self.mask
        if context is None:
            y_transformed = y - (1 - self.mask) * self.conditioner(y_masked)
        else:
            y_transformed = y - (1 - self.mask) * self.conditioner(y_masked, context)
        return y_transformed

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`forward`"""
        return self.forward(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`backward`"""
        return self.backward(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor
            
        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        return 0.0

    def sign(self) -> int:
        """ Computes the sign of the determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            int: sign of the determinant of the Jacobian of the transform
        """
        return 1.0

    def to(self, device):
        """ Moves the layer to a given device
        
        Args:
            device (torch.device): target device
        """
        self.mask = self.mask.to(device)
        
        return super().to(device)

class InverseTransform(BaseTransform):
    """Represents the inverse of a given transform. 
    This is useful for composing transforms"""
    
    def __init__(self, transform: Transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
        self.bijective = transform.bijective
        self.domain = transform.codomain
        self.codomain = transform.domain
        
    def forward(self, x: torch.Tensor, context = None) -> torch.Tensor:
        """ Computes the inverse transform
        Args:
            x (torch.Tensor): input tensor
            context (torch.Tensor): context tensor (ignored)
        """
        return self.transform.backward(x, context)
    
    def backward(self, y: torch.Tensor, context = None) -> torch.Tensor:
        """ Computes the forward transform
        Args:
            y (torch.Tensor): input tensor
            context (torch.Tensor): context tensor (ignored)
        """
        return self.transform.forward(y, context)
    
    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`forward`"""
        return self.forward(x)
    
    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`backward`"""
        return self.backward(y)
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor
            
        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        return -self.transform.log_abs_det_jacobian(x, y, context)

    def sign(self) -> int:
        """ Computes the sign of the determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            int: sign of the determinant of the Jacobian of the transform
        """
        return self.transform.sign()
    
class LeakyReLUTransform(BaseTransform):
    bijective = True
    domain = dist.constraints.real
    codomain = dist.constraints.real
    sign = 1

    def __init__(self, alpha: float = 0.01, *args, **kwargs) -> None:
        """ Initializes the LeakyReLU transform.
        
        Args:
            alpha (float, optional): slope of the negative part of the function. Defaults to 0.01.
        """
        if alpha == 0:
            raise ValueError("alpha must be positive")
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes the LeakyReLU transform
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            torch.Tensor: transformed tensor
        """
        return F.leaky_relu(x, negative_slope=self.alpha)

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """ Computes the inverse transform
        
        Args:
            y (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: transformed tensor
        """
        return F.leaky_relu(y, negative_slope=1 / self.alpha)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`forward`"""
        return self.forward(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`backward`"""
        return self.backward(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor
        
        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        return torch.log(y/x).sum()

class Rotation(BaseTransform):
    """Implements a rotation transform. The transform is defined by two 
    coordinate axes, defining a plane, and a rotation angle."""
    bijective = True
    domain = dist.constraints.real
    codomain = dist.constraints.real
    sign = 1
    ladj = 0
    
    
    def __init__(self, dim, plane: Tuple[int, int], angle: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if dim < 2:
            raise ValueError("dim must be at least 2")
        if plane[0] == plane[1]:
            raise ValueError("plane must be a tuple of different indices")
        if dim <= max(plane):
            raise ValueError("plane indices must be smaller than dim")
        
        self.dim = dim
        self.plane = plane
        self.angle = angle
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes the rotation transform
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            torch.Tensor: transformed tensor
        """
        y = x.clone()
        y[..., self.plane[0]] = x[..., self.plane[0]] * math.cos(self.angle) - x[..., self.plane[1]] * math.sin(self.angle)
        y[..., self.plane[1]] = x[..., self.plane[0]] * math.sin(self.angle) + x[..., self.plane[1]] * math.cos(self.angle)
        return y

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """ Computes the inverse transform
        
        Args:
            y (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: transformed tensor
        """
        y = y.clone()
        y[..., self.plane[0]] = y[..., self.plane[0]] * math.cos(self.angle) + y[..., self.plane[1]] * math.sin(self.angle)
        y[..., self.plane[1]] = -y[..., self.plane[0]] * math.sin(self.angle) + y[..., self.plane[1]] * math.cos(self.angle)
        return y
    
    def as_matrix(self) -> torch.Tensor:
        """ Returns the rotation matrix"""
        R = torch.eye(self.dim)
        R[self.plane[0], self.plane[0]] = math.cos(self.angle)
        R[self.plane[0], self.plane[1]] = -math.sin(self.angle)
        R[self.plane[1], self.plane[0]] = math.sin(self.angle)
        R[self.plane[1], self.plane[1]] = math.cos(self.angle)
        return R

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`forward`"""
        return self.forward(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`backward`"""
        return self.backward(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor
        
        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        return self.ladj
        
        

class CompositeRotation(BaseTransform):
    """Implements a composite rotation transform. The transform is defined by a 
    sequence of rotations  $R_1, \ldots, R_n$."""
    bijective = True
    domain = dist.constraints.real
    codomain = dist.constraints.real
    sign = 1
    ladj = 0
    
    def __init__(self, rotations: List[Rotation], *args, **kwargs) -> None:
        """ Initializes the composite rotation transform.
        
        Args:
            rotations (List[torch.Tensor]): list of rotation matrices
        """
        super().__init__(*args, **kwargs)
        self.rotations = rotations
        self.input_shape = rotations[0].dim
    
    def as_matrix(self) -> torch.Tensor:
        """ Returns the composite rotation matrix"""
        R = torch.eye(self.input_shape)
        for rot in self.rotations:
            R = torch.matmul(R, rot.as_matrix())
        return R
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the composite rotation transform by applying the rotations in sequence"""
        for rot in self.rotations:
            x = rot(x)
            
        return x
    
    def backward(self, y: torch.Tensor, context: Optional[Any] = None) -> torch.Tensor:
        for rot in self.rotations[::-1]:
            y = rot.backward(y)
            
        return y

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`forward`"""
        return self.forward(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`backward`"""
        return self.backward(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor
        
        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        return self.ladj


class AffineTransform(BaseTransform):
    """Interface for an affine transform that offers getters for the
    transformation matrix and the bias vector. 
    The transform is defined by a matrix $A$ and a vector $b$ such that 
    $y = Ax + b$.
    """
    
    bijective = True
    domain = dist.constraints.real_vector
    codomain = dist.constraints.real_vector

    def __init__(self, dim: int, *args, **kwargs):
        """ Initializes the affine transform.
        
        Args:
            dim (int): dimension of the input and output
        """
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.input_shape = dim

    @property
    @abstractmethod
    def matrix(self) -> torch.Tensor:
        """ Returns the transformation matrix"""
        pass

    @property
    @abstractmethod
    def bias(self) -> torch.Tensor:
        """ Returns the bias vector"""
        pass
    
    @abstractmethod
    def inverse_matrix(self) -> torch.Tensor:
        """ Returns the inverse transformation matrix"""
        pass

class HouseholderTransform(AffineTransform):
    """Implements a Householder transform. The transform is defined by a 
    Householder matrix $H = I - 2vv^T$, where $v$ is a vector.
    """
    bijective = True
    domain = dist.constraints.real
    codomain = dist.constraints.real
    sign = 1
    ladj = 0
    
    def __init__(
        self,
        dim: int,
        nvs: int = 1,
        device = "cpu",
        *args,
        **kwargs
    ) -> None:
        """ Initializes the Householder transform.
        
        Args:
            v (torch.Tensor): Householder vector
        """
        super().__init__(dim, *args, **kwargs)
        self.nvs = nvs
        self.dim = dim
        
        # initial random permutation
        indices = torch.randperm(dim)
        w = torch.zeros((dim, dim))
        w[torch.arange(dim), indices] = 1.0
        
        self.vk_householder = nn.Parameter(
            0.2 * torch.randn(nvs, dim, requires_grad=True),
            requires_grad=True,
        )
        self.w_0 = nn.Parameter(
            torch.FloatTensor(w), 
            requires_grad=False, 
        )
        
        self.to(device)
    
    def _construct_householder_permutation(self) -> torch.Tensor:
        """Compute permutation matrix from learned reflection vectors.

        Returns:
            torch.Tensor: Constructed permutation matrix
        """
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(
                w,
                torch.eye(self.dim).to(w.device) \
                    - 2 * torch.ger(vk, vk) / torch.dot(vk, vk)
            )

        return w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the Householder transform.
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            torch.Tensor: transformed tensor
        """
        w = self._construct_householder_permutation()
        w = w.transpose(0, 1).contiguous()
        return torch.matmul(x, w)
    
    def backward(
        self,
        y: torch.Tensor,
        context: Optional[Any] = None
    ) -> torch.Tensor:
        """Computes the inverse transform
        
        Args:
            y (torch.Tensor): input tensor
            
        Returns:
            torch.Tensor: transformed tensor
        """
        w = self._construct_householder_permutation()
        return torch.matmul(y, w.T)
    
    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`forward`"""
        return self.forward(x)
    
    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`backward`"""
        return self.backward(y)
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor
            
        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        return self.ladj
    
    def matrix(self) -> torch.Tensor:
        """ Returns the transformation matrix"""
        return self._construct_householder_permutation()
    
    def inverse_matrix(self) -> torch.Tensor:
        """ Returns the inverse transformation matrix"""
        w = self._construct_householder_permutation()
        w = w.transpose(0, 1).contiguous()
        return w
    
    def bias(self) -> torch.Tensor:
        """ Returns the bias vector"""
        return torch.zeros(self.dim).to(self.vk_householder.device)

class BlockAffineTransform(BaseTransform):
    """Implements a block affine transform. The transform is defined by a 
    block-diagonal matrix $A$ and a vector $b$ such that 
    $y = Ax + b$.
    """
    bijective = True
    domain = dist.constraints.real_vector
    codomain = dist.constraints.real_vector

    def __init__(
        self,
        in_dims: Iterable[int],
        block_transform: AffineTransform,
        *args,
        **kwargs
    ):
        """ Initializes the block affine transform.
        
        Args:
            in_dims (Iterable[int]): dimensions of the input
            block_transform (AffineTransform): block affine transform
        """
        super().__init__(*args, **kwargs)
        self.in_dims = in_dims
        
        if block_transform.dim != in_dims[0]:
            raise ValueError("block_transform dim must match input dim")
        self.block_size = in_dims[0]
        self.input_rank = len(in_dims) - 1
        self.n_blocks = math.prod(in_dims[1:])
        global_transform = {
            1: F.linear,
            2: F.conv1d,
            3: F.conv2d,
            4: F.conv3d
        }
        self.global_transform = global_transform[len(in_dims)]
        self.block_transform = block_transform
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes the block affine transform $y = Ax + b$.
        
        Args:
            x (torch.Tensor): input tensor
            context (torch.Tensor): context tensor (ignored)
            
        Returns:
            torch.Tensor: transformed tensor
        """
        w = self.block_transform.matrix().view(
            self.block_size,
            self.block_size,
            *([1] * self.input_rank)
        ).to(x.device)
        b = self.block_transform.bias().to(x.device)
        
        return self.global_transform(x, w, b)
    
    def backward(
        self,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes the inverse transform $y = Ax + b$.
        
        Args:
            y (torch.Tensor): input tensor
            context (torch.Tensor): context tensor (ignored)
            
        Returns:
            torch.Tensor: transformed tensor
        """
        w = self.block_transform.inverse_matrix().view(
            self.block_size,
            self.block_size,
            *([1] * self.input_rank)
        )
        b = self.block_transform.bias().view(
            self.block_size,
            *([1] * self.input_rank)
        )
        
        y = y - b
        y = self.global_transform(y, w)
        return y
    
    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor
            context (torch.Tensor): context tensor (ignored)
            
        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        return self.block_transform.log_abs_det_jacobian(x, y, context) * self.n_blocks
    
    def sign(self) -> int:
        """ Computes the sign of the determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            int: sign of the determinant of the Jacobian of the transform
        """
        return self.block_transform.sign() ** self.n_blocks
    
class LUTransform(AffineTransform):
    """Implementation of a linear bijection transform. Applies a transform $y = (\mathbf{L}\mathbf{U})^{-1}x$, where $\mathbf{L}$ is a
    lower triangular matrix with unit diagonal and $\mathbf{U}$ is an upper triangular matrix. Bijectivity is guaranteed by
    requiring that the diagonal elements of $\mathbf{U}$ are positive and the diagonal elements of  $\mathbf{L}$ are all $1$.

    *Note:* The implementation does not enforce the non-zero constraint of the diagonal elements of $\mathbf{U}$ during training.
    See :func:`add_jitter` and :func:`is_feasible` for a way to ensure that the transformation is invertible.
    """

    bijective = True
    volume_preserving = False
    domain = dist.constraints.real_vector
    codomain = dist.constraints.real_vector

    def __init__(self, dim: int, prior_scale: float = 1.0, *args, **kwargs,):
        """ Initializes the LU transform.
        
        Args:
            dim (int): dimension of the input and output
        """
        super().__init__(dim, *args, **kwargs)
        self.L_raw = torch.nn.Parameter(torch.empty(dim, dim)) 
        self.U_raw = torch.nn.Parameter(torch.empty(dim, dim)) 
        self.bias_vector = torch.nn.Parameter(torch.empty(dim)) 
        self.dim = dim
        self.prior_scale = prior_scale

        self.init_params()

        self.input_shape = dim

        self.L_mask = torch.tril(torch.ones(dim, dim), diagonal=-1)
        self.U_mask = torch.triu(torch.ones(dim, dim), diagonal=0)

        self.L_raw.register_hook(lambda grad: grad * self.L_mask)
        self.U_raw.register_hook(lambda grad: grad * self.U_mask)

    def init_params(self):
        """Parameter initialization
        Adopted from pytorch's Linear layer parameter initialization.
        """

        init.kaiming_uniform_(self.L_raw, nonlinearity="relu")
        with torch.no_grad():
            self.L_raw.copy_(self.L_raw.tril(diagonal=-1).fill_diagonal_(1))

        init.kaiming_uniform_(self.U_raw, nonlinearity="relu")
         
        with torch.no_grad():
            self.U_raw.fill_diagonal_(0) 
            #self.U_raw += torch.eye(self.dim)
            # TODO: Proper handling
            d = self.dim
            sign = -torch.ones(d) + 2 * torch.bernoulli(.5 * torch.ones(d))
            scale = self.prior_scale * torch.ones(d) * 1/self.dim if self.prior_scale is not None else torch.ones(d) 
            
            self.U_raw += sign * torch.normal(torch.zeros(self.dim), scale).exp().diag()
            self.U_raw.copy_(self.U_raw.triu())

        if self.bias_vector is not None:
            fan_in = self.dim
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_vector, -bound, bound)

    def forward(self, x: torch.Tensor, context = None) -> torch.Tensor:
        """Computes the affine transform $y = (LU)^{-1}x + \mathrm{bias}$.
        The value $y$ is computed by solving the linear equation system
        \begin{align*}
            Ly_0 &= x + LU\textrm{bias} \\
            Uy &= y_0  
        \end{align*}

        :param x: input tensor
        :type x: torch.Tensor
        :return: transformed tensor $(LU)x + \mathrm{bias}$
        """

        y = torch.functional.F.linear(x, self.matrix(), self.bias())
        return y

    def backward(self, y: torch.Tensor, context = None) -> torch.Tensor:
        """Computes the inverse transform $(LU)(y - \mathrm{bias})$

        :param y: input tensor
        :type y: torch.Tensor
        :return: transformed tensor $(LU)^{-1}(y - \mathrm{bias})$"""
        L_inv = torch.inverse(self.L)
        U_inv = torch.inverse(self.U)
        x = y - self.bias_vector
        x = torch.functional.F.linear(x, L_inv)
        x = torch.functional.F.linear(x, U_inv)
        return x

    @property
    def L(self) -> torch.Tensor:
        """The lower triangular matrix $\mathbf{L}$ of the layers LU decomposition"""
        return self.L_raw.tril(-1)  + torch.eye(self.dim).to(self.L_raw.device)

    @property
    def U(self) -> torch.Tensor:
        """The upper triangular matrix $\mathbf{U}$ of the layers LU decomposition"""
        return self.U_raw.triu()
    
    def matrix(self) -> torch.Tensor:
        """ Returns the transformation matrix"""
        return LA.matmul(self.L, self.U)
    
    def bias(self) -> torch.Tensor:
        """ Returns the bias vector"""
        return self.bias_vector
    
    def inverse_matrix(self) -> torch.Tensor:
        """ Returns the inverse transformation matrix"""
        L_inv = torch.inverse(self.L)
        U_inv = torch.inverse(self.U)
        return torch.matmul(U_inv, L_inv)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`forward`"""
        return self.forward(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`backward`"""
        return self.backward(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform $(LU)x + \mathrm{bias}$.
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): transformed tensor
            
        Returns:
            float: log absolute determinant of the Jacobian of the transform $(LU)x + \mathrm{bias}$
        """
        # log |Det(LU)| =  sum(log(|diag(U)|)) 
        # (as L is lower triangular with all 1s on the diag, i.e. log|Det(L)| = 0, and U is upper triangular)
        # However, since onnx export of diag() is currently not supported, we have use
        # a reformulation. Note dU keeps the quadratic structure but replace all values  
        # outside the diagonal with 1. Then sum(log(|diag(U)|)) = sum(log(|dU|))
        U = self.U
        dU = U - U.triu(1) + (torch.ones_like(U) - torch.eye(self.dim).to(U.device))
        return dU.abs().log().sum()

    def sign(self) -> int:
        """ Computes the sign of the determinant of the Jacobian of the transform $(LU)x + \mathrm{bias}$.
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            float: sign of the determinant of the Jacobian of the transform $(LU)x + \mathrm{bias}$
        """
        return self.L.diag().prod().sign() *  self.U.diag().prod().sign()

    def to(self, device) -> None:
        """ Moves the layer to a given device
        
        Args:
            device (torch.device): target device
        """
        self.L_mask = self.L_mask.to(device)
        self.U_mask = self.U_mask.to(device)
        # self.L_raw = self.L_raw.to(device)
        # self.U_raw = self.U_raw.to(device)
        # self.bias = self.bias.to(device)
        self.device = device
        return super().to(device)

    def is_feasible(self) -> bool:
        """Checks if the layer is feasible, i.e. if the diagonal elements of $\mathbf{U}$ are all positive"""
        return (self.U_raw.diag() != 0).all()

    def add_jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to the diagonal elements of $\mathbf{U}$. This is useful to ensure that the transformation 
        is invertible.
        
        Args:
            jitter (float, optional): jitter strength. Defaults to 1e-6.
        """
        perturbation = torch.randn(self.dim, device=self.U_raw.device) * jitter
        with torch.no_grad():
            self.U_raw.copy_(
                self.U_raw
                + perturbation * torch.eye(self.dim, device=self.U_raw.device)
            )
    
    def to_linear(self) -> BijectiveLinearTransform:
        """ Converts the transform to a linear transform"""
        M_inv = torch.matmul(self.L, self.U)
        M = torch.inverse(M_inv)
        return BijectiveLinearTransform(self.dim, M, self.bias_vector, M_inv)

class SequentialAffineTransform(AffineTransform):
    """Implements a sequential affine transform. The transform is defined by a sequence of affine transforms $y = A_1 A_2 \ldots A_n x + b$.
    """
    bijective = True
    domain = dist.constraints.real_vector
    codomain = dist.constraints.real_vector

    def __init__(self, transforms: Iterable[AffineTransform], *args, **kwargs) -> None:
        """ Initializes the sequential affine transform.
        
        Args:
            transforms (List[AffineTransform]): list of affine transforms
        """
        dim = transforms[0].dim
        if any([t.dim != dim for t in transforms]):
            raise ValueError("All transforms must have the same dimension")
        
        super().__init__(dim, *args, **kwargs)
        self.transforms = transforms
        
    def forward(self, x: torch.Tensor, context = None) -> torch.Tensor:
        """ Computes the sequential affine transform
        
        Args:
            x (torch.Tensor): input tensor
            context (torch.Tensor): context tensor (ignored)
            
        Returns:
            torch.Tensor: transformed tensor
        """
        for transform in self.transforms:
            x = transform(x, context)
        return x
    
    def backward(self, y: torch.Tensor, context = None) -> torch.Tensor:
        """ Computes the inverse transform
        
        Args:
            y (torch.Tensor): input tensor
            context (torch.Tensor): context tensor (ignored)
            
        Returns:
            torch.Tensor: transformed tensor
        """
        for transform in self.transforms[::-1]:
            y = transform.backward(y, context)
        return y
    
    def log_abs_det_jacobian(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        context = None
    ) -> float:
        """ Computes the log absolute determinant of the Jacobian of the transform
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor
            
        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        return sum(
            [transform.log_abs_det_jacobian(x, y, context) for transform in self.transforms]
        )
    
    def sign(self) -> int:
        """ Computes the sign of the determinant of the Jacobian of the transform
        Args:
            x (torch.Tensor): input tensor
        Returns:
            float: sign of the determinant of the Jacobian of the transform
        """
        return math.prod([transform.sign() for transform in self.transforms])
    
    def matrix(self) -> torch.Tensor:
        """ Returns the transformation matrix"""
        M = torch.eye(self.dim)
        for transform in self.transforms:
            M = torch.matmul(M, transform.matrix())
        return M
    
    def inverse_matrix(self) -> torch.Tensor:
        """ Returns the inverse transformation matrix"""
        M = torch.eye(self.dim)
        for transform in self.transforms[::-1]:
            M = torch.matmul(M, transform.inverse_matrix())
        return M
    
    def bias(self) -> torch.Tensor:
        """ Returns the bias vector"""
        b = torch.zeros(self.dim)
        for transform in self.transforms:
            b = torch.matmul(b, transform.matrix()) + transform.bias()
        return b
    
class BlockLUTransform(LUTransform):
    """Implementation of a tiled LU transform. The transform is defined by a block-diagonal matrix $\mathbf{L}\mathbf{U}$, where $\mathbf{L}$ is a
    lower triangular matrix with unit diagonal and $\mathbf{U}$ is an upper triangular matrix. The blocks are of size $b \times b$.
    Bijectivity is guaranteed by requiring that the diagonal elements of $\mathbf{U}$ are non-zero and the diagonal elements of  $\mathbf{L}$ are all $1$.
    """
    bijective = True
    volume_preserving = False
    domain = dist.constraints.real_vector
    codomain = dist.constraints.real_vector

    def __init__(self, in_dims: Iterable[int], prior_scale: float = 1.0, *args, **kwargs):
        """ Initializes the tiled LU transform.
        
        Args:
            in_dims (int): dimension of the input (and output)
            prior_scale (float): scale of the prior distribution
        """
        self.in_dims = in_dims
        self.block_size = in_dims[0]
        self.input_rank = len(in_dims) - 1
        self.n_blocks = math.prod(in_dims[1:])
        global_transform = {
            1: F.linear,
            2: F.conv1d,
            3: F.conv2d,
            4: F.conv3d
        }
        self.global_transform = global_transform[len(in_dims)]
        super().__init__(self.block_size, prior_scale, *args, **kwargs)
    
    def forward(self, x: torch.Tensor, context = None) -> torch.Tensor:
        """Computes the blockwise affine transform $y = (LU)^{-1}x + \mathrm{bias}$.
        The value $y$ is computed by solving the linear equation system
        \begin{align*}
            Ly_0 &= x + LU\textrm{bias} \\
            Uy &= y_0  
        \end{align*}

        :param x: input tensor
        :type x: torch.Tensor
        :return: transformed tensor $(LU)x + \mathrm{bias}$
        """
        
        w = LA.matmul(self.L, self.U).view(
            self.block_size,
            self.block_size,
            *([1] * self.input_rank)
        )
        b = self.bias_vector
        return self.global_transform(x, w, b)
    
    def backward(self, y: torch.Tensor, context = None) -> torch.Tensor:
        """Computes the inverse transform $(LU)(y - \mathrm{bias})$

        :param y: input tensor
        :type y: torch.Tensor
        :return: transformed tensor $(LU)^{-1}(y - \mathrm{bias})$"""
        
        L_inv = torch.inverse(self.L)
        U_inv = torch.inverse(self.U)
        w = torch.matmul(U_inv, L_inv).view(
            self.block_size,
            self.block_size,
            *([1] * self.input_rank)
        )
        b = self.bias_vector.view(
            self.block_size,
            *([1] * self.input_rank)
        )
        
        y = y - b
        y = self.global_transform(y, w)
        return y
        
    
    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`forward`"""
        return self.forward(x)
    
    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """ Alias for :func:`backward`"""
        return self.backward(y)
    
    def log_prior(self, correlated: bool = False) -> torch.Tensor:
        """Defines a log-normal prior on the diagonal elements of U Matrix,
        implicitply defining a log-normal prior on the absolute determinat
        of the transform."""
        precision = None
        d = self.block_size
        if correlated:
            # Pairwise negative correlation of 1/d
            covariance = -1 / d * torch.ones(d, d).to(self.device) + (1 + 1 / d) * torch.diag(
                torch.ones(d).to(self.device)
            )
            # Scaling
            covariance = covariance * (self.prior_scale**2)
        else:
            covariance = torch.eye(d).to(self.device)
            # Scaling
            covariance = covariance * (self.prior_scale**2 / (d))

        precision = torch.linalg.inv(covariance).to(self.device)

        # log-density of Normal in log-space
        x = self.U.diag().abs().log() 
        log_prior = -(x * (precision @ x)).sum()
        # Change of variables to input space
        log_prior += -x.sum()
        return log_prior
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context = None) -> float:
        """ Computes the log absolute determinant of the Jacobian of the 
        blockwise transform $(LU)x + \mathrm{bias}$. Since the Jacobian is block-diagonal,
        the determinant is the product of the determinants of the blocks.
        The log absolute determinant is the sum of the log absolute determinants of the blocks.
        Since all blocks use the same LU transform, we can use the log absolute determinant of 
        the LU transform multiplied with the number of blocks.
        
        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): transformed tensor
            
        Returns:
            float: log absolute determinant of the Jacobian of the transform $(LU)x + \mathrm{bias}$
        """
        return super().log_abs_det_jacobian(x, y, context) * self.n_blocks
    
    def sign(self) -> int:
        """ Computes the sign of the determinant of the Jacobian of the blockwise transform $(LU)x + \mathrm{bias}$.
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            float: sign of the determinant of the Jacobian of the blockwise transform $(LU)x + \mathrm{bias}$
        """
        return self.block_transform.sign() ** self.n_blocks
     