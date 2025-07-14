from typing import Dict, Iterable, Optional, Union
from src.usflows.transforms import Rotation, CompositeRotation
from src.usflows.linalg import random_orthonormal_matrix
from src.usflows.utils import inv_softplus
import torch
from torch.distributions import constraints
from torch.nn import ParameterDict
from torch.distributions import Distribution, Chi2
from torch.nn.functional import softplus
import pyro
from pyro.distributions.torch_distribution import TorchDistributionMixin
import math


class RotatedLaplace(torch.distributions.Distribution):
    """Implements a Laplace distribution that is rotated so that the bounding
    box of the density contours is of minimal (Euclidean) volume."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_enumerate_support = False

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.dim = loc.shape[0]
        self.loc = loc
        self.scale = scale
        self.rotation = random_orthonormal_matrix(self.dim)
        self.laplace = pyro.distributions.Laplace(self.loc, self.scale)
        batch_shape = self.laplace.batch_shape
        if len(batch_shape) > 0:
            self.laplace = pyro.distributions.Independent(
                self.laplace, len(batch_shape)
            )
        self.shape = self.laplace.event_shape
        super().__init__(event_shape=(self.dim,), validate_args=False)

    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples n points from the distribution."""
        if sample_shape is None:
            sample_shape = ()
        else:
            sample_shape = tuple(sample_shape)

        return torch.matmul(self.laplace.sample(sample_shape), self.rotation)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""
        return self.laplace.log_prob(torch.matmul(x, self.rotation.t()))


class Chi(Distribution):
    arg_constraints = {"df": constraints.positive}
    support = constraints.positive
    has_enumerate_support = False

    def __init__(self, df: int, validate_args=None):
        """
        Initialize the Chi distribution with degrees of freedom `df`.
        Args:
            df (Tensor): degrees of freedom.
            validate_args (bool, optional): Whether to validate input parameters. Default: None.
        """
        self.chi2 = Chi2(df)
        self.df = df
        super(Chi, self).__init__(
            self.chi2._batch_shape, self.chi2._event_shape, validate_args=validate_args
        )

    def sample(self, sample_shape=torch.Size()):
        """
        Generate samples from the Chi distribution.
        Args:
            sample_shape (torch.Size, optional): The size of the sample to draw. Default: torch.Size().
        Returns:
            Tensor: A sample of the specified shape.
        """
        return torch.sqrt(self.chi2.sample(sample_shape))

    def log_prob(self, value):
        """
        Calculate the log probability of a given value.
        Args:
            value (Tensor): The value at which to evaluate the log probability.
        Returns:
            Tensor: The log probability of the value.
        """
        y = value**2
        return self.chi2.log_prob(y) + torch.log(value * 2)

    def cdf(self, value):
        """
        Calculate the cumulative distribution function (CDF) at a given value.
        Args:
            value (Tensor): The value at which to evaluate the CDF.
        Returns:
            Tensor: The CDF of the value.
        """
        y = value**2
        return self.chi2.cdf(y)

    def entropy(self):
        """
        Calculate the entropy of the distribution.
        Returns:
            Tensor: The entropy of the distribution.
        """
        return self.chi2.entropy() / 2 + torch.log(torch.tensor(2))


class DistributionModule(torch.nn.Module, torch.distributions.Distribution):
    """Wrapper class to treat pyro distributions as PyTorch modules.


    Args:
        distribution_class: Pyro distribution to wrap.
        trainable_args: Dictionary of trainable parameters.
            Initial parameters need to be given as tensors.
        static_args: Dictionary of static (non-trainable) parameters.
        n_batch_dims: Number of batch dimensions.
    """

    def __init__(
        self,
        distribution_class: type[torch.distributions.Distribution],
        params: Dict[str, torch.tensor] = None,
        module_args: Dict[str, any] = None,
        n_batch_dims: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.distribution_class = distribution_class
        if params is None:
            params = {}
        if module_args is None:
            module_args = {}
        self.params = ParameterDict(
            {
                key: torch.nn.Parameter(value) if not isinstance(value, torch.nn.Parameter) else value
                for key, value in params.items()
            }
        )
        self.module_args = torch.nn.ModuleDict(module_args) 
        self.n_batch_dims = n_batch_dims
        self.generated_args = dict()

    @property
    def event_shape(self) -> torch.Size:
        """Returns the shape of the distribution."""
        return self.distribution.event_shape

    @property
    def batch_shape(self) -> torch.Size:
        """Returns the batch shape of the distribution."""
        return self.distribution.batch_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the distribution module. Synonymous to the
        distribution's log_prob method."""
        return self.distribution.log_prob(x)

    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples batch of shape sample_shape from the distribution."""
        if sample_shape is None:
            sample_shape = ()
        else:
            sample_shape = tuple(sample_shape)

        return self.distribution.sample(sample_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""
        return self.distribution.log_prob(x)

    @property
    def distribution(self) -> torch.distributions.Distribution:
        """Builds the distribution with the current parameters."""
        generator_args = {
            key: generator() for key, generator in self.generated_args.items()
        }
        d = self.distribution_class(
            **self.params,
            **self.module_args,
            **generator_args
        )

        nbatch_dims = len(d.batch_shape) - self.n_batch_dims
        if nbatch_dims > 0:
            d = torch.distributions.Independent(d, nbatch_dims)

        return d
    
    def register_generated_arg(self, name: str, generator: callable) -> None:
        """Register a parameter to the module."""
        self.generated_args[name] = generator
        

class Gamma(DistributionModule):
    """Wrapper class for the Gamma distribution."""

    def __init__(
        self,
        concentration: torch.Tensor,
        rate: torch.Tensor,
        device: str = "cpu",
        *args,
        **kwargs
    ):
        """Initializes the Gamma distribution."""
        super().__init__(
            torch.distributions.Gamma,
            *args,
            **kwargs
        )
        self._concentration_unconstrained = torch.nn.Parameter(
            inv_softplus(concentration),
            requires_grad=True
        )
        self._rate_unconstrained = torch.nn.Parameter(
            inv_softplus(rate),
            requires_grad=True
        )
        self.register_generated_arg(
            "concentration",
            lambda: softplus(self._concentration_unconstrained)
        )
        self.register_generated_arg(
            "rate",
            lambda: softplus(self._rate_unconstrained)
        )
        self.to(device)

class LogNormal(DistributionModule):
    """Wrapper class for the LogNormal distribution."""

    def __init__(
        self,
        loc: torch.Tensor, 
        scale: torch.Tensor, 
        device: str = "cpu",
        *args, 
        **kwargs
    ):
        """Initializes the LogNormal distribution."""
        super().__init__(
            torch.distributions.LogNormal,
            *args,
            **kwargs
        )
        self._loc = torch.nn.Parameter(loc)
        self._scale_unconstrained = torch.nn.Parameter(
            inv_softplus(scale)
        )
        self.register_generated_arg(
            "scale",
            lambda: softplus(self._scale_unconstrained)
        )
        self.register_generated_arg(
            "loc",
            lambda: self._loc
        )
        
        self.to(device)


class Laplace(DistributionModule):
    """Wrapper class for the Laplace distribution."""

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        device: str = "cpu",
        *args,
        **kwargs
    ):
        """Initializes the Laplace distribution."""
        super().__init__(
            torch.distributions.Laplace,
            *args,
            **kwargs
        )
        self._loc = torch.nn.Parameter(loc)
        self._scale_unconstrained = torch.nn.Parameter(
            inv_softplus(scale)
        )
        self.register_generated_arg(
            "scale",
            lambda: softplus(self._scale_unconstrained)
        )
        self.register_generated_arg(
            "loc",
            lambda: self._loc
        )
        
        self.to(device)
        


class Normal(DistributionModule):
    """Wrapper class for the Normal distribution."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, *args, **kwargs):
        """Initializes the Normal distribution."""
        super().__init__(
            torch.distributions.Normal,
            *args,
            **kwargs
        )
        self._loc = torch.nn.Parameter(loc)
        self._scale_unconstrained = torch.nn.Parameter(
            inv_softplus(scale)
        )
        self.register_generated_arg(
            "scale",
            lambda: softplus(self._scale_unconstrained)
        )
        self.register_generated_arg(
            "loc",
            lambda: self._loc
        )
        self.to(self.device)

class GMM(DistributionModule):
    """Wrapper class for the Gaussian Mixture Model (GMM) distribution."""

    def __init__(
        self, loc: torch.Tensor, scale: torch.Tensor, mixture_weights: torch.Tensor, device: str = "cpu", *args, **kwargs
    ):
        """Initializes the GMM distribution."""
        super().__init__(
            torch.distributions.MixtureSameFamily,
            *args,
            **kwargs
        )
        self.device = device
        self.normal_batch = Normal(loc, scale, n_batch_dims=1)
        self.mixture_distribution = Categorical(mixture_weights)
        self.register_generated_arg(
            "mixture_distribution",
            lambda: self.mixture_distribution
        )
        self.register_generated_arg(
            "component_distribution",
            lambda: self.normal_batch
        )
        self.to(self.device)
        



class UniformUnitLpBall(torch.distributions.Distribution):
    """Implements a uniform distribution on the unit ball."""

    support = constraints.real
    has_enumerate_support = False

    def __init__(self, dim: int, p: float):
        self.p = p
        self.dim = dim
        if self.p == 1:
            self.log_surface_area_unit_ball = (
                (3 / 2) * math.log(self.dim)
                + math.log(2) * self.dim
                - torch.log(torch.arange(1, self.dim + 1)).sum()
            )
        elif self.p == 2:
            self.log_surface_area_unit_ball = (
                math.log(2)
                + (self.dim / 2) * math.log(math.pi)
                - math.lgamma(self.dim / 2)
            )
        elif self.p == math.inf:
            self.log_surface_area_unit_ball = math.log(2) * self.dim + math.log(
                self.dim
            )
        else:
            raise ValueError("p must be 1, 2, or inf.")
        super().__init__(event_shape=(dim,), validate_args=False)

    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples batch of shape sample_shape from the distribution."""
        if sample_shape is None:
            sample_shape = ()
        else:
            sample_shape = tuple(sample_shape)

        if self.p == 1:
            x = pyro.distributions.Dirichlet(torch.ones(self.dim)).sample(sample_shape)
            dims = (
                pyro.distributions.Categorical(probs=torch.ones(2) / 2).sample(
                    sample_shape + (self.dim,)
                )
                * 2
                - 1
            )
            x = x * dims
        elif self.p == 2:
            x = pyro.distributions.Normal(0, 1).sample(sample_shape + (self.dim,))
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.p == math.inf:
            extremal_dims = pyro.distributions.Categorical(
                torch.ones(self.dim) / self.dim
            ).sample(sample_shape + (1,))
            mask = (
                torch.ones(sample_shape + (self.dim,)).cumsum(dim=-1) - 1
                == extremal_dims
            )

            boundary = torch.ones(sample_shape + (self.dim,))
            hyperplane_distribution = pyro.distributions.Uniform(-boundary, boundary)
            x = hyperplane_distribution.sample()
            x[mask] = 1.0
        else:
            raise ValueError("p must be 1, 2, or inf.")

        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""

        return -self.log_surface_area_unit_ball


class RadialDistribution(torch.nn.Module):
    """Implements radial distributions. More precisely, this class realizes
    Lp-radial distributions with specifiable redial distribution.

    Args:
        loc: Location of the distribution
        norm_distribution: Distribution of the radial component
        p: Exponent of the Lp norm used to define the distribution. Currently,
        p = 1, 2, and inf are supported.
    """

    arg_constraints = {"loc": constraints.real}
    support = constraints.real
    has_enumerate_support = False

    def __init__(
        self,
        loc: torch.Tensor,
        norm_distribution: DistributionModule,
        p: float,
        n_batch_dims: int = 0,
        device: str = "cpu",
    ):
        torch.nn.Module.__init__(self)
        self.norm_distribution = norm_distribution
        self.event_shape=loc.shape[n_batch_dims:]
        self.batch_shape=loc.shape[:n_batch_dims]
        
        if not isinstance(p, float):
            raise ValueError("p must be a float.")
        if p <= 0:
            raise ValueError("p must be positive.")

        self.device = device
        self.loc = torch.nn.Parameter(loc.to(device))
        self.p = p
        self.n_batch_dims = n_batch_dims
        self.dim = torch.prod(torch.tensor(loc.shape[self.n_batch_dims :]))
        self.shape = loc.shape
        self.unit_ball_distribution = UniformUnitLpBall(self.dim, p)
        self.to(self.device)
        
    
    def _merge_intervals(self, intervals: torch.Tensor) -> torch.Tensor:
            """returns start and end of consecutive intervals of indices"""
            if len(intervals) == 1:
                return torch.tensor([[intervals[0], intervals[0]]])
            
            intervals = intervals.sort().values
            merged = []
            start = intervals[0]
            end = intervals[0]
            for i in range(1, len(intervals)):
                if intervals[i] == end + 1:
                    end = intervals[i]
                else:
                    merged.append([start, end])
                    start = intervals[i]
                    end = intervals[i]
            merged.append([start, end])
            return torch.tensor(merged, device=self.device)
    

    def radial_udl_profile(self, q: Optional[float] = None, threshold: Optional[float] = None, r_max: float = 100000, n_samples: int = 10000) -> torch.Tensor:
        """Computes an approximate representation of the upper density level set of distribution as intervals radial intervals.
        
        Args:
            q: probability level.
        Returns:
            Tensor: Upper density level set of the distribution given as tensor of shape n_intervals x 2. Each row reresents an interval [a,b] where a is the lower bound and b is the upper bound of the radial component.
        """
        if q is not None and threshold is not None:
            raise ValueError("Only one of 'q' or 'threshold' can be provided.")
        if q is None and threshold is None:
            raise ValueError("Either 'q' or 'threshold' must be provided.")

        rs = torch.linspace(1e-20, r_max, n_samples, device=self.device).reshape(-1, 1)

        # radial profile function 
        profile = self.norm_distribution.log_prob(rs) - self.log_delta_volume(self.p, rs).flatten()
        
        # compute threshold
        if q is not None:
            sample = self.norm_distribution.sample((n_samples,)).to(self.device)
            logprob = self.norm_distribution.log_prob(sample) - self.log_delta_volume(self.p, sample).flatten()
            sorted_logprob, _ = torch.sort(logprob, descending=True)
            threshold_idx = int(n_samples * q)
            threshold = sorted_logprob[threshold_idx]

        # compute intervals
        indices = torch.arange(n_samples, device=self.device)
        indices = indices[profile > threshold]

    
        
        return rs.flatten()[self._merge_intervals(indices)]    
    
    def radial_ldl_profile(self, q: Optional[float] = None, threshold: Optional[float] = None, r_max: float = 100000, n_samples: int = 10000) -> torch.Tensor:
        """Computes an approximate representation of the lower density level set of distribution as intervals radial intervals.
        
        Args:
            q: probability level.
        Returns:
            Tensor: Lower density level set of the distribution given as tensor of shape n_intervals x 2. Each row reresents an interval [a,b] where a is the lower bound and b is the upper bound of the radial component.
        """
        if q is not None and threshold is not None:
            raise ValueError("Only one of 'q' or 'threshold' can be provided.")
        if q is None and threshold is None:
            raise ValueError("Either 'q' or 'threshold' must be provided.")

        rs = torch.linspace(1e-20, r_max, n_samples, device=self.device).reshape(-1, 1)

        # radial profile function 
        profile = self.norm_distribution.log_prob(rs) - self.log_delta_volume(self.p, rs).flatten()
        
        # compute threshold
        if q is not None:
            sample = self.norm_distribution.sample((n_samples,)).to(self.device)
            logprob = self.norm_distribution.log_prob(sample) - self.log_delta_volume(self.p, sample).flatten()
            sorted_logprob, _ = torch.sort(logprob, descending=False)
            threshold_idx = int(n_samples * q)
            threshold = sorted_logprob[threshold_idx]

        # compute intervals
        indices = torch.arange(n_samples, device=self.device)
        indices = indices[profile <= threshold]

        return rs.flatten()[self._merge_intervals(indices)]

    def r_profile(self, r: torch.Tensor) -> torch.Tensor:
        """Computes the radial profile of the distribution at radius r.
        
        Args:
            r: radius (batch)
        Returns:
            Tensor: Radial profile of the distribution at radius r.
        """
        if isinstance(r, torch.Tensor):
            r = r.to(self.device)
        else:
            r = torch.tensor(r, device=self.device)

        log_prob_norm = self.norm_distribution.log_prob(r.unsqueeze(-1)).squeeze(-1)
        log_dV = self.log_delta_volume(self.p, r)

        return log_prob_norm - log_dV      

    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples batch of shape sample_shape from the distribution."""
        peel = False
        if sample_shape is None:
            sample_shape = (1,)
            peel = True
        else:
            sample_shape = tuple(sample_shape)

        r = self.norm_distribution.sample(sample_shape).to(self.device)
        r = r.repeat(
            *[1 for _ in sample_shape],
            *[1 for _ in range(self.n_batch_dims)],
            *tuple(self.event_shape),
        )

        u = self.unit_ball_distribution.sample(
            sample_shape + tuple(self.batch_shape)
        ).to(self.device)
        u = u.reshape(*sample_shape, *self.shape)
        x = r * u

        if peel:
            x = x.squeeze(0)

        return x + self.loc

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""
        x = x - self.loc

        event_dims = tuple(range(x.dim() - len(self.event_shape), x.dim()))
        r = x.norm(p=self.p, dim=event_dims)

        log_prob_norm = self.norm_distribution.log_prob(r.unsqueeze(-1))
        log_dV = self.log_delta_volume(self.p, r)

        return log_prob_norm - log_dV

    def log_delta_volume(
        self, p: int, r: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """Computes the differential log-volume of an $L^p$ ball with radius r.
        Currently, $p=1,2,\text{ or }\infty$ is implemented

        Args:
            p: p norm
            r: radius (batch)
        Returns:
            Differential volume (batch)
        """
        if p == 1:
            # V_1^d'(r) = (2r)**(d-1) / (d-1)!
            log_denominator = sum([math.log(i) for i in range(1, self.dim)])
            log_dv = (
                math.log(2) * self.dim + torch.log(r) * (self.dim - 1) - log_denominator
            )
        elif p == 2:
            # V_2^d'(r) = d * (pi)^(d/2) * r^(d-1) / Gamma(d/2 + 1)
            log_numerator = (
                math.log(self.dim)
                + (self.dim / 2) * math.log(math.pi)
                + (self.dim - 1) * torch.log(r)
            )
            log_dv = log_numerator - math.lgamma((self.dim / 2) + 1)
        elif p == math.inf:
            # V_\infty^d'(r) = d * (pi)^(d/2) * r^(d-1) / Gamma(d/2 + 1)
            log_dv = (
                math.log(self.dim)
                + self.dim * math.log(2)
                + (self.dim - 1) * torch.log(r)
            )
        else:
            raise ValueError(f"p={p} not implemented. Use p=1,2, or infinity")

        return log_dv


class Categorical(DistributionModule, torch.distributions.Categorical):
    """Wrapper class for the Categorical distribution."""

    def __init__(
        self,
        logits: torch.Tensor,
        device: str = "cpu",
        *args,
        **kwargs
    ):
        """Initializes the Categorical distribution."""
        DistributionModule.__init__(self, torch.distributions.Categorical, *args, **kwargs)
        torch.distributions.Categorical.__init__(
            self,
            logits=logits,
            validate_args=False,
            *args,
            **kwargs
        )
        self._logits = torch.nn.Parameter(logits)
        self.register_generated_arg(
            "logits",
            lambda: self.logits
        )
        
        self.to(device)
        
    @property
    def logits(self) -> torch.Tensor:
        """Returns the logits of the distribution."""
        return self._logits
    @logits.setter
    def logits(self, value: torch.Tensor) -> None:
        self._logits = torch.nn.Parameter(value)
    
    @property
    def probs(self) -> torch.Tensor:
        """Returns the probabilities of the distribution."""
        return torch.nn.functional.softmax(self._logits, dim=-1)

class RadialMM(DistributionModule):

    def __init__(
        self,
        loc: torch.Tensor,
        norm_distribution: torch.distributions.Distribution,
        p: float,
        mixture_weights: torch.Tensor = None,
        n_batch_dims: int = 1,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """Builds a mixture of radial distributions.

        Args:
            loc: Location param. of (B,D)
            norm_distribution: Norm distributions of (B,D).
            p: Order of norm.
            device: Compute deivce.
        """
        
        super().__init__(
            torch.distributions.MixtureSameFamily,
            *args,
            **kwargs
        )
        self.radial_batch = RadialDistribution(
            loc, 
            norm_distribution, 
            p, 
            device=device, 
            n_batch_dims=n_batch_dims
        )

        if mixture_weights is None:
            mixture_weights = torch.ones(norm_distribution.batch_shape)
        else:
            assert isinstance(
                mixture_weights, torch.Tensor
            ), f"`mixture_weights` must be a tensor. Got {type(mixture_weights)}"

        # Move weights to the same device as the distribution
        mixture_weights = mixture_weights.to(device)
        self.mixture_distribution = Categorical(
            logits=mixture_weights
        )
        
        self.register_generated_arg(
            "mixture_distribution",
            lambda: self.mixture_distribution
        )
        self.register_generated_arg(
            "component_distribution",
            lambda: self.radial_batch
        )

        self.to(device)

   

class LMM(torch.nn.Module, torch.distributions.MixtureSameFamily):
    """Wrapper class for the Laplace Mixture Model (LMM) distribution."""

    def __init__(
        self, loc: torch.Tensor, scale: torch.Tensor, mixture_weights: torch.Tensor,
        device: str = "cpu", *args, **kwargs
    ):
        """Initializes the LMM distribution."""
        torch.nn.Module.__init__(self)
        self._laplace_batch = Laplace(loc, scale, n_batch_dims=1)
        self._mixture_distribution = Categorical(mixture_weights)
        torch.distributions.MixtureSameFamily.__init__(
            self,
            mixture_distribution=self._mixture_distribution,
            component_distribution=self._laplace_batch,
            validate_args=False,
            *args,
            **kwargs,
        )
        self.to(device)

class GammaMM(torch.nn.Module, torch.distributions.MixtureSameFamily):
    """Wrapper class for the Gamma Mixture Model (GMM) distribution."""

    def __init__(
        self, 
        concentration: torch.Tensor, 
        rate: torch.Tensor, 
        mixture_weights: torch.Tensor,
        n_batch_dims: int = 0,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """Initializes the GMM distribution."""
        torch.nn.Module.__init__(self)
        self._gamma_batch = Gamma(concentration, rate, n_batch_dims=n_batch_dims+1)
        self._mixture_distribution = Categorical(mixture_weights)
        torch.distributions.MixtureSameFamily.__init__(
            self,
            mixture_distribution=self._mixture_distribution,
            component_distribution=self._gamma_batch,
            validate_args=False,
            *args,
            **kwargs,
        )
        self.to(device)

class Independent(torch.nn.Module, torch.distributions.Independent):
    """Wrapper class for the Independent distribution."""

    def __init__(
        self,
        base_distribution: torch.distributions.Distribution,
        reinterpreted_batch_ndims: int = 0,
        *args,
        **kwargs
    ):
        """Initializes the Independent distribution."""
        torch.nn.Module.__init__(self)
        self._base_distribution = base_distribution
        torch.distributions.Independent.__init__(
            self,
            self._base_distribution,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            *args,
            **kwargs
        )