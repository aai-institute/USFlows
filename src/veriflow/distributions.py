from typing import Iterable
from src.veriflow.transforms import Rotation, CompositeRotation
from src.veriflow.linalg import random_orthonormal_matrix
import torch
from torch.distributions import constraints
import pyro
from pyro.distributions.torch_distribution import TorchDistributionMixin
import math

class RotatedLaplace(torch.distributions.Distribution):
    """Implements a Laplace distribution that is rotated so that the bounding box of the density contours is of minimal (Euclidean) volume."""
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
 
from torch.distributions import Distribution, Chi2

class Chi(Distribution):
    arg_constraints = {"df": constraints.positive}
    support = constraints.positive
    has_enumerate_support = False  
    
    def __init__(self, df, validate_args=None):
        """
        Initialize the Chi distribution with degrees of freedom `df`.
        Args:
            df (Tensor): degrees of freedom.
            validate_args (bool, optional): Whether to validate input parameters. Default: None.
        """
        self.chi2 = Chi2(df)
        self.df = df
        super(Chi, self).__init__(self.chi2._batch_shape, self.chi2._event_shape, validate_args=validate_args)
        
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
        y = value ** 2
        return self.chi2.log_prob(y) + torch.log(value * 2)
    
    def cdf(self, value):
        """
        Calculate the cumulative distribution function (CDF) at a given value.
        Args:
            value (Tensor): The value at which to evaluate the CDF.
        Returns:
            Tensor: The CDF of the value.
        """
        y = value ** 2
        return self.chi2.cdf(y)
    
    def entropy(self):
        """
        Calculate the entropy of the distribution.
        Returns:
            Tensor: The entropy of the distribution.
        """
        return self.chi2.entropy() / 2 + torch.log(torch.tensor(2))
 
   
class UniformUnitLpBall(torch.distributions.Distribution):
    """Implements a uniform distribution on the unit ball."""
    
    support = constraints.real
    has_enumerate_support = False  
    
    def __init__(self, dim: int, p: float):
        self.p = p
        self.dim = dim
        if self.p == 1: 
            self.log_surface_area_unit_ball = (3/2) * math.log(self.dim) + math.log(2) * self.dim  -  torch.log(torch.arange(1, self.dim + 1)).sum() 
        elif self.p == 2:
            self.log_surface_area_unit_ball = math.log(2) + (self.dim / 2) * math.log(math.pi) - torch.lgamma(self.dim / 2)
        elif self.p == math.inf:
            self.log_surface_area_unit_ball = math.log(2) * self.dim + math.log(self.dim) 
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
            dims = pyro.distributions.Categorical(probs=torch.ones(2) / 2).sample(sample_shape + (self.dim,)) * 2 - 1
            x = x * dims
        elif self.p == 2:
            x = pyro.distributions.Normal(0, 1).sample(sample_shape + (self.dim,))
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.p == math.inf:
            extremal_dims = pyro.distributions.Categorical(torch.ones(self.dim)/ self.dim).sample(sample_shape + (1,))
            mask = torch.ones(sample_shape + (self.dim,)).cumsum(dim=-1) - 1 == extremal_dims

            boundary = torch.ones(sample_shape + (self.dim,))
            hyperplane_distribution = pyro.distributions.Uniform(-boundary, boundary)
            x = hyperplane_distribution.sample()
            x[mask] = 1.
        else:
            raise ValueError("p must be 1, 2, or inf.")
        
        return x
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""
        
        return -self.log_surface_area_unit_ball
    
    
class RadialDistribution(torch.distributions.Distribution):
    """Implements radial distributions. More precisely, this class realizes Lp-radial distributions with specifiable redial distribution.
    
    Args:
        loc: Location of the distribution
        norm_distribution: Distribution of the radial component
        p: Exponent of the Lp norm used to define the distribution. Currently, p = 1, 2, and inf are supported.
    """
    arg_constraints = {"loc": constraints.real}
    support = constraints.positive
    has_enumerate_support = False  
    
    def __init__(self, loc: torch.Tensor, norm_distribution: torch.distributions.Distribution, p: float, device: str = "cpu"):
        if not isinstance(p, float):
            raise ValueError("p must be a float.")
        if p <= 0:
            raise ValueError("p must be positive.")
        
        self.device = device
        self.loc = loc.to(device)
        self.norm_distribution = norm_distribution
        self.p = p
        self.dim = loc.shape[0]
        self.unit_ball_distribution = UniformUnitLpBall(self.dim, p)
        
        super().__init__(event_shape=(loc.shape[0],), validate_args=False)
        
    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Samples batch of shape sample_shape from the distribution."""
        peel = False
        if sample_shape is None:
            sample_shape = (1,)
            peel = True
        else:
            sample_shape = tuple(sample_shape)
        
        r = self.norm_distribution.sample(sample_shape).to(self.device)
        r = r.repeat(*[1 for _ in sample_shape], self.dim)
        u = self.unit_ball_distribution.sample(sample_shape).to(self.device)
        x = r * u
 
        if peel:
            x = x.squeeze(0)
            
        return x + self.loc
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the points x under the distribution."""
        x = x - self.loc
        r = x.norm(dim=-1, p=self.p)
        log_prob_norm = self.norm_distribution.log_prob(r)
        log_dV = self.log_delta_volume(self.p, r)
        
        return log_prob_norm - log_dV
    
    def log_delta_volume(p: int, r: Union[float, torch.Tensor]) -> Union[float. torch.Tensor]:
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
           log_dv = math.log(2) * d + math.log(r) * (d-1) - ldfac
        elif p == 2:
           # V_2^d'(r) = d * (pi)^(d/2) * r^(d-1) / Gamma(d/2 + 1)
           log_numerator = (
               math.log(self.dim) + (self.dim / 2) * math.log(math.pi) + (d - 1) * math.log(r)
           )
           log_dv = log_numerator - math.lgamma((d / 2) + 1)
        elif p == math.inf:
           # V_\infty^d'(r) = d * (pi)^(d/2) * r^(d-1) / Gamma(d/2 + 1)
           log_dv = math.log(self.dim) + self.dim * math.log(2) + (self.dim - 1) * math.log(r)  
        else:
            raise ValueError(f"p={p} not implemented. Use p=1,2, or infinity")
        
        return log_dv