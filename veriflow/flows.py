import numpy as np

from torch.utils.data import Dataset

import pyro 
from pyro import distributions as dist
from pyro.distributions.transforms import LeakyReLUTransform
from pyro.nn import DenseNN
from pyro.distributions.transforms import AffineCoupling, LowerCholeskyAffine
from pyro.infer import SVI
from typing import List, Dict, Literal, Any, Iterable, Optional, Union, Tuple
import torch
from torch.utils.data import DataLoader

from sklearn.datasets import load_digits
from tqdm import tqdm
from veriflow.transforms import ScaleTransform, MaskedCoupling, Permute, LUTransform
from veriflow.networks import AdditiveAffineNN, ConvNet2D
from veriflow.experiments.utils import create_checkerboard_mask


class Flow(torch.nn.Module):

    def __init__(self, base_distribution, layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = layers
        self.trainable_layers = torch.nn.ModuleList([l for l in layers if isinstance(l, torch.nn.Module)])
        self.base_distribution = base_distribution

        self.transform = dist.TransformedDistribution(base_distribution, layers)

    def fit(
            self, 
            data_train: Dataset, 
            optim: torch.optim.Optimizer = torch.optim.Adam,
            optim_params: Dict[str, Any] = None,
            batch_size: int = 32,
            shuffe: bool = True,
            gradient_clip: float = None,
            device: torch.device = None,
            jitter: float = 1e-4,
            ) -> float:
        """
        Wrapper function for the fitting procedure. Allows basic configuration of the optimizer and other fitting parameters.

        @param data_train: training data.
        @param batch_size: number of samples per optimization step.
        @param optim: optimizer class. 
        @param optimizer_params: optimizer parameter dictionary.

        @returns loss curve (negative log-likelihood).
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        model = self.to(device)
        
        if optim_params is not None:
            optim = optim(model.trainable_layers.parameters(), **optim_params)
        else:
            optim = optim(model.trainable_layers.parameters())

        N = len(data_train)
        losses = []

        if shuffe:
            perm = np.random.choice(N, N, replace=False)
            data_train = data_train[perm]


        for idx in range(0, N, batch_size):
            idx_end = min(idx + batch_size, N)
            try:
                sample = torch.Tensor(data_train[idx:idx_end][0]).to(device)
            except:
                continue
            optim.zero_grad()
            while not self.is_feasible():
                self.add_jitter(jitter)
            loss = -model.transform.log_prob(sample).mean()
            losses.append(float(loss.detach()))
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optim.step()
            while not self.is_feasible():
                self.add_jitter(jitter)
            
            model.transform.clear_cache()
        
        return sum(losses) / len(losses)
    
    def log_prob(self, x: torch.Tensor):
        """Returns the models log-densities for the given samples

        @param x: sample tensor.
        """
        return self.transform.log_prob(x)
    
    def sample(self, sample_shape: Iterable[int] = None):
        """Returns n_sample samples from the distribution
        
        @param n_sample: sample shape.
        """
        if sample_shape is None:
            sample_shape = [1]

        return self.transform.sample(sample_shape)
    
    def to(self, device):
        """Moves the model to the given device"""
        self.device = device
        #self.layers = torch.nn.ModuleList([l.to(device) for l in self.layers])
        self.trainable_layers = torch.nn.ModuleList([l.to(device) for l in self.trainable_layers])
        return super().to(device)
    
    def is_feasible(self):
        return True
    
    def add_jitter(self, jitter):
        pass

Permutation = Literal["random", "half", "LU"]

class NiceFlow(Flow):
    """Implementation of the NICE flow architecture by using fully connected coupling layers"""
    def __init__(
            self,
            base_distribution: dist.Distribution,
            coupling_layers: int,
            coupling_nn_layers: List[int],
            split_dim: int,
            scale_every_coupling=False,
            nonlinearity: Optional[torch.nn.Module] = None,
            permutation: Permutation = "random",
            *args,
            **kwargs
        ) -> None:
        """Initialization

        @param base_distribution: base distribution,
        @param coupling_layers: number of coupling layers. All coupling layers share the same architecture but not the same weights.
        @param coupling_nn_layers: number of neurons in the hidden layers of the dense neural network that computes the coupling loc parameter.
        @param split_dim: split dimension for the coupling.
        @param scale_every_coupling: if True, a scale transform is applied after every coupling layer. Otherwise, a single scale transform is applied after all coupling layers.
        @param nonlinearity: nonlinearity of the coupling network.
        @param permutation: permutation type. Can be "random" or "half".
        """
        input_dim = base_distribution.sample().shape[0]
        self.input_dim = input_dim
        self.coupling_layers = coupling_layers
        self.coupling_nn_layers = coupling_nn_layers
        self.split_dim = split_dim

        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU

        rdim = input_dim - split_dim
        layers = []
        for i in range(coupling_layers):
            layers.append(self._get_permutation(permutation, i))
            layers.append(AffineCoupling(split_dim, AdditiveAffineNN(split_dim, coupling_nn_layers, rdim, nonlinearity=nonlinearity)))

            if scale_every_coupling:
                layers.append(ScaleTransform(input_dim))
        
        if not scale_every_coupling:
            layers.append(ScaleTransform(input_dim))

        super().__init__(base_distribution, layers, *args, **kwargs)
    
    def _get_permutation(self, permtype: Permutation, i=0):
        """Returns a permutation layer
        """
        if permtype == "random":
            return Permute(torch.randperm(self.input_dim, dtype=torch.long))
        elif permtype == "half":
            if i % 2 == 0: # every 2nd pixel
                perm = torch.arange(self.input_dim, dtype=torch.long)
                perm = perm.reshape(-1, 2).moveaxis(0, 1).reshape(-1)
            elif i % 2 == 1: # interchange conditioning variables and output variables
                perm = torch.arange(self.input_dim, dtype=torch.long)
                perm = perm.reshape(2, -1).flip(0).reshape(-1)
            else: # random permutation
                perm = torch.randperm(self.input_dim, dtype=torch.long)
            return Permute(perm)
        elif permtype == "LU":
            return LUTransform(self.input_dim)  
        else:
            raise ValueError(f"Unknown permutation type {permtype}")

class NiceMaskedConvFlow(Flow):
    """Implementation of the NICE flow architecture using fully connected coupling layers and a checkerboard permutation"""
    def __init__(
            self,
            base_distribution: dist.Distribution,
            coupling_layers: int,
            conv_layers: int,
            kernel_size: int,
            nonlinearity: Optional[torch.nn.Module] = None,
            c_hidden: int = 32,
            rescale_hidden: Union[int, Tuple[int]] = 4,
            *args,
            **kwargs
        ) -> None:
        """Initialization

        Args:
        
            base_distribution: base distribution,
            coupling_layers: number of coupling layers. All coupling layers share the same architecture but not the same weights.
            coupling_nn_layers: number of hidden convolutional layers of the network that computes the coupling loc parameter.
            kernel_size: kernel size of the convolutional layers.
            nonlinearity: nonlinearity of the convolutional layers.
            c_hidden: number of hidden channels of the convolutional layers.
            rescale_hidden: rescaling of hight and width for the hidden layers.
        """
        
        self.coupling_layers = coupling_layers

        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()

        c, h, w = base_distribution.sample().shape
        mask = create_checkerboard_mask(h, w)
        
        layers = []
        self.masks = []
        for i in range(coupling_layers):
            layers.append(
                MaskedCoupling(
                    mask, 
                    ConvNet2D(
                        mask.shape[0],
                        num_layers=conv_layers, 
                        nonlinearity=nonlinearity, 
                        kernel_size=kernel_size, 
                        c_hidden=c_hidden,
                        rescale_hidden=rescale_hidden
                    )
                )
            )
            self.masks.append(mask)
            mask = 1 - mask
        
        layers.append(ScaleTransform(mask.shape))

        super().__init__(base_distribution, layers, *args, **kwargs)
        
class LUFlow(Flow):
    """Implementation of the NICE flow architecture using fully connected coupling layers and a checkerboard permutation"""
    def __init__(
            self,
            base_distribution: dist.Distribution,
            n_layers: int,
            nonlinearity: Optional[torch.distributions.Transform] = None,
            *args,
            **kwargs
        ) -> None:
        """Initialization

        :param: base_distribution: base distribution,
        :param: n_layers: number of LU-layers. 
        :param: nonlinearity: nonlinearity of the convolutional layers. 
        """
        
        self.n_layers = n_layers

        if nonlinearity is None:
            nonlinearity = LeakyReLUTransform

        layers = []

        for i in range(n_layers):
            layers.append(
                LUTransform(
                    base_distribution.sample().shape[0],
                )
            )
            #layers.append(nonlinearity())

        super().__init__(base_distribution, layers, *args, **kwargs)
    
    def is_feasible(self):
        return all([l.is_feasible() for l in self.layers if isinstance(l, LUTransform)])
    
    def add_jitter(self, jitter):
        for layer in self.layers :
            if isinstance(layer, LUTransform) and not layer.is_feasible():
                layer.add_jitter(jitter)

