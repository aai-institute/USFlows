import numpy as np

from torch.utils.data import Dataset

import pyro
from pyro import distributions as dist
from pyro.distributions.transforms import SoftplusTransform
from pyro.nn import DenseNN
from pyro.distributions.transforms import AffineCoupling, LowerCholeskyAffine
from pyro.infer import SVI
from typing import List, Dict, Literal, Any, Iterable, Optional, Union, Tuple
import torch
from torch.utils.data import DataLoader

from sklearn.datasets import load_digits
from tqdm import tqdm
from src.veriflow.transforms import (
    ScaleTransform,
    MaskedCoupling,
    Permute,
    LUTransform,
    LeakyReLUTransform,
    BaseTransform,
)
from src.veriflow.networks import AdditiveAffineNN, ConvNet2D


class Flow(torch.nn.Module):
    """Base implementation of a flow model"""

    # Export mode determines whether the log_prob or the sample function is exported to onnx
    export_modes = Literal["log_prob", "sample"]
    export: export_modes = "log_prob"

    def forward(self, x: torch.Tensor):
        """Dummy implementation of forward method for onnx export. The self.export attribute
        determines whether the log_prob or the sample function is exported to onnx"""
        if self.export == "log_prob":
            return self.log_prob(x)
        elif self.export == "sample":
            return self.sample()
        else:
            raise ValueError(f"Unknown export mode {self.export}")

    def __init__(self, base_distribution, layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = layers
        self.trainable_layers = torch.nn.ModuleList(
            [l for l in layers if isinstance(l, torch.nn.Module)]
        )
        self.base_distribution = base_distribution
        
        # Redeclare all batch dimensions to event dimensions
        # This is a sanitary measure to avoid pyro from creating a batch of transforms
        # rather than a single transform.
        batch_shape = self.base_distribution.batch_shape
        if len(batch_shape) > 0:
            self.base_distribution = dist.Independent(
                self.base_distribution, len(batch_shape)
            )

        self.transform = dist.TransformedDistribution(self.base_distribution, layers)

    def log_prior(self) -> torch.Tensor:
        """Returns the log prior of the model parameters. The model is trained in maximum posterior fashion, i.e.
        $$argmax_{\\theta} \log p_{\\theta}(D) + \log p_{prior}(\\theta)$$ By default, this ia the constant zero, which amounts
        to maximum likelihood training (improper uniform prior).
        """
        return 0
        
    
    def fit(
        self,
        data_train: Dataset,
        optim: torch.optim.Optimizer = torch.optim.Adam,
        optim_params: Dict[str, Any] = None,
        batch_size: int = 32,
        shuffe: bool = True,
        gradient_clip: float = None,
        device: torch.device = None,
        jitter: float = 1e-6,
        epochs: int = 10
    ) -> float:
        """
        Wrapper function for the fitting procedure. Allows basic configuration of the optimizer and other
        fitting parameters.

        Args:
            data_train: training data.
            batch_size: number of samples per optimization step.
            optim: optimizer class.
            optimizer_params: optimizer parameter dictionary.
            jitter: Determines the amount of jitter that is added if the optimization leaves the feasible region.
            epochs: number of epochs.

        Returns:
            Loss curve .
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
        
        epoch_losses = []
        for _ in range(epochs):
            losses = []
            if shuffe:
                perm = np.random.choice(N, N, replace=False)
                data_train_shuffle = data_train[perm]

            for idx in range(0, N, batch_size):
                idx_end = min(idx + batch_size, N)
                try:
                    sample = torch.Tensor(data_train_shuffle[idx:idx_end][0]).to(device)
                except:
                    continue
                optim.zero_grad()
                while not self.is_feasible():
                    self.add_jitter(jitter)
                loss = -model.transform.log_prob(sample).mean() - model.log_prior()
                losses.append(float(loss.detach()))
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optim.step()
                while not self.is_feasible():
                    self.add_jitter(jitter)

                model.transform.clear_cache()
            epoch_losses.append(np.mean(losses))
            
        return epoch_losses

    def to_onnx(self, path: str, export_mode: export_modes = "log_prob") -> None:
        """Saves the model as onnx file

        Args:
            path: path to save the model.
            export_mode: export mode. Can be "log_prob" or "sample".
        """
        mode_cache = self.export
        self.export = export_mode
        dummy_input = self.base_distribution.sample()
        torch.onnx.export(self, dummy_input, path, verbose=True)
        self.export = mode_cache
        

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the models log-densities for the given samples

        Args:
            x: sample tensor.
        """
        return self.transform.log_prob(x)

    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Returns n_sample samples from the distribution

        Args:
            n_sample: sample shape.
        """
        if sample_shape is None:
            sample_shape = [1]

        return self.transform.sample(sample_shape)

    def to(self, device) -> None:
        """Moves the model to the given device"""
        self.device = device
        # self.layers = torch.nn.ModuleList([l.to(device) for l in self.layers])
        self.trainable_layers = torch.nn.ModuleList(
            [l.to(device) for l in self.trainable_layers]
        )
        return super().to(device)

    def is_feasible(self) -> bool:
        """Checks is the model parameters meet all constraints"""
        return all(
            [l.is_feasible() for l in self.layers if isinstance(l, BaseTransform)]
        )

    def add_jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to meet non-zero constraints"""
        for l in self.layers:
            if isinstance(l, BaseTransform) and not l.is_feasible():
                l.add_jitter(jitter)


class NiceFlow(Flow):
    mask = Literal["random", "half", "alternate"]

    """Implementation of the NICE flow architecture by using fully connected coupling layers"""

    def __init__(
        self,
        base_distribution: dist.Distribution,
        coupling_layers: int,
        coupling_nn_layers: List[int],
        split_dim: int,
        nonlinearity: Optional[torch.nn.Module] = None,
        masktype: mask = "half",
        use_lu: bool = True,
        prior_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        """Initialization

        Args:
            base_distribution: base distribution,
            coupling_layers: number of coupling layers. All coupling layers share the same architecture but not the same weights.
            coupling_nn_layers: number of neurons in the hidden layers of the dense neural network that computes the coupling loc parameter.
            split_dim: split dimension for the coupling, i.e. input dimension of the conditioner.
            scale_every_coupling: if True, a scale transform is applied after every coupling layer. Otherwise, a single scale transform is applied after all coupling layers.
            nonlinearity: nonlinearity of the coupling network.
            permutation: permutation type. Can be "random" or "half".
        """
        input_dim = base_distribution.sample().shape[0]
        self.input_dim = input_dim
        self.coupling_layers = coupling_layers
        self.coupling_nn_layers = coupling_nn_layers
        self.split_dim = split_dim
        self.perm_type = masktype
        self.prior_scale = torch.tensor(prior_scale)
        self.use_lu = use_lu

        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()

        rdim = input_dim - split_dim
        layers = []
        self.lu_layers = []
        for i in range(coupling_layers):
            if self.use_lu:
                layers.append(LUTransform(input_dim))
                self.lu_layers.append(layers[-1])
            mask = self._get_mask(masktype, i)
            layers.append(
                MaskedCoupling(
                    mask,
                    DenseNN(
                        input_dim,
                        coupling_nn_layers,
                        [input_dim],
                        nonlinearity=nonlinearity,
                    ),
                )
            )
            
        
        layers.append(ScaleTransform(input_dim))

        super().__init__(base_distribution, layers, *args, **kwargs)

    def _get_mask(self, masktype: mask, i=0):
        """Returns a mask for the i-th coupling
        """
        if masktype == "random":
            mask = torch.bernoulli(torch.tensor([self.split_dim/self.input_dim] * self.input_dim))
            return mask
        elif masktype == "half":
            mask = torch.zeros(self.input_dim)
            mask[:self.split_dim] = 1
            if i % 2 == 1: 
                mask =  1 - mask
            return mask
        elif masktype == "alternate":
            mask = [0, 1] * (self.input_dim // 2)
            if self.input_dim % 2 == 1:
                mask += [0]
            mask = torch.tensor(mask)
            
            if i % 2 == 1:  # every 2nd pixel
                mask =  1 - mask
            return mask
        else:
            raise ValueError(f"Unknown permutation type {masktype}")

    def log_prior(self) -> torch.Tensor:
        """Returns the log prior of the model parameters. If LU layers are used, 
        we directly regularize the determinant flows Jacobian by putting an 
        independent mirrored log-normal 
        prior on the diagonal elements of $U$ matrices. The normal has 
        mean $0$ and standard deviation $\sqrt{d}\sigma$, where $d$ is the data dimension.
        That means that we put a log-normal prior on the determinant of the Jacobian.
        
        Any additive constant is dropped in the optimization procedure.
        """

        if self.use_lu and self.prior_scale is not None:
            log_prior = 0
            for p in self.lu_layers:
                # log-density of Normal in log-space
                log_prior += -((p.U.diag().log()**2) / (self.prior_scale**2 / self.input_dim)).sum() 
                # Change of variables to input space
                log_prior += -p.U.diag().log().sum()
            return log_prior
        else:
            return 0

class NiceMaskedConvFlow(Flow):
    """Implementation of the NICE flow architecture using fully connected coupling layers
    and a checkerboard permutation"""

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
        **kwargs,
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
        mask = NiceMaskedConvFlow.create_checkerboard_mask(h, w)

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
                        rescale_hidden=rescale_hidden,
                    ),
                )
            )
            self.masks.append(mask)
            mask = 1 - mask

        layers.append(ScaleTransform(mask.shape))

        super().__init__(base_distribution, layers, *args, **kwargs)

    @classmethod
    def create_checkerboard_mask(
        cls, h: int, w: int, invert: bool = False
    ) -> torch.Tensor:
        """Creates a checkerboard mask of size $(h,w)$.

        Args:
            h (_type_): height
            w (_type_): width
            invert (bool, optional): If True, inverts the mask. Defaults to False.
        Returns:
            Checkerboard mask of height $h$ and width $w$.
        """
        x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        mask = torch.fmod(xx + yy, 2)
        mask = mask.to(torch.float32).view(1, 1, h, w)
        if invert:
            mask = 1 - mask
        return mask


class LUFlow(Flow):
    """Implementation of the NICE flow architecture using fully connected coupling layers
    and a checkerboard permutation
    """

    def __init__(
        self,
        base_distribution: dist.Distribution,
        n_layers: int,
        nonlinearity: Optional[torch.distributions.Transform] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialization

        Args:
            base_distribution: base distribution,
            n_layers: number of LU-layers.
            nonlinearity: nonlinearity of the convolutional layers.
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
            layers.append(nonlinearity())

        super().__init__(base_distribution, layers, *args, **kwargs)

    def is_feasible(self):
        """Checks if all LU layers are feasible"""
        return all([l.is_feasible() for l in self.layers if isinstance(l, LUTransform)])

    def add_jitter(self, jitter: float = 1e-6) -> None:
        for layer in self.layers:
            if isinstance(layer, LUTransform) and not layer.is_feasible():
                layer.add_jitter(jitter)
