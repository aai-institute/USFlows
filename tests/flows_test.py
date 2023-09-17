from sklearn.datasets import load_digits
import torch
from pyro.distributions.transforms import Permute, AffineCoupling
from pyro import distributions as dist
from veriflow.flows import Flow
from veriflow.transforms import ScaleTransform 
from veriflow.networks import AdditiveAffineNN

from matplotlib import pyplot as plt

def test_mnist():
    mnist = load_digits()

    input_dim = mnist["data"].shape[1]
    coupling_dim = 32
    additive_dim = input_dim - coupling_dim

    laplace = dist.Laplace(torch.zeros(input_dim),torch.ones(input_dim))
    layers = []
    trainable_layers = []
    for i in range(2):
        layers.append(Permute(torch.randperm(input_dim, dtype=torch.long)))
        layers.append(AffineCoupling(coupling_dim, AdditiveAffineNN(coupling_dim, [512], additive_dim)))
        trainable_layers.append(layers[-1])

    layers.append(ScaleTransform(input_dim))
    trainable_layers.append(layers[-1])

    flow = Flow(laplace, layers)

    data = torch.Tensor(mnist["data"])
    losses = flow.fit(
        data, 
        {"optim": torch.optim.Adam, "optim_params": {"lr": .000001}, "iters": 2000, "batch_size": 32}
        )
    
    plt.plot(losses)
    plt.set_ylabel("neg. log-likelihood")
    plt.show()

