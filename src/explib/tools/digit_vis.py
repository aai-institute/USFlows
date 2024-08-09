from glob import glob
from typing import Dict
from src.explib.config_parser import from_checkpoint
from src.veriflow.flows import Flow
import torch 
from matplotlib import pyplot as plt
import click
from src.explib.visualization import udl_multisample
from typing import Dict, Iterable, Literal
import numpy as np

Norm = Literal[-1, 1, 2]

# Authored by Faried and Mustafa

def nsample(model, n, reshape=[10, 10]):
    with torch.no_grad():
        sample = torch.clip(model.sample(torch.tensor([n, n])), 0, 1) * 255
        
    if reshape is not None:
        reshape = [n, n] + reshape
        sample = sample.reshape(reshape)

    sample = torch.cat([x for x in sample], dim=-1)
    sample = torch.cat([x for x in sample], dim=0)
    
    return sample


def plot_digits(models: Dict[str, Flow], sqrtn: int, save_to=None, res=[28, 28]):
    """ Plot the samples from the models in a grid.

    Args:
        models: A dictionary of models. The keys are the digits and the values are dictionaries of models.
        sqrtn: The number of samples to plot in each row and column.
        save_to: If not None, the plot is saved to this path.
    """
    with torch.no_grad():
        experiments = sorted([
            exp for exp in models.keys()
        ])
        num_experiments = len(experiments)

        ncols = 5
        nrows = 2
        figsize = (7 * ncols, 25)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        # Flatten axes for easy iteration
        axes = axes.flatten()

        for idx, (digit, ax) in enumerate(zip(experiments, axes)):
            if idx < num_experiments:
                model = models[digit]
                sample = nsample(model, sqrtn, reshape=res)
                ax.imshow(sample, cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(digit, fontsize=55)
            else:
                # Hide any unused axes
                ax.axis('off')
        plt.tight_layout()
        if save_to:
            plt.savefig(save_to)
        plt.show()

def norm(x: torch.Tensor, p: Norm):
    """ Computes the L_p norm of a tensor x.

    Args:
        x: A tensor of shape (..., dim)
        p: The norm to compute. If p is -1 or inf, the maximum norm is computed.

    Returns:
            A tensor of shape (...,)
    """
    if p == -1 or p == float("inf"):
        return x.abs().max(dim=-1)[0]
    elif p == 1:
        return x.abs().sum(dim=-1)
    elif p == 2:
        return torch.sqrt((x * x).sum(dim=-1))


def plot_digits_UDL(models: Dict[str, Flow], sqrtn: int, save_to=None, res=[28, 28],
                    sqrt_n_sample: int = 3,
                    n_udl_estimation: int = 100000,
                    saveto=None
                    ):
    with torch.no_grad():
        experiments = sorted(models.keys())
        num_experiments = len(experiments)

        # Calculate the number of rows and columns for the grid
        ncols = 5
        nrows = 2

        figsize = (7 * ncols, 7 * nrows)  # Adjusted figure size to keep aspect ratio similar
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Flatten axes for easy iteration
        axes = axes.flatten()

        for idx, (digit, ax) in enumerate(zip(experiments, axes)):
            if idx < num_experiments:
                model = models[digit]
                with torch.no_grad():
                    radius = norm(model.base_distribution.sample((n_udl_estimation,)), 1).quantile(0.1)
                sample = udl_multisample(model, radius, "conditional", 1, 3, (28, 28))
                ax.imshow(sample, cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(digit, fontsize=55)
            else:
                # Hide any unused axes
                ax.axis('off')

        plt.tight_layout()
        if save_to:
            plt.savefig(save_to)
        plt.show()


def evaluate(exp_dir, res = [28, 28], save_to = "./"):
    models = dict()
    query = f"{exp_dir}/*"
    exps = glob(query)
    PATHS = [ # Adjust the paths accordingly.
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/0_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/1_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/2_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/3_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/4_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/5_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/6_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/7_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/8_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/9digit9/experiment/"
    ]
    for exp in PATHS:
        exp_name = exp.split("/")[7][0]
        cfg = glob(f"{exp}/*.pkl")[-1]
        wghts = glob(f"{exp}/*.pt")[-1]
        onnx_model = glob(f"{exp}/forward.onnx")
        models[exp_name] = from_checkpoint(cfg, wghts)
    plot_digits_UDL(models, 3, save_to, res)
    plot_digits(models, 3, save_to, res)

@click.command()
@click.option("--dir", help="experients directory")
def run(dir):
    evaluate(dir)

if __name__ == "__main__":
    run()