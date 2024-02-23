import ray
import torch
from pyro.distributions import Normal as Laplace, Normal

import src.explib.hyperopt
from src.explib.base import ExperimentCollection
from src.explib.datasets import MnistSplit
from src.veriflow.flows import NiceFlow


def create_exp_nice(digit: int, name: str, base_distribution, permutation="LU"):
    return src.explib.hyperopt.HyperoptExperiment(
        name=name,
        scheduler=ray.tune.schedulers.ASHAScheduler(
            max_t=1000000,
            grace_period=1000000,
            reduction_factor=2),
        num_hyperopt_samples=20,
        gpus_per_trial=0,
        cpus_per_trial=1,
        tuner_params={
            "metric": "val_loss",
            "mode": "min"},
        trial_config={
            "dataset": MnistSplit(),
            "digit": digit,
            "epochs": 200000,
            "patience": 50,
            "batch_size": ray.tune.choice([32]),
            "optim_cfg": {
                "optimizer": torch.optim.Adam,
                "params": {
                    "lr": ray.tune.loguniform(1e-4, 1e-2),
                    "weight_decay": 0.0
                }
            },
            "model_cfg": {
                "type": NiceFlow,
                "params": {
                    "coupling_layers": ray.tune.choice([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "coupling_nn_layers": ray.tune.choice([[w] * l for l in [1, 2, 3, 4] for w in [10, 20, 50, 100, 200]]),
                    "nonlinearity": ray.tune.choice([torch.nn.ReLU()]),
                    "split_dim": ray.tune.choice([i for i in range(1, 51)]),
                    "base_distribution": base_distribution,
                    "permutation": permutation
                }
            }})


def create_digit_experiment(digit: int):
  return src.explib.base.ExperimentCollection(
    name="mnist_basedist_comparison",
    experiments=[
        create_exp_nice(digit, "mnist_nice_lu_laplace",
            Laplace(loc=torch.zeros(100), scale= torch.ones(100))),
        create_exp_nice(digit, "mnist_nice_lu_normal",
            Normal(loc=torch.zeros(100), scale=torch.ones(100))),
        create_exp_nice(digit, "mnist_nice_rand_laplace",
            Laplace(loc=torch.zeros(100), scale= torch.ones(100)),
            permutation="random"),
        create_exp_nice(digit, "mnist_nice_rand_normal",
            Normal(loc=torch.zeros(100), scale=torch.ones(100)),
            permutation="random")
  ])


if __name__ == '__main__':
    experiment_collection = ExperimentCollection(
            name="mnist_digit_basedist_comparison",
            experiments=[create_digit_experiment(digit) for digit in range(10)]) \
        .run_cli()