from laplace_flows.experiments.utils import config_from_yaml, read_config
from laplace_flows.experiments.base import Experiment
import torch

if __name__ == "__main__":
    experiment = read_config("config.yaml")
    experiment.conduct("./")