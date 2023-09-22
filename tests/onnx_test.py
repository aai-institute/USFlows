from veriflow.flows import NiceFlow
from pyro.distributions import Normal
import torch

def test_onnx():
    loc = torch.zeros(2)
    scale = torch.ones(2)
    model = NiceFlow(
        Normal(loc, scale),
        2,
        [10, 10],
        split_dim=1,
        permutation= "half"
    )
    model.to_onnx("log_prob.onnx")
    model.to_onnx("sample.onnx", export_mode="sample")