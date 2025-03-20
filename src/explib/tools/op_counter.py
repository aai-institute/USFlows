from onnx_opcounter import calculate_params
import onnx
from onnx import numpy_helper


def count_parameters(model):
    total_parameters = 0
    for initializer in model.graph.initializer:
        total_parameters += numpy_helper.to_array(initializer).size
    return total_parameters



if __name__ == '__main__':
    PATH ="/home/mustafa/repos/VeriFlow/experiments/verification/resources/radial/"
    NAMES = ["mnist_4_forward.onnx", "model_0_lognormal_decay_forward.onnx"]

    for name in NAMES:
        model = onnx.load_model(f'/home/mustafa/repos/counterfactuals/plausible_counterfactuals/experiment_sources/power/classifier_medium.onnx')
        params = calculate_params(model)
        print(f'Number of params {name}:', params)