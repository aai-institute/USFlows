from onnx_opcounter import calculate_params
import onnx
from onnx import numpy_helper


def count_parameters(model):
    total_parameters = 0
    for initializer in model.graph.initializer:
        total_parameters += numpy_helper.to_array(initializer).size
    return total_parameters



if __name__ == '__main__':
    PATH ="/home/mustafa/repos/ERAN/eran/veriflow/unscaled_mnist/experiments/"
    NAMES = ["merged_mnist_unscaled0.onnx",
             "merged_mnist_unscaled1.onnx",
             "merged_mnist_unscaled2.onnx",
             "merged_mnist_unscaled3.onnx",
             "merged_mnist_unscaled4.onnx",
             "merged_mnist_unscaled5.onnx"]

    for name in NAMES:
        model = onnx.load_model(f'{PATH}{name}')
        params = calculate_params(model)
        print(f'Number of params {name}:', params)
        params = count_parameters(model)
        print(f'Number of params {name}:', params)
