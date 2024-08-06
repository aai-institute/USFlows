import onnx
from onnx import version_converter, helper
import numpy
from onnx import numpy_helper


@staticmethod
def merge_models(modified_model, classifier):
    flow_output = modified_model.graph.output[0].name
    classifier_input = classifier.graph.input[0].name
    combined_model = onnx.compose.merge_models(
        modified_model, classifier,
        io_map=[(flow_output, classifier_input)],
        prefix1="flow",
        prefix2="classifier"
    )
    onnx.checker.check_model(model=combined_model, full_check=True)
    return combined_model



@staticmethod
def swap_mul_inputs(model):
    # Ensures that the second input to the mul node is the constant node (and not the variable).
    # This constraint is imposed by the marabou library that implicitly
    # assumes the first input to be the variable and the second the constant factor.
    for node in model.graph.node:
        if node.op_type == "Mul":
            if "Constant" in node.input[0]:
                node.input[0], node.input[1] = node.input[1], node.input[0]
    return model


def replace_mul_with_matmul(model_path, output_model_path):
    # Load the ONNX model
    target_name = "flowtrainable_layers.4.scale"
    model = onnx.load(model_path)

    # Iterate through all nodes in the model
    for node in model.graph.node:
        if node.name == "flow/Mul_4":
            for i in range(len(model.graph.initializer)):
                if model.graph.initializer[i].name == target_name:
                    print("manipulating the neural net")
                    weights = numpy_helper.to_array(model.graph.initializer[i])
                    diag_weights = numpy.diag(weights)
                    model.graph.initializer[i].CopyFrom(numpy_helper.from_array(diag_weights))
                    model.graph.initializer[i].name = target_name
            # Replace "Mul" node with "MatMul"
            node.op_type = "MatMul"
#[numpy.diag(numpy_helper.to_array(ini)) if ini.name=="trainable_layers.6.scale" else "" for ini in model.graph.initializer]
    # Save the modified model
    onnx.save(model, output_model_path)


if __name__ == '__main__':
    PATH = "models/classifiers_various_depth/"
    CLASSIFIERS = ["mnist_unscaled0.onnx","mnist_unscaled1.onnx","mnist_unscaled2.onnx","mnist_unscaled3.onnx","mnist_unscaled4.onnx","mnist_unscaled5.onnx"]
    for CLASSIFIER_NAME in CLASSIFIERS:
        FLOW_NAME = "full_mnist_best_fwd.onnx"
        RESULT_PATH = f'./models/classifiers_various_depth/merged_{CLASSIFIER_NAME}'
        model = onnx.load(PATH+FLOW_NAME)
        classifier = onnx.load(PATH+CLASSIFIER_NAME)
        modified_model = swap_mul_inputs(model)
        combined_model = merge_models(modified_model,classifier)
        combined_model.ir_version = 7
        onnx.save(combined_model, RESULT_PATH)

        replace_mul_with_matmul(RESULT_PATH, RESULT_PATH)