import onnx


@staticmethod
def merge_models(modified_model, classifier):
    flow_output = modified_model.graph.output[0].name
    classifier_input = classifier.graph.input[0].name
    combined_model = onnx.compose.merge_models(
        modified_model, classifier,
        io_map=[(flow_output, classifier_input)]
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


if __name__ == '__main__':
    PATH_FLOW = "matmuls_replaced_mnist_9_forward_IR_7.onnx"
    PATH_CLASSIFIER ="./MnistSimpleClassifier_IR_7.onnx"
    model = onnx.load(PATH_FLOW)
    classifier = onnx.load(PATH_CLASSIFIER)
    modified_model = swap_mul_inputs(model)
    combined_model = merge_models(modified_model,classifier)
    onnx.save(combined_model, "combined_model.onnx")