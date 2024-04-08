import onnx
from onnx import numpy_helper
from onnx import TensorProto, GraphProto
import numpy

def replace_mul_with_matmul(model_path, output_model_path):
    # Load the ONNX model
    target_name = "trainable_layers.6.scale"
    model = onnx.load(model_path)

    # Iterate through all nodes in the model
    for node in model.graph.node:
        if node.name == "/Mul_6":
            for i in range(len(model.graph.initializer)):
                if model.graph.initializer[i].name == target_name:
                    weights = numpy_helper.to_array(model.graph.initializer[i])
                    diag_weights = numpy.diag(weights)
                    model.graph.initializer[i].CopyFrom(numpy_helper.from_array(diag_weights))
                    model.graph.initializer[i].name = target_name
            # Replace "Mul" node with "MatMul"
            node.op_type = "MatMul"
    # Save the modified model
    onnx.save(model, output_model_path)

if __name__ == "__main__":
    # Provide the path to the input ONNX file and the desired output path
    input_model_path = "./ir_version_7_flow.onnx"
    output_model_path = "./matmuls_replaced_ir_version_7_flow.onnx"

    # Replace "Mul" with "MatMul" and save the modified model
    replace_mul_with_matmul(input_model_path, output_model_path)
