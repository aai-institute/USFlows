import onnx
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    # Load the ONNX model
    onnx_model_path = './merged_model.onnx'
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    # Create an inference session
    ort_session = ort.InferenceSession(onnx_model_path)

    # Sample input: a tensor of zeros with the expected input shape
    # Adjust the shape according to your model's expected input shape
    input_shape = ort_session.get_inputs()[0].shape
    for i in range(100):
        dummy_input = np.random.uniform(low=-0.1, high=0.1, size=784).astype(np.float32)
        # Run inference
        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input})
        output_image = outputs[0]
        print(np.argmax(output_image))