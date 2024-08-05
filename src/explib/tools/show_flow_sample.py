import onnx
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    # Load the ONNX model
    onnx_model_path = './forward_full_mnist_3.onnx'
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    # Create an inference session
    ort_session = ort.InferenceSession(onnx_model_path)

    # Sample input: a tensor of zeros with the expected input shape
    # Adjust the shape according to your model's expected input shape
    input_shape = ort_session.get_inputs()[0].shape
    for i in range(10):
        dummy_input = np.random.uniform(low=-0.1, high=0.1, size=784).astype(np.float32)
        # Run inference
        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input})

        # Assuming the model outputs a single tensor with the image data
        output_image = outputs[0]  # Remove any singleton dimensions if needed

        sample = np.uint8(np.clip(output_image, 0, 1) * 255)
        # Display the image
        plt.imshow(torch.tensor(sample).view(28, 28), cmap='gray')  # Adjust 'cmap' if the image is in color
        plt.axis('off')
        plt.show()