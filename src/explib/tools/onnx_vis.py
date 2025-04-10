import onnx
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    # Load the ONNX model
    onnx_model_path = '/home/mustafa/Documents/midas/conv/withtransagain/_trial_29411_00000_0_batch_size=1024,nonlinearity=ref_ph_842f7f0d_2025-04-02_11-30-48/forward.onnx'
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    # Create an inference session
    ort_session = ort.InferenceSession(onnx_model_path)

    # Sample input: a tensor of zeros with the expected input shape
    # Adjust the shape according to your model's expected input shape
    input_shape = ort_session.get_inputs()[0].shape
    for i in range(1):
        dummy_input = np.random.uniform(low=-0.001, high=0.001, size=input_shape).astype(np.float32)
        # Run inference
        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input})
        # Assuming the model outputs a single tensor with the image data
        output_image = outputs[0][0].reshape(2, 2, 14, 14).transpose(2, 0, 3, 1).reshape(28, 28)  # Remove any singleton dimensions if needed
        sample = np.uint8(np.clip(output_image, 0, 1) * 255)
        # Display the image
        plt.imshow(torch.tensor(sample).view(28, 28), cmap='gray')  # Adjust 'cmap' if the image is in color
        plt.axis('off')
        plt.savefig("flow_out.png")