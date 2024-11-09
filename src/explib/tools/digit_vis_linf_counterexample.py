import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

# Constants for reshaping and visualization
RESOLUTION = [14, 14]  # Adjust if different output shape is expected
RESHAPE = (RESOLUTION[0], RESOLUTION[1])
p = 0.000000001


def quantile_log_normal(p, mu=1, sigma=0.5):
    return math.exp(mu + sigma * norm.ppf(p))

def load_model(onnx_path):
    """Loads an ONNX model for inference."""
    return ort.InferenceSession(onnx_path)


def forward_random_vector(model):
    """Generates a random vector, forwards it through the model, and returns the output."""
    # Assume single input; generate a random input vector based on input shape
    input_name = model.get_inputs()[0].name
    input_shape = model.get_inputs()[0].shape
    print(f'p={p} and quantile={quantile_log_normal(p)}')
    random_input = np.random.choice([-quantile_log_normal(p), quantile_log_normal(p)], size=input_shape).astype(np.float32)

    # Perform inference
    output = model.run(None, {input_name: random_input})
    output_vector = output[0]

    # Apply clipping to the output vector (e.g., between 0 and 1)
    clipped_output = np.clip(output_vector, 0, 1)

    return clipped_output


def visualize_flow_output_border(output_vector, reshape=RESHAPE):
    """Visualizes the output vector as an image."""
    output_image = output_vector.reshape(reshape)
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')
    plt.title(f"UDL {p}")
    plt.show()

def visualize_flow_output_center(output_vector, predicted_class, reshape=RESHAPE):
    """Visualizes the output vector as an image."""
    output_image = output_vector.reshape(reshape)
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')
    plt.title(f"Zero vector, class prediction {predicted_class}")
    plt.show()

def check_class_of_zero_vector(classifier, model):
    input_name = model.get_inputs()[0].name
    input_shape = model.get_inputs()[0].shape
    random_input = np.zeros(input_shape).astype(np.float32)
    # Perform inference
    output = model.run(None, {input_name: random_input})
    output_vector = output[0]
    digit_predict = classifier.run(None, {model.get_inputs()[0].name: output_vector})[0].argmax()

    # Apply clipping to the output vector (e.g., between 0 and 1)
    clipped_output = np.clip(output_vector, 0, 1)
    return digit_predict, clipped_output

if __name__ == '__main__':
    # Example usage
    onnx_paths = [
        "/home/mustafa/Documents/midas/all_digits_lu_small/0_mnist_logNormal_linf_digit_0/forward.onnx",
        "/home/mustafa/Documents/midas/all_digits_lu_small/1_mnist_logNormal_linf_digit_9/forward.onnx",
        "/home/mustafa/Documents/midas/all_digits_lu_small/2_mnist_logNormal_linf_digit_8/forward.onnx",
        "/home/mustafa/Documents/midas/all_digits_lu_small/3_mnist_logNormal_linf_digit_7/forward.onnx",
        "/home/mustafa/Documents/midas/all_digits_lu_small/4_mnist_logNormal_linf_digit_6/forward.onnx",
        "/home/mustafa/Documents/midas/all_digits_lu_small/5_mnist_logNormal_linf_digit_5/forward.onnx",
        "/home/mustafa/Documents/midas/all_digits_lu_small/6_mnist_logNormal_linf_digit_4/forward.onnx",
        "/home/mustafa/Documents/midas/all_digits_lu_small/7_mnist_logNormal_linf_digit_3/forward.onnx",
        "/home/mustafa/Documents/midas/all_digits_lu_small/8_mnist_logNormal_linf_digit_2/forward.onnx",
        "/home/mustafa/Documents/midas/all_digits_lu_small/9_mnist_logNormal_linf_digit_1/forward.onnx",
    ]

    SHOW_CENTER = True

    classifier_path = "/home/mustafa/repos/VeriFlow/experiments/verification/resources/MnistClassifier_14_14.onnx"
    for onnx_path in onnx_paths:
        flow_model = load_model(onnx_path)
        classifier = load_model(classifier_path)
        output_vector = forward_random_vector(flow_model)
        visualize_flow_output_border(output_vector)
        if SHOW_CENTER:
            predicted_digit, center_clipped = check_class_of_zero_vector(classifier, flow_model)
            visualize_flow_output_center(center_clipped, predicted_digit)