import onnxruntime as ort
import numpy as np
import torch
import pickle


if __name__ == '__main__':

    ort_sess = ort.InferenceSession('/home/mustafa/repos/VeriFlow/scripts/reports/model.onnx')

    # Load the scalers
    with open('min_max_scaler.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)
    with open('standard_scaler.pkl', 'rb') as f:
        standard_scaler = pickle.load(f)

    for i in range(20):
        #x = torch.zeros(21)
        x = np.random.uniform(low=-0.01, high=0.01, size=(21,)).astype(np.float32)
        outputs = ort_sess.run(None, {'x.1': x})
        # Print Result
        print(f'Predicted: "{outputs[0]}"')
        # Inverse transform the standardization
        data_unstandardized = standard_scaler.inverse_transform(outputs[0].reshape(1, -1))
        # Inverse transform the scaling to [0, 1]
        original_data = min_max_scaler.inverse_transform(data_unstandardized)
        print(f'reconstructed: {original_data[0]}')



