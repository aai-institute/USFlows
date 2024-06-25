import onnxruntime as ort
import numpy as np
import torch
import pickle


if __name__ == '__main__':

    ort_sess = ort.InferenceSession('/home/mustafa/repos/VeriFlow/scripts/reports/model.onnx')

    # Load the scalers
    with open('min_max_scaler.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)
    #with open('standard_scaler.pkl', 'rb') as f:
    #    standard_scaler = pickle.load(f)
    samples = []
    for i in range(20):
        #x = torch.zeros(21)
        x = np.random.uniform(low=-0.01, high=0.01, size=(21,)).astype(np.float32)
        outputs = ort_sess.run(None, {'x.1': x})
        # Print Result
        print(f'Predicted: "{outputs[0]}"')
        # Inverse transform the standardization
        #data_unstandardized = standard_scaler.inverse_transform(outputs[0].reshape(1, -1))
        # Inverse transform the scaling to [0, 1]
        original_data = min_max_scaler.inverse_transform(outputs[0].reshape(1, -1))
        print(f'reconstructed: {original_data[0]}')
        samples.append([round(i) for i  in original_data[0]])
    for sample in samples:
        print(f'-1,{sample[0]},{sample[1]},{sample[2]},{sample[3]},{sample[4]},{sample[5]},{sample[6]},{sample[7]},{sample[8]},{sample[9]},{sample[10]},{sample[11]},{sample[12]},{sample[13]},{sample[14]},{sample[15]},{sample[16]},{sample[17]},{sample[18]},{sample[19]},{sample[20]}')



