import onnxruntime as ort
import numpy as np
import torch
import pickle
import csv




def samples_encoded_to_csv(samples, csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        column_names = "laufkont,laufzeit,moral,hoehe,sparkont,beszeit,rate,buerge,wohnzeit,verm,alter,weitkred,bishkred,beruf,pers,telef,gastarb,kredit,verw_0,verw_1,verw_2,verw_3,verw_4,verw_5,verw_6,verw_8,verw_9,verw_10,famges_1,famges_2,famges_3,famges_4,wohn_1,wohn_2,wohn_3".split(sep=",")
        writer.writerow(column_names)
        for one_hot_sample in samples:
            writer.writerow(one_hot_sample)


def samples_to_csv(samples, csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        columns_one_hot = 'verw_0,verw_1,verw_2,verw_3,verw_4,verw_5,verw_6,verw_8,verw_9,verw_10,famges_1,famges_2,famges_3,famges_4,wohn_1,wohn_2,wohn_3'.split(sep=",")
        column_names = "Id,laufkont,laufzeit,moral,verw,hoehe,sparkont,beszeit,rate,famges,buerge,wohnzeit,verm,alter,weitkred,wohn,bishkred,beruf,pers,telef,gastarb,kredit".split(sep=",")
        writer.writerow(column_names)
        for one_hot_sample in samples:
            writer.writerow(one_hot_sample)
            for column in columns_one_hot:
                column_name_number = column.split("_")
                encoded_column_name = column_name_number[0]
                encoded_column_val = column_name_number[1]


if __name__ == '__main__':
    model_name = 'model_forward.onnx'
    path = '/home/mustafa/repos/VeriFlow/scripts/reports/'
    ort_sess = ort.InferenceSession(path+model_name)

    # Load the scalers
    with open('min_max_scaler.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)
    #with open('standard_scaler.pkl', 'rb') as f:
    #    standard_scaler = pickle.load(f)
    samples = []
    for i in range(20):
        #x = torch.zeros(21)
        x = np.random.uniform(low=-0.1, high=0.1, size=(35,)).astype(np.float32)
        outputs = ort_sess.run(None, {'x.1': x})
        # Print Result
        print(f'Predicted: "{outputs[0]}"')
        # Inverse transform the standardization
        #data_unstandardized = standard_scaler.inverse_transform(outputs[0].reshape(1, -1))
        # Inverse transform the scaling to [0, 1]
        original_data = min_max_scaler.inverse_transform(outputs[0].reshape(1, -1))
        print(f'reconstructed: {original_data[0]}')
        samples.append([i for i in original_data[0]])

    #samples_encoded_to_csv(samples, path+'/sampled.csv')
    print("for csv file:")
    for sample in samples:
        print(f'{[i for i in sample]}')
    #    print(f'-1,{sample[0]},{sample[1]},{sample[2]},{sample[3]},{sample[4]},{sample[5]},{sample[6]},{sample[7]},{sample[8]},{sample[9]},{sample[10]},{sample[11]},{sample[12]},{sample[13]},{sample[14]},{sample[15]},{sample[16]},{sample[17]},{sample[18]},{sample[19]},{sample[20]}')



