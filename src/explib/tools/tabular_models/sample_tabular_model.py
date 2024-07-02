import onnxruntime as ort
import numpy as np
import pickle
import csv
import pandas as pd

feature_bounds = {
    'laufkont': (0, 3),
    'laufzeit': (0, 999999999),
    'moral': (0, 4),
    'verw': (0, 10),
    'hoehe': (0, 999999999),
    'sparkont': (0, 4),
    'beszeit': (0, 2),
    'rate': (0, 3),
    'famges': (0, 3),
    'buerge': (0, 2),
    'wohnzeit': (0, 3),
    'verm': (0, 3),
    'alter': (0, 999999999),
    'weitkred': (0, 2),
    'wohn': (0, 2),
    'bishkred': (0, 3),
    'beruf': (0, 3),
    'pers': (0, 1),
    'telef': (0, 1),
    'gastarb': (1, 2),
    'kredit': (0, 1)
}

encoded_var_verw = ("verw_0,verw_1,verw_2,verw_3,verw_4,verw_5,verw_6,verw_8,verw_9,verw_10").split(",")
encoded_var_famges = ("famges_1,famges_2,famges_3,famges_4").split(",")
encoded_var_wohn = ("wohn_1,wohn_2,wohn_3").split(",")

# The initial column names in the training data.
encoded_column_names = ("laufkont,laufzeit,moral,hoehe,sparkont,beszeit,rate,buerge,wohnzeit,verm,alter,weitkred,"
                        "bishkred,beruf,pers,telef,gastarb,kredit,verw_0,verw_1,verw_2,verw_3,verw_4,verw_5,verw_6,"
                        "verw_8,verw_9,verw_10,famges_1,famges_2,famges_3,famges_4,wohn_1,wohn_2,wohn_3").split(",")

# The target columns. I.e., the format to which we want to process the samples into.
final_columns = ("laufkont,laufzeit,moral,verw,hoehe,sparkont,beszeit,rate,famges,buerge,wohnzeit,verm,alter,"
                 "weitkred,wohn,bishkred,beruf,pers,telef,gastarb,kredit")

def samples_encoded_to_csv(samples, csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(encoded_column_names)
        for one_hot_sample in samples:
            writer.writerow(one_hot_sample)

def samples_decoded_to_csv(raw_outputs_path, column_names, output_path):
    data = pd.read_csv(raw_outputs_path)  # contains the samples of the flow - already inverse transformed scale
    data.columns = column_names

    data_copy = pd.read_csv(raw_outputs_path) # contains the samples of the flow - already inverse transformed scale
    data_copy.columns = column_names

    for encoded_var in [encoded_var_verw, encoded_var_famges, encoded_var_wohn]:
        # Find the argmax of each row
        argmax_indices = data[encoded_var].values.argmax(axis=1)
        decoded_categories = [encoded_var[idx].split("_")[1] for idx in argmax_indices]
        encoded_var_df = pd.DataFrame({encoded_var[0].split("_")[0]: decoded_categories})
        original_data_df = data.drop(columns=encoded_var)
        data = pd.concat([original_data_df, encoded_var_df], axis=1)

    data_reordered = data[final_columns.split(",")]
    data_reordered = data_reordered.round(0).astype(int)

    for column in final_columns.split(","):
        data_reordered[column] = data_reordered[column].clip(lower=feature_bounds[column][0], upper=feature_bounds[column][1])
    data_reordered.to_csv("./0-5.csv", index=False)



if __name__ == '__main__':
    model_name = 'model_forward_with_preprocessing.onnx'
    path = '/scripts/reports/'

    ort_sess = ort.InferenceSession(path+model_name)

    # Load the scalers
    with open('standard_scaler.pkl', 'rb') as f:
        standard_scaler = pickle.load(f)

    samples = []
    for i in range(100):
        x = np.random.uniform(low=-0.5, high=0.5, size=(35,)).astype(np.float32)
        outputs = ort_sess.run(None, {'x.1': x})
        original_data = standard_scaler.inverse_transform(outputs[0].reshape(1, -1))
        samples.append([i for i in original_data[0]])
    raw_outputs_path = path+'/sampled.csv'
    samples_encoded_to_csv(samples, raw_outputs_path)
    path_to_training_data = '/experiments/credit/dataset/credit/train_positive_processed_one_hot.csv'
    samples_decoded_to_csv(raw_outputs_path, encoded_column_names, raw_outputs_path)



