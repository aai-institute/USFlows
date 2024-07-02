import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

full_data_path = "/home/mustafa/repos/VeriFlow/experiments/credit/dataset/heloc/heloc_full.csv"
full_data_output = "/home/mustafa/repos/VeriFlow/experiments/credit/dataset/heloc/heloc_full_scaled.csv"
positive_data_output = "/home/mustafa/repos/VeriFlow/experiments/credit/dataset/heloc/heloc_positive_scaled.csv"
negative_data_output = "/home/mustafa/repos/VeriFlow/experiments/credit/dataset/heloc/heloc_negative_scaled.csv"


def scale_save_data(dataframe, scaler_name):
    # Scaling to range [0, 1]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dataframe)
    with open(scaler_name, 'wb') as f:
        pickle.dump(scaler, f)
    return pd.DataFrame(data_scaled, columns=dataframe.columns)


def preprocess_data():
    # Load the data
    if not os.path.exists(full_data_path):
        print(f"Dataset not found: {full_data_path}")
        return

    data = pd.read_csv(full_data_path)
    if data is None or data.empty:
        print("heloc data is none or empty")
        return

    full_dataset = scale_save_data(data,'standard_scaler_full.pkl')
    full_dataset.to_csv(full_data_output, index=False)
    positive_data = scale_save_data(data[data['RiskPerformance'] == 0.0], 'standard_scaler_positive.pkl')
    positive_data.to_csv(positive_data_output, index=False)
    negative_data = scale_save_data(data[data['RiskPerformance'] == 1.0], 'standard_scaler_negative.pkl')
    negative_data.to_csv(negative_data_output, index=False)
    print("asdf")

if __name__ == '__main__':
    preprocess_data()




