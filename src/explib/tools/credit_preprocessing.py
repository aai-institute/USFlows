import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import pickle

def preprocess_data(input_file, output_file):
    # Load the data
    if not os.path.exists(input_file):
        print(f"Dataset not found: {input_file}")
        return

    data = pd.read_csv(input_file)
    if data is None or data.empty:
        print("Credit data is none or empty")
        return

    # Drop the "Id" column if it exists
    if "Id" in data.columns:
        data = data.drop(["Id"], axis=1)

    # Scaling to range [0, 1]
    min_max_scaler = MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)

    # Standardizing the data
    standard_scaler = StandardScaler()
    data_standardized = standard_scaler.fit_transform(data_scaled)

    # Convert the standardized data back to DataFrame
    data_standardized_df = pd.DataFrame(data_standardized, columns=data.columns)

    # Save the processed data to CSV
    data_standardized_df.to_csv(output_file, index=False)

    # Save the scalers for inverse transform later
    with open('min_max_scaler.pkl', 'wb') as f:
        pickle.dump(min_max_scaler, f)
    with open('standard_scaler.pkl', 'wb') as f:
        pickle.dump(standard_scaler, f)

    print(f"Processed data saved to {output_file}")

def inverse_transform(processed_data_file):
    # Load the processed data
    data_processed = pd.read_csv(processed_data_file)

    # Load the scalers
    with open('min_max_scaler.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)
    with open('standard_scaler.pkl', 'rb') as f:
        standard_scaler = pickle.load(f)

    # Inverse transform the standardization
    data_unstandardized = standard_scaler.inverse_transform(data_processed)

    # Inverse transform the scaling to [0, 1]
    original_data = min_max_scaler.inverse_transform(data_unstandardized)

    # Convert back to DataFrame
    original_data_df = pd.DataFrame(original_data, columns=data_processed.columns)
    return original_data_df


# Example usage:
# Preprocess data
input_file = '/home/mustafa/repos/VeriFlow/experiments/credit/dataset/credit/train_positive.csv'
output_file = '/home/mustafa/repos/VeriFlow/experiments/credit/dataset/credit/train_positive_processed.csv'
preprocess_data(input_file, output_file)

# Inverse transform
reconstructed_file = '/home/mustafa/repos/VeriFlow/experiments/credit/dataset/credit/recontructed.csv'
original_data = inverse_transform(output_file)
# Save the processed data to CSV
original_data.to_csv(reconstructed_file, index=False)
print("Inverse transformed data:")
print(original_data)
