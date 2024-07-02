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


    # One-hot encode specified features
    one_hot_features = ['verw', 'famges', 'wohn']
    data = pd.get_dummies(data, columns=one_hot_features)

    # Scaling to range [0, 1]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    with open('standard_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Convert the standardized data back to DataFrame
    data_standardized_df = pd.DataFrame(data_scaled, columns=data.columns)
    # Save the processed data to CSV
    data_standardized_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

def inverse_transform(processed_data_file, input_file):
    # Load the processed data
    data_processed = pd.read_csv(processed_data_file)

    # Load the scalers
    with open('standard_scaler.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)

    # Inverse transform the scaling to [0, 1]
    original_data = min_max_scaler.inverse_transform(data_processed)
    original_data = original_data.round(0).astype(int)
    # Convert back to DataFrame
    original_data_df = pd.DataFrame(original_data, columns=data_processed.columns)
    encoded_vars = "verw_0,verw_1,verw_2,verw_3,verw_4,verw_5,verw_6,verw_8,verw_9,verw_10,famges_1,famges_2,famges_3,famges_4,wohn_1,wohn_2,wohn_3".split(",")
    original_data_df_decode_dummies = pd.from_dummies(original_data_df[encoded_vars], sep="_")
    original_data_df = original_data_df.drop(columns=encoded_vars)
    merged_data = pd.concat([original_data_df, original_data_df_decode_dummies], axis=1)

    data_processed = pd.read_csv(input_file)
    data_reconstructed_with_id = pd.concat([merged_data, data_processed["Id"]], axis=1)
    data_reconstructed_with_id = data_reconstructed_with_id["Id,laufkont,laufzeit,moral,verw,hoehe,sparkont,beszeit,rate,famges,buerge,wohnzeit,verm,alter,weitkred,wohn,bishkred,beruf,pers,telef,gastarb,kredit".split(",")]
    return data_reconstructed_with_id


# Example usage:
# Preprocess data
input_file = '/experiments/credit/dataset/credit/train_positive.csv'
output_file = '/experiments/credit/dataset/credit/train_positive_processed_one_hot.csv'
preprocess_data(input_file, output_file)

# Inverse transform
reconstructed_file = '/experiments/credit/dataset/credit/recontructed.csv'
original_data = inverse_transform(output_file, input_file)
# Save the processed data to CSV
original_data.to_csv(reconstructed_file, index=False)
print("Inverse transformed data:")
print(original_data)
