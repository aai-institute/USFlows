import onnxruntime as ort
import numpy as np
import pickle
import csv
import pandas as pd

encoded_column_names = ("ExternalRiskEstimate,MSinceOldestTradeOpen,MSinceMostRecentTradeOpen,AverageMInFile,"
                        "NumSatisfactoryTrades,NumTrades60Ever2DerogPubRec,NumTrades90Ever2DerogPubRec,"
                        "PercentTradesNeverDelq,MSinceMostRecentDelq,NumTotalTrades,NumTradesOpeninLast12M,"
                        "PercentInstallTrades,MSinceMostRecentInqexcl7days,NumInqLast6M,NumInqLast6Mexcl7days,"
                        "NetFractionRevolvingBurden,NetFractionInstallBurden,NumRevolvingTradesWBalance,"
                        "NumInstallTradesWBalance,NumBank2NatlTradesWHighUtilization,PercentTradesWBalance,"
                        "RiskPerformance").split(",")


def samples_encoded_to_csv(samples, csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(encoded_column_names)
        for one_hot_sample in samples:
            writer.writerow(one_hot_sample)


if __name__ == '__main__':
    model_path = '/scripts/reports/model_heloc_forward.onnx'
    ort_sess = ort.InferenceSession(model_path)
    # Load the scalers
    with open('../standard_scaler_positive.pkl', 'rb') as f:
        standard_scaler = pickle.load(f)

    samples = []
    for i in range(100):
        x = np.random.uniform(low=-0.001, high=0.001, size=(22,)).astype(np.float32)
        outputs = ort_sess.run(None, {'x.1': x})
        original_data = standard_scaler.inverse_transform(outputs[0].reshape(1, -1))
        samples.append([i for i in original_data[0]])
    raw_outputs_path = '/experiments/credit/dataset/heloc/samples.csv'
    samples_encoded_to_csv(samples, raw_outputs_path)



