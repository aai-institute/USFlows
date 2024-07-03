import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import onnxruntime as ort

if __name__ == '__main__':
    CHECK_LOF = False

    if CHECK_LOF:
        # Assuming heloc_data is your real dataset and generated_data is your synthetic dataset
        # heloc_data = pd.read_csv('heloc_dataset.csv')
        # generated_data = pd.read_csv('generated_dataset.csv')

        heloc_data = pd.read_csv('/home/mustafa/repos/VeriFlow/experiments/credit/dataset/heloc/heloc_positive_original.csv')
        generated_data = pd.read_csv('/home/mustafa/repos/VeriFlow/experiments/credit/dataset/heloc/samples.csv')
        heloc_data = heloc_data[["ExternalRiskEstimate","MSinceOldestTradeOpen", "NumTotalTrades"]]
        generated_data = generated_data[["ExternalRiskEstimate","MSinceOldestTradeOpen", "NumTotalTrades"]]
        # Combine the datasets
        #filtered_data =heloc_data.filter(items=["ExternalRiskEstimate","MSinceOldestTradeOpen"]).reset_index(drop=True)
        generated_scores = []
        # Train the LOF model on the combined dataset
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)  # You can adjust n_neighbors and contamination as needed
        lof.fit(heloc_data)
        # Calculate LOF scores (negative_outlier_factor_)
        lof_scores = lof.negative_outlier_factor_
        for index, row in generated_data.iterrows():
            generated_scores.append(lof.score_samples(row.to_frame().T))

        # Separate the scores for real and generated data
        real_scores = lof_scores
        # Analyze the results
        print(f'Real data LOF scores mean: {np.mean(real_scores)}, std: {np.std(real_scores)}')
        print(f'Generated data LOF scores mean: {np.mean(generated_scores)}, std: {np.std(generated_scores)}')
    else:
        # Model trained on positive samples maps samples to density:
        model_path = '/home/mustafa/repos/VeriFlow/experiments/credit/report/credit/0_credit_1/2024-07-0222:47:15.621328/model_heloc_backward.onnx'

        ort_sess = ort.InferenceSession(model_path)
        same_distribution_data = pd.read_csv("/home/mustafa/repos/VeriFlow/experiments/credit/dataset/heloc/heloc_positive_original.csv")
        # Data negative samples
        different_distribution_data = pd.read_csv("/home/mustafa/repos/VeriFlow/experiments/credit/dataset/heloc/heloc_negative_original.csv")

        avg_sum = 0
        for index, row in different_distribution_data.iterrows():
            outputs = ort_sess.run(None, {'x.1': row.to_numpy().astype(np.float32)})
            avg_sum = avg_sum + sum(abs(outputs[0]))
        avg_sum = avg_sum / len(different_distribution_data)
        print(f'average sum for different distribution data is {avg_sum}')

        avg_sum = 0
        for index, row in same_distribution_data.iterrows():
            outputs = ort_sess.run(None, {'x.1': row.to_numpy().astype(np.float32)})
            avg_sum = avg_sum + sum(abs(outputs[0]))
        avg_sum = avg_sum / len(same_distribution_data)
        print(f'average sum for same distribution data is {avg_sum}')

