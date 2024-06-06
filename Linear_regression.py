import os
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import fcntl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run simple model for ITE estimation with specified datasets')
parser.add_argument('--target', type=str, required=True, help='Path to the target dataset')
parser.add_argument('--task_id', type=int, required=True, help='SLURM array task ID')
args = parser.parse_args()

target_dataset_name = os.path.basename(args.target).replace('.csv', '')

output_file = '/home/tu/tu_tu/tu_zxojs39/doc/model_outputs.csv'

if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    if (existing_df['Target Dataset'] == target_dataset_name).any():
        print(f"Results for target {target_dataset_name} already exist. Skipping.")
        exit(0)
else:
    existing_df = pd.DataFrame()

# Data Loading and Preprocessing
target_df = pd.read_csv(args.target)

def preprocess(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.iloc[:, :5])
    y_factual = df['Y.t'].values
    treatment = df['A.t'].values
    return X, treatment, y_factual

X, treatment, y_factual = preprocess(target_df)

X_train, X_test, treatment_train, treatment_test, y_factual_train, y_factual_test = train_test_split(
    X, treatment, y_factual, test_size=0.3, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(treatment_train, dtype=torch.float32),
                              torch.tensor(y_factual_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(treatment_test, dtype=torch.float32),
                             torch.tensor(y_factual_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple Linear Regression Model for ITE estimation
class SimpleITEModel:
    def __init__(self):
        self.model_control = LinearRegression()
        self.model_treated = LinearRegression()

    def train(self, X, treatment, y):
        # Separate the data into control and treated groups
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        X_treated = X[treatment == 1]
        y_treated = y[treatment == 1]

        # Fit the linear regression models
        self.model_control.fit(X_control, y_control)
        self.model_treated.fit(X_treated, y_treated)

    def predict_ite(self, X):
        y0_pred = self.model_control.predict(X)
        y1_pred = self.model_treated.predict(X)
        ite_pred = y1_pred - y0_pred
        return y0_pred, y1_pred, ite_pred

# Training the simple ITE model
simple_model = SimpleITEModel()

simple_model.train(X_train, treatment_train, y_factual_train)


# Predicting ITE on the test set

y0_pred, y1_pred, ite_pred = simple_model.predict_ite(X_test)

# Calculating metrics
mean_ite = np.mean(ite_pred)
median_ite = np.median(ite_pred)
std_ite = np.std(ite_pred)

print(f"Mean ITE: {mean_ite}")
print(f"Median ITE: {median_ite}")
print(f"Standard Deviation of ITE: {std_ite}")

plt.figure(figsize=(5, 5))
plt.hist(ite_pred, bins=50, color='blue', edgecolor='black')
plt.xlabel('ITE')
plt.ylabel('Frequency')
plt.title('Histogram of Individual Treatment Effects')
plt.grid(True)
plt.show()

results = {
    'Task ID': [args.task_id],
    'Mean ITE': [mean_ite],
    'Median ITE': [median_ite],
    'Std ITE': [std_ite],
    'Target Dataset': [target_dataset_name]
}

df_results = pd.DataFrame(results)

with open(output_file, 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    if os.stat(output_file).st_size == 0:
        df_results.to_csv(f, index=False)
    else:
        df_results.to_csv(f, header=False, index=False)
    fcntl.flock(f, fcntl.LOCK_UN)

print("Script completed.")

