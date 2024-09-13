import os
import argparse
import logging
import pandas as pd
import numpy as np
import geomloss
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import fcntl
# Parse command line arguments
parser = argparse.ArgumentParser(description='Run TARNet model with specified datasets')
parser.add_argument('--source', type=str, required=True, help='/home/tu/tu_tu/tu_zxojs39/doc')
parser.add_argument('--target', type=str, required=True, help='/home/tu/tu_tu/tu_zxojs39/doc')
parser.add_argument('--task_id', type=int, required=True, help='SLURM array task ID')
args = parser.parse_args()

source_dataset_name = os.path.basename(args.source).replace('.csv', '')
target_dataset_name = os.path.basename(args.target).replace('.csv', '')

output_file = '/home/tu/tu_tu/tu_zxojs39/doc/model_outputs.csv'

if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    if ((existing_df['Source Dataset'] == source_dataset_name) & (existing_df['Target Dataset'] == target_dataset_name)).any():
        print(f"Results for source {source_dataset_name} and target {target_dataset_name} already exist. Skipping.")
        exit(0)
else:
    existing_df = pd.DataFrame()

# Load datasets
#source_df = pd.read_csv(args.source)
#target_df = pd.read_csv(args.target)


# The TARNet Aechitecture:
class CFR(nn.Module):
    def __init__(self, input_dim=5, output_dim=1, hidden_dim=10, classify=False):
        '''
        input_dim, output_dim:self-evident;
        hidden_dim: all neural networks in this model are using the same hidden layer dimension for simplicity
        '''
        super().__init__()

        self.classify = classify

        # func0 is used for predicting the outcome if the treatment is not applied (control group), and func1 for predicting the outcome if the treatment is applied (treatment group).

        encoder = [nn.Linear(input_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2),
                   nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2)]
        self.encoder = nn.Sequential(*encoder)
        func0 = [nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                 nn.Linear(hidden_dim, output_dim)]
        self.func0 = nn.Sequential(*func0)
        func1 = [nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                 nn.Linear(hidden_dim, output_dim)]
        self.func1 = nn.Sequential(*func1)

    def forward(self, X):
        # The input features (covariates) are first mapped in a hidden representation space
        # to measure the distance between Z0 and Z1
        Phi = self.encoder(X)

        # Pass the transformed features through treatments' predicting networks
        Y0 = self.func0(Phi)
        Y1 = self.func1(Phi)

        if self.classify:
            Y0 = torch.sigmoid(Y0)
            Y1 = torch.sigmoid(Y1)

        return Phi, Y0, Y1


# Define the Wassertein Distance
from geomloss import SamplesLoss


class Wassertein_Loss(nn.Module):
    def __init__(self, p=2, blur=0.01):
        super(Wassertein_Loss, self).__init__()
        self.p = p
        self.blur = blur

    def forward(self, phi1, phi0):
        samples_loss = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur, backend="tensorized")
        imbalance_loss = samples_loss(phi1, phi0)
        return imbalance_loss


from torch.nn import MSELoss as MSE

mse = MSE()
wass = Wassertein_Loss()

import pandas as pd
from io import StringIO
#df = pd.read_csv(
#    '/home/tu/tu_tu/tu_zxojs39/doc/source_data_5k.csv',


#logging.info(f"Reading dataset from {dataset_path}...")

df = pd.read_csv(args.source, header=None, skiprows=1)
column_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'A.t', 'Y0.t', 'Y1.t', 'Y.t']
df.columns = column_names
source_df = df.apply(pd.to_numeric)

def preprocess(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.iloc[:, :5])
    y_factual = df['Y.t'].values
    treatment = df['A.t'].values
    return X, treatment, y_factual


X, treatment, y_factual = preprocess(source_df)

X_train, X_test, treatment_train, treatment_test, y_factual_train, y_factual_test = train_test_split(
    X, treatment, y_factual, test_size=0.3, random_state=42)
# treatment_tensor = torch.tensor(treatment_train, dtype=torch.float32).to(device)
import torch
from torch.utils.data import TensorDataset
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(treatment_train, dtype=torch.float32),
                              torch.tensor(y_factual_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(treatment_test, dtype=torch.float32),
                             torch.tensor(y_factual_test, dtype=torch.float32))
from torch.utils.data import TensorDataset, DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

from torch.nn import MSELoss

mse_loss = MSELoss()
alpha = 10  # In the article there are four scenarios where alpha is 0, 1, 3 and 10.
# Alpha balances the contribution of the Wassertein loss and mse.

input_dim = 5
hidden_dim = 10
output_dim = 1
classify = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
source_model = CFR()
source_model.to(device)

wass = Wassertein_Loss()


def IPM(phi_control, phi_treated):
    wasserstein_distance = wass(phi_control, phi_treated)
    return wasserstein_distance


optimizer = torch.optim.Adam(source_model.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss()
wass = Wassertein_Loss()

logging.info("training is starting for source.")


def train_model(model, data_loader, optimizer, loss_function, IPM, alpha, device):
    model.train()
    total_loss = 0.0
    source_treatment_losses = []
    source_control_losses = []
    phi_source_control = []
    phi_source_treatment = []
    for batch in data_loader:
        x, a, y = [item.to(device) for item in batch]
        v = a.float().mean().item()
        weights = (a / (2 * v) + (1 - a) / (2 * (1 - v))).to(device)
        optimizer.zero_grad()
        phi_source, y0_predd, y1_predd = model(x)
        phi_source_control.append(phi_source[a == 0].detach())
        phi_source_treatment.append(phi_source[a == 1].detach())
        y0_pred = y0_predd.squeeze()
        y1_pred = y1_predd.squeeze()
        factual_loss_control = loss_function(y0_pred[a == 0], y[a == 0]) * weights[a == 0]
        factual_loss_treat = loss_function(y1_pred[a == 1], y[a == 1]) * weights[a == 1]
        source_control_losses.append(factual_loss_control.sum().item())
        source_treatment_losses.append(factual_loss_treat.sum().item())
        factual_loss = (factual_loss_control.sum() + factual_loss_treat.sum())
        source_factual_loss = factual_loss / x.size(0)
        ipm_loss = IPM(phi_source[a == 0], phi_source[a == 1])
        loss = source_factual_loss + alpha * ipm_loss
        # print("Loss requires grad:", loss.requires_grad)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    phi_source_control = torch.cat(phi_source_control, dim=0)
    phi_source_treatment = torch.cat(phi_source_treatment, dim=0)
    average_loss = total_loss / len(train_loader)
    # phi_sources = torch.cat(phi_sources, dim=0)
    return average_loss, source_treatment_losses, source_control_losses, phi_source_treatment, phi_source_control


num_epochs = 20
for epoch in range(num_epochs):
    train_loss, source_treatment_losses, source_control_losses, phi_source_treatment, phi_source_control = train_model(
        source_model, train_loader, optimizer, loss_function, IPM, alpha, device)
    print(f"Epoch {epoch}: Train Loss = {train_loss}")
    print(f"Average Treatment Loss: {sum(source_treatment_losses) / len(source_treatment_losses)}")
logging.info("traning finished for sourcefully.")
logging.info("CSV file read sucy.")

model_path = f'/home/tu/tu_tu/tu_zxojs39/doc/cfr_fine_tuned_model_{args.task_id}_version8.pth'
torch.save(source_model.state_dict(), model_path)


import numpy as np
import torch
from torch.utils.data import DataLoader
from geomloss import SamplesLoss

source_model.load_state_dict(torch.load(model_path, map_location=device))
source_model.to(device)


def test_model_with_ipm(model, test_loader, loss_function, IPM, device, alpha):
    model.eval()
    total_loss = 0.0
    count = 0
    ite_wotransfer_random_5k = []

    with torch.no_grad():
        for X_batch, treatment_batch, y_batch in test_loader:
            X_batch, treatment_batch, y_batch = X_batch.to(device), treatment_batch.to(device), y_batch.to(device)
            phi, Y0_pred, Y1_pred = model(X_batch)
            Y0_pred = Y0_pred.squeeze()
            Y1_pred = Y1_pred.squeeze()
            v = treatment_batch.float().mean().item()
            weights = (treatment_batch / (2 * v) + (1 - treatment_batch) / (2 * (1 - v))).to(device)
            mse_loss_control = loss_function(Y0_pred[treatment_batch == 0], y_batch[treatment_batch == 0]) * weights[
                treatment_batch == 0]
            mse_loss_treat = loss_function(Y1_pred[treatment_batch == 1], y_batch[treatment_batch == 1]) * weights[
                treatment_batch == 1]
            mse_loss = (mse_loss_control.sum() + mse_loss_treat.sum()) / X_batch.size(0)
            ipm_loss = IPM(phi[treatment_batch == 0], phi[treatment_batch == 1])
            combined_loss = mse_loss + alpha * ipm_loss
            total_loss += combined_loss.item() * X_batch.size(0)
            count += X_batch.size(0)
            ITE_batch = Y1_pred - Y0_pred
            ite_wotransfer_random_5k.extend(ITE_batch.tolist())
    average_loss = total_loss / count
    return average_loss, ite_wotransfer_random_5k


import matplotlib.pyplot as plt

average_loss, ite_wotransfer_random_5k = test_model_with_ipm(source_model, test_loader, mse_loss, wass, device, alpha)

ite_array = np.array(ite_wotransfer_random_5k)
print(f"Mean ITE: {ite_array.mean()}")
print(f"Median ITE: {np.median(ite_array)}")
print(f"Standard Deviation of ITE: {ite_array.std()}")

# Plotting the histogram of ITE
plt.figure(figsize=(5, 5))
plt.hist(ite_array, bins=50, color='blue', edgecolor='black')
plt.xlabel('ITE')
plt.ylabel('Frequency')
plt.title('Histogram of Individual Treatment Effects')
plt.grid(True)
plt.show()

# transfering
target_df = pd.read_csv(args.target)
#target_df = pd.read_csv(dataset_path_t) 
# column_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'A.t', 'Y0.t', 'Y1.t', 'Y.t']
# df.columns = column_names
# target_df = df.apply(pd.to_numeric)
X_target, treatment_target, y_factual_target = preprocess(target_df)

X_train_target, X_test_target, treatment_train_target, treatment_test_target, y_factual_train_target, y_factual_test_target = train_test_split(
    X_target, treatment_target, y_factual_target, test_size=0.3, random_state=42
)

train_dataset_target = TensorDataset(torch.tensor(X_train_target, dtype=torch.float32),
                                     torch.tensor(treatment_train_target, dtype=torch.float32),
                                     torch.tensor(y_factual_train_target, dtype=torch.float32))
test_dataset_target = TensorDataset(torch.tensor(X_test_target, dtype=torch.float32),
                                    torch.tensor(treatment_test_target, dtype=torch.float32),
                                    torch.tensor(y_factual_test_target, dtype=torch.float32))

train_loader_target = DataLoader(train_dataset_target, batch_size=32, shuffle=True)
test_loader_target = DataLoader(test_dataset_target, batch_size=32, shuffle=False)

all_target = TensorDataset(torch.tensor(X_target, dtype=torch.float32),
                           torch.tensor(treatment_target, dtype=torch.float32),
                           torch.tensor(y_factual_target, dtype=torch.float32))

loader_all_target = DataLoader(all_target, batch_size=32, shuffle=True)


# I have defined gamma star but I did not use it, because this is not part of the convergence graph of the loss function, this is constant.
def gamma_star(source_task, target_task, IPM, device):
    source_T = torch.Tensor(source_task['t']).to(device)
    target_T = torch.Tensor(target_task['t']).to(device)
    source_input = torch.Tensor(source_task['x']).to(device)
    target_input = torch.Tensor(target_task['x']).to(device)
    return np.mean(IPM(source_input[target_T == 0], target_input[target_T == 0]) + \
                   IPM(source_input[target_T == 1], target_input[target_T == 1]))


# The following distance function which is defined in this way in Lemma 4.4 and Theorem 4.5.
def IPM_distance(phi_source_treatment, phi_source_control, phi_target_treatment, phi_target_control, IPM, device):
    # batch_size = phi.size(0)
    # source_T = torch.Tensor(source_task['t']).to(device)
    # target_T = torch.Tensor(target_task['t']).to(device)
    # source_input = torch.Tensor(source_task['x']).to(device)
    # target_input = torch.Tensor(target_task['x']).to(device)
    # target_input.to(device)
    # phi_target,y0_predd, y1_predd= model(target_input)
    # phi_source,_, _ = model(source_input)
    # print(phi.shape)
    return IPM(phi_target_treatment, phi_source_treatment) + \
        IPM(phi_target_control, phi_source_control) + \
        IPM(phi_target_control, phi_target_treatment)


# Authors put the distance function in the following way, however they did not put phi as they put into the paper.
# When they did not put phi, that distance will be constant value rather than dynamic value.
# def IPM_distance(source_model,source_task, target_task,IPM, device):
#   source_T = torch.Tensor(source_task['t']).to(device)
#   target_T = torch.Tensor(target_task['t']).to(device)
#   source_input = torch.Tensor(source_task['x']).to(device)
#   target_input = torch.Tensor(target_task['x']).to(device)
#   # target_input.to(device)
#   phi,y0,y1 = source_model(target_input)
#   return IPM(source_input[source_T == 0],target_input[target_T==0]) + \
#   IPM(source_input[source_T == 1],target_input[target_T==1]) + \
#   IPM(phi[target_T==0],phi[target_T==1])


model = CFR().to(device)
model.load_state_dict(torch.load(model_path))
model.to(device)

for param in source_model.parameters():
    param.requires_grad = False
for param in list(source_model.func0.children())[-1].parameters():
    param.requires_grad = True
for param in list(source_model.func1.children())[-1].parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, source_model.parameters()), lr=0.001)

for name, param in source_model.named_parameters():
    print(f"{name} is {'trainable' if param.requires_grad else 'frozen'}")

X_tensor = torch.tensor(X_train).float().to(device)
treatment_tensor = torch.tensor(treatment_train).float().to(device)
X_target_tensor = torch.tensor(X_train_target).float().to(device)
treatment_target_tensor = torch.tensor(treatment_train_target).float().to(device)

source_task = {'t': treatment_tensor, 'x': X_tensor}
target_task = {'x': X_target_tensor, 't': treatment_target_tensor}


def fine_tune_model(model, optimizer, train_loader_target, IPM, alpha, device, epochs, phi_source_treatment,
                    phi_source_control):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        phi_target_control = []
        phi_target_treatment = []
        for batch in train_loader_target:
            x_, a, y = [item.to(device) for item in batch]
            optimizer.zero_grad()
            phi_target, y0_predd, y1_predd = model(x_)

            phi_target_control.append(phi_target[a == 0])
            phi_target_treatment.append(phi_target[a == 1])
            optimizer.step()

        phi_target_control_tensor = torch.cat(phi_target_control, dim=0)
        phi_target_treatment_tensor = torch.cat(phi_target_treatment, dim=0)
        # total_treatment_loss = sum(source_treatment_losses)
        # total_control_loss = sum(source_control_losses)

        loss_dist = IPM_distance(phi_source_treatment, phi_source_control, phi_target_treatment_tensor,
                                 phi_target_control_tensor, IPM, device)
        loss = loss_dist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader_target)}")

    return model, phi_target_treatment_tensor, phi_target_control_tensor


epochs = 20

fine_tuned_model, phi_target_treatment_tensor, phi_target_control_tensor = fine_tune_model(
    model,
    optimizer,
    train_loader_target,
    IPM,
    alpha,
    device,
    epochs,
    phi_source_treatment,
    phi_source_control
)

fine_tuned_model_path = f'/home/tu/tu_tu/tu_zxojs39/doc/cfr_fine_tuned_model_{args.task_id}_version6.pth'
torch.save(fine_tuned_model.state_dict(), fine_tuned_model_path)

import numpy as np
import torch
from torch.utils.data import DataLoader
from geomloss import SamplesLoss


fine_tuned_model.load_state_dict(torch.load(fine_tuned_model_path))
fine_tuned_model.to(device)

import torch
from torch.utils.data import DataLoader


def test_model_with_ipm_fine_tune(model, test_loader_target, IPM_distance, alpha, device, phi_source_treatment,
                                  phi_source_control, phi_target_treatment_tensor, phi_target_control_tensor):
    model.eval()
    total_loss = 0.0
    count = 0
    ite_wtransfer_random_5k = []

    with torch.no_grad():
        for X_batch_t, treatment_batch_t, y_batch_t in test_loader_target:
            X_batch_t, treatment_batch_t, y_batch_t = X_batch_t.to(device), treatment_batch_t.to(device), y_batch_t.to(
                device)
            phi, y0_pred, y1_pred = model(X_batch_t)
            y0_pred = y0_pred.squeeze()
            y1_pred = y1_pred.squeeze()

            factual_loss = mse_loss(y0_pred[treatment_batch_t == 0], y_batch_t[treatment_batch_t == 0]) + \
                           mse_loss(y1_pred[treatment_batch_t == 1], y_batch_t[treatment_batch_t == 1])

            phi_target_treatment = phi[treatment_batch_t == 1]
            phi_target_control = phi[treatment_batch_t == 0]

            # ipm_loss = IPM_distance(phi_source_treatment, phi_source_control, phi_target_treatment_tensor, phi_target_control_tensor, IPM, device)

            total_loss += factual_loss.item() * X_batch_t.size(0)
            count += X_batch_t.size(0)
            ITE_batch = y1_pred - y0_pred
            ite_wtransfer_random_5k.extend(ITE_batch.tolist())

    average_loss = total_loss / count
    return average_loss, ite_wtransfer_random_5k


average_loss, ite_wtransfer_random_5k = test_model_with_ipm_fine_tune(fine_tuned_model, test_loader_target,
                                                                      IPM_distance, alpha, device, phi_source_treatment,
                                                                      phi_source_control, phi_target_treatment_tensor,
                                                                      phi_target_control_tensor)

ite_array = np.array(ite_wtransfer_random_5k)
mean_ite = ite_array.mean()
median_ite = np.median(ite_array)
std_ite = ite_array.std()

print(f"Mean ITE: {mean_ite}")
print(f"Median ITE: {median_ite}")
print(f"Standard Deviation of ITE: {std_ite}")

plt.figure(figsize=(5, 5))
plt.hist(ite_array, bins=50, color='blue', edgecolor='black')
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
    'Source Dataset': [source_dataset_name],
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
