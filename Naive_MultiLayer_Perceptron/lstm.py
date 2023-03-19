import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from sklearn.linear_model import LogisticRegression
import steptoy as toy
from tqdm import tqdm
import matplotlib as plt
import pandas as pd
import torch.optim as optim
import torch.utils.data as data
BATCH_SIZE = 1024
N_WORKERS = 0
class toydataset(Dataset):
	def __init__(self, generated):
		self.features = generated['features'][:]
		self.labels = generated['labels'][:]
		self.features = self.features.astype('float32')
		self.labels = self.labels.astype('float32')
	
	def __len__(self):
		return len(self.features)
	
	def __getitem__(self, idx):
		return [self.features[idx], self.labels[idx]]

	def get_splits(self, n_test=0.33):
		# determine sizes
		test_size = round(n_test * len(self.features))
		train_size = len(self.features) - test_size
		# calculate the split
		return random_split(self, [train_size, test_size])
	
class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
dataa = toy.datagen(5000,5, True, 0.01)
dataset = toydataset(dataa)
train, test = dataset.get_splits()
a_to_x_train_loader = DataLoader(TensorDataset(torch.tensor([[o[0]] for o in train]), torch.tensor([[o[1]] for o in train], dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
a_to_x_test_loader = DataLoader(TensorDataset(torch.tensor([[o[0]] for o in test]), torch.tensor([[o[1]] for o in test], dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in enumerate(a_to_x_train_loader):
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(train[0])
        train_rmse = np.sqrt(loss_fn(y_pred, train[1]))
        y_pred = model(test[0])
        test_rmse = np.sqrt(loss_fn(y_pred, test[1]))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
 
with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
