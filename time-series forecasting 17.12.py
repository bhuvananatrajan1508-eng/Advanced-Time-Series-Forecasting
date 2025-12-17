# Step 1: Imports & Configuration

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Step 2: Load Kaggle CSV Dataset
DATA_PATH = "/data/input/jena-climate-2009-2016/jena_climate_2009_2016.csv"

df = pd.read_csv(DATA_PATH)
df['Date Time'] = pd.to_datetime(df['Date Time'])
df.set_index('Date Time', inplace=True)

# Select multivariate features
features = [
    'T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)', 'wd (deg)'
]

data = df[features]

# Step 3: Train–Validation Split (Time-Series Safe)
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)

train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size+val_size]
test_data = data.iloc[train_size+val_size:]

# Step 4: Scaling
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# Step 5: Sequence Dataset Builder
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=48):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len, 0]  # Forecast temperature
        return x, y
		
# Step 6: DataLoaders		
SEQ_LEN = 48
BATCH_SIZE = 64

train_ds = TimeSeriesDataset(train_scaled, SEQ_LEN)
val_ds = TimeSeriesDataset(val_scaled, SEQ_LEN)
test_ds = TimeSeriesDataset(test_scaled, SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Step 7: Baseline LSTM Model (No Attention)
class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# Step 8: LSTM with Temporal Attention (KEY PART)
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        attn_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1
        )

        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        output = self.fc(context)

        return output, attn_weights

# Step 9: Training Loop (Walk-Forward Compatible)
def train_model(model, loader, optimizer, criterion):
    model.train()
    losses = []

    for x, y in loader:
        optimizer.zero_grad()
        preds = model(x)[0] if isinstance(model(x), tuple) else model(x)
        loss = criterion(preds.squeeze(), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)
	
# Step 10: Model Training
input_dim = len(features)
hidden_dim = 64

baseline = LSTMBaseline(input_dim, hidden_dim)
attention_model = AttentionLSTM(input_dim, hidden_dim)

criterion = nn.MSELoss()
optimizer_base = torch.optim.Adam(baseline.parameters(), lr=0.001)
optimizer_attn = torch.optim.Adam(attention_model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
    base_loss = train_model(baseline, train_loader, optimizer_base, criterion)
    attn_loss = train_model(attention_model, train_loader, optimizer_attn, criterion)

    print(f"Epoch {epoch+1} | Baseline: {base_loss:.4f} | Attention: {attn_loss:.4f}")

# Step 11: Evaluation Metrics
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            output = model(x)[0] if isinstance(model(x), tuple) else model(x)
            preds.extend(output.squeeze().numpy())
            targets.extend(y.numpy())

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    return rmse, mae

# Step 12: Attention Weights Visualization (Deliverable)
x_sample, _ = next(iter(test_ds))
x_sample = x_sample.unsqueeze(0)

_, attention_weights = attention_model(x_sample)

plt.figure(figsize=(10,4))
plt.plot(attention_weights.squeeze().numpy())
plt.title("Temporal Attention Weights")
plt.xlabel("Historical Time Steps")
plt.ylabel("Importance")
plt.show()
