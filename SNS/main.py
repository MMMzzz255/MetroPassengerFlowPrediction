# Import required libraries
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
import torch  # Deep learning framework
import torch.nn as nn  # Neural network modules
from torch.utils.data import Dataset, DataLoader  # Data handling utilities
import torch.optim as optim  # Optimization algorithms
from sklearn.preprocessing import StandardScaler  # Feature standardization

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device：", device)

# Load and merge historical data from Excel files
data = pd.read_excel('2022.xlsx')
data2 = pd.read_excel('2023.xlsx')
data3 = pd.read_excel('2021.xlsx')
merged_data = pd.concat([data3, data, data2], axis=0, ignore_index=True)

data = merged_data  # Combined dataset

# Define feature and target columns
feature_cols = ["最高气温", "最低气温", "风力", "降雨",
                "是否周末"]  # Feature: max_temp, min_temp, wind, rain, is_weekend
target_col = "客流量"  # Target: passenger flow

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data[feature_cols])
target_values = data[target_col].values

# Create time series windows (7 days window)
window_size = 7
X, y = [], []
for i in range(len(data) - window_size):
    X.append(features_scaled[i:i + window_size])  # 7-day features window
    y.append(target_values[i + window_size])  # Next day's passenger flow
X = np.array(X)
y = np.array(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Add dimension [N, 1]


# Custom dataset class for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Split dataset into train/test sets
dataset = TimeSeriesDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders for batch training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # LSTM layer configuration
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM output sequence
        out = out[:, -1, :]  # Take last time step output
        out = self.fc(out)  # Final prediction
        return out


# Model hyperparameters
input_size = len(feature_cols)  # Number of features
hidden_size = 64  # LSTM hidden units
num_layers = 2  # Stacked LSTM layers
output_size = 1  # Regression output

# Initialize model and move to device
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    # Print epoch statistics
    train_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')

# Model evaluation on test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
test_loss /= len(test_loader.dataset)
print(f'Test Loss: {test_loss:.4f}')

# Save model weights
model_path = 'lstm_model_weights.pth'
torch.save(model.state_dict(), model_path)