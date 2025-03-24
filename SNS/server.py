# Import required libraries
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Set device for PyTorch (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and merge historical data from Excel files
data = pd.read_excel('2022.xlsx')
data2 = pd.read_excel('2023.xlsx')
data3 = pd.read_excel('2021.xlsx')
merged_data = pd.concat([data3, data, data2], axis=0, ignore_index=True)
data = merged_data

# Define feature columns and scale features
feature_cols = ["最高气温", "最低气温", "风力", "降雨", "是否周末"]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data[feature_cols])


# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # Process input sequence
        out = out[:, -1, :]  # Get last time step output
        out = self.fc(out)
        return out


# Model parameters
input_size = len(feature_cols)
hidden_size = 64
num_layers = 2
output_size = 1
window_size = 7

# Load pre-trained model
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model_path = 'lstm_model_weights.pth'
loaded_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()  # Set model to evaluation mode

# Flask application setup
app = Flask(__name__)


def predict(input_data):
    # Preprocess input and make prediction
    future_features_scaled = scaler.transform(input_data)
    last_window = features_scaled[-window_size:]  # Get last window from historical data
    current_window = np.vstack([last_window[1:], future_features_scaled[0]])
    input_tensor = torch.tensor(current_window[np.newaxis, :, :], dtype=torch.float32).to(device)

    with torch.no_grad():
        future_pred = loaded_model(input_tensor)
    return future_pred.item()


# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        input_data = data['input']  # Expected input format: list of feature values
        prediction = predict(input_data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})


# Start Flask server
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)