
# Metro Passenger Flow Prediction System

## Overview
This project predicts metro passenger flow using an LSTM-based deep learning model. It employs a client-server architecture:
- Server: A Flask API that loads a pre-trained LSTM model and provides predictions.
- Client: A CLI application to collect weather and date inputs, then queries the server for predictions.
- Model: Trained on historical data (2021-2023) with features like temperature, rainfall, and weekend flags.

- Project Structure
├── client.py          # CLI interface for predictions
├── main.py            # Model training script
├── server.py          # Flask prediction API
├── lstm_model_weights.pth  # Trained model weights
├── 2021.xlsx         # Historical data
├── 2022.xlsx         # Historical data
└── 2023.xlsx         # Historical data

## Group Member
24045050
24050344
24133917

## Key Libraries
- PyTorch: LSTM model implementation and training
- Flask: REST API server for predictions
- Pandas/Numpy: Data processing and transformation
- Scikit-learn: Feature standardization (`StandardScaler`)
- Requests: HTTP client communication

## How to Run 

### Prerequisites
- Python 3.8+
- Required data files: `2021.xlsx`, `2022.xlsx`, `2023.xlsx` in project root

### Usage
Step 1: Train the Mode

This will:
Merge historical data from Excel files.
Train the LSTM model for 50 epochsl.
Save model weights to lstm_model_weights.pth.

Step 2: Start the Server
The Flask API will start at http://127.0.0.1:5000

Step 3: Run the Client

