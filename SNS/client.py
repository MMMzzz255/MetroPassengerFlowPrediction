# Import required libraries
import requests  # For making HTTP requests
import json  # For handling JSON data
import numpy as np  # For numerical data handling

# Define the API endpoint URL
url = 'http://127.0.0.1:5000/predict'  # Local Flask server endpoint


def get_prediction(input_data):
    """
    Send POST request to prediction API and handle response

    Parameters:
    input_data (list): Preprocessed input features for prediction
    """
    try:
        # Prepare data payload in JSON format
        data = {'input': input_data}

        # Send POST request with JSON payload
        response = requests.post(url, json=data)

        # Check if request was successful (HTTP 200 OK)
        if response.status_code == 200:
            # Parse JSON response
            result = response.json()
            print("Prediction: ", result['prediction'])
        else:
            print("Request failed with status code: ", response.status_code)
    except Exception as e:
        print("An error occurred: ", str(e))


if __name__ == '__main__':
    # User interface for input collection
    print(" Hello, I’m the Oracle. Please enter the weather data for the next day:")

    # Collect user inputs with validation
    max_temp = float(input("Max temperature (°C): "))
    min_temp = float(input("Min temperature (°C): "))
    wind_speed = float(input("Wind speed (level): "))
    rainfall = float(input("Rainfall (mm): "))
    is_weekend = int(input("Is it the weekend? (0-No, 1-Yes): "))

    # Prepare input data in correct format for the model
    # Note: Using numpy array then converting to list for JSON compatibility
    input_data = np.array([[max_temp, min_temp, wind_speed, rainfall, is_weekend]])

    # Get and display prediction
    get_prediction(input_data.tolist())  # Convert numpy array to list for JSON serialization