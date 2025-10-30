from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys
import pickle 

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration (Set for Local VS Code Deployment) ---
# NOTE: Paths are relative to the root directory where you run 'python app.py'
MODEL_PATH = 'model/saved_model_autoencoder.pkl'
SCALER_PATH = 'model/scaler.pkl'
DATA_PATH = 'data/iiot_edge_computing_dataset.csv' 

# Set the anomaly threshold based on your model's evaluation
ANOMALY_THRESHOLD = 0.25 
SEQ_LEN = 24
FEATURE_NAMES = ['Temperature', 'Pressure', 'Vibration', 'Network_Latency', 'Edge_Processing_Time', 'Fuzzy_PID_Output', 'Predicted_Failure']

autoencoder = None
scaler = None

# --- Model Loading Function ---
def load_assets():
    global autoencoder, scaler
    try:
        # Load the Keras model (using compile=False for compatibility)
        autoencoder = pickle.load(MODEL_PATH) 
        # Load the fitted MinMaxScaler object
        scaler = joblib.load(SCALER_PATH)
        print(" Model and Scaler loaded successfully.")
        return True
    
    except Exception as e:
        print(f" FATAL ERROR: Model or Scaler failed to load. Check paths: {MODEL_PATH} and {SCALER_PATH}")
        print(f"Error details: {e}")
        
        # Attempting graceful failure if model/scaler are missing
        try:
            df = pd.read_csv(DATA_PATH).select_dtypes(include=[float, int]).dropna()
            scaler = MinMaxScaler()
            scaler.fit(df[FEATURE_NAMES].values)
            global_status = "Model Load Failed"
            print(" Running in debug mode with DUMMY SCALER. Inference is DISABLED.")
            autoencoder = None
            return False
        except:
            print(f" Could not load dummy scaler data from {DATA_PATH}. Application cannot run.")
            sys.exit(1)

# Load assets when the application starts
load_assets()


# --- Preprocessing Logic ---
def sliding_window(series, seq_len=SEQ_LEN):
    if series.shape[0] < seq_len:
        raise ValueError(f"Input sequence must contain exactly {seq_len} time steps.")
    X = series[-seq_len:]
    return np.expand_dims(X, axis=0)

def preprocess_data(data_list):
    input_df = pd.DataFrame(data_list, columns=FEATURE_NAMES)
    scaled_data = scaler.transform(input_df.values)
    return sliding_window(scaled_data, seq_len=SEQ_LEN)

# --- Web Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    model_status = "Model Loaded & Ready" if autoencoder else "Model NOT Loaded (Inference Disabled)"
    return render_template('index.html', model_status=model_status, threshold=ANOMALY_THRESHOLD)

@app.route('/detect', methods=['POST'])
def detect():
    """Handles the form submission and returns anomaly detection results as JSON."""
    if not autoencoder:
        return jsonify({'error': 'Anomaly detection model is not loaded. Cannot run inference.'}), 500

    try:
        data_string = request.form.get('timeseries_data')
        
        data_list = []
        for row in data_string.strip().split('\n'):
            if row.strip():
                data_list.append(list(map(float, row.strip().split(','))))

        if len(data_list) != SEQ_LEN:
            raise ValueError(f"Expected exactly {SEQ_LEN} time steps (rows), but received {len(data_list)}.")
        
        if len(data_list[0]) != len(FEATURE_NAMES):
            raise ValueError(f"Expected exactly {len(FEATURE_NAMES)} features (columns), but received {len(data_list[0])}.")

        # Preprocess and prepare for model
        preprocessed_data = preprocess_data(data_list)

        # Get reconstruction error
        reconstructions = autoencoder.predict(preprocessed_data, verbose=0) 
        reconstruction_error = np.mean(np.square(reconstructions - preprocessed_data), axis=(1, 2))[0]
        
        # Check for anomaly
        is_anomaly = reconstruction_error > ANOMALY_THRESHOLD

        return jsonify({
            'reconstruction_error': float(reconstruction_error),
            'threshold': ANOMALY_THRESHOLD,
            'is_anomaly': is_anomaly
        })

    except ValueError as ve:
        return jsonify({'error': f"Input Error: {str(ve)}"}), 400
    except Exception as e:
        print(f"An unexpected error occurred during detection: {e}")
        return jsonify({'error': 'An unexpected server error occurred.'}), 500

if __name__ == '__main__':
    # Running on localhost on port 5000 (default for Flask)
    app.run(host='0.0.0.0', port=81, debug=True)
