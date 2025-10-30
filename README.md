# Synthetic Time-Series Generation & Anomaly Detection using TimeGAN

Major Project @GNCIPL This project uses an IIoT Edge Computing dataset to:  Learn normal machine-sensor behavior from time-series data. Generate realistic synthetic “normal” sequences using TimeGAN. Train an LSTM Autoencoder anomaly detector to identify failures.

## Objective

This project leverages an IIoT Edge Computing dataset to:

- Learn normal machine-sensor behavior from time-series data
- Generate realistic synthetic “normal” sequences using *TimeGAN*
- Train an *LSTM Autoencoder* anomaly detector to identify failures

## Dataset

- Source: IIoT Edge Computing dataset (CSV format)
- Contains timestamped sensor readings: Temperature, Pressure, Vibration, Network Latency, Edge Processing Time, Fuzzy PID Output, and failure labels


## Project Structure

- finalproject.ipynb: Main Jupyter notebook with all code, analysis, and results
- README.md: Project overview and instructions

## Workflow

1. *Data Loading & Cleaning*: Load the dataset, handle missing values, convert timestamps, and preprocess features
2. *Exploratory Data Analysis (EDA)*: Visualize feature distributions, correlations, and time-series trends
3. *Windowing*: Transform data into sequences for time-series modeling
4. *Synthetic Data Generation (TimeGAN)*: Train a simplified TimeGAN to generate realistic normal sequences
5. *Anomaly Detection (LSTM Autoencoder)*: Train an autoencoder to detect failures based on reconstruction error
6. *Evaluation*: Assess model performance using ROC-AUC, precision, recall, F1-score, and confusion matrix
7. *Visualization*: Compare real vs synthetic sequences, plot reconstruction errors, and highlight anomalies
8. Deployment: develope a webpage using flask with template and backend file(app.py)

## Key Results

- TimeGAN successfully generates realistic synthetic time-series data
- LSTM Autoencoder achieves high accuracy (ROC AUC ~0.99) in detecting anomalies
- Synthetic data augmentation improves recall and reduces false negatives

## Extensions

- Integrate real-time anomaly detection dashboard
- Experiment with advanced GAN architectures for time-series
- Deploy models for live IIoT monitoring
- Add explainable AI for model transparency

## Tools & Libraries

- Python 3.x
- pandas, numpy (data handling)
- matplotlib, seaborn (visualization)
- scikit-learn (preprocessing, metrics)
- tensorflow, keras (TimeGAN, LSTM Autoencoder)

## Team Members

- Devasish Sai Pothumudi 
- Narasingu Sai Suchendar
- Saumya Singh 
- Kajal Tiwari 
- PranayNigam
- Sangeeth M
- Sharon jacquiline S

## Deployment output:
<img width="1884" height="862" alt="Screenshot 2025-10-29 213622" src="https://github.com/user-attachments/assets/b0d8242b-6138-4c59-90cf-ebcd12e8e067" />
