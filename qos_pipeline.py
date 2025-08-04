import os
import sys
import warnings
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
from sklearn.ensemble import IsolationForest
import joblib
import pyshark
import logging
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =======================
# Config
# =======================
CLASSIFIER_PATH = "traffic_classifier.pkl"
FEATURE_SCHEMA_PATH = "feature_columns.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
OUTPUT_CSV = "qos_results.csv"

# Configurable Prophet threshold (default 30 bins)
MIN_BINS_FOR_PROPHET = int(os.environ.get("MIN_BINS_FOR_PROPHET", 30))

# =======================
# Load Models
# =======================
try:
    clf = joblib.load(CLASSIFIER_PATH)
    feature_columns = joblib.load(FEATURE_SCHEMA_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    logging.info("✅ Models and encoders loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load models: {e}")
    sys.exit(1)

if not hasattr(clf, "predict"):
    logging.error("❌ Invalid classifier in traffic_classifier.pkl")
    sys.exit(1)

# =======================
# PCAP Parsing with Time Binning
# =======================
def extract_features_from_pcap(pcap_file, bin_size=1):
    logging.info(f"Extracting features from {pcap_file} with {bin_size}s bins...")

    cap = pyshark.FileCapture(pcap_file, only_summaries=True)
    packets = []
    for pkt in cap:
        try:
            packets.append({
                "timestamp": float(pkt.time),
                "length": int(pkt.length),
            })
        except:
            continue
    cap.close()

    df = pd.DataFrame(packets)
    if df.empty:
        raise ValueError("No packets extracted from PCAP.")

    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    df['time_bin'] = (df['timestamp'] // bin_size).astype(int)

    grouped = df.groupby('time_bin').agg(
        forward_pl_mean=('length', 'mean'),
        forward_pl_var=('length', 'var'),
        forward_piat_mean=('timestamp', lambda x: x.diff().mean()),
        forward_pps_mean=('length', 'count'),
        forward_bps_mean=('length', 'sum')
    ).fillna(0).reset_index()

    return grouped

# =======================
# Feature Alignment
# =======================
def align_features(df, feature_columns):
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df.fillna(0)

# =======================
# Classification per Time Bin
# =======================
def classify_traffic(df):
    preds = clf.predict(df)
    decoded = le.inverse_transform(preds)
    return decoded

# =======================
# Save Forecast Plot
# =======================
def save_forecast_plot(ts, forecast, output_file="forecast.png"):
    plt.figure(figsize=(10,5))
    plt.plot(ts['ds'], ts['y'], label="Observed Load")
    plt.plot(forecast['ds'], forecast['yhat'], label="Predicted Load", linestyle="--")
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Bandwidth (BPS)")
    plt.title("Congestion Forecast")
    plt.legend()
    plt.savefig(output_file)
    plt.close()
    logging.info(f"📈 Forecast plot saved to {output_file}")

# =======================
# Congestion Prediction
# =======================
def predict_congestion(time_bins, loads):
    ts = pd.DataFrame({
        "ds": pd.to_datetime(time_bins, unit="s"),
        "y": loads
    })

    # Case 1: Not enough bins → fallback to mean + simple plot
    if len(time_bins) < MIN_BINS_FOR_PROPHET:
        logging.warning(f"Not enough data for Prophet (<{MIN_BINS_FOR_PROPHET} bins). Falling back to mean load.")
        
        plt.figure(figsize=(10,5))
        plt.plot(ts['ds'], ts['y'], label="Observed Load", color="blue")
        plt.xlabel("Time")
        plt.ylabel("Bandwidth (BPS)")
        plt.title("Observed Congestion (Fallback)")
        plt.legend()
        fallback_file = f"{OUTPUT_CSV.replace('.csv', '')}_forecast.png"
        plt.savefig(fallback_file)
        plt.close()
        
        logging.info(f"📉 Fallback plot saved to {fallback_file}")
        return {"status": "fallback", "predicted_congestion": float(np.mean(loads))}
    
    # Case 2: Enough bins → run Prophet forecast
    model = Prophet(daily_seasonality=False)
    model.fit(ts)
    future = model.make_future_dataframe(periods=60, freq="S")
    forecast = model.predict(future)

    save_forecast_plot(ts, forecast, f"{OUTPUT_CSV.replace('.csv', '')}_forecast.png")

    return {"status": "prophet", "predicted_congestion": forecast['yhat'].iloc[-60:].mean()}


# =======================
# OOD Detection
# =======================
def check_ood(df):
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(df)
    anomalies = df[preds == -1]
    return anomalies

# =======================
# Scaling Suggestion
# =======================
def suggest_scaling(streaming_ratio, congestion_level):
    if congestion_level > 100:
        return "Scale UP VMs (+3)"
    elif congestion_level > 50:
        return "Scale UP VMs (+2)"
    elif congestion_level > 20:
        return "Scale UP VMs (+1)"
    elif streaming_ratio < 0.3 and congestion_level < 0.3:
        return "Scale DOWN VMs (-1)"
    else:
        return "Maintain current VM allocation"

# =======================
# Main Pipeline
# =======================
def run_pipeline(pcap_file):
    # Extract
    features = extract_features_from_pcap(pcap_file, bin_size=1)
    aligned = align_features(features.drop(columns=['time_bin']), feature_columns)
    
    # Classify per bin
    preds = classify_traffic(aligned)
    features['traffic_class'] = preds

    streaming_ratio = (features['traffic_class'] == "Streaming").mean()
    
    # Congestion (using BPS per bin)
    congestion = predict_congestion(features['time_bin'], features['forward_bps_mean'])
    congestion_level = congestion["predicted_congestion"]

    # OOD check
    anomalies = check_ood(aligned)

    # Scaling Suggestion
    scaling = suggest_scaling(streaming_ratio, congestion_level)

    # Save Results
    results = pd.DataFrame({
        "pcap_file": [pcap_file],
        "streaming_ratio": [streaming_ratio],
        "congestion_level": [congestion_level],
        "scaling_suggestion": [scaling],
        "ood_count": [len(anomalies)]
    })
    results.to_csv(OUTPUT_CSV, index=False)

    logging.info("QoS Pipeline Completed.")
    print(results)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: MIN_BINS_FOR_PROPHET=10 python qos_pipeline.py <pcap_file>")
        sys.exit(1)
    run_pipeline(sys.argv[1])
