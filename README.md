import os
import numpy as np
import pandas as pd
import pywt
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from scipy.spatial.distance import euclidean

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# File Paths
scaler_path = r"C:\Users\Lingesh Royan\Desktop\doc\eeg_scaler.pkl"
cnn_model_path = r"C:\Users\Lingesh Royan\Desktop\doc\eeg_cnn_model.h5"
rnn_model_path = r"C:\Users\Lingesh Royan\Desktop\doc\eeg_rnn_model.h5"
csv_data_path = r"C:\Users\Lingesh Royan\Desktop\eeg excel\eeg_data_with_emotions.csv"

# Load the scaler
scaler = joblib.load(scaler_path)

# Load Deep Learning Models
cnn_model = load_model(cnn_model_path)
rnn_model = load_model(rnn_model_path)

# Function to extract DWT features
def extract_dwt_features(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=min(3, pywt.dwt_max_level(len(signal), pywt.Wavelet('db4').dec_len)))
    features = [np.mean(c) for c in coeffs]
    return np.array(features)

# Function to preprocess EEG data
def preprocess_eeg_data(eeg_data):
    dwt_features = extract_dwt_features(eeg_data)
    return scaler.transform([dwt_features])

# EEG Channel Names
channel_names = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6", "O1", "O2", "P7", "P8", "T7", "T8"]
print("🔹 Enter EEG data for the following channels:")
eeg_input = [float(input(f"{channel}: ")) for channel in channel_names]

# Load CSV Data
try:
    df = pd.read_csv(csv_data_path)
    df_channels = df[channel_names]
except Exception as e:
    print(f"❌ Error loading CSV file: {e}")
    exit()

# Check for closest matching EEG rows (Euclidean distance)
user_row = np.array(eeg_input)
df["Distance"] = df_channels.apply(lambda row: euclidean(row.values, user_row), axis=1)
top_matches = df.nsmallest(5, "Distance")

print("\n🔍 Top 5 Closest EEG Matches:")
print(top_matches[["Distance", "Label"]])

# Show most common emotion among matches
most_common = top_matches["Label"].mode()[0]
print(f"✅ Most common emotion from top matches: {most_common}")

# ✅ Evaluate CNN and RNN Model Accuracy
print("\n📊 Evaluating model accuracy from dataset...")

# Filter valid rows
df = df.dropna(subset=channel_names + ["Label"])
X = df[channel_names].values
y = df["Label"].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Extract DWT features
X_features = np.array([extract_dwt_features(row) for row in X])
X_scaled = scaler.transform(X_features)

# ---- CNN Accuracy ----
cnn_preds_all = cnn_model.predict(X_scaled, verbose=0)
cnn_labels = np.argmax(cnn_preds_all, axis=1)
cnn_acc = accuracy_score(y_encoded, cnn_labels)

# ---- RNN Accuracy ----
X_rnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
rnn_preds_all = rnn_model.predict(X_rnn, verbose=0)
rnn_labels = np.argmax(rnn_preds_all, axis=1)
rnn_acc = accuracy_score(y_encoded, rnn_labels)

print(f"✅ CNN Model Accuracy: {cnn_acc * 100:.2f}%")
print(f"✅ RNN Model Accuracy: {rnn_acc * 100:.2f}%")

# Preprocess the input
eeg_processed = preprocess_eeg_data(user_row)

# Reshape for RNN model
rnn_input = eeg_processed.reshape((eeg_processed.shape[0], eeg_processed.shape[1], 1))

# Predict with models
cnn_probs = cnn_model.predict(eeg_processed, verbose=0)[0]
cnn_pred = np.argmax(cnn_probs)

rnn_probs = rnn_model.predict(rnn_input, verbose=0)[0]
rnn_pred = np.argmax(rnn_probs)

labels = ["Happy", "Sad", "Angry", "Relaxed", "Fear", "Excited", "Neutral"]

# Show Top-N Predictions
top_n = 3
print("\n🎯 Top CNN Predictions:")
top_cnn = np.argsort(cnn_probs)[-top_n:][::-1]
for i in top_cnn:
    print(f"   {labels[i]}: {cnn_probs[i]:.2f}")

print("\n🎯 Top RNN Predictions:")
top_rnn = np.argsort(rnn_probs)[-top_n:][::-1]
for i in top_rnn:
    print(f"   {labels[i]}: {rnn_probs[i]:.2f}")

# Final predicted emotion
print("\n🧠 Final CNN Predicted Emotion:", labels[cnn_pred])
print("🧠 Final RNN Predicted Emotion:", labels[rnn_pred])

# Plot CNN vs RNN Confidence Scores
plt.figure(figsize=(10, 6))
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, cnn_probs, width, label='CNN', color='skyblue')
plt.bar(x + width/2, rnn_probs, width, label='RNN', color='lightgreen')
plt.xticks(x, labels)
plt.ylabel("Confidence Score")
plt.title("CNN vs RNN Emotion Prediction Confidence")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
