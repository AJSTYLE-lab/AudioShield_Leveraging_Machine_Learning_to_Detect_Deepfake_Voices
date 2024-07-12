import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rmse = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
    features = np.array([chroma_stft, rmse, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, *mfcc])
    return features
model_path = "https://github.com/AJSTYLE-lab/AudioShield_Leveraging_Machine_Learning_to_Detect_Deepfake_Voices/blob/main/svm_model.pkl"
scaler_path="https://github.com/AJSTYLE-lab/AudioShield_Leveraging_Machine_Learning_to_Detect_Deepfake_Voices/blob/main/scaler.pkl"
svm_model=joblib.load(model_path)
scaler=joblib.load(scaler_path)
st.title("AudioShield: Leveraging Machine Learning to Detect Deepfake Voices")
st.write("Welcome to the DeepFake Audio Detection tool. This application leverages an SVM model to determine whether an audio file is real or fake.")
uploaded_file = st.file_uploader("Choose an audio file", type=["flac"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.flac') as temp_flac_file:
        temp_flac_file.write(uploaded_file.read())
        temp_flac_path = temp_flac_file.name
    features = extract_features(temp_flac_path)    
    if features is not None:
        features_scaled = scaler.transform([features])
        features_scaled = features_scaled.reshape(1, -1)
        prediction = svm_model.predict(features_scaled)[0]
        label = "Real" if prediction == 1 else "Fake"
        st.write(f"The audio file '{uploaded_file.name}' is predicted to be: {label}")
    else:
        st.error("Feature extraction failed. Please try again with a different file.")
