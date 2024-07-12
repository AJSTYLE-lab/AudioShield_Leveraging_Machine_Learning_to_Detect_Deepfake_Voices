import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import requests
def download_file(url, file_path):
    with open(file_path, "wb") as f:
        response = requests.get(url)
        f.write(response.content)
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
model_url = "https://github.com/AJSTYLE-lab/AudioShield_Leveraging_Machine_Learning_to_Detect_Deepfake_Voices/raw/main/svm_model.pkl"
scaler_url = "https://github.com/AJSTYLE-lab/AudioShield_Leveraging_Machine_Learning_to_Detect_Deepfake_Voices/raw/main/scaler.pkl"
model_path = "/tmp/svm_model.pkl"
scaler_path = "/tmp/scaler.pkl"
download_file(model_url, model_path)
download_file(scaler_url, scaler_path)
svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
st.markdown("## 🔗 Links")
st.markdown("""
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](http://datascienceportfol.io/Muhammad_Ahmed_Javed)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](http://www.linkedin.com/in/%20muhammad-ahmed-javedb33900247)
[![dagshub](https://img.shields.io/badge/dagshub-000?style=for-the-badge&logo=github&logoColor=white)](https://dagshub.com/AJSTYLE-lab)
[![github](https://img.shields.io/badge/github-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AJSTYLE-lab)
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; text-decoration: underline;'>AudioShield: Leveraging Machine Learning to Detect Deepfake Voices</h1>", unsafe_allow_html=True)
st.write("**Developer Name:** Muhammad Ahmed Javed")
st.image("https://github.com/AJSTYLE-lab/AudioShield_Leveraging_Machine_Learning_to_Detect_Deepfake_Voices/raw/main/Project-image.jfif")

st.write("""
Welcome to the DeepFake Audio Detection tool. This application leverages advanced deep learning techniques to determine whether an audio file is real or fake. 
We experimented with two different deep learning models: **Convolutional Neural Networks (CNN)**, **Long Short-Term Memory networks (LSTM)**. and **Support Vector Machine (SVM Model)**
After extensive testing and evaluation, the SVM model demonstrated superior performance in detecting deepfake audio, achieving higher accuracy (81.93%) and robustness.
""")

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
