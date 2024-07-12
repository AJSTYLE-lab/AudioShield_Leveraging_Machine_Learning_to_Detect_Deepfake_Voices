import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import logging
import tempfile

def extract_features(file_path, max_pad_len=26):
    try:
        audio_data, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None
    try:
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sr).T, axis=0)
        rmse = np.mean(librosa.feature.rms(y=audio_data).T, axis=0)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr).T, axis=0)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr).T, axis=0)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr).T, axis=0)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_data).T, axis=0)
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20).T, axis=0)
        features = np.hstack([chroma_stft, rmse, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, mfccs])
        pad_width = max_pad_len - features.shape[0]
        if pad_width > 0:
            features = np.pad(features, (0, pad_width), mode='constant')
        else:
            features = features[:max_pad_len]
        return features.reshape(1, max_pad_len, 1)
    except Exception as e:
        st.error(f"Error extracting features from audio file: {e}")
        return None

# Load the model
model_url = "https://github.com/AJSTYLE-lab/AudioShield_Leveraging_Machine_Learning_to_Detect_Deepfake_Voices/raw/main/Deep-Fake-Audio-Detection-Model.h5"
model_path = tf.keras.utils.get_file('Deep-Fake-Audio-Detection-Model.h5', model_url)
model = tf.keras.models.load_model(model_path)
print(f"Model url:{model_url}")
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

if uploaded_file and model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.flac') as temp_flac_file:
        temp_flac_file.write(uploaded_file.read())
        temp_flac_path = temp_flac_file.name

    features = extract_features(temp_flac_path)
    if features is not None:
        prediction = model.predict(features)
        if prediction > 0.7:
            st.markdown(f"Audio <b>{uploaded_file.name}</b> is: <b>Real</b>", unsafe_allow_html=True)
        else:
            st.markdown(f"Audio <b>{uploaded_file.name}</b> is: <b>Fake</b>", unsafe_allow_html=True)
    else:
        st.error("Feature extraction failed. Please try again with a different file.")

# LSTM Model Evaluations
st.sidebar.title("SVM Model Evaluations")
evaluation_option = st.sidebar.selectbox(
    "Choose an evaluation metric",
    ("Select an option", "SVM Model Report", "Actual vs Predicted Label Chart", "SVM Model Loss", "SVM Model Accuracy")
)
if evaluation_option == "SVM Model Report":
    st.write("SVM Model Report")
    st.image("https://github.com/AJSTYLE-lab/AudioShield_Leveraging_Machine_Learning_to_Detect_Deepfake_Voices/raw/main/svm-model-report.png", caption="SVM Model Report", use_column_width=True)
elif evaluation_option == "Actual vs Predicted Label Chart":
    st.image("https://github.com/AJSTYLE-lab/AudioShield_Leveraging_Machine_Learning_to_Detect_Deepfake_Voices/raw/main/svm-actual-vs-predict-label.png", caption="Actual vs Predicted Label Chart", use_column_width=True)    
elif evaluation_option == "SVM Model Loss":
    st.image("https://github.com/AJSTYLE-lab/AudioShield_Leveraging_Machine_Learning_to_Detect_Deepfake_Voices/raw/main/svm-model-loss.png", caption="SVM Model Loss", use_column_width=True)
elif evaluation_option == "SVM Model Accuracy":
    st.image("https://github.com/AJSTYLE-lab/AudioShield_Leveraging_Machine_Learning_to_Detect_Deepfake_Voices/blob/main/svm-model-accuracy.png", caption="SVM Model Accuracy", use_column_width=True)
