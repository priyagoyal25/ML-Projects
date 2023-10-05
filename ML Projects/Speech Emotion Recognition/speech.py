import streamlit as st
import librosa
import soundfile
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import speech_recognition as sr
import tempfile
import joblib
import requests


model_url = "emotion_model.pkl"
response = requests.get(model_url)
with open("emotion_model.pkl", "wb") as f:
    f.write(response.content)
# Function to extract audio features
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Load your trained model (replace with your actual model loading code)
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
# Load your trained model

model = joblib.load("emotion_model.pkl")



# Function to predict emotion from an audio file
def predict_emotion_from_audio(audio_file_path):
    try:
        # Load the audio file and extract features
        feature = extract_feature(audio_file_path, mfcc=True, chroma=True, mel=True)
        # Predict emotion
        emotion = model.predict([feature])[0]
        return emotion
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Streamlit app
def main():
    st.title("Speech Emotion Recognition")

    st.sidebar.header("Options")

    # Option to upload an audio file
    uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav"])
    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/wav")

        # Predict emotion for the uploaded audio
        if st.sidebar.button("Predict Emotion"):
            predicted_emotion = predict_emotion_from_audio(uploaded_audio)
            if predicted_emotion is not None:
                st.sidebar.write(f"Predicted Emotion: {predicted_emotion}")
            else:
                st.sidebar.write("Emotion prediction failed.")

    # Option to record and predict user's emotion
    if st.sidebar.button("Record and Predict Your Emotion"):
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            st.write("Please speak something...")
            try:
                audio = recognizer.listen(source, timeout=10)  # Capture audio for up to 10 seconds
                st.write("Audio recorded. Recognizing emotion...")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio.get_wav_data())

                # Preprocess the user's audio from the temporary file
                user_feature = extract_feature(f.name, mfcc=True, chroma=True, mel=True)

                # Predict emotion
                user_emotion = model.predict([user_feature])[0]
                st.write(f"Predicted Emotion: {user_emotion}")

            except sr.WaitTimeoutError:
                st.write("No audio detected within the timeout period.")
            except Exception as e:
                st.write(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
