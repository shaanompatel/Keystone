import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from src.process import audio_to_melspectrogram, preprocess_dataset
from src.models import BaselineModel, AdvancedModel, BaselineModel # Import both
import librosa
import soundfile as sf
import tempfile

# Config
DATA_DIR = "data/real_train"
MODELS_DIR = "models"
CLASSES_FILE = os.path.join(MODELS_DIR, "classes.txt")

st.set_page_config(page_title="Sim2Real Chord Analyzer", layout="wide")

@st.cache_resource
def load_classes():
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        return ["A_Minor", "C_Major", "F_Major", "G_Major"] # Fallback

@st.cache_resource
def load_baseline():
    path = os.path.join(MODELS_DIR, "baseline.pkl")
    if os.path.exists(path):
        model = BaselineModel()
        model.load(path)
        return model
    return None

@st.cache_resource
def load_advanced(num_classes):
    path = os.path.join(MODELS_DIR, "advanced.pth")
    if os.path.exists(path):
        model = AdvancedModel(num_classes=num_classes)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

def main():
    st.title("🎸 Sim2Real Chord Classifier")
    st.markdown("Train on **Synthetic** waves, Test on **Real** audio.")
    
    classes = load_classes()
    
    # Sidebar
    st.sidebar.header("Model Configuration")
    model_choice = st.sidebar.selectbox("Select Model", ["Baseline (Random Forest)", "Advanced (CNN)"])
    
    # Load Models
    if model_choice == "Baseline (Random Forest)":
        model = load_baseline()
    else:
        model = load_advanced(len(classes))
        
    if model is None:
        st.error(f"Model {model_choice} not found. Please run training script first.")
        st.info("Run `python src/train.py` in the terminal.")
        return

    # Input Section
    st.header("1. Input Audio")
    input_method = st.radio("Choose input method:", ["Upload File", "Microphone"], horizontal=True)
    
    audio_file = None
    
    if input_method == "Upload File":
        audio_file = st.file_uploader("Upload a .wav file", type=["wav"])
    else:
        # Experimental audio input
        audio_buffer = st.audio_input("Record a chord")
        if audio_buffer:
            audio_file = audio_buffer

    if audio_file:
        # Save to temp file to process with librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
            
        st.audio(tmp_path)
        
        # Process
        st.header("2. Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Mel-Spectrogram")
            try:
                spec = audio_to_melspectrogram(tmp_path)
                if spec is not None:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.heatmap(spec, ax=ax, cmap="magma", cbar=False)
                    ax.invert_yaxis()
                    st.pyplot(fig)
                else:
                    st.error("Failed to process audio.")
                    return
            except Exception as e:
                st.error(f"Error: {e}")
                return
                
        # Inference
        with col2:
            st.subheader("Prediction")
            
            # Prepare input
            X = np.array([spec])
            
            if "Baseline" in model_choice:
                probs = model.predict_proba(X)[0]
                pred_idx = np.argmax(probs)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
                with torch.no_grad():
                    outputs = model(X_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0].numpy()
                pred_idx = np.argmax(probs)
                
            pred_class = classes[pred_idx]
            confidence = probs[pred_idx]
            
            st.metric("Predicted Chord", pred_class.replace("_", " "), f"{confidence:.1%}")
            
            # Bar chart
            chart_data = {
                "Chord": [c.replace("_", " ") for c in classes],
                "Confidence": probs
            }
            st.bar_chart(chart_data, x="Chord", y="Confidence")
            
        # Cleanup
        os.unlink(tmp_path)

if __name__ == "__main__":
    main()
