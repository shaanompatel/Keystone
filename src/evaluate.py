import os
import torch
import numpy as np
import argparse
import glob
from process import audio_to_melspectrogram
from models import BaselineModel, AdvancedModel, BaselineModel # Re-import
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

MODELS_DIR = "models"
CLASSES_FILE = os.path.join(MODELS_DIR, "classes.txt")

def load_classes():
    with open(CLASSES_FILE, "r") as f:
        return [line.strip() for line in f.readlines()]

def evaluate_folder(folder_path, model_type='advanced'):
    classes = load_classes()
    
    # Load model
    if model_type == 'baseline':
        model = BaselineModel()
        model.load(os.path.join(MODELS_DIR, "baseline.pkl"))
    else:
        model = AdvancedModel(num_classes=len(classes))
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "advanced.pth")))
        model.eval()
        
    print(f"Evaluated with {model_type} model.")
    
    files = glob.glob(os.path.join(folder_path, "*.wav"))
    if not files:
        print("No .wav files found in directory.")
        return
        
    X = []
    filenames = []
    
    for f in files:
        feat = audio_to_melspectrogram(f)
        if feat is not None:
            X.append(feat)
            filenames.append(os.path.basename(f))
            
    if not X:
        print("No valid audio files processed.")
        return
        
    X = np.array(X)
    
    # Predict
    if model_type == 'baseline':
        preds = model.predict(X)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, preds = torch.max(outputs, 1)
        preds = preds.numpy()
        
    results = pd.DataFrame({
        "Filename": filenames,
        "Prediction": [classes[p] for p in preds]
    })
    
    print("\nPredictions:")
    print(results)
    
    # If filenames contain ground truth (e.g., "C_Major_noise.wav"), we can calc metrics
    # Simple heuristic: Check if class name is in filename
    y_true = []
    has_labels = True
    for fname in filenames:
        found = False
        for i, c in enumerate(classes):
            if c in fname:
                y_true.append(i)
                found = True
                break
        if not found:
            has_labels = False
            break
            
    if has_labels:
        print("\n--- Metrics ---")
        acc = accuracy_score(y_true, preds)
        print(f"Accuracy: {acc:.4f}")
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, preds)
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_true, preds, target_names=classes))
        
        # Plot CM
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix ({model_type})")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{model_type}.png")
        print(f"Saved confusion matrix plot to confusion_matrix_{model_type}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing .wav files")
    parser.add_argument("--model", choices=['baseline', 'advanced'], default='advanced')
    args = parser.parse_args()
    
    evaluate_folder(args.folder, args.model)
