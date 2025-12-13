import os
from models import RNNModel
import torch
import numpy as np
import argparse
import glob
from models import BaselineModel, AdvancedModel, BaselineModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from process import extract_features, load_and_process, apply_norm, audio_to_melspectrogram

MODELS_DIR = "models"
CLASSES_FILE = os.path.join(MODELS_DIR, "classes.txt")

def infer_arch_from_state_dict(sd: dict) -> str:
    keys = sd.keys()
    if any(k.startswith("lstm.") for k in keys):
        return "rnn"
    if any(k.startswith("conv1.") for k in keys) or any(k.startswith("bn1.") for k in keys):
        return "cnn"
    return "cnn"

def load_classes():
    with open(CLASSES_FILE, "r") as f:
        return [line.strip() for line in f.readlines()]

def evaluate_folder(folder_path, model_type='advanced'):
    classes = load_classes()

    # load model
    if model_type == 'baseline':
        model = BaselineModel()
        model.load(os.path.join(MODELS_DIR, "baseline.pkl"))
    else:
        ckpt_path = os.path.join(MODELS_DIR, "advanced.pth")
        state = torch.load(ckpt_path, map_location="cpu")

        arch = infer_arch_from_state_dict(state)
        if arch == "rnn":
            model = RNNModel(num_classes=len(classes), input_size=128)
        else:
            model = AdvancedModel(num_classes=len(classes))

        model.load_state_dict(state)
        model.eval()

    print(f"Evaluated with {model_type} model.")

    files = glob.glob(os.path.join(folder_path, "*.wav"))
    if not files:
        print("No .wav files found in directory.")
        return

    # load normalized stats
    norm_path = os.path.join(MODELS_DIR, "norm_stats.json")
    norm_stats = None
    if os.path.exists(norm_path):
        with open(norm_path, "r") as f:
            norm_stats = json.load(f)

    X = []
    filenames = []

    for fpath in files:
        y, sr = load_and_process(fpath)
        if y is None:
            continue
        feat = extract_features(y, sr, "melspec")
        if feat is None:
            continue
        feat = apply_norm(feat, norm_stats)
        X.append(feat)
        filenames.append(os.path.basename(fpath))

    if not X:
        print("No valid audio files processed.")
        return

    X = np.array(X)

    # predict
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

    # if ground truth label exists in filename
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
        cm = confusion_matrix(y_true, preds)

        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_true, preds, target_names=classes))

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
