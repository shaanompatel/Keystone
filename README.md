# Sim2Real Chord Classifier

## About
This project uses a **Sim2Real** approach to classify musical chords. It trains machine learning models on **synthetic audio** (generated mathematically) and reinforces them with **real-world data** from YouTube. This allows the system to robustly identify chords (C Major, G Major, A Minor, F Major) across different instruments and recording conditions.

## Getting Started

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Setup
Generate synthetic training data and (optionally) download real samples:
```bash
# Generate synthetic waveforms
python src/generator.py

# (Optional) Download real examples from YouTube
python src/youtube_scraper.py --query "C Major Guitar Strum" --class_name "C_Major"
```

### 3. Training
Train the Deep Learning models (CNN, RNN):
```bash
python src/train.py
```
*Models are automatically saved to the `models/` directory.*

### 4. Run App
Launch the interactive web interface to test the models:

## Result Highlights
### Model Comparison
![Model Accuracy Comparison](models/model_comparison_bar.png)

### Training Performance (CNN)
| Training Curve | Confusion Matrix |
|:---:|:---:|
| ![CNN Curve](models/CNN_Basic_curve.png) | ![CNN Confusion Matrix](models/cm_CNN_Basic.png) |

*Results show that Deep Learning models (CNN) trained on synthetic data generalize well to real-world examples.*
