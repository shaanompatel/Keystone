# Sim2Real Chord Classifier

This project trains machine learning models on synthetic audio data to classify musical chords in real-world recordings.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Generate Synthetic Data
Generate the training dataset (C Major, G Major, A Minor, F Major):
```bash
python src/generator.py --samples 500
```
This creates `data/synthetic`.

### 2. Train Models
Train the Baseline (Random Forest) and Advanced (CNN) models:
```bash
python src/train.py
```
Models are saved to `models/`.

### 3. Run Web App
Launch the Streamlit interface:
```bash
streamlit run app.py
```

### 4. Evaluate on Real Data
To evaluate on a folder of real `.wav` files:
```bash
python src/evaluate.py path/to/real/audio
```

## Project Structure
- `src/generator.py`: Synthetic data generation.
- `src/process.py`: Audio processing (Mel-Spectrograms).
- `src/models.py`: Model architectures.
- `src/train.py`: Training loop.
- `src/evaluate.py`: Evaluation script.
- `app.py`: Streamlit web application.
