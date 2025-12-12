import librosa
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor

# Feature extraction parameters
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
DURATION = 2.0  # Fixed duration for consistent input size
EXPECTED_SAMPLES = int(SAMPLE_RATE * DURATION)

def audio_to_melspectrogram(audio_path, n_mels=128):
    """
    Load audio and convert to Mel Spectrogram.
    Returns: numpy array of shape (n_mels, time_steps)
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or truncate to ensure consistent length
        if len(y) < EXPECTED_SAMPLES:
            y = np.pad(y, (0, EXPECTED_SAMPLES - len(y)))
        else:
            y = y[:EXPECTED_SAMPLES]
            
        # Compute Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        return S_dB
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def spec_augment(mel_spectrogram, time_masking_para=10, freq_masking_para=10):
    """
    Apply SpecAugment (Time and Frequency Masking).
    Input: (n_mels, time_steps)
    """
    aug_spec = mel_spectrogram.copy()
    n_mels, n_steps = aug_spec.shape
    
    # Frequency Masking
    f = np.random.randint(0, freq_masking_para)
    f0 = np.random.randint(0, n_mels - f)
    aug_spec[f0:f0+f, :] = aug_spec.min() # Mask with min value instead of 0 for log spec
    
    # Time Masking
    t = np.random.randint(0, time_masking_para)
    t0 = np.random.randint(0, n_steps - t)
    aug_spec[:, t0:t0+t] = aug_spec.min()
    
    return aug_spec

def preprocess_dataset(data_dir, augment=False):
    """
    Load all .wav files from data_dir/{class_name}/*.wav
    Returns: X (features), y (labels), classes (list of class names)
    """
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()
    
    X = []
    y = []
    
    file_paths = []
    labels = []
    
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        wav_files = glob.glob(os.path.join(class_dir, "*.wav"))
        print(f"Found {len(wav_files)} files for {class_name}")
        
        for f in wav_files:
            file_paths.append(f)
            labels.append(idx)
            
    # Process in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(audio_to_melspectrogram, file_paths))
        
    for feat, label in zip(results, labels):
        if feat is not None:
            X.append(feat)
            y.append(label)
            
            # Apply Augmentations if requested
            if augment:
                # Add an augmented version to training (doubling data effectively for this batch)
                # Or just augment in-place? Usually we augment ONLY training set.
                # Here we return raw list, augmentation logic might be better in Dataset class or train loop.
                # But to keep it simple, let's just create augmented copies here.
                # WARNING: This doubles memory usage.
                pass 
                
    return np.array(X), np.array(y), classes
