import librosa
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor

# feature extraction params
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

N_MFCC = 13
DURATION = 2.0
EXPECTED_SAMPLES = int(SAMPLE_RATE * DURATION)

def extract_features(audio, sr, feature_type='melspec'):

    try:
        # check length
        if len(audio) < EXPECTED_SAMPLES:
            audio = np.pad(audio, (0, EXPECTED_SAMPLES - len(audio)))
        else:
            audio = audio[:EXPECTED_SAMPLES]

        if feature_type == 'melspec':
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            return librosa.power_to_db(S, ref=np.max)
        
        elif feature_type == 'mfcc':
            return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
            
        elif feature_type == 'chroma':
            return librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
            
        elif feature_type == 'spectral_contrast':
            return librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
            
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
    except Exception as e:
        print(f"Error extracting {feature_type}: {e}")
        return None

def load_and_process(file_path):

    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        print(f"Error loading {file_path}: {e}")
        return None, None

def audio_to_melspectrogram(file_path):

    y, sr = load_and_process(file_path)
    if y is not None:
        return extract_features(y, sr, 'melspec')
    return None

class AudioAugmenter:

    @staticmethod
    def add_noise(data, noise_factor=0.005):
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        return augmented_data

    @staticmethod
    def shift_pitch(data, sr, n_steps=2):
        return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def time_stretch(data, rate=1.2):
        return librosa.effects.time_stretch(data, rate=rate)

def spec_augment(mel_spectrogram, time_masking_para=10, freq_masking_para=10):

    aug_spec = mel_spectrogram.copy()
    n_mels, n_steps = aug_spec.shape
    
    # mask frequency
    if freq_masking_para > 0:
        f = np.random.randint(0, freq_masking_para)
        f0 = np.random.randint(0, max(1, n_mels - f))
        aug_spec[f0:f0+f, :] = aug_spec.min()
    
    # mask time
    if time_masking_para > 0:
        t = np.random.randint(0, time_masking_para)
        t0 = np.random.randint(0, max(1, n_steps - t))
        aug_spec[:, t0:t0+t] = aug_spec.min()
    
    return aug_spec

def preprocess_dataset_lazy(data_dirs):

    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    
    all_classes = set()
    for d in data_dirs:
        if os.path.exists(d):
            subdirs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
            all_classes.update(subdirs)
            
    classes = sorted(list(all_classes))
    print(f"Classes found: {classes}")
    
    file_paths = []
    labels = []
    
    for d in data_dirs:
        if not os.path.exists(d):
            print(f"Warning: Directory {d} does not exist. Skipping.")
            continue
            
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(d, class_name)
            if not os.path.exists(class_dir):
                continue
                
            wav_files = glob.glob(os.path.join(class_dir, "*.wav"))
            print(f"Found {len(wav_files)} files for {class_name} in {d}")
            
            for f in wav_files:
                file_paths.append(f)
                labels.append(idx)
            
    return file_paths, labels, classes

def compute_dataset_norm(feature_arrays, eps=1e-6):
    """
    feature_arrays: iterable of np arrays shaped (F, T) or (1, F, T)
    Returns dict with mean/std as float32 scalars (global) for simplicity.
    """
    s1 = 0.0
    s2 = 0.0
    n = 0

    for x in feature_arrays:
        if x is None:
            continue
        x = np.asarray(x)
        if x.ndim == 3:
            x = x[0]
        s1 += float(x.sum())
        s2 += float((x * x).sum())
        n += x.size

    if n == 0:
        return {"mean": 0.0, "std": 1.0}

    mean = s1 / n
    var = max(0.0, (s2 / n) - mean * mean)
    std = float(np.sqrt(var) + eps)
    return {"mean": float(mean), "std": float(std)}

def apply_norm(x, norm_stats):
    if norm_stats is None:
        return x
    mean = norm_stats.get("mean", 0.0)
    std = norm_stats.get("std", 1.0)
    return (x - mean) / std
