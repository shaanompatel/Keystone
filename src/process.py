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
# MFCC parameters
N_MFCC = 13
# Duration
DURATION = 2.0  
EXPECTED_SAMPLES = int(SAMPLE_RATE * DURATION)

def extract_features(audio, sr, feature_type='melspec'):
    """
    Extract features from audio based on feature_type.
    feature_type: 'melspec', 'mfcc', 'chroma', 'spectral_contrast'
    Returns: numpy array of shape (n_features, time_steps)
    """
    try:
        # Ensure consistent length
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
    """
    Load audio file and return raw audio array.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        print(f"Error loading {file_path}: {e}")
        return None, None

def audio_to_melspectrogram(file_path):
    """
    Helper to load audio and converting to melspectrogram directly.
    """
    y, sr = load_and_process(file_path)
    if y is not None:
        return extract_features(y, sr, 'melspec')
    return None

class AudioAugmenter:
    """Class for raw audio augmentation."""
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
        # Note: Time stretch changes length, so we'd need to re-pad/crop.
        # For simplicity in this pipeline, sticking to pitch shift and noise 
        # which preserve length or are easier to handle before padding.
        # But let's support it and handle length fix in dataset.
        return librosa.effects.time_stretch(data, rate=rate)

def spec_augment(mel_spectrogram, time_masking_para=10, freq_masking_para=10):
    """
    Apply SpecAugment (Time and Frequency Masking) on Spectrograms.
    """
    aug_spec = mel_spectrogram.copy()
    n_mels, n_steps = aug_spec.shape
    
    # Frequency Masking
    if freq_masking_para > 0:
        f = np.random.randint(0, freq_masking_para)
        f0 = np.random.randint(0, max(1, n_mels - f))
        aug_spec[f0:f0+f, :] = aug_spec.min()
    
    # Time Masking
    if time_masking_para > 0:
        t = np.random.randint(0, time_masking_para)
        t0 = np.random.randint(0, max(1, n_steps - t))
        aug_spec[:, t0:t0+t] = aug_spec.min()
    
    return aug_spec

def preprocess_dataset_lazy(data_dirs):
    """
    Scan directories and return list of file paths and labels.
    data_dirs: List of root directories, e.g. ["data/real_train", "data/synthetic"]
               Each root must have subdirs named by class (e.g. "A_Minor").
    Does NOT load audio.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
        
    # First, scan one dir to get classes, or scan all and take union?
    # Classes should be consistent across all dirs.
    # Let's assume the first dir has all classes or we manually define them.
    # Safe approach: Scan all dirs, collect all unique class names.
    
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
