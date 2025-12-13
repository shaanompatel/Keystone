import numpy as np
import soundfile as sf
import os
import random
import argparse
from scipy import signal

# Constants
SAMPLE_RATE = 22050
DURATION = 2.0 # seconds
AMPLITUDE = 0.5

# note frequencies
CHORD_FREQS = {
    'C_Major': [261.63, 329.63, 392.00],
    'G_Major': [196.00, 246.94, 293.66],
    'A_Minor': [220.00, 261.63, 329.63],
    'F_Major': [174.61, 220.00, 261.63]
}

def generate_tone(freq, duration, sample_rate, wave_type='sine'):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if wave_type == 'sine':
        wave = np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        wave = signal.square(2 * np.pi * freq * t)
    elif wave_type == 'sawtooth':
        wave = signal.sawtooth(2 * np.pi * freq * t)
    else:
        raise ValueError(f"Unknown wave type: {wave_type}")
    
    return wave

def apply_envelope(wave, sample_rate, attack=0.1, release=0.5):
    n_samples = len(wave)
    attack_samples = int(attack * sample_rate)
    release_samples = int(release * sample_rate)
    
    envelope = np.ones(n_samples)
    
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
    return wave * envelope

def construct_chord(chord_name, duration, sample_rate, wave_type='sine'):
    freqs = CHORD_FREQS.get(chord_name)
    if not freqs:
        raise ValueError(f"Unknown chord: {chord_name}")
        
    mixed_wave = np.zeros(int(sample_rate * duration))
    
    for f in freqs:
        #add randomness, detuning is done for realism
        detune = random.uniform(-0.5, 0.5)
        tone = generate_tone(f + detune, duration, sample_rate, wave_type)
        mixed_wave += tone
        
    # normalize
    mixed_wave = mixed_wave / len(freqs)
    
    return apply_envelope(mixed_wave, sample_rate)

def add_noise(wave, noise_level=0.01):
    noise = np.random.normal(0, noise_level, len(wave))
    return wave + noise

def add_harmonics(wave, freq, sample_rate):
    # add another harmonic
    t = np.linspace(0, len(wave)/sample_rate, len(wave), endpoint=False)
    harmonic = 0.3 * np.sin(2 * np.pi * (freq * 2) * t)
    return wave + harmonic

def generate_dataset(output_dir, n_samples_per_chord=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    wave_types = ['sine', 'square', 'sawtooth']
    
    print(f"Generating dataset at {output_dir}...")
    
    for chord_name in CHORD_FREQS.keys():
        chord_dir = os.path.join(output_dir, chord_name)
        if not os.path.exists(chord_dir):
            os.makedirs(chord_dir)
            
        print(f"Generating {chord_name}...")
        
        for i in range(n_samples_per_chord):
            wave_type = random.choice(wave_types)
            is_noisy = random.random() > 0.5
            
            audio = construct_chord(chord_name, DURATION, SAMPLE_RATE, wave_type)
            
            # add augmentations
            if is_noisy:
                audio = add_noise(audio, noise_level=random.uniform(0.005, 0.05))
                audio = audio * random.uniform(0.5, 1.0)
            
            # save generated wav
            filename = f"{chord_name}_{i}_{wave_type}_{'noisy' if is_noisy else 'clean'}.wav"
            filepath = os.path.join(chord_dir, filename)
            sf.write(filepath, audio, SAMPLE_RATE)
            
    print("Generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Chord Audio Generator")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--samples", type=int, default=200, help="Samples per chord")
    
    args = parser.parse_args()
    
    generate_dataset(args.output, args.samples)
