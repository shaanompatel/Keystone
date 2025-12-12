import yt_dlp
import os
import librosa
import soundfile as sf
import numpy as np
import argparse
import time

# Config
DATA_DIR = "data/real_train"
SAMPLE_RATE = 22050
DURATION = 2.0  # Seconds per chunk
# Hardcoded for current environment
FFMPEG_LOCATION = r"C:\Users\shaan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"

# Search terms for each chord
# We try to be specific to get clean samples
# Comprehensive search queries for diverse dataset
SEARCH_QUERIES = {
    'C_Major': [
        'C Major guitar chord strum', 'C Major guitar chord clean', 
        'C Major chord acoustic guitar', 'C Major chord electric guitar',
        'C Major chord jazz guitar', 'C Major guitar chord tutorial', 
        'C Major guitar chord sound', 'C Major barre chord guitar'
    ],
    'G_Major': [
        'G Major guitar chord strum', 'G Major guitar chord clean',
        'G Major chord acoustic guitar', 'G Major chord electric guitar',
        'G Major chord jazz guitar', 'G Major guitar chord tutorial',
        'G Major guitar chord sound', 'G Major barre chord guitar'
    ],
    'A_Minor': [
        'A Minor guitar chord strum', 'A Minor guitar chord clean',
        'A Minor chord acoustic guitar', 'A Minor chord electric guitar',
        'A Minor chord jazz guitar', 'A Minor guitar chord tutorial',
        'A Minor guitar chord sound', 'A Minor barre chord guitar'
    ],
    'F_Major': [
        'F Major guitar chord strum', 'F Major guitar chord clean',
        'F Major chord acoustic guitar', 'F Major chord electric guitar',
        'F Major chord jazz guitar', 'F Major guitar chord tutorial',
        'F Major guitar chord sound', 'F Major barre chord guitar'
    ]
}

def download_audio(query, output_path, num_results=5):
    """
    Downloads audio from YouTube videos matching the query.
    Returns list of downloaded filenames.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_path}/%(id)s.%(ext)s',
        'quiet': True,
        'noplaylist': True,
        'default_search': 'ytsearch',
        'max_downloads': num_results,
        'ffmpeg_location': FFMPEG_LOCATION
    }
    
    print(f"Searching and downloading for: {query}")
    
    downloaded_files = []
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # We search for slightly more to ensure failures don't stop us
            search_query = f"ytsearch{num_results}:{query}"
            info = ydl.extract_info(search_query, download=True)
            
            if 'entries' in info:
                for entry in info['entries']:
                    if entry:
                        fname = f"{output_path}/{entry['id']}.wav"
                        # yt-dlp might leave it as .wav or .m4a depending on ffmpeg availability
                        # But we asked for wav conversion.
                        # Let's check what exists
                        base = f"{output_path}/{entry['id']}"
                        if os.path.exists(base + ".wav"):
                            downloaded_files.append(base + ".wav")
            else:
                # Single video
                fname = f"{output_path}/{info['id']}.wav"
                if os.path.exists(fname):
                    downloaded_files.append(fname)
                    
    except Exception as e:
        print(f"Error downloading {query}: {e}")
        
    return downloaded_files

def process_audio(file_path, chord_name, output_dir):
    """
    Loads audio, removes silence, and splits into 2-second chunks.
    Saves chunks to output_dir.
    """
    try:
        # Load
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Remove silence (simple energy based)
        # top_db=20 means any sound < max-20dB is considered silence
        intervals = librosa.effects.split(y, top_db=20)
        
        chunk_idx = 0
        samples_per_chunk = int(DURATION * SAMPLE_RATE)
        
        for start, end in intervals:
            segment = y[start:end]
            
            # Split segment into chunks
            for i in range(0, len(segment) - samples_per_chunk, samples_per_chunk):
                chunk = segment[i : i + samples_per_chunk]
                
                # Check energy to avoid empty chunks
                if np.mean(chunk**2) > 0.001:
                    filename = f"{chord_name}_{os.path.basename(file_path).split('.')[0]}_{chunk_idx}.wav"
                    sf.write(os.path.join(output_dir, filename), chunk, SAMPLE_RATE)
                    chunk_idx += 1
        
        print(f"Computed {chunk_idx} chunks from {os.path.basename(file_path)}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    for chord, queries in SEARCH_QUERIES.items():
        chord_dir = os.path.join(DATA_DIR, chord)
        if not os.path.exists(chord_dir):
            os.makedirs(chord_dir)
            
        print(f"\n--- Processing {chord} ---")
        
        # Temp dir for downloads
        temp_dir = os.path.join("temp_downloads", chord)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        for query in queries:
            download_audio(query, temp_dir, num_results=5)
            
        # Process all files in temp_dir
        raw_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.wav')]
        print(f"Found {len(raw_files)} raw files in {temp_dir}. Processing...")
        
        for rf in raw_files:
            process_audio(rf, chord, chord_dir)
            
if __name__ == "__main__":
    main()
