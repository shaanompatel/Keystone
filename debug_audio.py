import librosa
import os
import sys

# Path to a file I know exists from list_dir
file_path = r"C:\Users\shaan\OneDrive\Desktop\441proj\temp_downloads\C_Major\72MEHvT9JV4.wav"

if not os.path.exists(file_path):
    print("File not found")
    sys.exit(1)

print(f"Attr: {os.path.getsize(file_path)} bytes")

try:
    y, sr = librosa.load(file_path, sr=22050)
    print(f"Success! Loaded {len(y)} samples.")
except Exception as e:
    print(f"Error: {e}")
