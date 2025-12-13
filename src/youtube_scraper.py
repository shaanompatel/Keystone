import os
import glob
from pydub import AudioSegment
import yt_dlp
import shutil

def download_audio(query, limit=5, output_dir="data/downloads", class_name=None):

    if class_name:
        target_dir = os.path.join(output_dir, class_name)
    else:
        # get functional folder name
        target_dir = os.path.join(output_dir, query.replace(" ", "_").replace(":", "").replace("/", ""))
        
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': False,
        'playlistend': limit,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(target_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'default_search': f'ytsearch{limit}:',
    }
    
    print(f"Searching for '{query}' -> {target_dir}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([query])
        except Exception as e:
            print(f"Error downloading: {e}")
            
    return target_dir

def slice_audio(directory, chunk_duration_ms=2000):

    files = glob.glob(os.path.join(directory, "*.wav"))
    print(f"Slicing {len(files)} files in {directory}...")
    
    for f in files:
        if "_chunk_" in f:
            continue
            
        try:
            audio = AudioSegment.from_wav(f)
            base_name = os.path.basename(f).replace(".wav", "")
            
            # slice
            for i in range(0, len(audio), chunk_duration_ms):
                chunk = audio[i:i+chunk_duration_ms]
                if len(chunk) == chunk_duration_ms:
                    chunk_name = f"{base_name}_chunk_{i//1000}.wav"
                    chunk.export(os.path.join(directory, chunk_name), format="wav")
            
            try:
                os.remove(f)
            except PermissionError:
                print(f"Could not delete {f}, generic permission error (file might be in use).")
                
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Search query or Playlist URL")
    parser.add_argument("--limit", type=int, default=5, help="Number of videos to download")
    parser.add_argument("--class_name", type=str, default=None, help="Class label (e.g. A_Minor). If not set, uses query name.")
    args = parser.parse_args()
    
    download_dir = download_audio(args.query, limit=args.limit, class_name=args.class_name)
    slice_audio(download_dir)
    print(f"Done, Check {download_dir}")
