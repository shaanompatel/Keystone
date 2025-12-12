import os
import hashlib

def get_cache_path(file_path, feature_type, cache_dir="data/cache"):
    """
    Generate a unique cache path for a given file and feature type.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        
    # Create a hash of the file path to ensure uniqueness and safe filenames
    path_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()
    filename = f"{path_hash}_{feature_type}.npy"
    
    return os.path.join(cache_dir, filename)
