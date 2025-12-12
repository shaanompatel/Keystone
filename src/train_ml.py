import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile
import shutil

# Import from our modules
from process import preprocess_dataset_lazy, load_and_process, extract_features
from models import BaselineModel, GradientBoostingWrapper

# Config
DATA_DIRS = ["data/real_train", "data/synthetic", "data/downloads"]
MODELS_DIR = "models"
    
def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print("Indexing dataset...")
    # Note: We rely on preprocess_dataset_lazy to scan files
    file_paths, labels, classes = preprocess_dataset_lazy(DATA_DIRS)
    
    if len(file_paths) == 0:
        raise ValueError("No data found!")
        
    with open(os.path.join(MODELS_DIR, "classes.txt"), "w") as f:
        for c in classes:
            f.write(c + "\n")
            
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, random_state=42)
    
    results = {}

    print("\n--- Training ML Models ---")
    print("Extracting features for ML models (using memmap for memory efficiency)...")
    
    # We need to determine the shape of features first
    sample_shape = None
    for p in X_train_paths:
        y_dummy, sr_dummy = load_and_process(p)
        if y_dummy is not None:
             f_dummy = extract_features(y_dummy, sr_dummy, 'melspec')
             if f_dummy is not None:
                 sample_shape = f_dummy.shape
                 break
    
    if sample_shape is None:
        print("Could not extract features from any training file!")
        return

    print(f"Feature shape: {sample_shape}")
    
    # Create memmaps
    # Use a temp directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp dir for memmaps: {temp_dir}")
    
    train_memmap_path = os.path.join(temp_dir, "X_train.dat")
    test_memmap_path = os.path.join(temp_dir, "X_test.dat")
    
    # Pre-allocate memmaps
    # Shape: (N_samples, *sample_shape)
    num_train = len(X_train_paths)
    num_test = len(X_test_paths)
    
    X_train_ml = np.memmap(train_memmap_path, dtype='float32', mode='w+', shape=(num_train,) + sample_shape)
    y_train_ml = []
    
    X_test_ml = np.memmap(test_memmap_path, dtype='float32', mode='w+', shape=(num_test,) + sample_shape)
    y_test_ml = []

    # Fill Train
    train_ptr = 0
    y_train_clean = []
    
    print("Processing Training Data (Memmap)...")
    for p, l in tqdm(zip(X_train_paths, y_train), total=num_train, desc="Train Feats"):
        try:
            y, sr = load_and_process(p)
            if y is not None:
                f = extract_features(y, sr, 'melspec')
                if f is not None and f.shape == sample_shape:
                    X_train_ml[train_ptr] = f
                    y_train_clean.append(l)
                    train_ptr += 1
        except:
            continue

    # Create a view of the valid data
    X_train_final = X_train_ml[:train_ptr] # View
    y_train_ml = np.array(y_train_clean)
    
    # Test
    test_ptr = 0
    y_test_clean = []
    
    print("Processing Test Data (Memmap)...")
    for p, l in tqdm(zip(X_test_paths, y_test), total=num_test, desc="Test Feats"):
        try:
            y, sr = load_and_process(p)
            if y is not None:
                 f = extract_features(y, sr, 'melspec')
                 if f is not None and f.shape == sample_shape:
                    X_test_ml[test_ptr] = f
                    y_test_clean.append(l)
                    test_ptr += 1
        except:
            continue
            
    X_test_final = X_test_ml[:test_ptr] # View
    y_test_ml = np.array(y_test_clean)
    
    # Reassign for compatibility with existing variable names
    X_train_ml = X_train_final
    X_test_ml = X_test_final
    
    # --- Random Forest ---
    print("\nTraining Random Forest...")
    rf = BaselineModel()
    rf.fit(X_train_ml, y_train_ml)
    rf_preds = rf.predict(X_test_ml)
    rf_acc = np.mean(rf_preds == y_test_ml)
    print(f"Random Forest Acc: {rf_acc:.4f}")
    rf.save(os.path.join(MODELS_DIR, "baseline_rf.pkl"))
    results["RandomForest"] = {"best": rf_acc, "preds": rf_preds, "true": y_test_ml}
    
    # --- Gradient Boosting ---
    print("\nTraining Gradient Boosting...")
    gbm = GradientBoostingWrapper()
    gbm.fit(X_train_ml, y_train_ml)
    gbm_preds = gbm.predict(X_test_ml)
    gbm_acc = np.mean(gbm_preds == y_test_ml)
    print(f"Gradient Boosting Acc: {gbm_acc:.4f}")
    gbm.save(os.path.join(MODELS_DIR, "gbm.pkl"))
    results["GradientBoosting"] = {"best": gbm_acc, "preds": gbm_preds, "true": y_test_ml}
    
    print("\nGenerating Comparison Plots...")
    plt.figure(figsize=(12, 6))
    
    names = []
    scores = []
    
    for name, data in results.items():
        names.append(name)
        scores.append(data["best"])
        cm = confusion_matrix(data["true"], data["preds"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(os.path.join(MODELS_DIR, f"cm_{name}.png"))
        plt.close()
        
    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=scores)
    plt.title("Model Accuracy Comparison (ML Only)")
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(MODELS_DIR, "model_comparison_ml.png"))
    plt.close()
    
    best_model_name = max(results, key=lambda x: results[x]["best"])
    print(f"Winner: {best_model_name}")

    try:
        shutil.rmtree(temp_dir)
        print("Cleaned up temp files.")
    except:
        pass

if __name__ == "__main__":
    main()
