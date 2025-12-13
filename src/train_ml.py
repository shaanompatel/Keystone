import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")

import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile
import shutil

from process import preprocess_dataset_lazy, load_and_process, extract_features
from models import BaselineModel, GradientBoostingWrapper


DATA_DIRS = ["data/real_train", "data/synthetic", "data/downloads"]
MODELS_DIR = "models"
    
def parse_video_id(path: str):
    base = os.path.basename(path)
    parts = base.split("__")
    if len(parts) >= 3:
        return parts[1]
    return base  # fallback

def make_group_split(file_paths, labels, test_size=0.2, seed=42):
    groups = [parse_video_id(p) for p in file_paths]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(file_paths, labels, groups=groups))
    X_train = [file_paths[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test  = [file_paths[i] for i in test_idx]
    y_test  = [labels[i] for i in test_idx]
    return X_train, X_test, y_train, y_test

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print("Indexing dataset...")
    file_paths, labels, classes = preprocess_dataset_lazy(DATA_DIRS)
    
    if len(file_paths) == 0:
        raise ValueError("No data found!")
        
    with open(os.path.join(MODELS_DIR, "classes.txt"), "w") as f:
        for c in classes:
            f.write(c + "\n")
            
    X_train_paths, X_temp_paths, y_train, y_temp = make_group_split(
        file_paths, labels, test_size=0.20, seed=42
    )

    X_val_paths, X_test_paths, y_val, y_test = make_group_split(
        X_temp_paths, y_temp, test_size=0.50, seed=43
    )
    
    results = {}

    print("\nTraining ML Models")
    print("Extracting features for ML models...")
    
    # get features shape
    sample_shape = None
    for p in X_train_paths:
        y_dummy, sr_dummy = load_and_process(p)
        if y_dummy is not None:
             f_dummy = extract_features(y_dummy, sr_dummy, 'melspec')
             if f_dummy is not None:
                 sample_shape = f_dummy.shape
                 break
    
    if sample_shape is None:
        print("Could not extract features!")
        return

    print(f"Feature shape: {sample_shape}")
    
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp dir for memmaps: {temp_dir}")
    
    train_memmap_path = os.path.join(temp_dir, "X_train.dat")
    test_memmap_path = os.path.join(temp_dir, "X_test.dat")
    val_memmap_path = os.path.join(temp_dir, "X_val.dat")

    # preallocate
    num_train = len(X_train_paths)
    num_test = len(X_test_paths)
    num_val = len(X_val_paths)


    X_train_ml = np.memmap(train_memmap_path, dtype='float32', mode='w+', shape=(num_train,) + sample_shape)
    y_train_ml = []
    
    X_test_ml = np.memmap(test_memmap_path, dtype='float32', mode='w+', shape=(num_test,) + sample_shape)
    y_test_ml = []

    X_val_ml = np.memmap(val_memmap_path, dtype='float32', mode='w+', shape=(num_val,) + sample_shape)
    y_val_ml = []

    train_ptr = 0
    y_train_clean = []
    
    print("Processing Training Data")
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

    X_train_final = X_train_ml[:train_ptr]
    y_train_ml = np.array(y_train_clean)

    val_ptr = 0
    y_val_clean = []

    print("Processing Val Data")
    for p, l in tqdm(zip(X_val_paths, y_val), total=num_val, desc="Val Feats"):
        try:
            y, sr = load_and_process(p)
            if y is not None:
                f = extract_features(y, sr, 'melspec')
                if f is not None and f.shape == sample_shape:
                    X_val_ml[val_ptr] = f
                    y_val_clean.append(l)
                    val_ptr += 1
        except:
            continue

    X_val_final = X_val_ml[:val_ptr]
    y_val_ml = np.array(y_val_clean)
    
    test_ptr = 0
    y_test_clean = []
    
    print("Processing Test Data")
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
            
    X_test_final = X_test_ml[:test_ptr]
    y_test_ml = np.array(y_test_clean)
    
    X_train_ml = X_train_final
    X_val_ml = X_val_final
    X_test_ml = X_test_final
    
    # random forest
    print("\nTraining Random Forest")
    rf = BaselineModel()
    rf.fit(X_train_ml, y_train_ml)
    rf_val_preds = rf.predict(X_val_ml)
    rf_val_acc = np.mean(rf_val_preds == y_val_ml)
    print(f"Random Forest Acc: {rf_val_acc:.4f}")
    rf.save(os.path.join(MODELS_DIR, "baseline_rf.pkl"))
    results["RandomForest"] = {"best": rf_val_acc, "preds": rf_val_preds, "true": y_val_ml}
    
    # gradient boosting
    print("\nTraining Gradient Boosting")
    gbm = GradientBoostingWrapper()
    gbm.fit(X_train_ml, y_train_ml)
    gbm_preds = gbm.predict(X_val_ml)
    gbm_acc = np.mean(gbm_preds == y_val_ml)
    print(f"Gradient Boosting Acc: {gbm_acc:.4f}")
    gbm.save(os.path.join(MODELS_DIR, "gbm.pkl"))
    results["GradientBoosting"] = {"best": gbm_acc, "preds": gbm_preds, "true": y_val_ml}
    
    print("\nGenerating Comparison Plots")
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
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(MODELS_DIR, "model_comparison_ml.png"))
    plt.close()
    
    best_model_name = max(results, key=lambda x: results[x]["best"])
    print(f"Winner: {best_model_name}")

    print("\nFinal evaluation on TEST set...")
    X_trainval = np.concatenate([X_train_ml, X_val_ml], axis=0)
    y_trainval = np.concatenate([y_train_ml, y_val_ml], axis=0)

    if best_model_name == "RandomForest":
        final_model = BaselineModel()
        final_model.fit(X_trainval, y_trainval)
    elif best_model_name == "GradientBoosting":
        final_model = GradientBoostingWrapper()
        final_model.fit(X_trainval, y_trainval)

    test_preds = final_model.predict(X_test_ml)
    test_acc = np.mean(test_preds == y_test_ml)
    print(f"{best_model_name} Test Acc: {test_acc:.4f}")

    cm = confusion_matrix(y_test_ml, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix (TEST): {best_model_name}")
    plt.savefig(os.path.join(MODELS_DIR, f"cm_{best_model_name}_TEST.png"))
    plt.close()

    # final model
    if best_model_name == "RandomForest":
        final_model.save(os.path.join(MODELS_DIR, "baseline_rf.pkl"))
    else:
        final_model.save(os.path.join(MODELS_DIR, "gbm.pkl"))


    try:
        shutil.rmtree(temp_dir)
        print("Cleaned up temp files.")
    except:
        pass

if __name__ == "__main__":
    main()
