import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from process import preprocess_dataset_lazy, load_and_process, extract_features, spec_augment, AudioAugmenter
from models import BaselineModel, AdvancedModel, RNNModel, GradientBoostingWrapper
from utils import get_cache_path

DATA_DIRS = ["data/real_train", "data/synthetic", "data/downloads"]
MODELS_DIR = "models"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

def make_weighted_sampler(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    class_weights = 1.0 / counts
    sample_weights = class_weights[np.array(labels)]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

def parse_video_id(path: str):
    
    base = os.path.basename(path)
    parts = base.split("__")
    if len(parts) >= 3:
        return parts[1]
    return base # fallback

def make_group_split(file_paths, labels, test_size=0.2, seed=42):
    groups = [parse_video_id(p) for p in file_paths]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(file_paths, labels, groups=groups))
    X_train = [file_paths[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [file_paths[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    return X_train, X_test, y_train, y_test

def mixup_batch(x, y, alpha=0.4):
    
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target):
        # target int64 class label
        num_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        logp = torch.log_softmax(logits, dim=-1)
        return -(true_dist * logp).sum(dim=-1).mean()

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, feature_type='melspec', augment=False, cache=True, norm_stats=None):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_type = feature_type
        self.augment = augment
        self.cache = cache
        self.norm_stats = norm_stats
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        features = None
        
        if self.cache:
            cache_path = get_cache_path(path, self.feature_type)
            if os.path.exists(cache_path):
                try:
                    features = np.load(cache_path)
                except:
                    pass
        
        if features is None:
            try:
                y, sr = load_and_process(path)
                if y is None:
                     return torch.zeros(1), torch.tensor(label).long()
                
                features = extract_features(y, sr, feature_type=self.feature_type)
                
                if self.cache and features is not None:
                    np.save(cache_path, features)
                    
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return torch.zeros(1), torch.tensor(label).long()

        if features is None:
             return torch.zeros(1), torch.tensor(label).long()

        if self.augment and self.feature_type == 'melspec':
            features = spec_augment(features)
        
        if features is not None and self.norm_stats is not None:
            from process import apply_norm
            features = apply_norm(features, self.norm_stats)
        
        features_t = torch.tensor(features, dtype=torch.float32)
        return features_t, torch.tensor(label, dtype=torch.long)

def train_dl_model(model_name, model, train_loader, test_loader, device, epochs=EPOCHS, use_mixup=False, mixup_alpha=0.4):
    print(f"\nTraining {model_name} for {epochs} epochs")

    criterion = SoftTargetCrossEntropy(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1
    )

    early_patience = 10
    best_epoch = -1
    best_acc = 0.0

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            if model_name.startswith("CNN") and inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            optimizer.zero_grad()

            if use_mixup:
                mixed, y_a, y_b, lam = mixup_batch(inputs, labels, alpha=mixup_alpha)
                outputs = model(mixed)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / max(1, len(train_loader))
        train_losses.append(epoch_loss)

        # validation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                if model_name.startswith("CNN") and inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / max(1, total)
        val_accuracies.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}_best.pth"))

        if (epoch - best_epoch) >= early_patience:
            tqdm.write(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch+1})")
            break

        tqdm.write(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Val Acc={val_acc:.4f} (Best: {best_acc:.4f})")

    print(f"Finished {model_name}. Best Val Acc: {best_acc:.4f}")
    return train_losses, val_accuracies, best_acc, all_preds, all_labels

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
            
    X_train_paths, X_test_paths, y_train, y_test = make_group_split(file_paths, labels, test_size=0.2, seed=42)
    
    # get norm stats
    from process import compute_dataset_norm

    tmp_feats = []
    for p in X_train_paths[:min(400, len(X_train_paths))]:
        y_raw, sr_raw = load_and_process(p)
        if y_raw is None:
            continue
        f = extract_features(y_raw, sr_raw, 'melspec')
        if f is not None:
            tmp_feats.append(f)

    norm_stats = compute_dataset_norm(tmp_feats)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f)
    print("Saved norm stats:", norm_stats)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    experiments = [
        {
            "name": "CNN_Basic",
            "type": "dl",
            "model_cls": AdvancedModel,
            "args": {"num_classes": len(classes)},
            "augment": False,
            "feature": "melspec"
        },
        {
            "name": "CNN_Augmented",
            "type": "dl",
            "model_cls": AdvancedModel,
            "args": {"num_classes": len(classes)},
            "augment": True,
            "feature": "melspec"
        },
        {
            "name": "RNN_Mel",
            "type": "dl",
            "model_cls": RNNModel,
            "args": {"num_classes": len(classes), "input_size": 128}, 
            "augment": True,
            "feature": "melspec",
            "epochs": 150
        }
    ]
    
    for exp in experiments:
        name = exp["name"]
        augment = exp["augment"]
        feature = exp["feature"]
        epochs = exp.get("epochs", EPOCHS)
        
        train_ds = AudioDataset(X_train_paths, y_train, feature_type=feature, augment=augment, cache=True, norm_stats=norm_stats)
        test_ds = AudioDataset(X_test_paths, y_test, feature_type=feature, augment=False, cache=True, norm_stats=norm_stats)
        
        # sampling with balancing varying class sizes
        class_counts = np.bincount(np.array(y_train), minlength=len(classes))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[y] for y in y_train]
        sampler = make_weighted_sampler(y_train, num_classes=len(classes))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        model = exp["model_cls"](**exp["args"]).to(device)
        
        losses, val_accs, best, preds, true_labels = train_dl_model(name, model, train_loader, test_loader, device, epochs=epochs)
        results[name] = {"val_accs": val_accs, "best": best, "preds": preds, "true": true_labels}
        
        plt.figure()
        plt.plot(losses, label='Train Loss')
        plt.plot(val_accs, label='Val Acc')
        plt.title(f"{name} Training Curve")
        plt.legend()
        plt.savefig(os.path.join(MODELS_DIR, f"{name}_curve.png"))
        plt.close()

    print("\n--- Training ML Models ---")
    print("Extracting features for ML models (using memmap for memory efficiency)...")
    
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
    
    # create memmaps
    import tempfile
    
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp dir for memmaps: {temp_dir}")
    
    train_memmap_path = os.path.join(temp_dir, "X_train.dat")
    test_memmap_path = os.path.join(temp_dir, "X_test.dat")
    
    num_train = len(X_train_paths)
    num_test = len(X_test_paths)
    
    X_train_ml = np.memmap(train_memmap_path, dtype='float32', mode='w+', shape=(num_train,) + sample_shape)
    y_train_ml = []
    
    X_test_ml = np.memmap(test_memmap_path, dtype='float32', mode='w+', shape=(num_test,) + sample_shape)
    y_test_ml = []

    # train
    valid_train_indices = []
    print("Processing Training Data...")
    for idx, (p, l) in enumerate(tqdm(zip(X_train_paths, y_train), total=num_train, desc="Train Feats")):
        y, sr = load_and_process(p)
        if y is not None:
            f = extract_features(y, sr, 'melspec')
            if f is not None:

                if f.shape == sample_shape:
                    X_train_ml[idx] = f
                    y_train_ml.append(l)
                    valid_train_indices.append(idx)
                else:
                    print(f"Shape mismatch: {p} got {f.shape}, expected {sample_shape}")
                    
    # test
    valid_test_indices = []
    print("Processing Test Data...")
    for idx, (p, l) in enumerate(tqdm(zip(X_test_paths, y_test), total=num_test, desc="Test Feats")):
        y, sr = load_and_process(p)
        if y is not None:
             f = extract_features(y, sr, 'melspec')
             if f is not None:
                if f.shape == sample_shape:
                    X_test_ml[idx] = f
                    y_test_ml.append(l)
                    valid_test_indices.append(idx)
    
    if len(y_train_ml) < num_train:
         print(f"Warning: Dropped {num_train - len(y_train_ml)} training samples due to errors.")
         pass
         
    y_train_ml = np.array(y_train_ml)
    y_test_ml = np.array(y_test_ml)
    
    pass
    
    # train
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

    X_train_final = X_train_ml[:train_ptr] # view
    y_train_ml = np.array(y_train_clean)
    
    # test
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
            
    X_test_final = X_test_ml[:test_ptr] # view
    y_test_ml = np.array(y_test_clean)
    
    X_train_ml = X_train_final
    X_test_ml = X_test_final

    
    rf = BaselineModel()
    rf.fit(X_train_ml, y_train_ml)
    rf_preds = rf.predict(X_test_ml)
    rf_acc = np.mean(rf_preds == y_test_ml)
    print(f"Random Forest Acc: {rf_acc:.4f}")
    rf.save(os.path.join(MODELS_DIR, "baseline_rf.pkl"))
    results["RandomForest"] = {"best": rf_acc, "preds": rf_preds, "true": y_test_ml}
    
    gbm = GradientBoostingWrapper()
    gbm.fit(X_train_ml, y_train_ml)
    gbm_preds = gbm.predict(X_test_ml)
    gbm_acc = np.mean(gbm_preds == y_test_ml)
    print(f"Gradient Boosting Acc: {gbm_acc:.4f}")
    gbm.save(os.path.join(MODELS_DIR, "gbm.pkl"))
    results["GradientBoosting"] = {"best": gbm_acc, "preds": gbm_preds, "true": y_test_ml}
    
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
    plt.savefig(os.path.join(MODELS_DIR, "model_comparison_bar.png"))
    plt.close()
    
    best_model_name = max(results, key=lambda x: results[x]["best"])
    print(f"Winner: {best_model_name}")
    
    if best_model_name in ["CNN_Basic", "CNN_Augmented", "RNN_Mel"]:
        import shutil
        shutil.copy(os.path.join(MODELS_DIR, f"{best_model_name}_best.pth"), os.path.join(MODELS_DIR, "advanced.pth"))
        print(f"Copied {best_model_name} to advanced.pth")

    try:
        import shutil
        shutil.rmtree(temp_dir)
        print("Cleaned up temp files.")
    except:
        pass

if __name__ == "__main__":
    main()
