import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from process import preprocess_dataset, spec_augment
from models import BaselineModel, AdvancedModel
import matplotlib.pyplot as plt

# Config
DATA_DIR = "data/real_train"
MODELS_DIR = "models"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

def train_experiment(model_name, X_train, y_train, X_test, y_test, num_classes, augment=False):
    print(f"\n--- Starting Experiment: {model_name} (Augment={augment}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # CNN Data Prep
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # If augmentation is enabled, we apply it here to the training tensors?
    # SpecAugment usually applied on the fly. 
    # For simplicity, let's augment the numpy array before tensor conversion if augment=True
    if augment:
        print("Applying SpecAugment to training data...")
        X_aug = []
        for x in X_train:
            X_aug.append(spec_augment(x))
        # Concatenate original + augmented
        X_train_combined = np.concatenate((X_train, np.array(X_aug)), axis=0)
        y_train_combined = np.concatenate((y_train, y_train), axis=0)
        
        X_train_t = torch.tensor(X_train_combined, dtype=torch.float32).unsqueeze(1).to(device)
        y_train_t = torch.tensor(y_train_combined, dtype=torch.long).to(device)
    else:
         X_train_t = X_train_t.to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = AdvancedModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    val_accuracies = []
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs.data, 1)
            total = y_test_t.size(0)
            correct = (predicted == y_test_t).sum().item()
            
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Save if it's the best so far for this experiment
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}_best.pth"))
        
        scheduler.step(epoch_loss)
        
        # print(f"Ep {epoch+1}: Loss={epoch_loss:.3f}, Val={val_acc:.3f}")

    print(f"Finished {model_name}. Best Val Acc: {best_acc:.4f}")
    return val_accuracies, best_acc

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print("Loading dataset...")
    X, y, classes = preprocess_dataset(DATA_DIR)
    
    if len(X) == 0:
        raise ValueError("No data found!")
        
    # Save class names
    with open(os.path.join(MODELS_DIR, "classes.txt"), "w") as f:
        for c in classes:
            f.write(c + "\n")
            
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # Experiment 1: Baseline RF
    print("\n--- Experiment: Baseline Random Forest ---")
    rf = BaselineModel()
    rf.fit(X_train, y_train)
    rf_acc = rf.clf.score(X_test.reshape(X_test.shape[0], -1), y_test)
    print(f"RF CV Accuracy: {rf_acc:.4f}")
    rf.save(os.path.join(MODELS_DIR, "baseline.pkl"))
    # RF has no epochs, so we plot a straight line? Or just bar chart.
    
    # Experiment 2: CNN Basic
    res_basic, best_basic = train_experiment("CNN_Basic", X_train, y_train, X_test, y_test, len(classes), augment=False)
    results["CNN_Basic"] = res_basic
    
    # Experiment 3: CNN Augmented
    res_aug, best_aug = train_experiment("CNN_Augmented", X_train, y_train, X_test, y_test, len(classes), augment=True)
    results["CNN_Augmented"] = res_aug
    
    # Plot Comparison
    plt.figure(figsize=(10, 6))
    for name, accs in results.items():
        plt.plot(accs, label=f"{name} (Best: {max(accs):.2%})")
        
    plt.axhline(y=rf_acc, color='r', linestyle='--', label=f"Random Forest ({rf_acc:.2%})")
    
    plt.title("Model Comparison: Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODELS_DIR, "model_comparison.png"))
    print("Saved model_comparison.png")
    
    # Determine winner
    winner = "CNN_Basic"
    if best_aug > best_basic:
        winner = "CNN_Augmented"
        
    print(f"\nWinning Model: {winner}")
    # Rename winner to advanced.pth for app compatibility
    import shutil
    shutil.copy(os.path.join(MODELS_DIR, f"{winner}_best.pth"), os.path.join(MODELS_DIR, "advanced.pth"))
    print(f"Promoted {winner} to advanced.pth")

if __name__ == "__main__":
    main()
