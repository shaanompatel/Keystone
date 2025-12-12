import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import numpy as np
import joblib
from tqdm import tqdm

class BaselineModel:
    def __init__(self):
        # Enable warm_start for incremental training
        self.n_estimators = 100
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=10, random_state=42, warm_start=True)
        
    def fit(self, X, y):
        # Flatten features: (batch, n_mels, time) -> (batch, n_mels * time)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Reset internal state by setting n_estimators to 0 (effectively) or clearing
        # Sklearn doesn't have a clear "reset" for warm_start, so we rely on managing n_estimators.
        # However, to be safe and ensure fresh training if called multiple times, 
        # we start from a small number.
        
        # Actually simplest way to "reset" is to create a fresh classifier if logic allowed,
        # but here we just manage the loop.
        
        # Clear existing trees if any (hacky but works for warm_start=True reuse)
        self.clf.n_estimators = 0
        if hasattr(self.clf, 'estimators_'):
             self.clf.estimators_ = []
             
        step = 5
        pbar = tqdm(total=self.n_estimators, desc="Training Random Forest")
        
        curr_est = 0
        while curr_est < self.n_estimators:
            curr_est += step
            if curr_est > self.n_estimators:
                curr_est = self.n_estimators
                
            self.clf.n_estimators = curr_est
            self.clf.fit(X_flat, y)
            pbar.update(step)
            
        pbar.close()
        
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.clf.predict(X_flat)
        
    def predict_proba(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.clf.predict_proba(X_flat)
        
    def save(self, path):
        joblib.dump(self.clf, path)
        
    def load(self, path):
        self.clf = joblib.load(path)

class GradientBoostingWrapper:
    def __init__(self):
        # HistGradientBoostingClassifier is much faster (O(n_samples))
        # uses max_iter instead of n_estimators
        self.max_iter = 100
        self.clf = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=0.1, max_depth=5, random_state=42, warm_start=True, early_stopping=False)

    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        
        # For HistGradientBoostingClassifier, we manage max_iter
        # To reset, we can't easily clear internal state if we reuse the object with warm_start=True.
        # But if we assume fresh usage or we just want to train *up to* max_iter:
        
        # Logic: Set max_iter to step, fit, increase, fit.
        # To ensure we start from scratch if called again, we theoretically need a new object,
        # but resetting max_iter to 0 isn't valid.
        # If we set warm_start=True, it continues where it left off.
        # If we want to restart, we should re-init.
        # Let's re-init the internal clf to be safe, ensuring fresh training.
        
        self.clf = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=0.1, max_depth=5, random_state=42, warm_start=True, early_stopping=False)
        
        step = 5
        curr_iter = 0
        pbar = tqdm(total=self.max_iter, desc="Training HistGradientBoosting")
        
        # We start with max_iter = step
        self.clf.max_iter = 0 # Hack to start loop
        
        while curr_iter < self.max_iter:
            curr_iter += step
            if curr_iter > self.max_iter:
                curr_iter = self.max_iter
            
            self.clf.max_iter = curr_iter
            self.clf.fit(X_flat, y)
            pbar.update(step)
            
        pbar.close()

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.clf.predict(X_flat)

    def predict_proba(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.clf.predict_proba(X_flat)

    def save(self, path):
        joblib.dump(self.clf, path)
    
    def load(self, path):
        self.clf = joblib.load(path)

class AdvancedModel(nn.Module):
    def __init__(self, num_classes=4, input_shape=None):
        super(AdvancedModel, self).__init__()
        
        # Assuming input is (1, 128, 86) roughly
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, num_classes=4, input_size=128, hidden_size=128, num_layers=2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Input shape: (batch, time, features) - we will permute in forward
        # LSTM input: (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time) for CNN compat, or (batch, n_mels, time)
        # We need (batch, time, n_mels)
        if x.dim() == 4:
            x = x.squeeze(1) # (batch, n_mels, time)
            
        x = x.permute(0, 2, 1) # (batch, time, n_mels)
        
        # Initialize hidden state (optional, defaults to 0)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out
