import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import numpy as np
import joblib
from tqdm import tqdm

class BaselineModel:
    def __init__(self):

        self.n_estimators = 100
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=10, random_state=42, warm_start=True)
        
    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        
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

        self.max_iter = 100
        self.clf = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=0.1, max_depth=5, random_state=42, warm_start=True, early_stopping=False)

    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        
        self.clf = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=0.1, max_depth=5, random_state=42, warm_start=True, early_stopping=False)
        
        step = 5
        curr_iter = 0
        pbar = tqdm(total=self.max_iter, desc="Training HistGradientBoosting")
        
        self.clf.max_iter = 0 # start loop
        
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):

        if x.dim() == 4:
            x = x.squeeze(1)
            
        x = x.permute(0, 2, 1)
        
        out, _ = self.lstm(x)
        
        out = out[:, -1, :]
        out = self.fc(out)
        return out
