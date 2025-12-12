import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

class BaselineModel:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
    def fit(self, X, y):
        # Flatten features: (batch, n_mels, time) -> (batch, n_mels * time)
        X_flat = X.reshape(X.shape[0], -1)
        self.clf.fit(X_flat, y)
        
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
