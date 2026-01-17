"""RCAEval model training pipeline."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

class RCAEvalTrainer:
    """Train models on RCAEval datasets."""
    
    def __init__(self, data_dir='data/rcaeval_processed'):
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_data(self, dataset='RE1'):
        """Load and preprocess data."""
        if dataset == 'RE1':
            df = pd.read_csv(self.data_dir / 'simple_data.csv')
            inject_time = int(open(self.data_dir / 'inject_time.txt').read().strip())
        elif dataset == 'RE2':
            df = pd.read_csv(self.data_dir / 'simple_metrics.csv')
            inject_time = int(open(self.data_dir / 'RE2_inject_time.txt').read().strip())
        else:  # RE3
            df = pd.read_csv(self.data_dir / 'RE3_simple_metrics.csv')
            inject_time = int(open(self.data_dir / 'RE3_inject_time.txt').read().strip())
        
        # Get numeric features
        X = df.select_dtypes(include=[np.number]).values
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Create labels - last 30% as anomaly
        labels = np.zeros(len(X))
        labels[int(len(X)*0.7):] = 1
        
        return X, labels, scaler
    
    def create_dataloaders(self, X, y, batch_size=32):
        """Create train/val/test loaders."""
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader

class SimpleAnomalyDetector(nn.Module):
    """Simple baseline model."""
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

if __name__ == '__main__':
    trainer = RCAEvalTrainer()
    
    print("=== Training on RCAEval Datasets ===\n")
    
    for dataset in ['RE1', 'RE2', 'RE3']:
        print(f"Training on {dataset}...")
        X, y, scaler = trainer.load_data(dataset)
        train_loader, val_loader, test_loader = trainer.create_dataloaders(X, y)
        
        model = SimpleAnomalyDetector(X.shape[1]).to(trainer.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        for epoch in range(10):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, trainer.device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, trainer.device)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, trainer.device)
        print(f"  Test Accuracy: {test_acc:.4f}\n")
