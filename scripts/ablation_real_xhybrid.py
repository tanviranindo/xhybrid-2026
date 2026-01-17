#!/usr/bin/env python3
"""
Ablation Study for Real xHybrid
Test individual components: Grammar, GNN, LSTM, and combinations
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from pathlib import Path
import json

# Import modules from train_real_xhybrid
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_real_xhybrid import GrammarModule, GNNModule, LSTMModule, AttentionFusion, SimpleDataset

# Variant models
class GrammarOnly(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.grammar = GrammarModule(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        feat = self.grammar(x)
        return self.classifier(feat), None

class GNNOnly(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.gnn = GNNModule(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        feat = self.gnn(x)
        return self.classifier(feat), None

class LSTMOnly(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.lstm = LSTMModule(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        feat = self.lstm(x)
        return self.classifier(feat), None

class GrammarGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.grammar = GrammarModule(input_dim, hidden_dim)
        self.gnn = GNNModule(input_dim, hidden_dim)
        self.fusion = AttentionFusion(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        g_feat = self.grammar(x)
        gnn_feat = self.gnn(x)
        fused, attn = self.fusion([g_feat, gnn_feat])
        return self.classifier(fused), attn

class GrammarLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.grammar = GrammarModule(input_dim, hidden_dim)
        self.lstm = LSTMModule(input_dim, hidden_dim)
        self.fusion = AttentionFusion(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        g_feat = self.grammar(x)
        l_feat = self.lstm(x)
        fused, attn = self.fusion([g_feat, l_feat])
        return self.classifier(fused), attn

class GNNLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.gnn = GNNModule(input_dim, hidden_dim)
        self.lstm = LSTMModule(input_dim, hidden_dim)
        self.fusion = AttentionFusion(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        gnn_feat = self.gnn(x)
        l_feat = self.lstm(x)
        fused, attn = self.fusion([gnn_feat, l_feat])
        return self.classifier(fused), attn

class FullXHybrid(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.grammar = GrammarModule(input_dim, hidden_dim)
        self.gnn = GNNModule(input_dim, hidden_dim)
        self.lstm = LSTMModule(input_dim, hidden_dim)
        self.fusion = AttentionFusion(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        g_feat = self.grammar(x)
        gnn_feat = self.gnn(x)
        l_feat = self.lstm(x)
        fused, attn = self.fusion([g_feat, gnn_feat, l_feat])
        return self.classifier(fused), attn

def train_and_evaluate(model, X_train, y_train, X_val, y_val, device, epochs=20):
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            logits, _ = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return f1, auc

def main():
    print("=" * 70)
    print("ABLATION STUDY: Real xHybrid Components")
    print("=" * 70)
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'processed'
    df = pd.read_parquet(data_path / 'traces_clean.parquet')
    df = df.sample(n=min(10000, len(df)), random_state=42)
    
    feature_cols = ['num_spans', 'avg_duration', 'total_duration', 'max_duration', 'num_services']
    X = df[feature_cols].fillna(0).values
    threshold = df['avg_duration'].quantile(0.7)
    y = (df['avg_duration'] > threshold).astype(int).values
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    variants = {
        'Grammar Only': GrammarOnly,
        'GNN Only': GNNOnly,
        'LSTM Only': LSTMOnly,
        'Grammar + GNN': GrammarGNN,
        'Grammar + LSTM': GrammarLSTM,
        'GNN + LSTM': GNNLSTM,
        'Full xHybrid': FullXHybrid
    }
    
    results = {name: {'f1': [], 'auc': []} for name in variants}
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\nFold {fold}/5")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        for name, ModelClass in variants.items():
            model = ModelClass(input_dim=5, hidden_dim=32).to(device)
            f1, auc = train_and_evaluate(model, X_train, y_train, X_val, y_val, device)
            results[name]['f1'].append(f1)
            results[name]['auc'].append(auc)
            print(f"  {name:20s}: F1={f1:.4f}, AUC={auc:.4f}")
    
    # Aggregate
    print(f"\n{'='*70}")
    print("FINAL ABLATION RESULTS")
    print(f"{'='*70}")
    
    summary = []
    for name in variants:
        f1_mean = np.mean(results[name]['f1'])
        f1_std = np.std(results[name]['f1'])
        auc_mean = np.mean(results[name]['auc'])
        auc_std = np.std(results[name]['auc'])
        
        print(f"{name:20s}: F1={f1_mean:.4f}±{f1_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
        
        summary.append({
            'variant': name,
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'auc_mean': auc_mean,
            'auc_std': auc_std
        })
    
    # Save
    output_dir = Path(__file__).parent.parent / 'results' / 'real_xhybrid'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    df_results = pd.DataFrame(summary)
    df_results.to_csv(output_dir / 'ablation_results.csv', index=False)
    
    print(f"\n✅ Results saved to {output_dir}")

if __name__ == "__main__":
    main()
