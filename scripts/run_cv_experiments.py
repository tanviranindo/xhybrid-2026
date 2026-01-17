#!/usr/bin/env python3
"""
5-Fold Cross-Validation for All Models
Final rigorous evaluation with statistical significance

CSE713 - Pattern Recognition (Fall 2025)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class SimpleHybridModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        return self.fusion(torch.cat([self.branch1(x), self.branch2(x)], dim=1))


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_hybrid(X_train, y_train, X_test, y_test, epochs=30):
    """Train hybrid model for one fold."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleHybridModel().to(device)
    
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Train
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate sklearn model."""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_score) if len(set(y_test)) > 1 else 0.5
    }


def main():
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION - ALL MODELS")
    print("=" * 70)
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'processed'
    df = pd.read_parquet(data_path / 'traces_clean.parquet')
    df = df.sample(n=min(10000, len(df)), random_state=42)
    
    feature_cols = ['num_spans', 'avg_duration', 'total_duration', 'max_duration', 'num_services']
    X = df[feature_cols].fillna(0).values
    threshold = df['avg_duration'].quantile(0.7)
    y = (df['avg_duration'] > threshold).astype(int).values
    
    print(f"\nData: {len(X)} samples, {X.shape[1]} features")
    print(f"Anomaly rate: {y.mean():.1%}")
    
    # 5-fold CV
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        'hybrid': [],
        'logistic_regression': [],
        'random_forest': [],
        'isolation_forest': []
    }
    
    print("\n" + "=" * 70)
    print("Running 5-Fold Cross-Validation...")
    print("=" * 70)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\nFold {fold}/5:")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Hybrid Model
        print("  Training Hybrid Model...", end=" ")
        metrics = train_hybrid(X_train_scaled, y_train, X_test_scaled, y_test)
        results['hybrid'].append(metrics)
        print(f"F1={metrics['f1']:.4f}")
        
        # 2. Logistic Regression
        print("  Training Logistic Regression...", end=" ")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        metrics = evaluate_model(lr, X_test_scaled, y_test)
        results['logistic_regression'].append(metrics)
        print(f"F1={metrics['f1']:.4f}")
        
        # 3. Random Forest
        print("  Training Random Forest...", end=" ")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train_scaled, y_train)
        metrics = evaluate_model(rf, X_test_scaled, y_test)
        results['random_forest'].append(metrics)
        print(f"F1={metrics['f1']:.4f}")
        
        # 4. Isolation Forest
        print("  Training Isolation Forest...", end=" ")
        iso = IsolationForest(contamination=y_train.mean(), random_state=42)
        iso.fit(X_train_scaled)
        y_pred = (iso.predict(X_test_scaled) == -1).astype(int)
        y_score = -iso.score_samples(X_test_scaled)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_score) if len(set(y_test)) > 1 else 0.5
        }
        results['isolation_forest'].append(metrics)
        print(f"F1={metrics['f1']:.4f}")
    
    # Compute statistics
    print("\n" + "=" * 70)
    print("FINAL RESULTS - 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    summary = {}
    for model_name, fold_results in results.items():
        f1_scores = [r['f1'] for r in fold_results]
        auc_scores = [r['roc_auc'] for r in fold_results]
        
        summary[model_name] = {
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'fold_results': fold_results
        }
        
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print(f"  F1 Score:  {summary[model_name]['f1_mean']:.4f} ± {summary[model_name]['f1_std']:.4f}")
        print(f"  ROC-AUC:   {summary[model_name]['auc_mean']:.4f} ± {summary[model_name]['auc_std']:.4f}")
    
    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'complete_comparison'
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'cv_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': [k.replace('_', ' ').title() for k in summary.keys()],
        'F1 Mean': [v['f1_mean'] for v in summary.values()],
        'F1 Std': [v['f1_std'] for v in summary.values()],
        'AUC Mean': [v['auc_mean'] for v in summary.values()],
        'AUC Std': [v['auc_std'] for v in summary.values()]
    })
    comparison_df = comparison_df.sort_values('F1 Mean', ascending=False)
    comparison_df.to_csv(output_path / 'comparison_table.csv', index=False)
    
    print("\n" + "=" * 70)
    print("RANKING:")
    print("=" * 70)
    for i, row in comparison_df.iterrows():
        print(f"{i+1}. {row['Model']}: F1={row['F1 Mean']:.4f}±{row['F1 Std']:.4f}")
    
    print(f"\n✅ Results saved to: {output_path}")
    print("   - cv_results.json")
    print("   - comparison_table.csv")
    
    return summary


if __name__ == "__main__":
    results = main()
