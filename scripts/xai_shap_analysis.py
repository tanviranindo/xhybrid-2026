#!/usr/bin/env python3
"""SHAP analysis for feature importance in anomaly detection models."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data/processed/traces_clean.parquet"
RESULTS_DIR = BASE_DIR / "results/xai_analysis"
RESULTS_DIR.mkdir(exist_ok=True)

# Feature names
FEATURE_NAMES = ['num_spans', 'avg_duration', 'total_duration', 'max_duration', 'num_services']

def load_data():
    """Load and prepare data."""
    df = pd.read_parquet(DATA_PATH)
    
    # Sample for faster computation
    df = df.sample(n=min(10000, len(df)), random_state=42)
    
    X = df[FEATURE_NAMES].fillna(0).values
    
    # Create labels based on duration threshold (same as CV experiments)
    threshold = df['avg_duration'].quantile(0.7)
    y = (df['avg_duration'] > threshold).astype(int).values
    
    return X, y

def train_models(X_train, y_train):
    """Train models for SHAP analysis."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_scaled, y_train)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y_train)
    
    return {'rf': rf, 'lr': lr}, scaler

def compute_shap_values(models, X, scaler):
    """Compute SHAP values for all models."""
    X_scaled = scaler.transform(X)
    shap_values = {}
    
    # Random Forest
    print("Computing SHAP for Random Forest...")
    explainer_rf = shap.TreeExplainer(models['rf'])
    shap_values['rf'] = explainer_rf.shap_values(X_scaled)
    
    # Logistic Regression
    print("Computing SHAP for Logistic Regression...")
    explainer_lr = shap.LinearExplainer(models['lr'], X_scaled)
    shap_values['lr'] = explainer_lr.shap_values(X_scaled)
    
    return shap_values

def plot_shap_summary(shap_values, X, model_name):
    """Create SHAP summary plot."""
    plt.figure(figsize=(10, 6))
    
    # For binary classification, use class 1 (anomaly)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    shap.summary_plot(shap_vals, X, feature_names=FEATURE_NAMES, show=False)
    plt.title(f'SHAP Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'shap_summary_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def compute_feature_importance(shap_values, model_name):
    """Compute mean absolute SHAP values for feature importance."""
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    importance = np.abs(shap_vals).mean(axis=0)
    
    # Flatten if needed
    if importance.ndim > 1:
        importance = importance.flatten()
    
    results = []
    for i, feat in enumerate(FEATURE_NAMES):
        results.append({
            'feature': feat,
            'importance': float(importance[i]),
            'rank': 0
        })
    
    # Sort by importance
    results = sorted(results, key=lambda x: x['importance'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1
    
    return results

def main():
    print("Loading data...")
    X, y = load_data()
    
    # Use stratified sampling to ensure both classes
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    print("Training models...")
    models, scaler = train_models(X_train, y_train)
    
    print("Computing SHAP values...")
    shap_values = compute_shap_values(models, X_test, scaler)
    
    # Generate plots and importance rankings
    all_results = {}
    
    for model_key, model_name in [('rf', 'Random Forest'), ('lr', 'Logistic Regression')]:
        print(f"\nProcessing {model_name}...")
        
        # Plot
        plot_shap_summary(shap_values[model_key], scaler.transform(X_test), model_name)
        
        # Compute importance
        importance = compute_feature_importance(shap_values[model_key], model_name)
        all_results[model_name] = importance
        
        print(f"\n{model_name} Feature Importance:")
        for item in importance:
            print(f"  {item['rank']}. {item['feature']}: {item['importance']:.4f}")
    
    # Save results
    with open(RESULTS_DIR / 'shap_importance.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison table
    df_results = []
    for model_name, importance in all_results.items():
        for item in importance:
            df_results.append({
                'Model': model_name,
                'Feature': item['feature'],
                'Importance': item['importance'],
                'Rank': item['rank']
            })
    
    df = pd.DataFrame(df_results)
    df.to_csv(RESULTS_DIR / 'shap_importance.csv', index=False)
    
    print(f"\n✅ SHAP analysis complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"  - shap_summary_random_forest.png")
    print(f"  - shap_summary_logistic_regression.png")
    print(f"  - shap_importance.json")
    print(f"  - shap_importance.csv")

if __name__ == "__main__":
    main()
