"""Evaluate all models on RCAEval datasets."""
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from pathlib import Path
from rcaeval_trainer import RCAEvalTrainer, SimpleAnomalyDetector, train_epoch, evaluate
from advanced_models import LSTMDetector, AutoEncoder, HybridDetector

class ModelEvaluator:
    """Evaluate multiple models."""
    
    def __init__(self):
        self.trainer = RCAEvalTrainer(data_dir='../../data/rcaeval_processed')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
    
    def train_and_evaluate(self, model_class, model_name, dataset, epochs=20):
        """Train and evaluate a model."""
        X, y, _ = self.trainer.load_data(dataset)
        train_loader, val_loader, test_loader = self.trainer.create_dataloaders(X, y)
        
        model = model_class(X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, self.device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, self.device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, self.device)
        
        # Get predictions for detailed metrics
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                logits = model(X_batch)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0)
        }
        
        return metrics
    
    def evaluate_all(self):
        """Evaluate all models on all datasets."""
        models = [
            (SimpleAnomalyDetector, 'SimpleAnomalyDetector'),
            (LSTMDetector, 'LSTMDetector'),
            (HybridDetector, 'HybridDetector')
        ]
        
        datasets = ['RE1', 'RE2', 'RE3']
        
        print("=== Model Evaluation Results ===\n")
        
        for dataset in datasets:
            print(f"\n{dataset} Dataset:")
            print("-" * 70)
            print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
            print("-" * 70)
            
            for model_class, model_name in models:
                try:
                    metrics = self.train_and_evaluate(model_class, model_name, dataset)
                    print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
                    
                    key = f"{dataset}_{model_name}"
                    self.results[key] = metrics
                except Exception as e:
                    print(f"{model_name:<25} Error: {str(e)[:40]}")
        
        return self.results

if __name__ == '__main__':
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for dataset in ['RE1', 'RE2', 'RE3']:
        avg_acc = np.mean([results[f"{dataset}_{m}"]["accuracy"] for m in ["SimpleAnomalyDetector", "LSTMDetector", "HybridDetector"] if f"{dataset}_{m}" in results])
        print(f"{dataset}: Average Accuracy = {avg_acc:.4f}")
