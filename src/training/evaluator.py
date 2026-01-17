"""
Evaluator Module
Metrics computation and evaluation utilities.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    # AUC metrics require probabilities
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.0

        try:
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auc_pr'] = 0.0

    return metrics


class Evaluator:
    """
    Evaluator class for comprehensive model evaluation.
    """

    def __init__(self):
        self.history = []

    def compute_metrics(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute and store metrics."""
        metrics = compute_metrics(y_true, y_pred, y_prob)
        self.history.append(metrics)
        return metrics

    def get_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    def get_classification_report(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   target_names: List[str] = None) -> str:
        """Get detailed classification report."""
        if target_names is None:
            target_names = ['Normal', 'Anomaly']
        return classification_report(y_true, y_pred, target_names=target_names)

    def get_roc_curve(self,
                      y_true: np.ndarray,
                      y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data."""
        return roc_curve(y_true, y_prob)

    def get_pr_curve(self,
                     y_true: np.ndarray,
                     y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get Precision-Recall curve data."""
        return precision_recall_curve(y_true, y_prob)

    def find_optimal_threshold(self,
                                y_true: np.ndarray,
                                y_prob: np.ndarray,
                                metric: str = 'f1') -> Tuple[float, float]:
        """
        Find optimal classification threshold.

        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')

        Returns:
            (optimal_threshold, metric_at_threshold)
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_metric = 0.0

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_metric:
                best_metric = score
                best_threshold = threshold

        return best_threshold, best_metric


class CrossValidator:
    """
    Cross-validation evaluator.
    """

    def __init__(self,
                 model_class,
                 model_kwargs: Dict,
                 n_folds: int = 5,
                 stratified: bool = True,
                 random_state: int = 42):
        """
        Args:
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model
            n_folds: Number of folds
            stratified: Use stratified sampling
            random_state: Random seed
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.n_folds = n_folds
        self.stratified = stratified
        self.random_state = random_state

        self.fold_results = []

    def cross_validate(self,
                       dataset,
                       labels: np.ndarray,
                       train_fn,
                       eval_fn,
                       device: str = 'cuda') -> Dict[str, Tuple[float, float]]:
        """
        Perform k-fold cross-validation.

        Args:
            dataset: Full dataset
            labels: Labels for stratification
            train_fn: Function to train model (model, train_loader, val_loader) -> None
            eval_fn: Function to evaluate model (model, loader) -> metrics_dict
            device: Device to use

        Returns:
            Dictionary of (mean, std) for each metric
        """
        if self.stratified:
            kfold = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            )
            splits = kfold.split(range(len(dataset)), labels)
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            )
            splits = kfold.split(range(len(dataset)))

        self.fold_results = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{self.n_folds}")
            print(f"{'='*50}")

            # Create data loaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=32, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_subset, batch_size=32, shuffle=False, num_workers=0
            )

            # Create fresh model
            model = self.model_class(**self.model_kwargs)
            model.to(device)

            # Train
            train_fn(model, train_loader, val_loader)

            # Evaluate
            metrics = eval_fn(model, val_loader)
            self.fold_results.append(metrics)

            print(f"Fold {fold + 1} Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        # Aggregate results
        aggregated = {}
        for metric in self.fold_results[0].keys():
            values = [r[metric] for r in self.fold_results]
            aggregated[metric] = (np.mean(values), np.std(values))

        print(f"\n{'='*50}")
        print("Cross-Validation Summary:")
        print(f"{'='*50}")
        for metric, (mean, std) in aggregated.items():
            print(f"  {metric}: {mean:.4f} (+/- {std:.4f})")

        return aggregated


class AblationStudy:
    """
    Ablation study evaluator.
    """

    def __init__(self,
                 base_model_class,
                 base_model_kwargs: Dict):
        """
        Args:
            base_model_class: Base model class
            base_model_kwargs: Base model configuration
        """
        self.base_model_class = base_model_class
        self.base_model_kwargs = base_model_kwargs.copy()
        self.results = {}

    def run_variant(self,
                    variant_name: str,
                    variant_kwargs: Dict,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    train_fn,
                    eval_fn,
                    device: str = 'cuda') -> Dict[str, float]:
        """
        Run a single ablation variant.

        Args:
            variant_name: Name of this variant
            variant_kwargs: Modified kwargs for variant
            train_loader: Training data loader
            val_loader: Validation data loader
            train_fn: Training function
            eval_fn: Evaluation function
            device: Device

        Returns:
            Metrics for this variant
        """
        print(f"\nRunning ablation: {variant_name}")

        # Merge with base kwargs
        kwargs = self.base_model_kwargs.copy()
        kwargs.update(variant_kwargs)

        # Create model
        model = self.base_model_class(**kwargs)
        model.to(device)

        # Train
        train_fn(model, train_loader, val_loader)

        # Evaluate
        metrics = eval_fn(model, val_loader)
        self.results[variant_name] = metrics

        return metrics

    def get_summary(self) -> str:
        """Get formatted summary of ablation results."""
        if not self.results:
            return "No results yet."

        lines = ["Ablation Study Results", "=" * 60]

        # Get all metrics
        metrics = list(next(iter(self.results.values())).keys())

        # Header
        header = f"{'Variant':<25}"
        for m in metrics:
            header += f"{m:>12}"
        lines.append(header)
        lines.append("-" * 60)

        # Results
        for variant, result in self.results.items():
            row = f"{variant:<25}"
            for m in metrics:
                row += f"{result[m]:>12.4f}"
            lines.append(row)

        return "\n".join(lines)


class BaselineComparator:
    """
    Compare against baseline methods.
    """

    def __init__(self):
        self.results = {}

    def add_baseline(self, name: str, metrics: Dict[str, float]):
        """Add baseline result."""
        self.results[name] = metrics

    def compare(self, model_name: str, model_metrics: Dict[str, float]) -> str:
        """
        Compare model against baselines.

        Returns:
            Formatted comparison string
        """
        all_results = {model_name: model_metrics, **self.results}

        lines = ["Model Comparison", "=" * 80]

        # Get all metrics
        metrics = list(model_metrics.keys())

        # Header
        header = f"{'Model':<25}"
        for m in metrics:
            header += f"{m:>12}"
        lines.append(header)
        lines.append("-" * 80)

        # Find best for each metric
        best = {}
        for m in metrics:
            values = [(name, r.get(m, 0)) for name, r in all_results.items()]
            best[m] = max(values, key=lambda x: x[1])[0]

        # Results
        for name, result in all_results.items():
            row = f"{name:<25}"
            for m in metrics:
                val = result.get(m, 0)
                if name == best[m]:
                    row += f"  *{val:>9.4f}"  # Mark best
                else:
                    row += f"{val:>12.4f}"
            lines.append(row)

        lines.append("")
        lines.append("* indicates best performance for that metric")

        return "\n".join(lines)

    def run_isolation_forest(self,
                              X_train: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              contamination: float = 0.1) -> Dict[str, float]:
        """Run Isolation Forest baseline."""
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        clf.fit(X_train)

        # Predict (-1 for anomaly, 1 for normal)
        y_pred = clf.predict(X_test)
        y_pred = (y_pred == -1).astype(int)  # Convert to 0/1

        # Scores for AUC
        y_scores = -clf.decision_function(X_test)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

        metrics = compute_metrics(y_test, y_pred, y_scores)
        self.results['Isolation Forest'] = metrics

        return metrics

    def run_one_class_svm(self,
                          X_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray) -> Dict[str, float]:
        """Run One-Class SVM baseline."""
        from sklearn.svm import OneClassSVM

        clf = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        clf.fit(X_train)

        y_pred = clf.predict(X_test)
        y_pred = (y_pred == -1).astype(int)

        y_scores = -clf.decision_function(X_test)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)

        metrics = compute_metrics(y_test, y_pred, y_scores)
        self.results['One-Class SVM'] = metrics

        return metrics
