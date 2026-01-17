"""
Trainer Module
Main training loop for hybrid anomaly detection.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..models.classifier import FocalLoss
from .evaluator import Evaluator


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic settings
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 0.00001

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 0.001
    monitor: str = "val_f1"

    # Loss
    loss_type: str = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None

    # Gradient
    gradient_clip: float = 1.0

    # Device
    device: str = "cuda"

    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_interval: int = 10
    save_best: bool = True


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Main trainer class for hybrid anomaly detection.
    """

    def __init__(self,
                 model: nn.Module,
                 config: TrainingConfig,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 grammar_extractor=None):
        """
        Args:
            model: The anomaly detection model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            grammar_extractor: Grammar feature extractor
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.grammar_extractor = grammar_extractor

        # Setup device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Setup loss function
        if config.loss_type == "focal":
            self.criterion = FocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma
            )
        else:
            weight = None
            if config.class_weights is not None:
                weight = torch.tensor(config.class_weights, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight)

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Setup scheduler
        if config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs - config.warmup_epochs,
                eta_min=config.min_lr
            )
        elif config.scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif config.scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=config.min_lr
            )
        else:
            self.scheduler = None

        # Early stopping
        if config.early_stopping:
            self.early_stopper = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta,
                mode='max'
            )
        else:
            self.early_stopper = None

        # Evaluator
        self.evaluator = Evaluator()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': []
        }

        # Best model tracking
        self.best_val_score = 0.0
        self.best_epoch = 0

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _warmup_lr(self, epoch: int):
        """Linear warmup of learning rate."""
        if epoch < self.config.warmup_epochs:
            warmup_factor = (epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * warmup_factor

    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """
        Train for one epoch.

        Returns:
            (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            sequences = batch['sequence'].to(self.device)
            latencies = batch['latencies'].to(self.device)
            labels = batch['label'].to(self.device)

            # Get grammar features if available
            if 'grammar_features' in batch:
                grammar_features = batch['grammar_features'].to(self.device)
            else:
                # Placeholder zeros
                grammar_features = torch.zeros(
                    sequences.size(0), 100, device=self.device
                )

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(sequences, latencies, grammar_features)

            # Loss
            loss = self.criterion(logits, labels.long())

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            # Track
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Log
            if (batch_idx + 1) % self.config.log_interval == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        metrics = self.evaluator.compute_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )

        return avg_loss, metrics

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """
        Validate the model.

        Returns:
            (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        for batch in self.val_loader:
            sequences = batch['sequence'].to(self.device)
            latencies = batch['latencies'].to(self.device)
            labels = batch['label'].to(self.device)

            if 'grammar_features' in batch:
                grammar_features = batch['grammar_features'].to(self.device)
            else:
                grammar_features = torch.zeros(
                    sequences.size(0), 100, device=self.device
                )

            logits = self.model(sequences, latencies, grammar_features)
            loss = self.criterion(logits, labels.long())

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        metrics = self.evaluator.compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        return avg_loss, metrics

    def train(self) -> Dict:
        """
        Full training loop.

        Returns:
            Training history
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Warmup
            self._warmup_lr(epoch)

            # Train
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate()

            # Update scheduler
            if self.scheduler is not None and epoch >= self.config.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log
            epoch_time = time.time() - epoch_start
            print(f"  Train Loss: {train_loss:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"AUC-ROC: {val_metrics.get('auc_roc', 0):.4f}")
            print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['lr'].append(current_lr)

            # Save best model
            val_score = val_metrics[self.config.monitor.replace('val_', '')]
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.best_epoch = epoch + 1
                if self.config.save_best:
                    self.save_checkpoint('best_model.pt')
                print(f"  New best model! {self.config.monitor}: {val_score:.4f}")

            # Early stopping
            if self.early_stopper is not None:
                if self.early_stopper(val_score):
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.1f} minutes")
        print(f"Best {self.config.monitor}: {self.best_val_score:.4f} at epoch {self.best_epoch}")

        # Save final model and history
        self.save_checkpoint('final_model.pt')
        self.save_history()

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_score': self.best_val_score,
            'best_epoch': self.best_epoch
        }

        path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_score = checkpoint.get('best_val_score', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)

    def save_history(self):
        """Save training history."""
        path = Path(self.config.output_dir) / 'training_history.json'

        # Convert numpy types for JSON serialization
        history_json = {}
        for key, values in self.history.items():
            if key.endswith('_metrics'):
                history_json[key] = [
                    {k: float(v) for k, v in m.items()}
                    for m in values
                ]
            else:
                history_json[key] = [float(v) for v in values]

        with open(path, 'w') as f:
            json.dump(history_json, f, indent=2)

    @torch.no_grad()
    def test(self) -> Dict:
        """
        Evaluate on test set.

        Returns:
            Test metrics
        """
        if self.test_loader is None:
            print("No test loader provided")
            return {}

        # Load best model
        self.load_checkpoint('best_model.pt')

        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        for batch in self.test_loader:
            sequences = batch['sequence'].to(self.device)
            latencies = batch['latencies'].to(self.device)
            labels = batch['label'].to(self.device)

            if 'grammar_features' in batch:
                grammar_features = batch['grammar_features'].to(self.device)
            else:
                grammar_features = torch.zeros(
                    sequences.size(0), 100, device=self.device
                )

            logits = self.model(sequences, latencies, grammar_features)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        metrics = self.evaluator.compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        print("\nTest Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return metrics


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
