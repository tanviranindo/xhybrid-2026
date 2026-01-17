"""
Training Utilities
Helper functions and classes for training.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import logging


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = 'max',
                 restore_best: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' to minimize
            restore_best: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self,
                 score: float,
                 model: Optional[nn.Module] = None) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value
            model: Model to save state from

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best:
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best:
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_weights(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None


class LRScheduler:
    """
    Learning rate scheduler wrapper with warmup support.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 scheduler_type: str = 'cosine',
                 total_epochs: int = 100,
                 warmup_epochs: int = 5,
                 min_lr: float = 1e-6,
                 **kwargs):
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: 'cosine', 'step', 'plateau', or 'linear'
            total_epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs
            min_lr: Minimum learning rate
            **kwargs: Additional scheduler arguments
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

        # Create scheduler
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'max'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                min_lr=min_lr
            )
        elif scheduler_type == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=min_lr / self.base_lr,
                total_iters=total_epochs - warmup_epochs
            )
        else:
            self.scheduler = None

        self.current_epoch = 0

    def step(self, metric: Optional[float] = None):
        """
        Update learning rate.

        Args:
            metric: Metric value for ReduceLROnPlateau
        """
        self.current_epoch += 1

        # Warmup phase
        if self.current_epoch <= self.warmup_epochs:
            warmup_factor = self.current_epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * warmup_factor
            return

        # Regular scheduling
        if self.scheduler is not None:
            if self.scheduler_type == 'plateau':
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class CheckpointManager:
    """
    Manages model checkpoints.
    """

    def __init__(self,
                 checkpoint_dir: str,
                 max_checkpoints: int = 5,
                 save_best_only: bool = True):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save when metric improves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only

        self.best_score = None
        self.checkpoint_files = []

    def save(self,
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             epoch: int,
             score: float,
             filename: Optional[str] = None,
             extra_info: Optional[Dict] = None) -> Optional[str]:
        """
        Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            score: Current metric score
            filename: Custom filename (optional)
            extra_info: Additional info to save

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        # Check if should save
        if self.save_best_only:
            if self.best_score is not None and score <= self.best_score:
                return None
            self.best_score = score

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score
        }

        if extra_info:
            checkpoint.update(extra_info)

        # Generate filename
        if filename is None:
            filename = f"checkpoint_epoch{epoch}_score{score:.4f}.pt"

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        # Track and cleanup old checkpoints
        self.checkpoint_files.append(filepath)
        self._cleanup_old_checkpoints()

        return str(filepath)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding max limit."""
        while len(self.checkpoint_files) > self.max_checkpoints:
            old_file = self.checkpoint_files.pop(0)
            if old_file.exists():
                old_file.unlink()

    def load(self,
             filepath: str,
             model: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             device: str = 'cuda') -> Dict:
        """
        Load checkpoint.

        Args:
            filepath: Path to checkpoint
            model: Model to load into
            optimizer: Optimizer to load into (optional)
            device: Device to load to

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if not self.checkpoint_files:
            return None
        return str(self.checkpoint_files[-1])


class MetricTracker:
    """
    Tracks metrics during training.
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], epoch: int):
        """
        Update metrics.

        Args:
            metrics: Dictionary of metric values
            epoch: Current epoch
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
                self.best_metrics[name] = value
                self.best_epoch[name] = epoch

            self.metrics[name].append(value)

            # Track best (assuming higher is better)
            if value > self.best_metrics[name]:
                self.best_metrics[name] = value
                self.best_epoch[name] = epoch

    def get_history(self, metric_name: str) -> List[float]:
        """Get history for a metric."""
        return self.metrics.get(metric_name, [])

    def get_best(self, metric_name: str) -> Optional[float]:
        """Get best value for a metric."""
        return self.best_metrics.get(metric_name)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        for name in self.metrics:
            summary[name] = {
                'last': self.metrics[name][-1] if self.metrics[name] else None,
                'best': self.best_metrics[name],
                'best_epoch': self.best_epoch[name],
                'mean': np.mean(self.metrics[name]),
                'std': np.std(self.metrics[name])
            }
        return summary

    def save(self, filepath: str):
        """Save metrics to JSON file."""
        data = {
            'metrics': self.metrics,
            'best_metrics': self.best_metrics,
            'best_epoch': self.best_epoch
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.metrics = data['metrics']
        self.best_metrics = data['best_metrics']
        self.best_epoch = data['best_epoch']


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name: Logger name
        log_file: Optional file to write logs to

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively move data to device.

    Args:
        data: Data to move (tensor, dict, list, etc.)
        device: Target device

    Returns:
        Data on device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data
