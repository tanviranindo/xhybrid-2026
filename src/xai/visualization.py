"""
Visualization Module for XAI
Creates visual explanations for anomaly detection.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_shap_summary(shap_values: np.ndarray,
                      feature_names: List[str],
                      max_features: int = 20,
                      output_path: Optional[str] = None,
                      title: str = "SHAP Feature Importance") -> Optional[plt.Figure]:
    """
    Plot SHAP summary showing feature importance.

    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: List of feature names
        max_features: Maximum features to display
        output_path: Path to save figure (SVG format)
        title: Plot title

    Returns:
        Matplotlib figure if matplotlib is available
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization")
        return None

    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Sort and select top features
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_features]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, max_features * 0.3)))

    # Plot horizontal bar chart
    y_pos = np.arange(len(sorted_idx))
    values = mean_abs_shap[sorted_idx]
    names = [feature_names[i] for i in sorted_idx]

    colors = plt.cm.RdYlBu_r(values / values.max())
    ax.barh(y_pos, values, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)

    return fig


def plot_attention_heatmap(attention_weights: np.ndarray,
                           x_labels: Optional[List[str]] = None,
                           y_labels: Optional[List[str]] = None,
                           output_path: Optional[str] = None,
                           title: str = "Attention Weights",
                           cmap: str = "YlOrRd") -> Optional[plt.Figure]:
    """
    Plot attention weights as heatmap.

    Args:
        attention_weights: 2D attention matrix
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        output_path: Path to save figure
        title: Plot title
        cmap: Colormap name

    Returns:
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    if HAS_SEABORN:
        sns.heatmap(
            attention_weights,
            xticklabels=x_labels if x_labels else False,
            yticklabels=y_labels if y_labels else False,
            cmap=cmap,
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
    else:
        im = ax.imshow(attention_weights, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Attention Weight')

        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)

    return fig


def plot_grammar_activations(activations: List[Dict],
                             output_path: Optional[str] = None,
                             max_patterns: int = 15,
                             title: str = "Grammar Pattern Activations") -> Optional[plt.Figure]:
    """
    Plot grammar rule activations.

    Args:
        activations: List of activation dictionaries
        output_path: Path to save figure
        max_patterns: Maximum patterns to show
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        return None

    # Filter and sort
    activations = activations[:max_patterns]

    fig, ax = plt.subplots(figsize=(12, max(6, len(activations) * 0.4)))

    patterns = [a['pattern'][:40] + '...' if len(a['pattern']) > 40 else a['pattern']
                for a in activations]
    supports = [a['support'] for a in activations]
    is_active = [a['is_active'] for a in activations]
    deviations = [a.get('deviation', 'normal') for a in activations]

    y_pos = np.arange(len(patterns))

    # Color based on deviation type
    colors = []
    for i, (active, dev) in enumerate(zip(is_active, deviations)):
        if dev == 'unexpected':
            colors.append('#ff6b6b')  # Red for unexpected
        elif dev == 'missing':
            colors.append('#ffd93d')  # Yellow for missing
        elif active:
            colors.append('#6bcb77')  # Green for active normal
        else:
            colors.append('#cccccc')  # Gray for inactive

    ax.barh(y_pos, supports, color=colors, edgecolor='black', linewidth=0.5)

    # Add activation indicators
    for i, active in enumerate(is_active):
        marker = '✓' if active else '✗'
        ax.text(supports[i] + 0.02, i, marker, va='center', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(patterns, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Pattern Support')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#6bcb77', label='Active (Normal)'),
        mpatches.Patch(color='#ff6b6b', label='Unexpected'),
        mpatches.Patch(color='#ffd93d', label='Missing'),
        mpatches.Patch(color='#cccccc', label='Inactive')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)

    return fig


def plot_service_contributions(service_importance: Dict[str, float],
                               output_path: Optional[str] = None,
                               max_services: int = 15,
                               title: str = "Service Contributions to Anomaly") -> Optional[plt.Figure]:
    """
    Plot per-service contribution to anomaly detection.

    Args:
        service_importance: Dictionary mapping service names to importance scores
        output_path: Path to save figure
        max_services: Maximum services to show
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        return None

    # Sort and select top services
    sorted_services = sorted(service_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_services = sorted_services[:max_services]

    services = [s[0] for s in sorted_services]
    importance = [s[1] for s in sorted_services]

    fig, ax = plt.subplots(figsize=(10, max(6, len(services) * 0.4)))

    y_pos = np.arange(len(services))

    # Color gradient based on importance
    colors = plt.cm.Reds(np.array(importance) / max(importance) * 0.8 + 0.2)

    ax.barh(y_pos, importance, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(services)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)

    return fig


def plot_component_contributions(contributions: Dict[str, float],
                                 output_path: Optional[str] = None,
                                 title: str = "Component Contributions") -> Optional[plt.Figure]:
    """
    Plot pie chart of component contributions.

    Args:
        contributions: Dictionary mapping component names to contribution values
        output_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    labels = list(contributions.keys())
    values = list(contributions.values())

    # Colors for each component
    color_map = {
        'syntactic': '#ff6b6b',
        'graph': '#4ecdc4',
        'temporal': '#45b7d1'
    }
    colors = [color_map.get(l, '#cccccc') for l in labels]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.05] * len(values),
        startangle=90
    )

    # Style
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)

    return fig


def create_explanation_report(explanations: Dict[str, Any],
                              output_dir: str,
                              trace_id: str = "unknown") -> Dict[str, str]:
    """
    Create a complete visual explanation report.

    Args:
        explanations: Dictionary of explanation objects
        output_dir: Directory to save figures
        trace_id: Trace identifier

    Returns:
        Dictionary mapping explanation types to saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # SHAP summary
    if 'shap' in explanations:
        exp = explanations['shap']
        if hasattr(exp, 'raw_shap_values') and exp.raw_shap_values is not None:
            path = output_dir / f"{trace_id}_shap_summary.svg"
            # Note: Would need feature names stored
            saved_files['shap'] = str(path)

    # Attention heatmap
    if 'attention' in explanations:
        exp = explanations['attention']
        if exp.attention_weights:
            for name, weights in exp.attention_weights.items():
                if weights is not None and isinstance(weights, np.ndarray):
                    path = output_dir / f"{trace_id}_attention_{name}.svg"
                    plot_attention_heatmap(
                        weights if len(weights.shape) == 2 else weights[0],
                        output_path=str(path),
                        title=f"{name.title()} Attention"
                    )
                    saved_files[f'attention_{name}'] = str(path)

    # Grammar activations
    if 'grammar' in explanations:
        exp = explanations['grammar']
        if exp.grammar_activations:
            path = output_dir / f"{trace_id}_grammar_activations.svg"
            plot_grammar_activations(
                exp.grammar_activations,
                output_path=str(path)
            )
            saved_files['grammar'] = str(path)

    # Service contributions
    if 'service' in explanations:
        exp = explanations['service']
        if exp.service_importance:
            path = output_dir / f"{trace_id}_service_contributions.svg"
            plot_service_contributions(
                exp.service_importance,
                output_path=str(path)
            )
            saved_files['service'] = str(path)

    # Component contributions (from any explanation that has them)
    for exp in explanations.values():
        if hasattr(exp, 'component_contributions') and exp.component_contributions:
            path = output_dir / f"{trace_id}_components.svg"
            plot_component_contributions(
                exp.component_contributions,
                output_path=str(path)
            )
            saved_files['components'] = str(path)
            break

    return saved_files


def plot_trace_timeline(spans: List[Dict],
                        service_importance: Optional[Dict[str, float]] = None,
                        output_path: Optional[str] = None,
                        title: str = "Trace Timeline") -> Optional[plt.Figure]:
    """
    Plot trace spans as timeline with importance highlighting.

    Args:
        spans: List of span dictionaries with start_time, duration, service_name
        service_importance: Optional importance scores for highlighting
        output_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        return None

    # Sort spans by start time
    spans = sorted(spans, key=lambda x: x.get('start_time', 0))

    # Get unique services for y-axis
    services = list(set(s.get('service_name', '') for s in spans))
    service_to_y = {s: i for i, s in enumerate(services)}

    fig, ax = plt.subplots(figsize=(14, max(4, len(services) * 0.5)))

    # Get time range
    if spans:
        min_time = min(s.get('start_time', 0) for s in spans)
        max_time = max(s.get('start_time', 0) + s.get('duration', 0) for s in spans)
    else:
        min_time, max_time = 0, 1

    # Plot each span
    for span in spans:
        service = span.get('service_name', '')
        start = span.get('start_time', 0) - min_time
        duration = span.get('duration', 0)
        y = service_to_y.get(service, 0)

        # Color based on importance
        if service_importance:
            importance = service_importance.get(service, 0)
            color = plt.cm.Reds(importance * 0.8 + 0.2)
        else:
            color = 'steelblue'

        ax.barh(y, duration, left=start, height=0.6, color=color,
                edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(services)))
    ax.set_yticklabels(services)
    ax.set_xlabel('Time (μs)')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)

    return fig


def plot_anomaly_comparison(normal_patterns: Dict[str, float],
                            anomaly_patterns: Dict[str, float],
                            output_path: Optional[str] = None,
                            title: str = "Normal vs Anomaly Pattern Comparison") -> Optional[plt.Figure]:
    """
    Compare pattern frequencies between normal and anomaly traces.

    Args:
        normal_patterns: Pattern frequencies in normal traces
        anomaly_patterns: Pattern frequencies in anomaly trace
        output_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        return None

    # Get all patterns
    all_patterns = set(normal_patterns.keys()) | set(anomaly_patterns.keys())
    patterns = sorted(all_patterns)[:20]  # Limit to top 20

    normal_vals = [normal_patterns.get(p, 0) for p in patterns]
    anomaly_vals = [anomaly_patterns.get(p, 0) for p in patterns]

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(patterns))
    width = 0.35

    ax.bar(x - width/2, normal_vals, width, label='Normal', color='#6bcb77')
    ax.bar(x + width/2, anomaly_vals, width, label='Anomaly', color='#ff6b6b')

    ax.set_xlabel('Pattern')
    ax.set_ylabel('Frequency')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)

    return fig
