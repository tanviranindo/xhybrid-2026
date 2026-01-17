"""
Explainable AI Module
Provides interpretability for hybrid anomaly detection.
"""

from .explainer import (
    HybridExplainer,
    SHAPExplainer,
    AttentionExplainer,
    GrammarExplainer,
    ServiceLevelExplainer
)
from .visualization import (
    plot_shap_summary,
    plot_attention_heatmap,
    plot_grammar_activations,
    plot_service_contributions,
    create_explanation_report
)

__all__ = [
    'HybridExplainer',
    'SHAPExplainer',
    'AttentionExplainer',
    'GrammarExplainer',
    'ServiceLevelExplainer',
    'plot_shap_summary',
    'plot_attention_heatmap',
    'plot_grammar_activations',
    'plot_service_contributions',
    'create_explanation_report'
]
