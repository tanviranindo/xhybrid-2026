"""
Professional Publication-Quality Figures
Style: NeurIPS/ICML conference standard
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json

# ============================================================================
# PROFESSIONAL STYLE SETTINGS
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
})

# Load data
with open('results/complete_comparison/cv_results.json', 'r') as f:
    cv_results = json.load(f)
with open('results/real_xhybrid/ablation_results.json', 'r') as f:
    ablation_results = json.load(f)
with open('results/xai_analysis/shap_importance.json', 'r') as f:
    shap_results = json.load(f)

# Professional color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'success': '#3A7D44',
    'light': '#E8E8E8',
    'dark': '#333333'
}

print("="*60)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("="*60)

# ============================================================================
# 1. ARCHITECTURE DIAGRAM (Standard academic style)
# ============================================================================
print("\n[1/5] Architecture...")

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

# Standard rectangular boxes
def rect(x, y, w, h, text, color='white'):
    r = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=color)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold')

# Standard arrow
def arr(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Input
rect(0.5, 1.5, 1.2, 1, 'Trace\nFeatures', '#f0f0f0')

# Three parallel branches
rect(2.5, 2.8, 1.5, 0.8, 'Grammar', '#d4edda')
rect(2.5, 1.6, 1.5, 0.8, 'GNN', '#cce5ff')
rect(2.5, 0.4, 1.5, 0.8, 'LSTM', '#fff3cd')

# Arrows from input to branches
arr(1.7, 2.0, 2.5, 3.2)
arr(1.7, 2.0, 2.5, 2.0)
arr(1.7, 2.0, 2.5, 0.8)

# Fusion
rect(5.0, 1.5, 1.5, 1, 'Attention\nFusion', '#e2d5f1')

# Arrows to fusion
arr(4.0, 3.2, 5.0, 2.2)
arr(4.0, 2.0, 5.0, 2.0)
arr(4.0, 0.8, 5.0, 1.8)

# Classifier
rect(7.5, 1.5, 1.5, 1, 'Classifier', '#f8d7da')

# Arrow to classifier
arr(6.5, 2.0, 7.5, 2.0)

# Output arrow
arr(9.0, 2.0, 9.8, 2.0)
ax.text(9.9, 2.0, 'Output', fontsize=11, va='center')

# Title
ax.text(5, 3.8, 'xHybrid Architecture', fontsize=14, fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('results/complete_comparison/architecture_simple.pdf', bbox_inches='tight', dpi=300)
plt.savefig('submissions/task3a/figures/architecture_simple.pdf', bbox_inches='tight', dpi=300)
print("  ✓ Saved")
plt.close()

# ============================================================================
# 2. FAULT DISTRIBUTION (replaces PNG)
# ============================================================================
print("[2/5] Fault Distribution...")

fault_types = ['CPU', 'Memory', 'Disk', 'Network\nDelay', 'Packet\nLoss', 'Socket']
fault_counts = [95657, 95657, 95657, 95658, 95658, 95658]  # ~equal distribution

fig, ax = plt.subplots(figsize=(8, 4))
colors = plt.cm.Set2(np.linspace(0, 1, 6))
bars = ax.bar(fault_types, fault_counts, color=colors, edgecolor='#333333', linewidth=1.5)

for bar, count in zip(bars, fault_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500,
           f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Number of Traces', fontsize=12, fontweight='bold')
ax.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
ax.set_title('Fault Type Distribution in RE2-TT Dataset', fontsize=14, fontweight='bold', pad=10)
ax.set_ylim(0, 110000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/complete_comparison/fault_distribution.pdf', bbox_inches='tight', dpi=300)
plt.savefig('submissions/task3a/figures/fault_distribution.pdf', bbox_inches='tight', dpi=300)
print("  ✓ Saved")
plt.close()

# ============================================================================
# 3. PERFORMANCE COMPARISON
# ============================================================================
print("[3/5] Performance...")

# Get xHybrid from ablation results (Full xHybrid = 99.43%)
xhybrid_data = next(item for item in ablation_results if item['variant'] == 'Full xHybrid')

models_data = [
    ('Logistic Reg.', cv_results['logistic_regression']['f1_mean'], cv_results['logistic_regression']['f1_std']),
    ('xHybrid', xhybrid_data['f1_mean'], xhybrid_data['f1_std']),
    ('Random Forest', cv_results['random_forest']['f1_mean'], cv_results['random_forest']['f1_std']),
]
models_data.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(8, 5))

names = [m[0] for m in models_data]
scores = [m[1] for m in models_data]
stds = [m[2] for m in models_data]
colors = [COLORS['primary'], COLORS['secondary'], COLORS['success']]

bars = ax.barh(names, scores, xerr=stds, height=0.6, 
               color=colors, edgecolor='#333333', linewidth=2,
               capsize=5, error_kw={'linewidth': 2})

# Value labels
for i, (bar, score, std) in enumerate(zip(bars, scores, stds)):
    ax.text(score + std + 0.005, bar.get_y() + bar.get_height()/2,
           f'{score:.1%}', va='center', fontsize=13, fontweight='bold')

ax.set_xlabel('F1 Score', fontsize=14, fontweight='bold')
ax.set_xlim(0.93, 1.02)
ax.set_title('Model Performance (5-Fold CV)', fontsize=16, fontweight='bold', pad=15)

# Clean up
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/complete_comparison/performance_simple.pdf', bbox_inches='tight', dpi=300)
plt.savefig('submissions/task3a/figures/performance_simple.pdf', bbox_inches='tight', dpi=300)
print("  ✓ Saved")
plt.close()

# ============================================================================
# 4. ABLATION STUDY
# ============================================================================
print("[4/5] Ablation...")

ablation_data = [(item['variant'], item['f1_mean']) for item in ablation_results]
ablation_data.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(10, 5))

names = [d[0] for d in ablation_data]
scores = [d[1] for d in ablation_data]

# Color gradient
cmap = plt.cm.Blues
colors = [cmap(0.3 + 0.6 * i / len(names)) for i in range(len(names))]

bars = ax.barh(names, scores, height=0.65, color=colors, 
               edgecolor='#333333', linewidth=2)

# Highlight best
bars[-1].set_edgecolor(COLORS['quaternary'])
bars[-1].set_linewidth(3)

# Value labels
for bar, score in zip(bars, scores):
    ax.text(score + 0.001, bar.get_y() + bar.get_height()/2,
           f'{score:.2%}', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('F1 Score', fontsize=14, fontweight='bold')
ax.set_xlim(0.975, 1.0)
ax.set_title('Ablation Study: Component Contributions', fontsize=16, fontweight='bold', pad=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/complete_comparison/ablation_simple.pdf', bbox_inches='tight', dpi=300)
plt.savefig('submissions/task3a/figures/ablation_simple.pdf', bbox_inches='tight', dpi=300)
print("  ✓ Saved")
plt.close()

# ============================================================================
# 5. SHAP IMPORTANCE
# ============================================================================
print("[5/5] SHAP...")

shap_data = [(item['feature'].replace('_', ' ').title(), item['importance']) 
             for item in shap_results['Random Forest']]
shap_data.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(8, 5))

names = [d[0] for d in shap_data]
values = [d[1] for d in shap_data]

# Color by importance
cmap = plt.cm.RdYlBu_r
colors = [cmap(0.3 + 0.5 * v / max(values)) for v in values]

bars = ax.barh(names, values, height=0.6, color=colors,
               edgecolor='#333333', linewidth=2)

# Value labels
for bar, val in zip(bars, values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
           f'{val:.3f}', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('SHAP Importance', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(values) * 1.25)
ax.set_title('Feature Importance (Random Forest)', fontsize=16, fontweight='bold', pad=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/complete_comparison/shap_simple.pdf', bbox_inches='tight', dpi=300)
plt.savefig('submissions/task3a/figures/shap_simple.pdf', bbox_inches='tight', dpi=300)
print("  ✓ Saved")
plt.close()

print("\n" + "="*60)
print("ALL FIGURES COMPLETE ✓")
print("="*60)
