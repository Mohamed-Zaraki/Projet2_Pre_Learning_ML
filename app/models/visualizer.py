"""
visualizer.py — Generates base64-encoded PNG plots from experiment results.
Supports Phase II visualizations:
  - Noise & Outlier Impact Comparison
  - Hyperparameter Sensitivity Curves (k=?, n_estimators=?, kernel=?)
  - Confusion Matrices, ROC Curves, Benchmark & Overfitting Diagnoses.
"""

import io, os, tempfile, base64
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mpl_cache')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODEL_COLORS = {
    'KNN':      '#3b82f6',
    'Ensemble': '#f59e0b',
    'SVM':      '#8b5cf6',
}


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f'data:image/png;base64,{b64}'


# ── Phase II: Noise & Outlier Impact Plot ──────────────────────────────────

def noise_outlier_plot(noise_outlier_data):
    """
    Renders grouped bar chart comparing model accuracies under 5 conditions:
      - Baseline
      - Without Outliers (IQR cleaned)
      - With Feature Noise
      - Noise → Gaussian Smoothed (noise removed via gaussian_filter1d)
      - With Injected Outliers
    """
    fig, ax = plt.subplots(figsize=(10, 4.8))
    comparison = noise_outlier_data['comparison']

    models = [c['model_name'].split('(')[0].strip() for c in comparison]
    x = np.arange(len(models))
    width = 0.15

    scenarios = [
        ('baseline_acc',         'Baseline (Standard)',        '#3b82f6'),
        ('without_outliers_acc', 'Without Outliers (IQR)',     '#10b981'),
        ('noisy_acc',            'With Feature Noise',         '#ef4444'),
        ('smoothed_acc',         'Noise → Gaussian Smoothed',  '#a855f7'),
        ('outliers_acc',         'With Injected Outliers',     '#f97316'),
    ]

    for idx, (metric_key, label, color) in enumerate(scenarios):
        vals = [c.get(metric_key, 0) * 100 for c in comparison]
        offset = (idx - 2) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.9)
        for b in bars:
            height = b.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Test Accuracy (%)', fontsize=10, fontweight='bold')
    ax.set_title('Phase II: Noise & Outlier Influence + Gaussian Smoothing Recovery',
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 118)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9, ncol=2)

    return _b64(fig)


# ── Phase II: Model Parameter Sensitivity Curves ───────────────────────────

def param_sensitivity_plot(sensitivity_dict):
    """
    Renders 3 subplots showing the influence of parameters:
      1. KNN (k = ?)
      2. Ensemble (estimators = ?)
      3. SVM (kernel = ...)
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # 1. KNN (k)
    knn_curve = sensitivity_dict.get('KNN', [])
    if knn_curve:
        # Aggregate by k
        k_scores = {}
        for item in knn_curve:
            k = item['params'].get('n_neighbors', 5)
            k_scores.setdefault(k, []).append(item['cv_mean'])
        k_vals = sorted(k_scores.keys())
        k_means = [np.mean(k_scores[k]) * 100 for k in k_vals]

        axes[0].plot(k_vals, k_means, marker='o', color='#3b82f6', lw=2.2, markersize=5)
        for k, sc in zip(k_vals, k_means):
            axes[0].annotate(f'{sc:.1f}%', (k, sc), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7.5)
        axes[0].set_title('KNN: Influence of k (k = ?)', fontsize=10, fontweight='bold')
        axes[0].set_xlabel('Number of Neighbors (k)', fontsize=9)
        axes[0].set_ylabel('5-Fold CV Accuracy (%)', fontsize=9)
        axes[0].grid(True, linestyle=':', alpha=0.6)

    # 2. Ensemble (n_estimators)
    ens_curve = sensitivity_dict.get('Ensemble', [])
    if ens_curve:
        n_vals = [item['params'].get('n_estimators', 50) for item in ens_curve]
        n_means = [item['cv_mean'] * 100 for item in ens_curve]

        axes[1].plot(n_vals, n_means, marker='s', color='#f59e0b', lw=2.2, markersize=5)
        for n, sc in zip(n_vals, n_means):
            axes[1].annotate(f'{sc:.1f}%', (n, sc), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7.5)
        axes[1].set_title('Ensemble: Influence of Estimators', fontsize=10, fontweight='bold')
        axes[1].set_xlabel('n_estimators', fontsize=9)
        axes[1].set_ylabel('5-Fold CV Accuracy (%)', fontsize=9)
        axes[1].grid(True, linestyle=':', alpha=0.6)

    # 3. SVM (kernel)
    svm_curve = sensitivity_dict.get('SVM', [])
    if svm_curve:
        kernel_scores = {}
        for item in svm_curve:
            k_name = item['params'].get('kernel', 'rbf')
            kernel_scores.setdefault(k_name, []).append(item['cv_mean'])
        kernels = list(kernel_scores.keys())
        k_means = [np.mean(kernel_scores[k]) * 100 for k in kernels]

        bars = axes[2].bar(kernels, k_means, color='#8b5cf6', width=0.45, alpha=0.85)
        for b in bars:
            height = b.get_height()
            axes[2].annotate(f'{height:.1f}%',
                             xy=(b.get_x() + b.get_width() / 2, height),
                             xytext=(0, 3), textcoords='offset points',
                             ha='center', va='bottom', fontsize=8, fontweight='bold')
        axes[2].set_title('SVM: Influence of Kernel', fontsize=10, fontweight='bold')
        axes[2].set_xlabel('Kernel Function', fontsize=9)
        axes[2].set_ylabel('5-Fold CV Accuracy (%)', fontsize=9)
        axes[2].grid(axis='y', linestyle=':', alpha=0.6)

    fig.tight_layout()
    return _b64(fig)


# ── Standard Plots ────────────────────────────────────────────────────────

def roc_curve_plot(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0,1],[0,1],'--', color='#94a3b8', lw=1.5, label='Random (AUC=0.50)')
    for key, res in results.items():
        ax.plot(res['roc']['fpr'], res['roc']['tpr'],
                color=MODEL_COLORS.get(key, '#64748b'), lw=2.5,
                label=f"{res['name']} (AUC={res['auc']:.3f})")
    ax.set(xlim=[-0.02,1.02], ylim=[-0.02,1.05],
           xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='ROC Curve Comparison')
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9)
    return _b64(fig)


def confusion_matrix_plot(model_name, cm_dict, model_key):
    tn, fp, fn, tp = cm_dict['tn'], cm_dict['fp'], cm_dict['fn'], cm_dict['tp']
    total = max(1, tn + fp + fn + tp)
    matrix = np.array([[tn, fp], [fn, tp]])
    cmap = {'KNN': 'Blues', 'Ensemble': 'YlOrBr', 'SVM': 'Purples'}.get(model_key, 'Blues')

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(matrix, cmap=cmap)
    labels = [[f'TN\n{tn}\n({tn/total*100:.1f}%)', f'FP\n{fp}\n({fp/total*100:.1f}%)'],
              [f'FN\n{fn}\n({fn/total*100:.1f}%)', f'TP\n{tp}\n({tp/total*100:.1f}%)']]
    thresh = matrix.max() / 2 if matrix.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha='center', va='center',
                    color='white' if matrix[i, j] > thresh else '#1e293b',
                    fontsize=10, fontweight='bold')
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Pred Neg', 'Pred Pos'],
           yticklabels=['True Neg', 'True Pos'],
           title=model_name)
    return _b64(fig)


def benchmark_bar_plot(results):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    keys  = list(results)
    names = [results[k]['name'].split('(')[0].strip() for k in keys]
    x, w  = np.arange(len(keys)), 0.18
    for idx, (mkey, mlbl, col) in enumerate(zip(
            ['test_accuracy', 'precision', 'recall', 'f1'],
            ['Accuracy', 'Precision', 'Recall', 'F1'],
            ['#3b82f6', '#06b6d4', '#f59e0b', '#10b981'])):
        vals = [results[k][mkey] * 100 for k in keys]
        bars = ax.bar(x + (idx - 1.5) * w, vals, w, label=mlbl, color=col)
        for b in bars:
            ax.annotate(f'{b.get_height():.0f}%',
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7.5)
    ax.set(ylabel='Score (%)', title='Model Benchmark Comparison',
           xticks=x, xticklabels=names, ylim=[0, 115])
    ax.grid(axis='y', ls=':', alpha=0.7)
    ax.legend(ncol=4, fontsize=9)
    return _b64(fig)


def overfit_bar_plot(results):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    keys  = list(results)
    names = [results[k]['name'].split('(')[0].strip() for k in keys]
    x, w  = np.arange(len(keys)), 0.25
    for idx, (label, col, val_key) in enumerate(zip(
            ['Train Accuracy', '5-Fold CV Accuracy', 'Test Accuracy'],
            ['#93c5fd', '#818cf8', '#34d399'],
            ['train_accuracy', 'cv_mean', 'test_accuracy'])):
        vals = [results[k][val_key] * 100 for k in keys]
        bars = ax.bar(x + (idx - 1) * w, vals, w, label=label, color=col)
        for b in bars:
            ax.annotate(f'{b.get_height():.1f}%',
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
    ax.set(ylabel='Accuracy (%)', title='Overfitting & Generalization (Train vs CV vs Test)',
           xticks=x, xticklabels=names, ylim=[0, 115])
    ax.grid(axis='y', ls=':', alpha=0.7)
    ax.legend(fontsize=9)
    return _b64(fig)


def generate_all(results, sensitivity_dict=None, noise_outlier_data=None):
    """
    Generate all plots for one experiment run, returning dict keyed by plot_type.
    """
    plots = {
        'roc':       roc_curve_plot(results),
        'benchmark': benchmark_bar_plot(results),
        'overfit':   overfit_bar_plot(results),
    }

    if noise_outlier_data:
        plots['noise_outlier'] = noise_outlier_plot(noise_outlier_data)

    if sensitivity_dict:
        plots['param_sensitivity'] = param_sensitivity_plot(sensitivity_dict)

    for key, res in results.items():
        plots[f'cm_{key}'] = confusion_matrix_plot(res['name'], res['confusion_matrix'], key)

    return plots
