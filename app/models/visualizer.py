"""
visualizer.py — Generates base64-encoded PNG plots from experiment results.
Called only by train_and_save.py (offline), never from Flask.
"""

import io, os, tempfile, base64
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mpl_cache')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODEL_COLORS = {
    'KNN':           '#3b82f6',
    'SVM':           '#8b5cf6',
    'Decision_Tree': '#10b981',
}


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f'data:image/png;base64,{b64}'


def roc_curve_plot(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0,1],[0,1],'--', color='#94a3b8', lw=1.5, label='Random (AUC=0.50)')
    for key, res in results.items():
        ax.plot(res['roc']['fpr'], res['roc']['tpr'],
                color=MODEL_COLORS.get(key,'#64748b'), lw=2.5,
                label=f"{res['name']} (AUC={res['auc']:.3f})")
    ax.set(xlim=[-0.02,1.02], ylim=[-0.02,1.05],
           xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='ROC Curve Comparison')
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9)
    return _b64(fig)


def confusion_matrix_plot(model_name, cm_dict, model_key):
    tn,fp,fn,tp = cm_dict['tn'], cm_dict['fp'], cm_dict['fn'], cm_dict['tp']
    total = tn+fp+fn+tp
    matrix = np.array([[tn,fp],[fn,tp]])
    cmap = {'KNN':'Blues','SVM':'Purples','Decision_Tree':'Greens'}.get(model_key,'Blues')

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(matrix, cmap=cmap)
    labels = [[f'TN\n{tn}\n({tn/total*100:.1f}%)', f'FP\n{fp}\n({fp/total*100:.1f}%)'],
              [f'FN\n{fn}\n({fn/total*100:.1f}%)', f'TP\n{tp}\n({tp/total*100:.1f}%)']]
    thresh = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha='center', va='center',
                    color='white' if matrix[i,j] > thresh else '#1e293b',
                    fontsize=10, fontweight='bold')
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=['Pred Neg','Pred Pos'],
           yticklabels=['True Neg','True Pos'],
           title=model_name)
    return _b64(fig)


def benchmark_bar_plot(results):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    keys  = list(results)
    names = [results[k]['name'].split('(')[0].strip() for k in keys]
    x, w  = np.arange(len(keys)), 0.18
    for idx,(mkey,mlbl,col) in enumerate(zip(
            ['test_accuracy','precision','recall','f1'],
            ['Accuracy','Precision','Recall','F1'],
            ['#3b82f6','#06b6d4','#f59e0b','#10b981'])):
        vals = [results[k][mkey]*100 for k in keys]
        bars = ax.bar(x+(idx-1.5)*w, vals, w, label=mlbl, color=col)
        for b in bars:
            ax.annotate(f'{b.get_height():.0f}%',
                        xy=(b.get_x()+b.get_width()/2, b.get_height()),
                        xytext=(0,2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7.5)
    ax.set(ylabel='Score (%)', title='Model Benchmark Comparison',
           xticks=x, xticklabels=names, ylim=[0,115])
    ax.grid(axis='y', ls=':', alpha=0.7)
    ax.legend(ncol=4, fontsize=9)
    return _b64(fig)


def overfit_bar_plot(results):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    keys  = list(results)
    names = [results[k]['name'].split('(')[0].strip() for k in keys]
    x, w  = np.arange(len(keys)), 0.25
    for idx,(label,col,val_key) in enumerate(zip(
            ['Train Accuracy','5-Fold CV Accuracy','Test Accuracy'],
            ['#93c5fd','#818cf8','#34d399'],
            ['train_accuracy','cv_mean','test_accuracy'])):
        vals = [results[k][val_key]*100 for k in keys]
        bars = ax.bar(x+(idx-1)*w, vals, w, label=label, color=col)
        for b in bars:
            ax.annotate(f'{b.get_height():.1f}%',
                        xy=(b.get_x()+b.get_width()/2, b.get_height()),
                        xytext=(0,2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
    ax.set(ylabel='Accuracy (%)', title='Overfitting & Generalization (Train vs CV vs Test)',
           xticks=x, xticklabels=names, ylim=[0,115])
    ax.grid(axis='y', ls=':', alpha=0.7)
    ax.legend(fontsize=9)
    return _b64(fig)


def generate_all(results):
    """
    Generate all plots for one experiment run.
    Returns a dict keyed by plot_type.
    """
    plots = {
        'roc':       roc_curve_plot(results),
        'benchmark': benchmark_bar_plot(results),
        'overfit':   overfit_bar_plot(results),
    }
    for key, res in results.items():
        plots[f'cm_{key}'] = confusion_matrix_plot(res['name'], res['confusion_matrix'], key)
    return plots
