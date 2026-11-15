"""
ml_manager.py — Phase II: Manual Hyperparameter Optimization & Noise/Outlier Evaluation.

Models (from whiteboard):
  - KNN (evaluating k = ?)
  - Ensemble: Random Forest (evaluating n_estimators = ?)
  - SVM (evaluating kernel = ...)

All tuning is performed manually via cross-validation loops (no automated grid search).
Includes Phase II experimental framework for evaluating the influence of noise and outliers.
"""

import os, tempfile
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import numpy as np
from sklearn.compose        import ColumnTransformer
from sklearn.impute         import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics        import (accuracy_score, precision_score, recall_score,
                                    f1_score, confusion_matrix, roc_curve, roc_auc_score)
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.svm            import SVC
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler, MinMaxScaler, RobustScaler

from app.models.dataset_loader import (detect_and_remove_outliers_iqr,
                                       inject_noise,
                                       smooth_with_gaussian,
                                       inject_extreme_outliers)

# ── Scaler Selection (Auto-benchmarked) ────────────────────────────────────

def _best_scaler(X_train, y_train, feature_names, imputer_strategy, cv=3):
    """Select the scaler that maximizes baseline cross-validation accuracy."""
    best_name, best_score = 'standard', -1.0
    for name, scaler in [('standard', StandardScaler()),
                         ('minmax',   MinMaxScaler()),
                         ('robust',   RobustScaler())]:
        pipe = Pipeline([
            ('pre', ColumnTransformer([
                ('num', Pipeline([('imp', SimpleImputer(strategy=imputer_strategy)),
                                  ('sc',  scaler)]),
                 feature_names)
            ])),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ])
        score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1).mean()
        if score > best_score:
            best_score, best_name = score, name
    return best_name


SCALER_CLASSES = {
    'standard': StandardScaler,
    'minmax':   MinMaxScaler,
    'robust':   RobustScaler,
}

# ── Model Definitions (From Whiteboard) ───────────────────────────────────

ESTIMATORS = {
    'KNN':      ('K-Nearest Neighbors (KNN)',    KNeighborsClassifier),
    'Ensemble': ('Ensemble (Random Forest)',     RandomForestClassifier),
    'SVM':      ('Support Vector Machine (SVM)', SVC),
}

# Manual parameter candidates to evaluate sensitivity
MANUAL_CANDIDATES = {
    'KNN': [
        {'n_neighbors': k, 'metric': m, 'weights': 'distance'}
        for k in [1, 3, 5, 7, 11, 15]
        for m in ['manhattan', 'euclidean']
    ],
    'Ensemble': [
        {'n_estimators': n, 'max_depth': 8, 'random_state': 42}
        for n in [10, 25, 50, 100]
    ],
    'SVM': [
        {'kernel': k, 'C': c, 'random_state': 42, 'max_iter': 2500}
        for k in ['rbf', 'linear', 'poly']
        for c in [0.5, 1.0, 5.0]
    ],
}


def _build_pipeline(feature_names, estimator_instance, scaler_name, imputer_strategy):
    return Pipeline([
        ('pre', ColumnTransformer([
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy=imputer_strategy)),
                ('sc',  SCALER_CLASSES[scaler_name]()),
            ]), feature_names)
        ])),
        ('clf', estimator_instance),
    ])


def _overfitting_label(train_acc, test_acc, cv_mean):
    gap = train_acc - test_acc
    if gap > 0.15:
        return ('High Overfitting Risk',    'danger',
                f'Train {train_acc*100:.1f}% >> Test {test_acc*100:.1f}%.')
    if gap > 0.07:
        return ('Moderate Overfitting',     'warning',
                f'Train {train_acc*100:.1f}% > Test {test_acc*100:.1f}%.')
    if gap < -0.05:
        return ('Underfitting',             'info',
                f'Test {test_acc*100:.1f}% >= Train {train_acc*100:.1f}%.')
    return     ('Well-Balanced',            'success',
                f'Train {train_acc*100:.1f}% ~ CV {cv_mean*100:.1f}% ~ Test {test_acc*100:.1f}%.')


# ── Manual Hyperparameter Optimization Loop (Offline) ─────────────────────

def manual_tune_model(model_key, feature_names, scaler_name, imputer_strategy,
                      X_train, y_train, cv):
    """
    Iterate manually through candidate hyperparameters, record cross-validation
    scores to study parameter influence, and pick the best performing set.
    """
    est_cls = ESTIMATORS[model_key][1]
    candidates = MANUAL_CANDIDATES[model_key]

    best_score = -1.0
    best_params = candidates[0]
    sensitivity_curve = []

    for params in candidates:
        est_inst = est_cls(**params)
        pipe = _build_pipeline(feature_names, est_inst, scaler_name, imputer_strategy)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
        mean_s = float(scores.mean())
        std_s = float(scores.std())

        sensitivity_curve.append({
            'params': params,
            'cv_mean': mean_s,
            'cv_std': std_s,
        })

        if mean_s > best_score:
            best_score = mean_s
            best_params = params

    # Refit optimal pipeline on full training data (enable probability for SVM)
    final_params = dict(best_params)
    if model_key == 'SVM':
        final_params['probability'] = True
    best_est_inst = est_cls(**final_params)
    final_pipeline = _build_pipeline(feature_names, best_est_inst, scaler_name, imputer_strategy)
    final_pipeline.fit(X_train, y_train)

    return final_pipeline, best_params, sensitivity_curve


# ── Phase II: Evaluation of the Influence of Noise and Outliers ───────────

def evaluate_noise_and_outliers(models_dict, feature_names, scaler_name, imputer_strategy,
                                X_train, y_train, X_test, y_test):
    """
    Phase II: Evaluate model performance across 4 data conditions:
      1. Baseline: Standard training data
      2. Without Outliers: Outliers filtered out via IQR method
      3. With Noise: Features perturbed by Gaussian noise
      4. With Outliers: Extreme synthetic outlier spikes injected

    All variations are evaluated against the same clean test set.
    """
    # 1. Prepare alternative training sets
    X_tr_no_outliers, y_tr_no_outliers, out_removed = detect_and_remove_outliers_iqr(X_train, y_train)
    X_tr_noisy = inject_noise(X_train, noise_level=0.18, random_state=42)
    X_tr_smoothed = smooth_with_gaussian(X_tr_noisy, sigma=1.2)   # Noise → then Gaussian smoothed
    X_tr_outliers, out_injected = inject_extreme_outliers(X_train, contamination=0.08, factor=3.5, random_state=42)

    scenarios = {
        'baseline': {
            'label': 'Baseline (Standard)',
            'X_tr': X_train, 'y_tr': y_train,
            'desc': 'Standard deduplicated training set.'
        },
        'without_outliers': {
            'label': 'Without Outliers (IQR)',
            'X_tr': X_tr_no_outliers, 'y_tr': y_tr_no_outliers,
            'desc': f'Filtered {out_removed} outlier samples using IQR.'
        },
        'with_noise': {
            'label': 'With Feature Noise',
            'X_tr': X_tr_noisy, 'y_tr': y_train,
            'desc': '18% Gaussian perturbation added to continuous features.'
        },
        'noise_smoothed': {
            'label': 'Noise → Gaussian Smoothed',
            'X_tr': X_tr_smoothed, 'y_tr': y_train,
            'desc': 'Gaussian noise injected then removed via gaussian_filter1d (σ=1.2).'
        },
        'with_outliers': {
            'label': 'With Injected Outliers',
            'X_tr': X_tr_outliers, 'y_tr': y_train,
            'desc': f'Injected {out_injected} extreme anomaly spikes.'
        },
    }

    results = {}
    for sc_key, sc_info in scenarios.items():
        results[sc_key] = {
            'label': sc_info['label'],
            'desc':  sc_info['desc'],
            'scores': {}
        }
        for m_key, fitted_pipe in models_dict.items():
            best_params = fitted_pipe.named_steps['clf'].get_params()
            est_cls = ESTIMATORS[m_key][1]
            init_params = {k: v for k, v in best_params.items() if not k.startswith('_')}
            est_inst = est_cls(**init_params)

            pipe = _build_pipeline(feature_names, est_inst, scaler_name, imputer_strategy)
            pipe.fit(sc_info['X_tr'], sc_info['y_tr'])

            preds = pipe.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, zero_division=0))

            results[sc_key]['scores'][m_key] = {
                'accuracy': acc,
                'f1': f1,
            }

    # Calculate degradation / resilience delta compared to baseline
    comparison = []
    for m_key in models_dict.keys():
        base_acc = results['baseline']['scores'][m_key]['accuracy']
        no_out_acc = results['without_outliers']['scores'][m_key]['accuracy']
        noisy_acc = results['with_noise']['scores'][m_key]['accuracy']
        smoothed_acc = results['noise_smoothed']['scores'][m_key]['accuracy']
        outlier_acc = results['with_outliers']['scores'][m_key]['accuracy']

        comparison.append({
            'model_key': m_key,
            'model_name': ESTIMATORS[m_key][0],
            'baseline_acc': base_acc,
            'without_outliers_acc': no_out_acc,
            'noisy_acc': noisy_acc,
            'smoothed_acc': smoothed_acc,
            'outliers_acc': outlier_acc,
            'noise_delta': float(noisy_acc - base_acc),
            'smooth_recovery': float(smoothed_acc - noisy_acc),  # how much smoothing recovered
            'outlier_delta': float(outlier_acc - base_acc),
            'iqr_cleanup_delta': float(no_out_acc - base_acc),
        })

    return {
        'scenarios': results,
        'comparison': comparison,
        'stats': {
            'outliers_removed': out_removed,
            'outliers_injected': out_injected,
        }
    }


# ── Full Training Routine ──────────────────────────────────────────────────

def train_all(X_train, X_test, y_train, y_test, feature_names,
              imputer_strategy='median', cv_folds=5):
    """
    Main offline trainer:
      1. Determines best scaler via baseline CV.
      2. Executes manual hyperparameter evaluation for KNN, Ensemble, and SVM.
      3. Performs Phase II Noise and Outlier sensitivity analysis.
      4. Compiles all metrics and confusion matrices.
    """
    scaler_name = _best_scaler(X_train, y_train, feature_names, imputer_strategy)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results = {}
    models_dict = {}
    sensitivity_dict = {}

    for key, (name, _) in ESTIMATORS.items():
        best_pipe, best_params, sens_curve = manual_tune_model(
            key, feature_names, scaler_name, imputer_strategy, X_train, y_train, cv
        )
        models_dict[key] = best_pipe
        sensitivity_dict[key] = sens_curve

        # Predictions on test & train sets
        y_pred = best_pipe.predict(X_test)
        y_train_pred = best_pipe.predict(X_train)

        # Probabilities for ROC
        try:
            y_prob = best_pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = np.zeros(len(y_test))

        test_acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_train_pred)

        cv_scores = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = (int(cm[0,0]), int(cm[0,1]),
                          int(cm[1,0]), int(cm[1,1])) if cm.shape == (2,2) else (0,0,0,0)

        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = float(roc_auc_score(y_test, y_prob))
        except Exception:
            fpr, tpr = [0, 1], [0, 1]
            auc_val = 0.5

        ov_status, ov_badge, ov_desc = _overfitting_label(train_acc, test_acc, cv_scores.mean())

        clean_params = {k: v for k, v in best_params.items() if k not in ['random_state', 'probability', 'max_iter', 'n_jobs']}

        results[key] = {
            'key':                key,
            'name':               name,
            'best_params':        clean_params,
            'test_accuracy':      float(test_acc),
            'train_accuracy':     float(train_acc),
            'cv_mean':            float(cv_scores.mean()),
            'cv_std':             float(cv_scores.std()),
            'precision':          float(precision_score(y_test, y_pred, zero_division=0)),
            'recall':             float(recall_score(y_test,    y_pred, zero_division=0)),
            'f1':                 float(f1_score(y_test,        y_pred, zero_division=0)),
            'auc':                auc_val,
            'overfitting_status': ov_status,
            'overfitting_badge':  ov_badge,
            'overfitting_desc':   ov_desc,
            'confusion_matrix':   {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'roc':                {'fpr': list(fpr), 'tpr': list(tpr)},
        }

    best_key = max(results, key=lambda k: results[k]['test_accuracy'])

    # Phase II: Noise & Outlier Evaluation
    noise_outlier_data = evaluate_noise_and_outliers(
        models_dict, feature_names, scaler_name, imputer_strategy,
        X_train, y_train, X_test, y_test
    )

    return results, models_dict, best_key, scaler_name, sensitivity_dict, noise_outlier_data
