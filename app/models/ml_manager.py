"""
ml_manager.py — Fits KNN, SVM, and Decision Tree with GridSearchCV.

Each model is wrapped in a sklearn Pipeline:
  ColumnTransformer(SimpleImputer → Scaler) → Classifier

GridSearchCV explores hyperparameter grids and returns the best estimator.
"""

import os, tempfile
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import numpy as np
from sklearn.compose        import ColumnTransformer
from sklearn.impute         import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics        import (accuracy_score, precision_score, recall_score,
                                    f1_score, confusion_matrix, roc_curve, roc_auc_score)
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm            import SVC
from sklearn.tree           import DecisionTreeClassifier


# ── Scaler selection (auto) ────────────────────────────────────────────────

def _best_scaler(X_train, y_train, feature_names, imputer_strategy, cv=3):
    """Pick the scaler that maximises KNN accuracy under 3-fold CV."""
    best_name, best_score = 'standard', -1
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

# ── GridSearch grids ───────────────────────────────────────────────────────

PARAM_GRIDS = {
    'KNN': {
        'clf__n_neighbors': [3, 5, 7],
        'clf__weights':     ['uniform', 'distance'],
        'clf__metric':      ['euclidean', 'manhattan'],
    },
    'SVM': {
        'clf__C':      [0.1, 1.0, 10.0],
        'clf__kernel': ['rbf'],
        'clf__gamma':  ['scale', 'auto'],
    },
    'Decision_Tree': {
        'clf__max_depth':        [3, 5, 10, None],
        'clf__criterion':        ['gini', 'entropy'],
    },
}

ESTIMATORS = {
    'KNN':           ('K-Nearest Neighbors (KNN)',     KNeighborsClassifier()),
    'SVM':           ('Support Vector Machine (SVM)',   SVC(probability=True, random_state=42, max_iter=3000)),
    'Decision_Tree': ('Decision Tree Classifier',       DecisionTreeClassifier(random_state=42)),
}


def _build_pipeline(feature_names, estimator, scaler_name, imputer_strategy):
    return Pipeline([
        ('pre', ColumnTransformer([
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy=imputer_strategy)),
                ('sc',  SCALER_CLASSES[scaler_name]()),
            ]), feature_names)
        ])),
        ('clf', estimator),
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
                f'Test {test_acc*100:.1f}% ≥ Train {train_acc*100:.1f}%.')
    return     ('Well-Balanced',            'success',
                f'Train {train_acc*100:.1f}% ≈ CV {cv_mean*100:.1f}% ≈ Test {test_acc*100:.1f}%.')


def train_all(X_train, X_test, y_train, y_test, feature_names,
              imputer_strategy='median', cv_folds=5):
    """
    Run GridSearchCV for every model and return metrics + fitted pipelines.

    Returns
    -------
    results      : {model_key: {...all metrics...}}
    models_dict  : {model_key: fitted best Pipeline}
    best_key     : key of the top model
    scaler_used  : name of the chosen scaler
    """
    scaler_name = _best_scaler(X_train, y_train, feature_names, imputer_strategy)
    cv          = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results     = {}
    models_dict = {}

    for key, (name, base_est) in ESTIMATORS.items():
        pipeline = _build_pipeline(feature_names, base_est, scaler_name, imputer_strategy)

        # Prefix param names with 'clf__' (already done in PARAM_GRIDS)
        gs = GridSearchCV(pipeline, PARAM_GRIDS[key], cv=cv,
                          scoring='accuracy', n_jobs=1, refit=True)
        gs.fit(X_train, y_train)

        best_pipe = gs.best_estimator_
        models_dict[key] = best_pipe

        # Predictions
        y_pred       = best_pipe.predict(X_test)
        y_train_pred = best_pipe.predict(X_train)
        y_prob       = best_pipe.predict_proba(X_test)[:, 1]

        # Metrics
        test_acc  = accuracy_score(y_test,  y_pred)
        train_acc = accuracy_score(y_train, y_train_pred)

        cv_scores = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = (int(cm[0,0]), int(cm[0,1]),
                          int(cm[1,0]), int(cm[1,1])) if cm.shape == (2,2) else (0,0,0,0)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val     = roc_auc_score(y_test, y_prob)

        ov_status, ov_badge, ov_desc = _overfitting_label(train_acc, test_acc, cv_scores.mean())

        # Strip 'clf__' prefix from best params for clean display
        best_params = {k.replace('clf__', ''): v for k, v in gs.best_params_.items()}

        results[key] = {
            'key':              key,
            'name':             name,
            'best_params':      best_params,
            'test_accuracy':    float(test_acc),
            'train_accuracy':   float(train_acc),
            'cv_mean':          float(cv_scores.mean()),
            'cv_std':           float(cv_scores.std()),
            'precision':        float(precision_score(y_test, y_pred, zero_division=0)),
            'recall':           float(recall_score(y_test,    y_pred, zero_division=0)),
            'f1':               float(f1_score(y_test,        y_pred, zero_division=0)),
            'auc':              float(auc_val),
            'overfitting_status': ov_status,
            'overfitting_badge':  ov_badge,
            'overfitting_desc':   ov_desc,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        }

    best_key = max(results, key=lambda k: results[k]['test_accuracy'])
    return results, models_dict, best_key, scaler_name
