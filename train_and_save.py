"""
train_and_save.py — Offline Phase II Training & Evaluation Script.

Run this script ONCE before running the Flask application.

What it does for each dataset:
  1. Load, deduplicate, and split (dataset_loader)
  2. Auto-select best scaler (StandardScaler / MinMaxScaler / RobustScaler)
  3. Manual Hyperparameter Optimization for:
       - KNN: evaluates k = ? (1, 3, 5, 7, 11, 15, 21)
       - Ensemble: Random Forest evaluating n_estimators = ? (10, 25, 50, 100, 150, 200)
       - SVM: evaluating kernel = ... (rbf, linear, poly)
  4. Phase II: Evaluate Influence of Noise and Outliers:
       - Baseline (Standard data)
       - Without Outliers (IQR filtered)
       - With Feature Noise (Gaussian perturbation)
       - With Injected Outliers (Synthetic anomaly spikes)
  5. Generate standalone base64 PNG visualizations:
       - Noise & Outlier Impact Comparison
       - Parameter Sensitivity Curves (k, estimators, kernel)
       - ROC Curves, Confusion Matrices, Overfitting Analysis
  6. Persist to SQLite (instance/ml_workbench.db) & Pickle files (storage/)
"""

import os, sys, tempfile, warnings

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

os.environ['MPLCONFIGDIR']   = os.path.join(tempfile.gettempdir(), 'mpl_cache')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
warnings.filterwarnings('ignore')

from app.models.dataset_loader import DATASETS, load_and_prepare
from app.models.ml_manager    import train_all
from app.models.visualizer    import generate_all
from app.models.db_manager    import init_db, save_run

SEP = '=' * 66


def run_dataset(dataset_key):
    cfg = DATASETS[dataset_key]
    print(f'\n{SEP}')
    print(f'  Dataset : {cfg["name"]}')
    print(SEP)

    # 1. Load & prepare split
    X_train, X_test, y_train, y_test, meta = load_and_prepare(dataset_key)
    print(f'  Loaded   : {meta["raw_rows"]} raw rows -> {meta["final_rows"]} after dedup '
          f'({meta["duplicates_removed"]} removed)')
    print(f'  Split    : {meta["train_rows"]} train / {meta["test_rows"]} test  '
          f'| {meta["features_count"]} features')
    if meta['total_nulls']:
        print(f'  NaN cols : {meta["null_per_col"]}')

    # 2. Manual Parameter Tuning & Model Fitting
    print('  [Phase II] Evaluating Manual Hyperparameters (KNN, Ensemble, SVM) ...')
    results, models_dict, best_key, scaler_used, sensitivity_dict, noise_outlier_data = train_all(
        X_train, X_test, y_train, y_test, meta['feature_names'],
        imputer_strategy='median', cv_folds=5)

    # 3. Print Model Benchmark
    print(f'  Scaler selected : {scaler_used.capitalize()}Scaler')
    for key, res in results.items():
        star = ' <-- BEST' if key == best_key else ''
        print(f'  [{key:<10s}]  '
              f'Test={res["test_accuracy"]*100:.2f}%  '
              f'CV={res["cv_mean"]*100:.1f}% +/- {res["cv_std"]*100:.1f}%  '
              f'AUC={res["auc"]:.3f}  '
              f'{res["overfitting_status"]}{star}')
        print(f'               Chosen params: {res["best_params"]}')

    # 4. Print Phase II: Noise & Outlier Results
    print('\n  [Phase II: Influence of Noise & Outliers]')
    print(f'  IQR Outliers Filtered : {noise_outlier_data["stats"]["outliers_removed"]} samples')
    print(f'  Outliers Injected     : {noise_outlier_data["stats"]["outliers_injected"]} samples')
    print('  ' + '-' * 76)
    print(f'  {"Model":<12} | {"Baseline":<9} | {"No Outliers":<11} | {"With Noise":<10} | {"Smoothed":<8} | {"With Outliers":<12}')
    print('  ' + '-' * 76)
    for row in noise_outlier_data['comparison']:
        print(f'  {row["model_key"]:<12} | {row["baseline_acc"]*100:6.1f}%   | '
              f'{row["without_outliers_acc"]*100:8.1f}%   | '
              f'{row["noisy_acc"]*100:7.1f}%   | '
              f'{row["smoothed_acc"]*100:5.1f}%   | '
              f'{row["outliers_acc"]*100:9.1f}%')
    print('  ' + '-' * 76)

    # 5. Generate Visualizations
    print('  Generating Phase II plots ...')
    plots = generate_all(results, sensitivity_dict, noise_outlier_data)

    # 6. Persist to SQLite + pickle
    run_id, split_path = save_run(
        dataset_id        = dataset_key,
        dataset_name      = meta['dataset_name'],
        raw_rows          = meta['raw_rows'],
        final_rows        = meta['final_rows'],
        features_count    = meta['features_count'],
        feature_names     = meta['feature_names'],
        duplicates_removed= meta['duplicates_removed'],
        missing_strategy  = 'median_imputer',
        scaler_used       = scaler_used,
        test_size         = 0.2,
        cv_folds          = 5,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        results           = results,
        best_model_key    = best_key,
        models_dict       = models_dict,
        plots_dict        = plots,
        sensitivity_dict  = sensitivity_dict,
        noise_outlier_data= noise_outlier_data
    )
    print(f'  Saved    : SQLite run #{run_id}')
    print(f'  Pickle   : {split_path}')


if __name__ == '__main__':
    init_db()
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    for key in targets:
        if key not in DATASETS:
            print(f'Unknown dataset key: {key}  (valid: {list(DATASETS.keys())})')
            sys.exit(1)
        run_dataset(key)

    print(f'\n{SEP}')
    print('  PHASE II COMPLETE -- Launch the Flask app: python run.py')
    print(SEP)
