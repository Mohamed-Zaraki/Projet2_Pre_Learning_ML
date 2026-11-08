"""
train_and_save.py
=================
Offline training script — run this ONCE before starting the Flask app.

What it does for each dataset:
  1. Load & deduplicate via dataset_loader
  2. Auto-select the best scaler (StandardScaler / MinMaxScaler / RobustScaler)
  3. Build sklearn Pipeline: ColumnTransformer(Imputer → Scaler) → Classifier
  4. Run GridSearchCV for KNN, SVM, and Decision Tree
  5. Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix,
               5-Fold Cross-Validation, Overfitting diagnosis
  6. Generate and store all plots as base64 PNGs
  7. Persist everything to SQLite (instance/ml_workbench.db)
     and serialise train/test splits to storage/splits/*.pkl
     and fitted pipelines to storage/models/*.pkl

Usage
-----
  python train_and_save.py           # trains all 4 datasets
  python train_and_save.py alcohol   # trains a single dataset
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

SEP = '=' * 62


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

    # 2. GridSearchCV → train
    print('  GridSearchCV running (KNN / SVM / Decision Tree) ...')
    results, models_dict, best_key, scaler_used = train_all(
        X_train, X_test, y_train, y_test, meta['feature_names'],
        imputer_strategy='median', cv_folds=5)

    # 3. Print results summary
    print(f'  Scaler selected : {scaler_used}')
    for key, res in results.items():
        star = ' <-- BEST' if key == best_key else ''
        print(f'  [{key:<13s}]  '
              f'Test={res["test_accuracy"]*100:.2f}%  '
              f'CV={res["cv_mean"]*100:.1f}% +/- {res["cv_std"]*100:.1f}%  '
              f'AUC={res["auc"]:.3f}  '
              f'{res["overfitting_status"]}{star}')
        print(f'               Best params: {res["best_params"]}')

    # 4. Plots
    print('  Generating plots ...')
    plots = generate_all(results)

    # 5. Persist to SQLite + pickle
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
    print('  ALL DONE -- launch the Flask app: python run.py')
    print(SEP)
