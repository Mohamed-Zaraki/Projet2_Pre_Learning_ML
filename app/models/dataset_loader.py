"""
dataset_loader.py — Loads, merges, deduplicates and splits each dataset.
No sklearn fitting happens here — only pandas operations.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATASETS = {
    'alcohol': {
        'id':          'alcohol',
        'name':        'Alcohol Consumption & Biometrics',
        'path':        'Datasets/alcohol.csv',
        'target_col':  'CLASS',
        'drop_cols':   [],
        'is_combined': False,
    },
    'covid19': {
        'id':          'covid19',
        'name':        'COVID-19 Patient Clinical Laboratory',
        'path':        'Datasets/dataset2-covid19.csv',
        'target_col':  'SARSCov',
        'drop_cols':   [],
        'is_combined': False,
    },
    'heart': {
        'id':          'heart',
        'name':        'Heart Disease (Cleveland / Hungary / Statlog)',
        'path':        'Datasets/heart_statlog_cleveland_hungary_final.csv',
        'target_col':  'target',
        'drop_cols':   [],
        'is_combined': False,
    },
    'parkinson': {
        'id':          'parkinson',
        'name':        "Parkinson's Disease Biomarkers",
        'path_1':      'Datasets/parkinson/data1.csv',
        'path_2':      'Datasets/parkinson/data2.csv',
        'target_col':  'Group (Normal/PD/SWEDD - 1/2/3)',
        'drop_cols':   ['Subject_id'],
        'is_combined': True,
    },
}


def load_and_prepare(dataset_key, test_size=0.2, random_state=42):
    """
    Load a dataset, deduplicate, return split arrays + metadata dict.

    Missing values are NOT imputed here — that happens inside the
    sklearn Pipeline during training.

    Returns
    -------
    X_train, X_test, y_train, y_test, meta (dict)
    """
    cfg = DATASETS[dataset_key]

    # ── Load ──────────────────────────────────────────────────────────
    if cfg['is_combined']:
        df = pd.concat([pd.read_csv(cfg['path_1']),
                        pd.read_csv(cfg['path_2'])],
                       ignore_index=True)
    else:
        df = pd.read_csv(cfg['path'])

    raw_rows      = len(df)
    total_nulls   = int(df.isnull().sum().sum())
    null_per_col  = {c: int(n) for c, n in df.isnull().sum().items() if n > 0}

    # ── Drop identifier columns ────────────────────────────────────────
    df = df.drop(columns=[c for c in cfg['drop_cols'] if c in df.columns])

    # ── Deduplicate ────────────────────────────────────────────────────
    dups_removed  = int(df.duplicated().sum())
    df            = df.drop_duplicates().reset_index(drop=True)

    target_col    = cfg['target_col']
    X             = df.drop(columns=[target_col])
    y             = df[target_col]

    # ── Stratified split ───────────────────────────────────────────────
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    meta = {
        'dataset_id':        dataset_key,
        'dataset_name':      cfg['name'],
        'is_combined':       cfg['is_combined'],
        'target_col':        target_col,
        'feature_names':     list(X.columns),
        'raw_rows':          raw_rows,
        'final_rows':        len(df),
        'features_count':    X.shape[1],
        'duplicates_removed': dups_removed,
        'total_nulls':       total_nulls,
        'null_per_col':      null_per_col,
        'train_rows':        len(X_train),
        'test_rows':         len(X_test),
        'class_dist':        {str(k): int(v) for k, v in y.value_counts().items()},
    }
    return X_train, X_test, y_train, y_test, meta
