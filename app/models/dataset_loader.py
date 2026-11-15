"""
dataset_loader.py — Loads, merges, deduplicates and splits each dataset.
Includes Phase II functions for Noise and Outlier evaluation.
No sklearn fitting happens here — only data preparation.
"""

import numpy as np
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


# ── Phase II: Noise & Outlier Processing ───────────────────────────────────

def detect_and_remove_outliers_iqr(X, y, factor=1.5):
    """
    Remove rows where numerical features are extreme outliers using IQR method.
    Identifies bounds: Q1 - factor*IQR to Q3 + factor*IQR.
    Returns: X_clean, y_clean, outliers_removed_count
    """
    X_df = X.copy()
    y_s = y.copy()

    Q1 = X_df.quantile(0.25)
    Q3 = X_df.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    is_outlier = (X_df < lower_bound) | (X_df > upper_bound)
    thresh = max(1, int(X_df.shape[1] * 0.15))
    outlier_mask = is_outlier.sum(axis=1) >= thresh
    outliers_removed = int(outlier_mask.sum())

    # Protect against discarding too much data
    if outliers_removed > 0 and (len(X_df) - outliers_removed) >= int(0.5 * len(X_df)):
        X_clean = X_df[~outlier_mask].reset_index(drop=True)
        y_clean = y_s[~outlier_mask].reset_index(drop=True)
    else:
        outlier_mask = is_outlier.sum(axis=1) >= max(2, int(X_df.shape[1] * 0.25))
        outliers_removed = int(outlier_mask.sum())
        X_clean = X_df[~outlier_mask].reset_index(drop=True)
        y_clean = y_s[~outlier_mask].reset_index(drop=True)

    return X_clean, y_clean, outliers_removed


def inject_noise(X, noise_level=0.15, random_state=42):
    """
    Add zero-mean Gaussian noise scaled by each numerical feature's std.
    X_noisy = X + N(0, (noise_level * std)^2)
    """
    rng = np.random.RandomState(random_state)
    X_noisy = X.copy()
    std = X_noisy.std(numeric_only=True).replace(0, 1.0)
    noise = rng.normal(0, noise_level * std.values, size=X_noisy.shape)
    X_noisy = X_noisy + noise
    return X_noisy


def smooth_with_gaussian(X, sigma=1.2):
    """
    Apply Gaussian smoothing column-by-column to reduce noise.
    Uses scipy.ndimage.gaussian_filter1d with a Gaussian kernel of width sigma.

    This models the realistic signal-processing step of smoothing a noisy
    dataset before training, to evaluate whether smoothing can recover
    performance lost due to noise injection.

    Parameters
    ----------
    X     : DataFrame of noisy features
    sigma : Standard deviation of the Gaussian kernel (larger = more smoothing)

    Returns
    -------
    X_smoothed : DataFrame with smoothed feature values
    """
    from scipy.ndimage import gaussian_filter1d

    X_smoothed = X.copy().reset_index(drop=True)
    for col in X_smoothed.select_dtypes(include='number').columns:
        X_smoothed[col] = gaussian_filter1d(X_smoothed[col].values.astype(float), sigma=sigma)
    return X_smoothed


def inject_extreme_outliers(X, contamination=0.08, factor=3.5, random_state=42):
    """
    Inject synthetic extreme outlier spikes into a random subset of rows
    to evaluate model vulnerability to extreme anomalies.
    """
    rng = np.random.RandomState(random_state)
    X_out = X.copy().reset_index(drop=True)
    n_samples = len(X_out)
    n_outliers = max(2, int(n_samples * contamination))
    outlier_idx = rng.choice(n_samples, size=n_outliers, replace=False)

    std = X_out.std(numeric_only=True).replace(0, 1.0).values
    for idx in outlier_idx:
        cols_to_corrupt = rng.choice(X_out.shape[1], size=min(3, X_out.shape[1]), replace=False)
        signs = rng.choice([-1, 1], size=len(cols_to_corrupt))
        for col, sign in zip(cols_to_corrupt, signs):
            X_out.iloc[idx, col] += sign * factor * std[col]

    return X_out, n_outliers
