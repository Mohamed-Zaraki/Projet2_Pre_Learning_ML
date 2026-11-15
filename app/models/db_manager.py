"""
db_manager.py — SQLite persistence layer for Phase II.

Responsible for:
  - Initialising the schema (init_db)
  - Writing completed offline experiment runs (save_run), including
    Phase II noise/outlier evaluations and parameter sensitivity
  - Reading experiments for the Flask read-only frontend
"""

import json
import os
import pickle
import sqlite3
from datetime import datetime

DB_PATH   = os.path.join('instance', 'ml_workbench.db')
SPLITS_DIR = os.path.join('storage', 'splits')
MODELS_DIR = os.path.join('storage', 'models')


def _conn():
    os.makedirs('instance', exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Create all tables if they don't exist yet."""
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    with _conn() as c:
        c.executescript('''
            CREATE TABLE IF NOT EXISTS training_runs (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp          TEXT    NOT NULL,
                dataset_id         TEXT    NOT NULL,
                dataset_name       TEXT    NOT NULL,
                raw_rows           INTEGER NOT NULL,
                final_rows         INTEGER NOT NULL,
                features_count     INTEGER NOT NULL,
                feature_names      TEXT    NOT NULL,   -- JSON list
                duplicates_removed INTEGER NOT NULL,
                missing_strategy   TEXT    NOT NULL,
                scaler_used        TEXT    NOT NULL,
                test_size          REAL    NOT NULL,
                cv_folds           INTEGER NOT NULL,
                split_pickle_path  TEXT    NOT NULL,
                best_model_key     TEXT    NOT NULL,
                noise_outlier_json TEXT                -- Phase II JSON
            );

            CREATE TABLE IF NOT EXISTS model_evaluations (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id             INTEGER NOT NULL,
                model_key          TEXT    NOT NULL,
                model_name         TEXT    NOT NULL,
                best_params        TEXT    NOT NULL,  -- JSON
                test_accuracy      REAL    NOT NULL,
                train_accuracy     REAL    NOT NULL,
                cv_mean            REAL    NOT NULL,
                cv_std             REAL    NOT NULL,
                precision_val      REAL    NOT NULL,
                recall_val         REAL    NOT NULL,
                f1_val             REAL    NOT NULL,
                auc_val            REAL    NOT NULL,
                overfitting_status TEXT    NOT NULL,
                overfitting_badge  TEXT    NOT NULL,
                overfitting_desc   TEXT    NOT NULL,
                tn INTEGER NOT NULL,
                fp INTEGER NOT NULL,
                fn INTEGER NOT NULL,
                tp INTEGER NOT NULL,
                model_pickle_path  TEXT,
                param_curve        TEXT,              -- Phase II sensitivity curve JSON
                FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS plots (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     INTEGER NOT NULL,
                plot_type  TEXT    NOT NULL,   -- 'roc' | 'benchmark' | 'noise_outlier' | ...
                plot_b64   TEXT    NOT NULL,   -- base64 PNG data URI
                FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
            );
        ''')

        # Add columns if migrating an older database
        try:
            c.execute('ALTER TABLE training_runs ADD COLUMN noise_outlier_json TEXT')
        except sqlite3.OperationalError:
            pass
        try:
            c.execute('ALTER TABLE model_evaluations ADD COLUMN param_curve TEXT')
        except sqlite3.OperationalError:
            pass


# ── Write path (called only by train_and_save.py offline) ──────────────────

def save_run(dataset_id, dataset_name, raw_rows, final_rows,
             features_count, feature_names, duplicates_removed,
             missing_strategy, scaler_used, test_size, cv_folds,
             X_train, X_test, y_train, y_test,
             results: dict, best_model_key: str,
             models_dict: dict, plots_dict: dict,
             sensitivity_dict: dict = None,
             noise_outlier_data: dict = None):
    """
    Persist one complete experiment to SQLite + pickle files.
    """
    init_db()
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    code = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Pickle split
    split_path = os.path.join(SPLITS_DIR, f'{dataset_id}_{code}_split.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump({
            'dataset_id':    dataset_id,
            'timestamp':     ts,
            'feature_names': feature_names,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
        }, f)

    # 2. Pickle each fitted pipeline
    model_paths = {}
    for key, pipe in models_dict.items():
        path = os.path.join(MODELS_DIR, f'{dataset_id}_{key}_{code}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(pipe, f)
        model_paths[key] = path

    # 3. Insert into SQLite
    with _conn() as c:
        cur = c.execute('''
            INSERT INTO training_runs
            (timestamp, dataset_id, dataset_name, raw_rows, final_rows,
             features_count, feature_names, duplicates_removed,
             missing_strategy, scaler_used, test_size, cv_folds,
             split_pickle_path, best_model_key, noise_outlier_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (ts, dataset_id, dataset_name, raw_rows, final_rows,
              features_count, json.dumps(feature_names), duplicates_removed,
              missing_strategy, scaler_used, test_size, cv_folds,
              split_path, best_model_key,
              json.dumps(noise_outlier_data or {})))
        run_id = cur.lastrowid

        # Model benchmarks
        for key, res in results.items():
            cm = res['confusion_matrix']
            curve = sensitivity_dict.get(key, []) if sensitivity_dict else []
            c.execute('''
                INSERT INTO model_evaluations
                (run_id, model_key, model_name, best_params,
                 test_accuracy, train_accuracy, cv_mean, cv_std,
                 precision_val, recall_val, f1_val, auc_val,
                 overfitting_status, overfitting_badge, overfitting_desc,
                 tn, fp, fn, tp, model_pickle_path, param_curve)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (run_id, key, res['name'],
                  json.dumps(res.get('best_params', {})),
                  res['test_accuracy'], res['train_accuracy'],
                  res['cv_mean'], res['cv_std'],
                  res['precision'], res['recall'],
                  res['f1'], res['auc'],
                  res['overfitting_status'], res['overfitting_badge'],
                  res['overfitting_desc'],
                  cm['tn'], cm['fp'], cm['fn'], cm['tp'],
                  model_paths.get(key, ''),
                  json.dumps(curve)))

        # Plots (base64)
        for ptype, b64 in plots_dict.items():
            c.execute(
                'INSERT INTO plots (run_id, plot_type, plot_b64) VALUES (?,?,?)',
                (run_id, ptype, b64))

    return run_id, split_path


# ── Read path (called by Flask controllers) ─────────────────────────────────

def get_all_runs(limit=50):
    init_db()
    with _conn() as c:
        return [dict(r) for r in c.execute(
            'SELECT * FROM training_runs ORDER BY id DESC LIMIT ?', (limit,))]


def _parse_run_record(row, c):
    run = dict(row)
    run['feature_names'] = json.loads(run['feature_names'])
    run['noise_outlier_data'] = json.loads(run.get('noise_outlier_json') or '{}')

    run['models'] = [
        dict(m) | {
            'best_params': json.loads(m['best_params']),
            'param_curve': json.loads(m['param_curve'] or '[]')
        }
        for m in c.execute(
            'SELECT * FROM model_evaluations WHERE run_id=? ORDER BY test_accuracy DESC',
            (run['id'],))
    ]

    run['plots'] = {
        r['plot_type']: r['plot_b64']
        for r in c.execute('SELECT plot_type, plot_b64 FROM plots WHERE run_id=?', (run['id'],))
    }
    return run


def get_run(run_id):
    init_db()
    with _conn() as c:
        row = c.execute('SELECT * FROM training_runs WHERE id=?', (run_id,)).fetchone()
        if not row:
            return None
        return _parse_run_record(row, c)


def get_latest_run_per_dataset():
    """Returns the most recent run for each dataset_id."""
    init_db()
    with _conn() as c:
        rows = c.execute('''
            SELECT t.* FROM training_runs t
            INNER JOIN (
                SELECT dataset_id, MAX(id) as max_id
                FROM training_runs GROUP BY dataset_id
            ) g ON t.id = g.max_id
            ORDER BY t.id DESC
        ''').fetchall()
        return [_parse_run_record(row, c) for row in rows]
