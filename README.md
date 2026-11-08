# Machine Learning Preprocessing & Modeling Platform (MVC)

An end-to-end Machine Learning web platform built with **Flask**, **scikit-learn (`Pipeline` & `ColumnTransformer`)**, **GridSearchCV**, **SQLite**, and **Pickle** following the **MVC (Model-View-Controller)** architectural pattern.

---

## 🎯 Architecture Overview (Two-Phase Design)

As required, model training and hyperparameter search are strictly decoupled from the web application:

1. **Offline Training & Optimization Phase (`train_and_save.py`)**:
   - Runs **GridSearchCV** on KNN, SVM, and Decision Tree with 5-fold cross-validation.
   - Automatically evaluates and applies the best scaler (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).
   - Implements `SimpleImputer` inside `Pipeline` and `ColumnTransformer` (prevents data leakage).
   - Handles deduplication and merges Parkinson's `data1.csv` & `data2.csv` (dropping `Subject_id`).
   - Generates standalone, non-scrolling base64 visualizations.
   - Saves all results, hyperparameters, metrics, and plots into **SQLite** (`instance/ml_workbench.db`).
   - Serializes train/test splits to **Pickle** (`storage/splits/*.pkl`) and fitted pipelines to (`storage/models/*.pkl`).

2. **Web Presentation Phase (`run.py` / Flask)**:
   - **100% Read-Only**: The website does **not** train models, execute training jobs, or take training form inputs.
   - It simply reads the pre-computed results from SQLite and renders a clean, modern dashboard with responsive tabs and cards.

---

## 📂 Project Structure (MVC)

```
Projet2_Pre_Learning_ML/
├── app/
│   ├── models/                    # MODEL LAYER
│   │   ├── dataset_loader.py      # Loads datasets, combines Parkinson data1+data2, dedupes, splits
│   │   ├── ml_manager.py          # Pipeline + ColumnTransformer + GridSearchCV (KNN, SVM, Decision Tree)
│   │   ├── visualizer.py          # Generates base64 PNGs (Confusion Matrices, ROC, Benchmark, Overfitting)
│   │   └── db_manager.py          # SQLite database interface & Pickle split serialization
│   │
│   ├── controllers/               # CONTROLLER LAYER (Read-Only)
│   │   └── main_controller.py     # Routes: Dashboard (/), Dataset Results (/dataset/<id>), History (/history)
│   │
│   ├── templates/                 # VIEW LAYER (Jinja2 Templates)
│   │   ├── base.html              # Layout with dark navbar
│   │   ├── index.html             # Dashboard summarizing all 4 datasets & winning models
│   │   ├── dataset_view.html      # Comprehensive metrics table, winner banner, modular plot tabs
│   │   └── history_view.html      # Full table of saved SQLite experiment runs
│   │
│   └── static/css/style.css       # Clean typography, cards, badges, no-scroll plot rules
│
├── Datasets/                      # Raw datasets
│   ├── alcohol.csv
│   ├── dataset2-covid19.csv
│   ├── heart_statlog_cleveland_hungary_final.csv
│   └── parkinson/ (data1.csv & data2.csv)
│
├── instance/
│   └── ml_workbench.db            # SQLite database
├── storage/
│   ├── splits/                    # Pickled train/test splits (.pkl)
│   └── models/                    # Pickled trained pipelines (.pkl)
│
├── train_and_save.py              # Offline training & GridSearchCV script
├── run.py                         # Flask web application entry point
└── requirements.txt
```

---

## 📊 Summary of Results from GridSearchCV

| Dataset | Raw Rows | After Dedup | Selected Scaler | Winning Model | Test Accuracy | ROC-AUC | 5-Fold CV Score |
|---|---|---|---|---|---|---|---|
| **Alcohol Biometrics** | 390 | 390 (0 dups) | `MinMaxScaler` | **Decision Tree** (`max_depth=3`) | **97.44%** | **0.997** | 98.7% ± 0.6% |
| **COVID-19 Laboratory** | 1,736 | 1,727 (9 dups) | `StandardScaler` | **SVM** (`C=1.0, RBF`) | **77.17%** | **0.848** | 81.5% ± 3.4% |
| **Heart Disease** | 1,190 | 918 (272 dups) | `StandardScaler` | **KNN** (`k=7, manhattan`) | **91.30%** | **0.934** | 87.3% ± 3.5% |
| **Parkinson's Disease** | 1,168 | 584 (584 dups) | `MinMaxScaler` | **KNN** (`k=7, manhattan`) | **96.58%** | **0.975** | 95.7% ± 2.8% |

---

## 🚀 How to Run

### 1. Run the Offline Training & GridSearchCV (Already done, can re-run anytime)
```bash
python train_and_save.py
```
*(Trains all 4 datasets, optimizes hyperparameters via GridSearchCV, saves to SQLite and Pickle)*

### 2. Start the Read-Only Flask Web Application
```bash
python run.py
```

### 3. Open in Your Browser
Navigate to:
```
http://127.0.0.1:5000
```
