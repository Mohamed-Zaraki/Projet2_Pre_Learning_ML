# Machine Learning Preprocessing & Modeling Platform (MVC) — Phase II

An end-to-end Machine Learning web platform built with **Flask**, **scikit-learn (`Pipeline` & `ColumnTransformer`)**, **SQLite**, and **Pickle** following the **MVC (Model-View-Controller)** architectural pattern.

---

## 🎯 Architecture Overview (Two-Phase Design)

As required, model training and hyperparameter search are strictly decoupled from the web application:

1. **Offline Training & Optimization Phase (`train_and_save.py`)**:
   - Evaluates the **Influence of Noise and Outliers**:
     - *Baseline* (Standard clean data).
     - *Without Outliers* (Outliers detected and removed using the statistical Interquartile Range / IQR method).
     - *With Feature Noise* (Continuous numerical features perturbed with Gaussian noise).
     - *With Outliers* (Synthetic extreme outlier spikes injected to test failure points).
   - Evaluates the **Influence of Model Parameters manually**:
     - **KNN**: Evaluates neighbor choices ($k \in [1, 21]$).
     - **Ensemble (Random Forest)**: Evaluates estimator counts (`n_estimators` $\in [10, 200]$).
     - **SVM**: Evaluates kernel influence (`rbf`, `linear`, `poly`).
   - Automatically benchmarks and applies the best scaler (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).
   - Implements `SimpleImputer` inside `Pipeline` and `ColumnTransformer` (prevents data leakage).
   - Handles deduplication and merges Parkinson's `data1.csv` & `data2.csv` (dropping `Subject_id`).
   - Generates standalone base64 visualizations (Noise & Outlier Comparison, Parameter Sensitivity Curves, Confusion Matrices, ROC, Overfitting).
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
│   │   ├── dataset_loader.py      # Loads datasets, handles deduplication, splits, IQR outliers, noise injection
│   │   ├── ml_manager.py          # Pipeline + ColumnTransformer + Manual parameter tuning + Noise/Outlier evaluation
│   │   ├── visualizer.py          # Generates base64 PNGs (Noise/Outlier plot, Parameter Sensitivity, CM, ROC)
│   │   └── db_manager.py          # SQLite database interface & Pickle split serialization
│   │
│   ├── controllers/               # CONTROLLER LAYER (Read-Only)
│   │   └── main_controller.py     # Routes: Dashboard (/), Dataset Results (/dataset/<id>), History (/history)
│   │
│   ├── templates/                 # VIEW LAYER (Jinja2 Templates)
│   │   ├── base.html              # Layout with dark navbar
│   │   ├── index.html             # Dashboard summarizing all 4 datasets & winning models
│   │   ├── dataset_view.html      # Phase II metrics table, noise/outliers audit, winner banner, modular plot tabs
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
├── train_and_save.py              # Offline training & Phase II evaluation script
├── run.py                         # Flask web application entry point
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Run the Offline Training & Phase II Evaluation
```bash
python train_and_save.py
```
*(Trains all 4 datasets, evaluates noise/outliers & manual parameters, saves to SQLite and Pickle)*

### 2. Start the Read-Only Flask Web Application
```bash
python run.py
```

### 3. Open in Your Browser
Navigate to:
```
http://127.0.0.1:5000
```
