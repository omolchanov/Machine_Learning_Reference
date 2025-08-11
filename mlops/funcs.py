import pickle
import json
import csv
from datetime import datetime
import re
import shutil
from pathlib import Path

import duckdb
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore")

# Local
# DF_PATH = "data/stagins_sales_df.csv"
# MODEL_PATH = "models/"
# EXPERIMENT_FILE_PATH = "models/exp.csv"

# Airflow
DF_PATH = "/opt/airflow/dags/data/stagins_sales_df.csv"
MODEL_PATH = "/opt/airflow/dags/models/"
EXPERIMENT_FILE_PATH = "/opt/airflow/dags/models/exp.csv"


def func_extract_data():
    df = pd.read_csv(DF_PATH)
    return df


def func_preprocess_data(data):
    df = data
    logging.info(f"Dataset shape before processing: {df.shape}")

    # Drop rows with missing 'amount'
    df_clean = df.dropna(subset=['amount', 'churn', 'new_customer'])
    logging.info(f"Dataset shape after dropping NULL values: {df_clean.shape}")

    # Remove outliers
    logging.info(f"Dataset shape before removing outliers: {df_clean.shape}")

    Q1 = df_clean['amount'].quantile(0.25)
    Q3 = df_clean['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_no_out = df_clean[(df_clean['amount'] >= lower_bound) & (df_clean['amount'] <= upper_bound)]
    logging.info(f"Dataset shape after removing outliers: {df_clean.shape}")

    return df_no_out


def func_train_model(data):
    df = data

    # Prepare features/target
    X = df[["amount", "new_customer"]]
    y = df["churn"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_score = 0.0
    best_model_name = ""

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        logging.info(f"{name} accuracy: {score:.3f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name

    logging.info(f"Best model: {best_model_name} with accuracy {best_score:.3f}")
    return best_model


def func_conduct_experiment(data, model):
    df = data

    X = df[["amount", "new_customer"]]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = None

    if isinstance(model, LogisticRegression):
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100, 10000],
            "solver": ["saga", "liblinear"],
            "penalty": ["l1", "l2"],
            "max_iter": [50000]
        }

    logging.info(f"Starting an expirement with model {model}")
    grid = GridSearchCV(model, param_grid, cv=2, scoring="accuracy", verbose=2)
    grid.fit(X_train, y_train)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save each parameter combination and corresponding mean test score
    with open(EXPERIMENT_FILE_PATH, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "params", "mean_test_score"])
        for i in range(len(grid.cv_results_["params"])):
            writer.writerow([
                timestamp,
                grid.cv_results_["params"][i],
                grid.cv_results_["mean_test_score"][i]
            ])

    logging.info(f"Saved the experiment data to {EXPERIMENT_FILE_PATH}")


def func_evaluate_model(data, model):
    df = data

    X = df[["amount", "new_customer"]]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    mean_acc_score = cv_scores.mean()

    logging.info(f"CV score: {mean_acc_score:.3f}")

    # Holdout Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    logging.info(classification_report(y_test, y_pred))
    logging.info(confusion_matrix(y_test, y_pred))
    logging.info(f"ROC-AUC: {roc_auc:.3f}")

    return mean_acc_score, roc_auc


def func_package_model(model):

    # Save the model for deployment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{MODEL_PATH}clf_{timestamp}.pkl"

    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    logging.info(f"Model saved as {filepath}")


def func_deploy_model():
    files = [f for f in Path(MODEL_PATH).iterdir() if f.is_file()]

    if not files:
        raise ValueError(f"Model directory {MODEL_PATH} does not contain evaluated models")

    latest_model = max(files, key=lambda f: f.stat().st_mtime)
    prod_model = Path(f"{MODEL_PATH}clf_prod.pkl")

    if prod_model.exists():
        prod_model.unlink()

    latest_model.rename(prod_model)

    logging.info(f"Model {latest_model} is depoyed to PROD with filename {prod_model}")


def load_prod_model():
    filepath = f"{MODEL_PATH}clf_prod.pkl"

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    return model
