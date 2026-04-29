import json
import re
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "emi_prediction_dataset.csv"
CLASSIFIER_PATH = BASE_DIR / "best_classification_model.pkl"
REGRESSOR_PATH = BASE_DIR / "best_regression_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"
MLRUNS_DIR = BASE_DIR / "mlruns"
EXPERIMENT_NAME = "EMIPredict AI"
RANDOM_STATE = 42
SAMPLE_SIZE = 120000

NUMERIC_COLUMNS = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
]

ENGINEERED_COLUMNS = [
    "total_expenses",
    "expense_to_income",
    "debt_to_income",
    "savings_to_income",
    "emergency_fund_ratio",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ENGINEERED_COLUMNS


def normalize_category(column, value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "<na>"}:
        return np.nan

    lowered = text.lower()
    if column == "gender":
        mapping = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
        return mapping.get(lowered, text.title())
    if column == "existing_loans":
        mapping = {"yes": "Yes", "no": "No"}
        return mapping.get(lowered, text.title())
    if column == "employment_type":
        mapping = {
            "government": "Government",
            "private": "Private",
            "self-employed": "Self-employed",
            "self employed": "Self-employed",
        }
        return mapping.get(lowered, text.title())
    if column == "company_type":
        mapping = {"mnc": "MNC"}
        return mapping.get(lowered, text.title())
    return text.title()


def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    cleaned = str(value).strip().replace(",", "")
    if cleaned == "" or cleaned.lower() == "nan":
        return np.nan

    cleaned = re.sub(r"(\.0)+$", "", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return pd.to_numeric(cleaned, errors="coerce")


def load_dataset():
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    for column in NUMERIC_COLUMNS + ["max_monthly_emi"]:
        df[column] = df[column].apply(clean_numeric_value)

    for column in CATEGORICAL_COLUMNS + ["emi_eligibility"]:
        df[column] = df[column].apply(lambda value: normalize_category(column, value))

    df["total_expenses"] = (
        df["school_fees"].fillna(0)
        + df["college_fees"].fillna(0)
        + df["travel_expenses"].fillna(0)
        + df["groceries_utilities"].fillna(0)
        + df["other_monthly_expenses"].fillna(0)
        + df["current_emi_amount"].fillna(0)
        + df["monthly_rent"].fillna(0)
    )

    salary = df["monthly_salary"].replace(0, np.nan)
    df["expense_to_income"] = df["total_expenses"] / salary
    df["debt_to_income"] = df["current_emi_amount"] / salary
    df["savings_to_income"] = df["bank_balance"] / salary
    df["emergency_fund_ratio"] = df["emergency_fund"] / salary

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["emi_eligibility", "max_monthly_emi"])
    df = df[df["max_monthly_emi"] >= 0].copy()

    if len(df) > SAMPLE_SIZE:
        strata = df["emi_eligibility"]
        df, _ = train_test_split(
            df,
            train_size=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
            stratify=strata,
        )

    return df.reset_index(drop=True)


def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS + ENGINEERED_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def evaluate_classifier(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
    }


def evaluate_regressor(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    return {
        "rmse": rmse,
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def train_and_log_models(df):
    x = df[FEATURE_COLUMNS]
    y_class = df["emi_eligibility"]
    y_reg = df["max_monthly_emi"]

    x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(
        x,
        y_class,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )
    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        x,
        y_reg,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    classifier_candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=None),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "extra_trees_classifier": ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    regressor_candidates = {
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "extra_trees_regressor": ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_classifier = None
    best_classifier_metrics = None
    best_classifier_name = None

    best_regressor = None
    best_regressor_metrics = None
    best_regressor_name = None

    for name, estimator in classifier_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        with mlflow.start_run(run_name=name):
            metrics = evaluate_classifier(pipeline, x_train_cls, x_test_cls, y_train_cls, y_test_cls)
            mlflow.log_param("task", "classification")
            mlflow.log_param("model_name", name)
            mlflow.log_param("sample_size", len(df))
            mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, name=f"classifier_{name}")

        if best_classifier_metrics is None or metrics["weighted_f1"] > best_classifier_metrics["weighted_f1"]:
            best_classifier = pipeline
            best_classifier_metrics = metrics
            best_classifier_name = name

    for name, estimator in regressor_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        with mlflow.start_run(run_name=name):
            metrics = evaluate_regressor(pipeline, x_train_reg, x_test_reg, y_train_reg, y_test_reg)
            mlflow.log_param("task", "regression")
            mlflow.log_param("model_name", name)
            mlflow.log_param("sample_size", len(df))
            mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, name=f"regressor_{name}")

        if best_regressor_metrics is None or metrics["rmse"] < best_regressor_metrics["rmse"]:
            best_regressor = pipeline
            best_regressor_metrics = metrics
            best_regressor_name = name

    with mlflow.start_run(run_name="training_summary"):
        mlflow.log_param("task", "summary")
        mlflow.log_param("sample_size", len(df))
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("best_classifier", best_classifier_name)
        mlflow.log_param("best_regressor", best_regressor_name)
        mlflow.log_metrics(
            {
                "best_classifier_accuracy": best_classifier_metrics["accuracy"],
                "best_classifier_weighted_f1": best_classifier_metrics["weighted_f1"],
                "best_regressor_rmse": best_regressor_metrics["rmse"],
                "best_regressor_mae": best_regressor_metrics["mae"],
                "best_regressor_r2": best_regressor_metrics["r2"],
            }
        )

    return best_classifier, best_classifier_name, best_classifier_metrics, best_regressor, best_regressor_name, best_regressor_metrics


def save_outputs(df, classifier, regressor, summary):
    joblib.dump(classifier, CLASSIFIER_PATH)
    joblib.dump(regressor, REGRESSOR_PATH)

    defaults = {}
    for column in NUMERIC_COLUMNS:
        defaults[column] = float(df[column].median())
    for column in CATEGORICAL_COLUMNS:
        defaults[column] = str(df[column].mode(dropna=True).iloc[0])

    metadata = {
        "defaults": defaults,
        "choices": {
            column: sorted(df[column].dropna().astype(str).unique().tolist()) for column in CATEGORICAL_COLUMNS
        },
        "features": FEATURE_COLUMNS,
        "summary": summary,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    df = load_dataset()
    (
        classifier,
        classifier_name,
        classifier_metrics,
        regressor,
        regressor_name,
        regressor_metrics,
    ) = train_and_log_models(df)

    summary = {
        "dataset_rows_used": len(df),
        "best_classifier": {
            "name": classifier_name,
            "metrics": classifier_metrics,
        },
        "best_regressor": {
            "name": regressor_name,
            "metrics": regressor_metrics,
        },
    }
    save_outputs(df, classifier, regressor, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
