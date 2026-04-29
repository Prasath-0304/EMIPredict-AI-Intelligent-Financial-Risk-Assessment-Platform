# EMIPredict AI

EMIPredict AI is a machine learning project for intelligent financial risk assessment. It predicts:

1. EMI eligibility class:
   `Eligible`, `High_Risk`, or `Not_Eligible`
2. Maximum safe monthly EMI amount for a customer

This project uses the dataset [emi_prediction_dataset.csv](d:/GUVI/EMI_Prediction_Project/emi_prediction_dataset.csv:1), a training pipeline in [train_models.py](d:/GUVI/EMI_Prediction_Project/train_models.py:1), and a Streamlit web application in [app.py](d:/GUVI/EMI_Prediction_Project/app.py:1).

## Project Status

This project is completed for local development and submission use.

Completed items:
- Dataset-driven training pipeline
- Data cleaning and feature engineering
- Multi-model comparison for classification and regression
- MLflow experiment tracking
- Best model artifact saving
- Streamlit prediction application
- End-to-end run documentation

## Folder Structure

- [app.py](d:/GUVI/EMI_Prediction_Project/app.py:1): Streamlit web app
- [train_models.py](d:/GUVI/EMI_Prediction_Project/train_models.py:1): model training and MLflow logging
- [emi_prediction_dataset.csv](d:/GUVI/EMI_Prediction_Project/emi_prediction_dataset.csv:1): source dataset
- [best_classification_model.pkl](d:/GUVI/EMI_Prediction_Project/best_classification_model.pkl:1): best saved classification pipeline
- [best_regression_model.pkl](d:/GUVI/EMI_Prediction_Project/best_regression_model.pkl:1): best saved regression pipeline
- [model_metadata.json](d:/GUVI/EMI_Prediction_Project/model_metadata.json:1): app defaults, category choices, summary
- [requirements.txt](d:/GUVI/EMI_Prediction_Project/requirements.txt:1): project dependencies
- [mlruns](d:/GUVI/EMI_Prediction_Project/mlruns): local MLflow experiment data

## Problem Statement

Financial institutions need a way to estimate whether a customer can safely take a new EMI commitment. This project evaluates customer income, expenses, liabilities, savings, and financial profile to:

- classify EMI risk level
- estimate the maximum safe EMI amount

## Dataset Overview

Dataset file:
- `emi_prediction_dataset.csv`

Main raw features include:
- personal details: age, gender, marital status, education
- employment profile: employment type, years of employment, company type
- housing and family details: house type, rent, family size, dependents
- monthly expenses: school fees, college fees, travel, groceries, utilities, other expenses
- current liabilities: existing loans, current EMI amount
- financial strength: credit score, bank balance, emergency fund
- request details: requested amount, requested tenure, EMI scenario

Target variables:
- classification target: `emi_eligibility`
- regression target: `max_monthly_emi`

## Data Cleaning

The dataset contained inconsistent values in some numeric columns, for example values like:
- `58.0.0`
- `64300.0.0`
- `nan` stored as text

Cleaning performed in [train_models.py](d:/GUVI/EMI_Prediction_Project/train_models.py:1):
- converted malformed numeric strings into valid numbers
- removed invalid target rows
- normalized categorical text values
- standardized labels such as `Male`, `Female`, `Yes`, `No`
- handled missing values through sklearn imputers

## Feature Engineering

Additional derived features are created before training and prediction:

- `total_expenses`
- `expense_to_income`
- `debt_to_income`
- `savings_to_income`
- `emergency_fund_ratio`

These engineered features improve the model's ability to represent repayment capacity and financial pressure.

## Machine Learning Approach

The project contains two separate ML tasks.

### 1. Classification

Goal:
- predict EMI eligibility class

Models compared:
- Logistic Regression
- Random Forest Classifier
- Extra Trees Classifier

Evaluation metrics:
- Accuracy
- Weighted F1-score

Best classification model from the latest run:
- `RandomForestClassifier`
- Accuracy: `0.9219`
- Weighted F1-score: `0.9009`

### 2. Regression

Goal:
- predict maximum safe monthly EMI

Models compared:
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor

Evaluation metrics:
- RMSE
- MAE
- R2 score

Best regression model from the latest run:
- `RandomForestRegressor`
- RMSE: `1143.48`
- MAE: `453.27`
- R2: `0.9785`

## Preprocessing Technique

The training pipeline uses `ColumnTransformer` and `Pipeline` from scikit-learn.

Numeric preprocessing:
- median imputation
- standard scaling

Categorical preprocessing:
- most frequent imputation
- one-hot encoding

This ensures the same preprocessing is used during both training and prediction.

## MLflow Integration

MLflow is integrated in [train_models.py](d:/GUVI/EMI_Prediction_Project/train_models.py:1).

What is tracked:
- experiment name
- model name
- task type
- dataset sample size
- metrics for each model
- logged sklearn model artifacts
- best model summary

Local MLflow tracking directory:
- [mlruns](d:/GUVI/EMI_Prediction_Project/mlruns)

## Technologies and Libraries Used

Core libraries:
- `pandas`: data loading and cleaning
- `numpy`: numerical operations
- `scikit-learn`: preprocessing, model training, evaluation, pipelines
- `mlflow`: experiment tracking and model logging
- `joblib`: saving and loading trained pipelines
- `streamlit`: web application UI

Python concepts and methods used:
- data preprocessing
- feature engineering
- supervised machine learning
- classification
- regression
- model comparison
- experiment tracking
- pipeline-based inference

## How the Project Works

### Training Flow

1. Read `emi_prediction_dataset.csv`
2. Clean malformed numeric and categorical values
3. Create engineered financial features
4. Split data into train and test sets
5. Train 3 classification models
6. Train 3 regression models
7. Log every run to MLflow
8. Select best models using evaluation metrics
9. Save best model pipelines and metadata

### Prediction Flow

1. User enters customer information in Streamlit UI
2. App creates the same engineered features used in training
3. Saved classification model predicts EMI eligibility
4. Saved regression model predicts maximum safe EMI
5. Result is displayed to the user

## How To Run This Project

Use PowerShell in the project folder:

```powershell
cd D:\GUVI\EMI_Prediction_Project
```

### 1. Create and activate virtual environment

If the virtual environment already exists, you can skip creation.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Train models

This step reads the dataset, logs MLflow runs, and saves the best models.

```powershell
python train_models.py
```

Generated files after training:
- `best_classification_model.pkl`
- `best_regression_model.pkl`
- `model_metadata.json`
- `mlruns/`

### 4. Run the Streamlit app

```powershell
streamlit run app.py
```

After running, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

Important:
- do not run `streamlit run yourscript.py`
- this project uses `app.py`, so the correct command is:

```powershell
streamlit run app.py
```

### 5. Open MLflow UI

To inspect experiments and metrics:

```powershell
python -m mlflow ui --backend-store-uri .\mlruns
```

Then open:

```text
http://localhost:5000
```

Important note:
- start MLflow from the activated project virtual environment
- if the global `mlflow` command opens the wrong Python installation or hangs during import, use:

```powershell
.\venv\Scripts\Activate.ps1
python -m mlflow ui --backend-store-uri .\mlruns
```

Alternative explicit command:

```powershell
.\venv\Scripts\mlflow.exe ui --backend-store-uri .\mlruns
```

### 6. Run Streamlit and MLflow together

To launch both services together from one command:

```powershell
.\run_project.ps1
```

This starts:
- Streamlit on `http://localhost:8501`
- MLflow UI on `http://localhost:5000`

## Expected Output

The Streamlit app shows:
- EMI eligibility class
- maximum safe monthly EMI
- summary of the entered customer features

The MLflow UI shows:
- each classification model run
- each regression model run
- tracked metrics
- model artifacts

## Model Artifacts

Saved artifacts:
- [best_classification_model.pkl](d:/GUVI/EMI_Prediction_Project/best_classification_model.pkl:1)
- [best_regression_model.pkl](d:/GUVI/EMI_Prediction_Project/best_regression_model.pkl:1)
- [model_metadata.json](d:/GUVI/EMI_Prediction_Project/model_metadata.json:1)

These are full sklearn pipelines, so preprocessing and model inference are bundled together.

## Submission Summary

This project demonstrates:
- end-to-end machine learning workflow
- dataset cleaning and transformation
- dual-task ML design: classification and regression
- experiment tracking with MLflow
- model artifact persistence
- interactive user-facing prediction app with Streamlit

## Limitations

- MLflow currently uses local file-based tracking in `mlruns`
- deployment is documented but not auto-configured in this repo
- model training uses a capped sample size of `120000` rows for practical local runtime

## Future Improvements

- use full dataset training on stronger hardware
- add hyperparameter tuning with Optuna or GridSearchCV
- register models in a database-backed MLflow server
- deploy the Streamlit app to cloud
- add unit tests and input validation tests
- add explainability with SHAP or feature importance charts

## Verification Completed

The following checks were completed locally:
- `python train_models.py`
- saved model generation
- MLflow run generation
- sample prediction using saved artifacts
- Python compile check for `app.py` and `train_models.py`

## Quick Commands

Train:

```powershell
python train_models.py
```

Run app:

```powershell
streamlit run app.py
```

Run MLflow UI:

```powershell
python -m mlflow ui --backend-store-uri .\mlruns
```
.\run_project.ps1
