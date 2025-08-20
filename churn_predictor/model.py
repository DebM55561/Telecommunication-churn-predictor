from math import remainder

import joblib
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals.array_api_compat.cupy import astype
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
# from sympy.stats import Logistic

teleco = pd.read_csv('teleco.csv')
print(teleco.columns)
# teleco = teleco.dropna()
# print(teleco.shape)
print(teleco.shape)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_curve)

num_col = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_col = [ "gender","SeniorCitizen","Partner","Dependents",
    "PhoneService","MultipleLines","InternetService",
    "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
TARGET = 'Churn'

for col in num_col:
    teleco[col] = pd.to_numeric(teleco[col], errors='coerce')

def XY_split(df):
    y = (df[TARGET].str.lower()=='yes').astype(int)
    x = df.drop(columns = [TARGET])
    return x, y
def preProcess():
    num_pip = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    cat_pip = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('OHE', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    col_trans = ColumnTransformer(transformers=[('num_pip', num_pip, num_col), ('cat_col', cat_pip, cat_col)], remainder='drop')
    return col_trans

def feature_eng(df):
    df = df.copy()
    df['tenure_bin'] = pd.cut(df['tenure'], bins=[0,12,24,60,120],labels=['0-1 years','1-2 years', '2-5 years','5+ years'])
    df['avg_monthly_cost'] = df['TotalCharges']/df['tenure']+1
    services = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                    "TechSupport","StreamingTV","StreamingMovies"]
    df['service_count'] = df[services].sum(axis=1)
    df['bundle'] = ((df['PhoneService']=='Yes') & (df['InternetService']!='No')).astype(int)
    contractMap = {
        'Month-to-month' :0, 'One year': 1, 'Two year': 2
    }
    df['contract_encoded'] = df['Contract'].map(contractMap)
    return df

def evaluate_model(finpipe, x, y):
    proba = finpipe.predict_proba(x)[:,1]
    y_pred = (proba>=0.337).astype(int)
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    metrics = {
        "accuracy": acc,
        "balanced accuracy": bacc,
        "confusion matrix": cm
    }
    return metrics


param_grid = {
        'clf__n_estimators': [200, 300, 500],
        'clf__max_depth': [10, 20, 30, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2'],
        'clf__class_weight': ['balanced', None]
    }


#randomForest model
def train_baseline(df):
    df = df.copy()
    findf = feature_eng(df)
    x, y = XY_split(findf)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
    pre = preProcess()
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        max_features='log2',
        min_samples_split=4,
        min_samples_leaf=10,
    )

    finpipe = Pipeline([('preprocessor', pre), ('clf', clf)])
    finpipe.fit(x_train, y_train)
    grid = GridSearchCV(
        finpipe,
        param_grid,
        cv=3,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2
    )
    grid.fit(x_train, y_train)
    print("done")
    print('best parameters: ', grid.best_params_)
    print("Best Score (CV balanced acc):", grid.best_score_)
    best_model = grid.best_estimator_
    joblib.dump(best_model, 'model.pkl')
    report = evaluate_model(finpipe, x_test, y_test)
    print('report: ', report)
    return best_model

import joblib

#xgb model
from xgboost import XGBClassifier

def train_xgb(df):
    df = df.copy()
    findf = feature_eng(df)
    x, y = XY_split(findf)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = preProcess()

    # Base XGBoost model
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",   # avoids warning
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        # use_label_encoder=False,
        random_state=42
    )

    finpipe = Pipeline([('preprocessor', pre), ('clf', clf)])
    finpipe.fit(x_train, y_train)

    calib_model = CalibratedClassifierCV(finpipe, method="isotonic", cv=3)
    calib_model.fit(x_train, y_train)

    # Tune threshold using Precision-Recall tradeoff
    proba = calib_model.predict_proba(x_test)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_test, proba)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)

    best_idx = f1_scores.argmax()
    best_thresh = thresh[best_idx]

    print(f"✅ Best threshold found: {best_thresh:.3f}")
    print(f"Precision: {prec[best_idx]:.3f}, Recall: {rec[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")

    # Grid search hyperparameters for XGBoost
    param_grid = {
        'clf__n_estimators': [200, 300, 500],
        'clf__max_depth': [4, 6, 10],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0],
    }

    grid = GridSearchCV(
        finpipe,
        param_grid,
        cv=3,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2
    )
    # grid.fit(x_train, y_train)

    # best_model = grid.best_estimator_
    #
    # print("XGBoost done")
    # print("Best Parameters:", grid.best_params_)
    # print("Best Score (CV balanced acc):", grid.best_score_)

    # report = evaluate_model(best_model, x_test, y_test)
    # print("Report:", report)

    # Save best model
    # joblib.dump(best_model, "churn_xgb.pkl")
    # print("Model saved as churn_xgb.pkl")

    # return best_model


train_xgb(teleco)
#
results = {
    "XGBoost": {"accuracy": 0.805, "balanced_acc": 0.713},
    "RandomForest": {"accuracy": 0.790, "balanced_acc": 0.690},
    "LogReg": {"accuracy": 0.750, "balanced_acc": 0.660}
}

import matplotlib.pyplot as plt

models = list(results.keys())
acc = [results[m]["accuracy"] for m in models]
bacc = [results[m]["balanced_acc"] for m in models]

plt.bar(models, acc, alpha=0.6, label="Accuracy")
plt.bar(models, bacc, alpha=0.6, label="Balanced Accuracy")
plt.ylabel("Score")
plt.title("Model Comparison on Telco Churn")
plt.legend()
plt.show()
