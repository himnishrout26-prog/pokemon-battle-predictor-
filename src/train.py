"""
Trains a logistic regression baseline plus XGBoost and LightGBM models,
compares them, and saves the best one to models/.

Run `pip install -r requirements.txt` first (needs xgboost + lightgbm).
This script degrades gracefully to sklearn's HistGradientBoostingClassifier
if xgboost/lightgbm aren't installed, purely so you can sanity-check the
pipeline runs end-to-end -- for the real project, install the real libraries.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "data/features.csv"
MODEL_DIR = "models"


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def eval_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f}  ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, preds, digits=3))
    return acc, auc


def main():
    X_train, X_test, y_train, y_test = load_data()
    results = {}

    # 1. Baseline: logistic regression (interpretable, fast, sets the floor).
    # Scaled since differential features vary wildly in range (e.g. speed_diff
    # vs ttk_diff) and unscaled inputs slow/break lbfgs convergence.
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    baseline.fit(X_train, y_train)
    results["logistic_regression"] = (baseline, *eval_model("Logistic Regression", baseline, X_test, y_test))

    # 2. XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        results["xgboost"] = (xgb, *eval_model("XGBoost", xgb, X_test, y_test))
    except Exception as e:
        print(f"\n[xgboost unavailable ({type(e).__name__}: {e}). "
              "If this is a library-load error on Mac, run `brew install libomp` and retry. "
              "Using sklearn HistGradientBoostingClassifier as a stand-in to validate the pipeline.]")
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)
        hgb.fit(X_train, y_train)
        results["xgboost_standin"] = (hgb, *eval_model("HistGradientBoosting (xgboost stand-in)", hgb, X_test, y_test))

    # 3. LightGBM
    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        lgbm.fit(X_train, y_train)
        results["lightgbm"] = (lgbm, *eval_model("LightGBM", lgbm, X_test, y_test))
    except Exception as e:
        print(f"\n[lightgbm unavailable ({type(e).__name__}: {e}). Skipping.]")

    # pick best by ROC-AUC and save
    best_name = max(results, key=lambda k: results[k][2])
    best_model = results[best_name][0]
    print(f"\nBest model: {best_name} (ROC-AUC {results[best_name][2]:.3f}) -> saving to {MODEL_DIR}/best_model.joblib")
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.joblib")
    joblib.dump(list(X_train.columns), f"{MODEL_DIR}/feature_columns.joblib")

    return results


if __name__ == "__main__":
    main()
