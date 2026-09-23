"""
Trains ONLY a logistic regression model -- deliberately simple. This is
what powers the custom HTML/CSS/JS web app, since the explanation for
"why" a prediction was made falls out of the model's own coefficients
(coefficient x scaled feature value = that feature's contribution to the
log-odds). No extra interpretability library needed.
"""
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = ROOT / "data" / "features.csv"
MODEL_PATH = ROOT / "models" / "simple_model.joblib"
FEATURE_COLUMNS_PATH = ROOT / "models" / "simple_feature_columns.joblib"


def cv(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, aucs, briers = [], [], []
    for tr, va in skf.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[va])
        pr = m.predict_proba(X.iloc[va])[:, 1]
        accs.append(accuracy_score(y.iloc[va], p))
        aucs.append(roc_auc_score(y.iloc[va], pr))
        briers.append(brier_score_loss(y.iloc[va], pr))
    return np.mean(accs), np.mean(aucs), np.mean(briers)


def main():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    acc, auc, brier = cv(None, X, y)
    print(f"CV (5-fold): acc {acc:.3f}   auc {auc:.3f}   brier {brier:.3f}")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"\nHeld-out: acc {accuracy_score(y_test, preds):.3f}   "
          f"auc {roc_auc_score(y_test, proba):.3f}   "
          f"brier {brier_score_loss(y_test, proba):.3f}")
    print(classification_report(y_test, preds, digits=3))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURE_COLUMNS_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()