"""
Trains a stat-only logistic regression baseline, a full-feature logistic
regression baseline, and (when available) XGBoost and LightGBM. Compares
them with 5-fold cross-validation plus Brier score, generates a
calibration plot, saves the best model by mean CV ROC-AUC.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import STAT_ONLY_FEATURES

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "features.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
N_SPLITS = 5


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def cv_scores(model, X, y, n_splits=N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, aucs, briers = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = m.predict(X.iloc[val_idx])
        proba = m.predict_proba(X.iloc[val_idx])[:, 1]
        accs.append(accuracy_score(y.iloc[val_idx], preds))
        aucs.append(roc_auc_score(y.iloc[val_idx], proba))
        briers.append(brier_score_loss(y.iloc[val_idx], proba))
    return {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "brier_mean": float(np.mean(briers)), "brier_std": float(np.std(briers)),
    }


def print_cv(name, s):
    print(f"{name:<28} acc {s['acc_mean']:.3f} ± {s['acc_std']:.3f}   "
          f"auc {s['auc_mean']:.3f} ± {s['auc_std']:.3f}   "
          f"brier {s['brier_mean']:.3f} ± {s['brier_std']:.3f}")


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n=== 5-fold CV (mean ± std) ===")
    candidates = {}

    # 1. Stat-only baseline: no move info at all.
    stat_cols = [c for c in STAT_ONLY_FEATURES if c in X.columns]
    stat_only = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    stat_only_scores = cv_scores(stat_only, X[stat_cols], y)
    print_cv("stat-only logistic", stat_only_scores)

    # 2. Full-feature logistic regression.
    full_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr_scores = cv_scores(full_lr, X, y)
    print_cv("full logistic", lr_scores)
    candidates["logistic_regression"] = (full_lr, lr_scores)

    # 3. XGBoost (fall back to sklearn HGB if xgboost is missing).
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42,
        )
        xgb_scores = cv_scores(xgb, X, y)
        print_cv("xgboost", xgb_scores)
        candidates["xgboost"] = (xgb, xgb_scores)
    except Exception as e:
        print(f"[xgboost unavailable ({type(e).__name__}: {e}). Using sklearn stand-in.]")
        hgb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)
        hgb_scores = cv_scores(hgb, X, y)
        print_cv("hist-gradient (xgboost stand-in)", hgb_scores)
        candidates["xgboost_standin"] = (hgb, hgb_scores)

    # 4. LightGBM.
    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        lgbm_scores = cv_scores(lgbm, X, y)
        print_cv("lightgbm", lgbm_scores)
        candidates["lightgbm"] = (lgbm, lgbm_scores)
    except Exception as e:
        print(f"[lightgbm unavailable ({type(e).__name__}: {e}). Skipping.]")

    # Pick the best by CV mean AUC and refit on the full training split.
    best_name = max(candidates, key=lambda k: candidates[k][1]["auc_mean"])
    best_model = candidates[best_name][0]
    print(f"\nBest by CV AUC: {best_name}")

    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    print(f"\n=== Held-out test ({best_name}) ===")
    print(f"Accuracy : {accuracy_score(y_test, preds):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.3f}")
    print(f"Brier    : {brier_score_loss(y_test, proba):.3f}")
    print(classification_report(y_test, preds, digits=3))

    # Calibration plot
    fig, ax = plt.subplots(figsize=(5, 5))
    CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, ax=ax)
    ax.set_title(f"Calibration: {best_name}")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "calibration.png", dpi=150)
    print(f"Saved {MODEL_DIR / 'calibration.png'}")
    plt.close(fig)
    print(f"Saved {MODEL_DIR}/calibration.png")

    # Permutation importance on the held-out set
    perm = permutation_importance(best_model, X_test, y_test,
                                  n_repeats=10, random_state=42, scoring="roc_auc")
    order = np.argsort(perm.importances_mean)[::-1]
    print("\nPermutation importance (drop in ROC-AUC):")
    for i in order:
        print(f"  {X.columns[i]:<28} {perm.importances_mean[i]:+.4f} ± {perm.importances_std[i]:.4f}")

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")
    print(f"\nSaved {MODEL_DIR / 'best_model.joblib'} ({best_name})")
    print(f"\nSaved {MODEL_DIR}/best_model.joblib ({best_name})")

    return candidates


if __name__ == "__main__":
    main()