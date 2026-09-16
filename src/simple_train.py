"""
Trains ONLY a logistic regression model -- deliberately simple, no
gradient boosting, no SHAP. This is what powers the custom HTML/CSS/JS
web app, since the explanation for "why" a prediction was made just
falls out of the model's own coefficients (coefficient x scaled feature
value = that feature's contribution to the log-odds). No extra
interpretability library needed.

This is a separate, smaller model from src/train.py (which is the
XGBoost/LightGBM version for the main resume-facing pipeline) -- keep
both if you want to talk about the tradeoff between them in an interview.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "data/features.csv"
MODEL_PATH = "models/simple_model.joblib"
FEATURE_COLUMNS_PATH = "models/simple_feature_columns.joblib"


def main():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, preds, digits=3))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURE_COLUMNS_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
