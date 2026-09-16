"""
SHAP-based explainability for the trained model.

Two outputs:
1. A global summary plot (models/shap_summary.png) -- which features matter
   most across all battles. Good for the "why does this model work" story.
2. explain_prediction(p1, p2) -- a per-battle explanation used by the
   Streamlit app to show why THIS matchup was predicted the way it was.

Requires `pip install shap` (not available in the build sandbox -- this
script is written against the real shap API and should be run locally).
"""
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from features import build_features

MODEL_PATH = "models/best_model.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns.joblib"


def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


def global_summary(features_csv="data/features.csv", sample_size=1000):
    model, feature_columns = load_model()
    df = pd.read_csv(features_csv).drop(columns=["label"]).sample(
        n=min(sample_size, 100000), random_state=42
    )
    explainer = shap.Explainer(model.predict_proba, df)
    shap_values = explainer(df)

    shap.summary_plot(shap_values[..., 1], df, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=150)
    plt.close()
    print("Saved models/shap_summary.png")


def explain_prediction(p1_row, p2_row):
    """p1_row, p2_row: dict-like Pokemon stat rows. Returns (win_prob_p1, list of
    (feature_name, shap_value, human_label) sorted by |impact|, for use in the app."""
    model, feature_columns = load_model()
    feats = build_features(p1_row, p2_row)
    X = pd.DataFrame([feats])[feature_columns]

    win_prob_p1 = model.predict_proba(X)[0, 1]

    explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer(X)
    contributions = list(zip(feature_columns, shap_values[0, :, 1].values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    labels = {
        "speed_diff": "Speed advantage",
        "type_advantage_diff": "Type matchup",
        "effective_power_diff": "Damage output edge",
        "effective_power_ratio": "Damage output ratio",
        "hp_diff": "HP pool difference",
        "defense_diff": "Defense difference",
        "base_stat_total_diff": "Overall stat total",
        "ttk_diff": "Speed-to-kill estimate",
    }
    readable = [(labels.get(name, name), val) for name, val in contributions[:6]]
    return win_prob_p1, readable


if __name__ == "__main__":
    global_summary()
