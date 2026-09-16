"""
Flask backend for the custom HTML/CSS/JS Pokemon battle predictor UI.

Uses the plain logistic regression model from src/simple_train.py.
Explanations come straight from the model's own coefficients -- no SHAP,
no extra library: contribution = coefficient x scaled_feature_value.
That's basic, honest linear-model interpretability.

Run with: python3 webapp/app.py   (from the project root)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template

from features import build_features
from movesets import best_move

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "simple_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "models", "simple_feature_columns.joblib")
STATS_PATH = os.path.join(BASE_DIR, "data", "pokemon_stats.csv")

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
stats_df = pd.read_csv(STATS_PATH)
stats_lookup = stats_df.set_index("name").to_dict("index")
for name, row in stats_lookup.items():
    row["name"] = name

FEATURE_LABELS = {
    "hp_diff": "HP difference",
    "attack_diff": "Attack difference",
    "defense_diff": "Defense difference",
    "sp_atk_diff": "Sp. Atk difference",
    "sp_def_diff": "Sp. Def difference",
    "speed_diff": "Speed difference",
    "speed_advantage": "Speed advantage",
    "base_stat_total_diff": "Overall stat total",
    "p1_type_advantage": "Pokemon 1 type matchup",
    "p2_type_advantage": "Pokemon 2 type matchup",
    "type_advantage_diff": "Type matchup edge",
    "p1_effective_power": "Pokemon 1 damage output",
    "p2_effective_power": "Pokemon 2 damage output",
    "effective_power_diff": "Damage output edge",
    "effective_power_ratio": "Damage output ratio",
    "p1_best_move_multiplier": "Pokemon 1 best-move effectiveness",
    "p2_best_move_multiplier": "Pokemon 2 best-move effectiveness",
    "best_move_multiplier_diff": "Best-move effectiveness edge",
    "p1_ttk_estimate": "Pokemon 1 turns-to-KO",
    "p2_ttk_estimate": "Pokemon 2 turns-to-KO",
    "ttk_diff": "Turns-to-KO edge",
}


def slugify(name):
    return name.lower().replace(".", "").replace("'", "").replace(" ", "-")


def effectiveness_label(multiplier):
    if multiplier == 0:
        return "No effect"
    if multiplier >= 2:
        return "Super effective"
    if multiplier <= 0.5:
        return "Not very effective"
    return "Normal effectiveness"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/pokemon")
def list_pokemon():
    out = []
    for name, row in sorted(stats_lookup.items()):
        type2 = row.get("type2")
        sprite_url = row.get("sprite_url")
        out.append({
            "name": name,
            "type1": row["type1"],
            "type2": None if pd.isna(type2) else type2,
            "slug": slugify(name),
            "sprite_url": None if pd.isna(sprite_url) else sprite_url,
        })
    return jsonify(out)


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    p1_name = body.get("pokemon1")
    p2_name = body.get("pokemon2")

    if p1_name not in stats_lookup or p2_name not in stats_lookup:
        return jsonify({"error": "Unknown Pokemon name"}), 400
    if p1_name == p2_name:
        return jsonify({"error": "Pick two different Pokemon"}), 400

    p1 = stats_lookup[p1_name]
    p2 = stats_lookup[p2_name]

    feats = build_features(p1, p2)
    X = pd.DataFrame([feats])[feature_columns]

    win_prob_p1 = float(model.predict_proba(X)[0, 1])

    scaler = model.named_steps["standardscaler"]
    logreg = model.named_steps["logisticregression"]
    scaled_values = scaler.transform(X)[0]
    coefs = logreg.coef_[0]
    contributions = [
        {
            "feature": feature_columns[i],
            "label": FEATURE_LABELS.get(feature_columns[i], feature_columns[i]),
            "contribution": float(coefs[i] * scaled_values[i]),
        }
        for i in range(len(feature_columns))
    ]
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    # move analysis: which move each side would actually pick against
    # THIS opponent, and how effective it is -- the "analyze moves against
    # the other Pokemon" feature.
    p1_move = best_move(p1, p2)
    p2_move = best_move(p2, p1)

    return jsonify({
        "pokemon1": p1_name,
        "pokemon2": p2_name,
        "win_prob_1": win_prob_p1,
        "win_prob_2": 1 - win_prob_p1,
        "winner": p1_name if win_prob_p1 >= 0.5 else p2_name,
        "top_factors": contributions[:5],
        "move_analysis": {
            "pokemon1": {
                "move_type": p1_move["move_type"],
                "category": p1_move["category"],
                "multiplier": p1_move["multiplier"],
                "effectiveness": effectiveness_label(p1_move["multiplier"]),
                "estimated_damage": round(p1_move["damage"], 1),
            },
            "pokemon2": {
                "move_type": p2_move["move_type"],
                "category": p2_move["category"],
                "multiplier": p2_move["multiplier"],
                "effectiveness": effectiveness_label(p2_move["multiplier"]),
                "estimated_damage": round(p2_move["damage"], 1),
            },
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
