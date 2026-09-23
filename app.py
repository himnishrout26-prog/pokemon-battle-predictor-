"""
Flask backend for the custom HTML/CSS/JS Pokemon battle predictor UI.

Uses the plain logistic regression model from src/simple_train.py.
Explanations come straight from the model's coefficients:
    contribution = coefficient x scaled_feature_value (per feature)
which is exactly that feature's share of the log-odds toward P(p1 wins).

Endpoints:
  GET  /api/pokemon    -> roster for autocomplete
  POST /api/predict    -> prediction + charts + replay + N-battle simulation

Run with: python3 webapp/app.py   (from the project root)
"""
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import build_features
from movesets import all_moves
from battle_simulator import simulate_battle

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "simple_model.joblib"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "simple_feature_columns.joblib"
STATS_PATH = BASE_DIR / "data" / "pokemon_stats.csv"

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
    "type_advantage_diff": "Type matchup edge",
}


def slugify(name):
    return name.lower().replace(".", "").replace("'", "").replace(" ", "-")


def effectiveness_label(mult):
    if mult == 0:
        return "No effect"
    if mult >= 2:
        return "Super effective"
    if mult <= 0.5:
        return "Not very effective"
    return "Normal effectiveness"


def clean_row(row, name):
    """Dict-safe copy of a stats row with name filled in and NaN -> None."""
    out = {}
    for k, v in row.items():
        if isinstance(v, float) and v != v:
            out[k] = None
        else:
            out[k] = v
    out["name"] = name
    return out


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
    try:
        body = request.get_json(force=True) or {}
        p1_name = body.get("pokemon1")
        p2_name = body.get("pokemon2")
        n_sim = int(body.get("n_sim", 200))
        n_sim = max(20, min(n_sim, 2000))

        if p1_name not in stats_lookup or p2_name not in stats_lookup:
            return jsonify({"error": "Unknown Pokemon name"}), 400
        if p1_name == p2_name:
            return jsonify({"error": "Pick two different Pokemon"}), 400

        p1 = clean_row(stats_lookup[p1_name], p1_name)
        p2 = clean_row(stats_lookup[p2_name], p2_name)

        # ---- Model prediction + linear-model contributions ----
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

        # ---- Move analysis (all moves, not just the best) ----
        p1_moves = all_moves(p1, p2)
        p2_moves = all_moves(p2, p1)
        for m in p1_moves:
            m["effectiveness"] = effectiveness_label(m["multiplier"])
        for m in p2_moves:
            m["effectiveness"] = effectiveness_label(m["multiplier"])

        # ---- Deterministic replay (one battle, full turn log) ----
        replay_rng = random.Random()  # fresh seed each request -> different replay each time
        replay_winner, replay_turns, replay_log = simulate_battle(
            p1, p2, replay_rng, return_log=True
        )

        # ---- Monte Carlo: N simulator runs, histogram of turns + win rates ----
        sim_rng = random.Random()
        p1_wins = 0
        turns_list = []
        for _ in range(n_sim):
            w, t = simulate_battle(p1, p2, sim_rng)
            turns_list.append(t)
            if w == p1_name:
                p1_wins += 1

        turns_hist = Counter(turns_list)
        turns_histogram = {
            str(k): turns_hist.get(k, 0)
            for k in range(1, max(turns_list) + 1)
        }

        stat_labels = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
        stat_keys = ["hp", "attack", "defense", "sp_atk", "sp_def", "speed"]

        return jsonify({
            "pokemon1": {"name": p1_name, "type1": p1["type1"], "type2": p1["type2"]},
            "pokemon2": {"name": p2_name, "type1": p2["type1"], "type2": p2["type2"]},
            "win_prob_1": win_prob_p1,
            "win_prob_2": 1 - win_prob_p1,
            "winner": p1_name if win_prob_p1 >= 0.5 else p2_name,
            "top_factors": contributions[:8],
            "moves": {"pokemon1": p1_moves, "pokemon2": p2_moves},
            "stat_radar": {
                "labels": stat_labels,
                "pokemon1": [p1[k] for k in stat_keys],
                "pokemon2": [p2[k] for k in stat_keys],
            },
            "replay": {
                "winner": replay_winner,
                "turns": replay_turns,
                "log": replay_log,
            },
            "simulation": {
                "n": n_sim,
                "p1_wins": p1_wins,
                "p2_wins": n_sim - p1_wins,
                "p1_win_rate": p1_wins / n_sim,
                "mean_turns": sum(turns_list) / len(turns_list),
                "turns_histogram": turns_histogram,
            },
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)