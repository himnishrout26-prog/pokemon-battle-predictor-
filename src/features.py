"""
Feature engineering for the battle predictor.

Deliberately does NOT feed raw stats straight into the model. Instead it
builds differentials and matchup scores, the same quantities a player
would actually reason about ("who's faster", "who hits harder", "which
move actually connects best here"), because those transfer better than
raw numbers and are far more interpretable later.

The "best move" features come from src/movesets.py: for each side, we
work out which move in its (synthetic) movepool does the most damage to
THIS SPECIFIC opponent, not just a generic STAB estimate. That's the
"analyze moves against the other Pokemon" piece.
"""
import pandas as pd
from type_chart import type_multiplier
from movesets import best_move


def build_features(p1, p2):
    """p1, p2: dict-like rows (name/type1/type2/hp/attack/defense/sp_atk/sp_def/speed).
    Returns a flat dict of features, always framed as p1-vs-p2 so it's symmetric-safe
    (swap p1/p2 and every feature flips sign / inverts, which we rely on for augmentation)."""
    f = {}

    f["hp_diff"] = p1["hp"] - p2["hp"]
    f["attack_diff"] = p1["attack"] - p2["attack"]
    f["defense_diff"] = p1["defense"] - p2["defense"]
    f["sp_atk_diff"] = p1["sp_atk"] - p2["sp_atk"]
    f["sp_def_diff"] = p1["sp_def"] - p2["sp_def"]
    f["speed_diff"] = p1["speed"] - p2["speed"]
    f["speed_advantage"] = 1 if p1["speed"] > p2["speed"] else (-1 if p1["speed"] < p2["speed"] else 0)

    bst1 = p1["hp"] + p1["attack"] + p1["defense"] + p1["sp_atk"] + p1["sp_def"] + p1["speed"]
    bst2 = p2["hp"] + p2["attack"] + p2["defense"] + p2["sp_atk"] + p2["sp_def"] + p2["speed"]
    f["base_stat_total_diff"] = bst1 - bst2

    # type matchup: how well p1's primary type hits p2, and vice versa
    p1_types = [p1["type1"], p1.get("type2")]
    p2_types = [p2["type1"], p2.get("type2")]
    f["p1_type_advantage"] = type_multiplier(p1["type1"], p2_types)
    f["p2_type_advantage"] = type_multiplier(p2["type1"], p1_types)
    f["type_advantage_diff"] = f["p1_type_advantage"] - f["p2_type_advantage"]

    # best-move analysis: which move from each side's movepool hits hardest
    # against THIS specific opponent -- this replaces the old generic
    # "effective power" estimate with the actual optimal-move logic.
    p1_best = best_move(p1, p2)
    p2_best = best_move(p2, p1)
    f["p1_effective_power"] = p1_best["damage"]
    f["p2_effective_power"] = p2_best["damage"]
    f["effective_power_diff"] = p1_best["damage"] - p2_best["damage"]
    f["effective_power_ratio"] = p1_best["damage"] / max(p2_best["damage"], 0.01)
    f["p1_best_move_multiplier"] = p1_best["multiplier"]
    f["p2_best_move_multiplier"] = p2_best["multiplier"]
    f["best_move_multiplier_diff"] = p1_best["multiplier"] - p2_best["multiplier"]

    # rough "turns to KO" estimate -- ties best-move damage to opponent's HP pool
    f["p1_ttk_estimate"] = p2["hp"] * 2 / max(p1_best["damage"], 0.01)
    f["p2_ttk_estimate"] = p1["hp"] * 2 / max(p2_best["damage"], 0.01)
    f["ttk_diff"] = f["p2_ttk_estimate"] - f["p1_ttk_estimate"]  # positive favors p1 (p1 kills faster)

    return f


def build_feature_dataframe(battles_df, stats_df):
    stats_lookup = stats_df.set_index("name").to_dict("index")
    for name, row in stats_lookup.items():
        row["name"] = name

    feature_rows = []
    for _, battle in battles_df.iterrows():
        p1 = stats_lookup[battle["pokemon_1"]]
        p2 = stats_lookup[battle["pokemon_2"]]
        feats = build_features(p1, p2)
        feats["label"] = battle["label"]
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)


if __name__ == "__main__":
    stats = pd.read_csv("data/pokemon_stats.csv")
    battles = pd.read_csv("data/battles.csv")
    features = build_feature_dataframe(battles, stats)
    features.to_csv("data/features.csv", index=False)
    print(f"Built {len(features)} feature rows, {features.shape[1] - 1} features")
    print(features.columns.tolist())
