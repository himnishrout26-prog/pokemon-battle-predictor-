"""
Feature engineering for the battle predictor.

Deliberately does NOT feed raw stats straight into the model. It builds
differentials and a type-matchup score -- the same quantities a player
would reason about ("who's faster", "whose typing lines up better here").

Design note on what is and isn't included:

- We DO include the type-effectiveness multiplier (from type_chart.py),
  because it's a static property of the two Pokemon's typings. The
  simulator still has to combine it with stats to compute damage.
- We DO NOT include best_move()["damage"] or any other output of the
  simulator's damage formula. Those caused the model to just re-derive
  the simulator's own math instead of learning to predict outcomes.
- We DO NOT include base_stat_total_diff -- it's a linear combination of
  the six stat diffs already present, and letting trees split on it
  collapses the model back into a single-feature stat lookup.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from type_chart import type_multiplier

ROOT = Path(__file__).resolve().parent.parent

STAT_NAMES = ("hp", "attack", "defense", "sp_atk", "sp_def", "speed")


def _defending_types(pokemon):
    return [pokemon["type1"], pokemon.get("type2")]


def build_features(p1, p2):
    """p1, p2: dict-like rows with name/type1/type2/<stats>.

    Returns a flat dict of features framed as p1-vs-p2. Every signed
    feature flips sign under a p1/p2 swap, which the tests rely on."""
    f = {}

    for stat in STAT_NAMES:
        f[f"{stat}_diff"] = p1[stat] - p2[stat]

    f["speed_advantage"] = (p1["speed"] > p2["speed"]) - (p1["speed"] < p2["speed"])

    # Type matchup edge. Uses ONLY the type chart -- no damage formula,
    # no best_move() call -- so the model can't read the simulator's
    # damage output off its inputs. It has to combine this with stat
    # diffs itself.
    p1_adv = type_multiplier(p1["type1"], _defending_types(p2))
    p2_adv = type_multiplier(p2["type1"], _defending_types(p1))
    f["type_advantage_diff"] = p1_adv - p2_adv

    return f


# Subset used for the "stat-only baseline" comparison in train.py.
STAT_ONLY_FEATURES = [
    "hp_diff", "attack_diff", "defense_diff",
    "sp_atk_diff", "sp_def_diff", "speed_diff",
    "speed_advantage",
]


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
    stats = pd.read_csv(ROOT / "data" / "pokemon_stats.csv")
    battles = pd.read_csv(ROOT / "data" / "battles.csv")
    features = build_feature_dataframe(battles, stats)
    features.to_csv(ROOT / "data" / "features.csv", index=False)
    print(f"Built {len(features)} feature rows, {features.shape[1] - 1} features")
    print(features.columns.tolist())