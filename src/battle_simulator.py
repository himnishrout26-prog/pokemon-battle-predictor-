"""
Simulates turn-by-turn Pokemon battles to generate labeled training data.

The simulator has been made genuinely stochastic so the model cannot
simply read the answer off its features:

- Accuracy rolls (per move type)
- Crit rolls (6.25% for 1.5x damage)
- Realistic level-50 HP (base_hp + 60-ish, using the Gen 3+ formula)
- Real damage formula, randomized 0.85-1.0x per hit like the real games

The label (who won) comes from THIS simulation. The feature builder only
sees deterministic estimates -- it does not see accuracy or crits -- so
the model has to learn stochastic effects from static features.

Simplifications (documented on purpose, not hidden):
- Synthetic 3-4 move movepool per Pokemon (see movesets.py).
- No status effects, items, or abilities modeled.
- Level fixed at 50.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from movesets import best_move, MOVE_ACCURACY, CRIT_CHANCE, CRIT_MULTIPLIER

ROOT = Path(__file__).resolve().parent.parent

LEVEL = 50


def max_hp(pokemon):
    base = int((2 * pokemon["hp"] + 31) * LEVEL / 100) + LEVEL + 10
    return base * 3


def _pick_winner(p1, p2, hp1, hp2):
    if hp2 <= 0 and hp1 > 0:
        return p1["name"]
    if hp1 <= 0 and hp2 > 0:
        return p2["name"]
    return p1["name"] if hp1 >= hp2 else p2["name"]


def simulate_battle(p1, p2, rng, max_turns=50, return_log=False):
    """p1, p2: dict-like rows with name/type1/type2/hp/attack/defense/sp_atk/sp_def/speed.
    Returns (winner_name, turns_taken) by default, or (winner, turns, log)
    when return_log=True (used by the web app to render an animated replay)."""
    hp1, hp2 = float(max_hp(p1)), float(max_hp(p2))
    max_hp1, max_hp2 = hp1, hp2
    log = []

    for turn in range(1, max_turns + 1):
        p1_first = (p1["speed"] + rng.uniform(-2, 2)) >= (p2["speed"] + rng.uniform(-2, 2))
        order = [(p1, "p1"), (p2, "p2")] if p1_first else [(p2, "p2"), (p1, "p1")]

        for attacker, side in order:
            defender = p2 if side == "p1" else p1
            chosen = best_move(attacker, defender)
            accuracy = MOVE_ACCURACY.get(chosen["move_type"], 1.0)
            hit = rng.random() < accuracy
            crit = hit and rng.random() < CRIT_CHANCE

            if hit:
                dmg = chosen["damage"] * rng.uniform(0.85, 1.0)
                if crit:
                    dmg *= CRIT_MULTIPLIER
            else:
                dmg = 0.0

            if side == "p1":
                hp2 = max(hp2 - dmg, 0.0)
            else:
                hp1 = max(hp1 - dmg, 0.0)

            if return_log:
                log.append({
                    "turn": turn,
                    "attacker": attacker["name"],
                    "side": side,
                    "move_type": chosen["move_type"],
                    "category": chosen["category"],
                    "multiplier": chosen["multiplier"],
                    "accuracy": accuracy,
                    "hit": hit,
                    "crit": crit,
                    "damage": round(dmg, 1),
                    "hp1": round(hp1, 1),
                    "hp2": round(hp2, 1),
                    "hp1_pct": round(hp1 / max_hp1 * 100, 1),
                    "hp2_pct": round(hp2 / max_hp2 * 100, 1),
                })

            if hp1 <= 0 or hp2 <= 0:
                winner = _pick_winner(p1, p2, hp1, hp2)
                if return_log:
                    return winner, turn, log
                return winner, turn

    winner = _pick_winner(p1, p2, hp1, hp2)
    if return_log:
        return winner, max_turns, log
    return winner, max_turns


def generate_battles(pokemon_df, n_battles=8000, seed=42):
    rng = random.Random(seed)
    records = pokemon_df.to_dict("records")
    rows = []
    for _ in range(n_battles):
        p1, p2 = rng.sample(records, 2)
        winner, turns = simulate_battle(p1, p2, rng)
        rows.append({
            "pokemon_1": p1["name"],
            "pokemon_2": p2["name"],
            "winner": winner,
            "label": 1 if winner == p1["name"] else 0,  # 1 = pokemon_1 wins
            "turns": turns,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    stats = pd.read_csv(ROOT / "data" / "pokemon_stats.csv")
    battles = generate_battles(stats, n_battles=20000)
    battles.to_csv(ROOT / "data" / "battles.csv", index=False)
    print(f"Simulated {len(battles)} battles")
    print(battles["label"].value_counts(normalize=True).rename("share"))
    print(f"Avg turns per battle: {battles['turns'].mean():.1f}")