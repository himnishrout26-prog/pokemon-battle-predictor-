"""
Simulates turn-by-turn Pokemon battles to generate labeled training data.

This is the key design choice that makes the project more than a stat
lookup table: instead of labeling the winner by comparing raw stats, we
actually simulate combat (speed -> turn order -> best-move selection ->
damage -> HP depletion) and use the simulated outcome as the label. The
model then has to learn to approximate that dynamic from static
pre-battle stats alone, the same way a real "who wins" predictor would
have to.

Each turn, the attacker picks the single most effective move from its
movepool against the CURRENT opponent (see src/movesets.py) -- this is
what makes it "optimal play" rather than spamming one fixed move: a
Fire-type facing a Grass-type picks its Fire move; facing a Water-type
instead, it switches to a coverage move if one hits harder.

Simplifications (documented on purpose, not hidden):
- Synthetic 3-4 move movepool per Pokemon (see movesets.py), not real
  in-game movesets.
- No status effects, items, or abilities modeled (documented as a
  stretch goal in the README).
- Level fixed at 50 for all Pokemon, standard Gen-line damage formula,
  with per-hit damage randomized 0.85-1.0x like the real games.
"""
import random
import pandas as pd
from movesets import best_move


def simulate_battle(p1, p2, rng, max_turns=50):
    """p1, p2: dict-like rows with name/type1/type2/hp/attack/defense/sp_atk/sp_def/speed.
    Returns (winner_name, turns_taken)."""
    hp1, hp2 = float(p1["hp"]) * 2, float(p2["hp"]) * 2  # scale HP like real games (~x2 at lvl 50)

    for turn in range(1, max_turns + 1):
        # turn order: faster goes first, small random factor breaks ties like real Speed stat rolls
        p1_first = (p1["speed"] + rng.uniform(-2, 2)) >= (p2["speed"] + rng.uniform(-2, 2))
        order = [(p1, "p1"), (p2, "p2")] if p1_first else [(p2, "p2"), (p1, "p1")]

        for attacker, side in order:
            defender = p2 if side == "p1" else p1
            chosen = best_move(attacker, defender)
            dmg = chosen["damage"] * rng.uniform(0.85, 1.0)

            if side == "p1":
                hp2 -= dmg
            else:
                hp1 -= dmg

            if hp1 <= 0 or hp2 <= 0:
                winner = p1["name"] if hp2 <= 0 and hp1 > 0 else (
                    p2["name"] if hp1 <= 0 and hp2 > 0 else
                    (p1["name"] if hp1 >= hp2 else p2["name"])  # simultaneous KO -> higher remaining hp wins
                )
                return winner, turn

    # safety net: shouldn't normally trigger, but avoid infinite loops
    return (p1["name"] if hp1 >= hp2 else p2["name"]), max_turns


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
            "winner": p1["name"] if winner == p1["name"] else p2["name"],
            "label": 1 if winner == p1["name"] else 0,  # 1 = pokemon_1 wins
            "turns": turns,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    stats = pd.read_csv("data/pokemon_stats.csv")
    battles = generate_battles(stats, n_battles=20000)
    battles.to_csv("data/battles.csv", index=False)
    print(f"Simulated {len(battles)} battles")
    print(battles["label"].value_counts(normalize=True).rename("share"))
    print(f"Avg turns per battle: {battles['turns'].mean():.1f}")
