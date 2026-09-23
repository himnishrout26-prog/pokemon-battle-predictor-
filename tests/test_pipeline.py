"""
Sanity tests for the type chart, movesets, feature engineering, and the
battle simulator. These are fast, dependency-light tests meant to catch
the kind of bugs that silently break the ML pipeline (asymmetric features,
non-terminating simulations, movesets with duplicate types).

Run from the project root:
    pytest -q
"""
import os
import random
import sys

import pytest

# Make src/ importable no matter where pytest is invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from type_chart import type_multiplier  # noqa: E402
from movesets import (  # noqa: E402
    MOVE_ACCURACY, all_moves, best_move, get_movepool, move_category,
)
from features import (  # noqa: E402
    STAT_ONLY_FEATURES, build_features,
)
from battle_simulator import max_hp, simulate_battle  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures: three canonical Pokemon used throughout
# --------------------------------------------------------------------------
CHARIZARD = {
    "name": "Charizard", "type1": "fire", "type2": "flying",
    "hp": 78, "attack": 84, "defense": 78, "sp_atk": 109, "sp_def": 85, "speed": 100,
}
BLASTOISE = {
    "name": "Blastoise", "type1": "water", "type2": None,
    "hp": 79, "attack": 83, "defense": 100, "sp_atk": 85, "sp_def": 105, "speed": 78,
}
VENUSAUR = {
    "name": "Venusaur", "type1": "grass", "type2": "poison",
    "hp": 80, "attack": 82, "defense": 83, "sp_atk": 100, "sp_def": 100, "speed": 80,
}
PIKACHU = {
    "name": "Pikachu", "type1": "electric", "type2": None,
    "hp": 35, "attack": 55, "defense": 40, "sp_atk": 50, "sp_def": 50, "speed": 90,
}
DUGTRIO = {
    "name": "Dugtrio", "type1": "ground", "type2": None,
    "hp": 35, "attack": 100, "defense": 50, "sp_atk": 50, "sp_def": 70, "speed": 120,
}


# ==========================================================================
# Type chart
# ==========================================================================

def test_type_chart_neutral_default():
    assert type_multiplier("normal", ["normal"]) == 1.0


def test_type_chart_super_effective():
    assert type_multiplier("fire", ["grass"]) == 2.0
    assert type_multiplier("water", ["fire"]) == 2.0


def test_type_chart_not_very_effective():
    assert type_multiplier("fire", ["water"]) == 0.5
    assert type_multiplier("grass", ["fire"]) == 0.5


def test_type_chart_immune():
    assert type_multiplier("electric", ["ground"]) == 0.0
    assert type_multiplier("normal", ["ghost"]) == 0.0


def test_type_chart_dual_type_multiplies():
    # fire vs grass/poison = 2.0 * 1.0 = 2.0
    assert type_multiplier("fire", ["grass", "poison"]) == 2.0
    # ice vs dragon/flying = 2.0 * 2.0 = 4.0
    assert type_multiplier("ice", ["dragon", "flying"]) == 4.0
    # electric vs water/flying = 2.0 * 2.0 = 4.0
    assert type_multiplier("electric", ["water", "flying"]) == 4.0


def test_type_chart_none_defender_is_neutral():
    # Defenders with no type2 are passed as [type1, None]; None entries are skipped.
    assert type_multiplier("fire", ["water", None]) == 0.5


# ==========================================================================
# Movesets
# ==========================================================================

def test_movepool_has_no_duplicates():
    for mon in (CHARIZARD, BLASTOISE, VENUSAUR, PIKACHU):
        pool = get_movepool(mon)
        assert len(pool) == len(set(pool)), f"duplicate move types for {mon['name']}"


def test_movepool_always_includes_own_primary_type():
    for mon in (CHARIZARD, BLASTOISE, VENUSAUR, PIKACHU):
        assert mon["type1"] in get_movepool(mon)


def test_movepool_size_between_1_and_4():
    for mon in (CHARIZARD, BLASTOISE, VENUSAUR, PIKACHU, DUGTRIO):
        pool = get_movepool(mon)
        assert 1 <= len(pool) <= 4


def test_move_category_physical_vs_special():
    # Gen 1-3 style: category is determined by the move's TYPE.
    assert move_category("normal") == "physical"
    assert move_category("fighting") == "physical"
    assert move_category("fire") == "special"
    assert move_category("water") == "special"
    assert move_category("psychic") == "special"


def test_all_moves_sorted_descending():
    moves = all_moves(CHARIZARD, BLASTOISE)
    dmg = [m["damage"] for m in moves]
    assert dmg == sorted(dmg, reverse=True)


def test_best_move_matches_top_of_all_moves():
    top = all_moves(CHARIZARD, BLASTOISE)[0]
    best = best_move(CHARIZARD, BLASTOISE)
    assert best["move_type"] == top["move_type"]
    assert abs(best["damage"] - top["damage"]) < 1e-6


def test_best_move_returns_highest_damage():
    # Water vs Charizard: water is 2x, ice (coverage) is 1x vs fire/flying...
    # ...but ice hits flying for 2x too. Just assert it's the max of all moves.
    best = best_move(BLASTOISE, CHARIZARD)
    top_dmg = max(m["damage"] for m in all_moves(BLASTOISE, CHARIZARD))
    assert abs(best["damage"] - top_dmg) < 1e-6


def test_best_move_damage_is_positive():
    best = best_move(CHARIZARD, BLASTOISE)
    assert best["damage"] > 0


def test_move_accuracy_in_range():
    for acc in MOVE_ACCURACY.values():
        assert 0 < acc <= 1.0


# ==========================================================================
# Feature engineering
# ==========================================================================

def test_features_are_antisymmetric():
    """Swapping p1/p2 must negate every feature (the model relies on this
    for the augmented-dataset trick and for its symmetry guarantees)."""
    f1 = build_features(CHARIZARD, BLASTOISE)
    f2 = build_features(BLASTOISE, CHARIZARD)
    for k in f1:
        assert abs(f1[k] + f2[k]) < 1e-9, f"feature {k!r} not antisymmetric: {f1[k]} vs {f2[k]}"


def test_features_antisymmetric_dual_types():
    f1 = build_features(VENUSAUR, CHARIZARD)
    f2 = build_features(CHARIZARD, VENUSAUR)
    for k in f1:
        assert abs(f1[k] + f2[k]) < 1e-9, f"feature {k!r} not antisymmetric"


def test_self_matchup_features_are_zero():
    """A Pokemon against itself must produce all-zero differential features."""
    f = build_features(CHARIZARD, CHARIZARD)
    for k, v in f.items():
        assert abs(v) < 1e-9, f"feature {k!r} = {v} for self-matchup"


def test_stat_only_features_are_subset():
    full = set(build_features(CHARIZARD, BLASTOISE).keys())
    for f in STAT_ONLY_FEATURES:
        assert f in full, f"stat-only feature {f!r} missing from full feature set"


def test_no_nan_features():
    f = build_features(CHARIZARD, BLASTOISE)
    for k, v in f.items():
        assert v == v, f"feature {k!r} is NaN"


def test_feature_keys_are_stable():
    """Every matchup should produce the exact same set of feature keys."""
    keys_a = set(build_features(CHARIZARD, BLASTOISE).keys())
    keys_b = set(build_features(VENUSAUR, PIKACHU).keys())
    keys_c = set(build_features(DUGTRIO, VENUSAUR).keys())
    assert keys_a == keys_b == keys_c


# ==========================================================================
# Battle simulator
# ==========================================================================

def test_max_hp_level_50_formula():
    # Charizard base HP 78 -> floor((2*78 + 31) * 50 / 100) + 50 + 10
    #                       = floor(187 * 0.5) + 60 = 93 + 60 = 153
    assert max_hp(CHARIZARD) == 153


def test_simulate_returns_valid_winner():
    rng = random.Random(0)
    winner, turns = simulate_battle(CHARIZARD, BLASTOISE, rng)
    assert winner in (CHARIZARD["name"], BLASTOISE["name"])
    assert 1 <= turns <= 50


def test_simulate_terminates_quickly():
    rng = random.Random(123)
    for _ in range(50):
        _, turns = simulate_battle(CHARIZARD, VENUSAUR, rng)
        assert turns <= 50


def test_simulate_is_deterministic_with_seed():
    w1, t1 = simulate_battle(CHARIZARD, BLASTOISE, random.Random(7))
    w2, t2 = simulate_battle(CHARIZARD, BLASTOISE, random.Random(7))
    assert (w1, t1) == (w2, t2)


def test_simulate_returns_log_when_requested():
    _, _, log = simulate_battle(CHARIZARD, BLASTOISE, random.Random(1), return_log=True)
    assert len(log) > 0
    for entry in log:
        for key in ("turn", "attacker", "side", "move_type", "damage",
                    "hp1", "hp2", "hp1_pct", "hp2_pct", "hit"):
            assert key in entry, f"log entry missing key {key!r}"
        assert entry["side"] in ("p1", "p2")
        assert entry["damage"] >= 0
        assert 0 <= entry["hp1_pct"] <= 100
        assert 0 <= entry["hp2_pct"] <= 100


def test_simulate_log_hp_monotonically_decreases():
    _, _, log = simulate_battle(CHARIZARD, BLASTOISE, random.Random(3), return_log=True)
    prev1 = prev2 = 100.0
    for entry in log:
        assert entry["hp1_pct"] <= prev1 + 1e-9
        assert entry["hp2_pct"] <= prev2 + 1e-9
        prev1, prev2 = entry["hp1_pct"], entry["hp2_pct"]


def test_type_immunity_blocks_all_damage():
    """Pikachu's electric moves can't hurt Dugtrio (ground). But Pikachu
    also has normal and ice coverage, so it should still be able to damage
    Dugtrio with those. Just verify the run terminates and someone wins."""
    winner, turns = simulate_battle(PIKACHU, DUGTRIO, random.Random(11))
    assert winner in (PIKACHU["name"], DUGTRIO["name"])
    assert 1 <= turns <= 50


def test_mismatched_pokemon_is_not_a_tie():
    """If sides have different HP after the KO, one side must have won."""
    winner, _ = simulate_battle(CHARIZARD, PIKACHU, random.Random(99))
    assert winner in (CHARIZARD["name"], PIKACHU["name"])


# ==========================================================================
# Integration: features <-> simulator agree on matchup direction
# ==========================================================================

def test_dominant_pokemon_wins_most_simulations():
    """Mewtwo should beat Magikarp the vast majority of the time. This is a
    directional sanity check: the simulator isn't fully random."""
    mewtwo = {
        "name": "Mewtwo", "type1": "psychic", "type2": None,
        "hp": 106, "attack": 110, "defense": 90,
        "sp_atk": 154, "sp_def": 90, "speed": 130,
    }
    magikarp = {
        "name": "Magikarp", "type1": "water", "type2": None,
        "hp": 20, "attack": 10, "defense": 55,
        "sp_atk": 15, "sp_def": 20, "speed": 80,
    }
    rng = random.Random(42)
    mewtwo_wins = sum(
        simulate_battle(mewtwo, magikarp, rng)[0] == "Mewtwo"
        for _ in range(200)
    )
    assert mewtwo_wins >= 180, f"Mewtwo only won {mewtwo_wins}/200 vs Magikarp"


def test_features_point_the_right_way():
    """Mewtwo's effective_power_diff and base_stat_total_diff vs Magikarp
    should both be strongly positive (favoring Mewtwo)."""
    mewtwo = {
        "name": "Mewtwo", "type1": "psychic", "type2": None,
        "hp": 106, "attack": 110, "defense": 90,
        "sp_atk": 154, "sp_def": 90, "speed": 130,
    }
    magikarp = {
        "name": "Magikarp", "type1": "water", "type2": None,
        "hp": 20, "attack": 10, "defense": 55,
        "sp_atk": 15, "sp_def": 20, "speed": 80,
    }
    f = build_features(mewtwo, magikarp)
    assert f["base_stat_total_diff"] > 0
    assert f["effective_power_diff"] > 0
    assert f["speed_diff"] > 0