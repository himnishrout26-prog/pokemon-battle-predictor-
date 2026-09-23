"""
Simplified moveset model.

Real Pokemon have huge, hand-picked movepools. For this project each
Pokemon gets a small SYNTHETIC movepool built from its own type(s) plus a
couple of common coverage types -- documented here as a deliberate
simplification, not hidden.

Move category (physical vs special) follows the classic Gen 1-3 rule,
where category is determined by the move's TYPE rather than the
individual move. This is real Pokemon game history, not a shortcut.
"""
from type_chart import type_multiplier

PHYSICAL_TYPES = {
    "normal", "fighting", "flying", "ground", "rock", "bug", "ghost", "poison", "steel",
}

MOVE_POWER = 80
LEVEL = 50
COVERAGE_TYPE = "ice"  # generic 4th-slot filler for movepool variety

# Per-type accuracy flavour, loosely modeled on common Gen-1 moves.
# The simulator rolls against these; the feature builder deliberately does
# NOT know about accuracy, so the model has to learn stochastic effects.
MOVE_ACCURACY = {
    "normal": 1.00, "fire": 1.00, "water": 1.00, "electric": 1.00,
    "grass": 1.00, "ice": 0.90, "fighting": 1.00, "poison": 1.00,
    "ground": 1.00, "flying": 0.95, "psychic": 1.00, "bug": 1.00,
    "rock": 0.90, "ghost": 1.00, "dragon": 0.90, "dark": 1.00,
    "steel": 0.90, "fairy": 1.00,
}

CRIT_CHANCE = 0.0625
CRIT_MULTIPLIER = 1.5


def move_category(move_type):
    return "physical" if move_type in PHYSICAL_TYPES else "special"


def _clean_type2(pokemon):
    t2 = pokemon.get("type2")
    if t2 is None:
        return None
    try:
        if t2 != t2:  # NaN check without importing pandas here
            return None
    except TypeError:
        pass
    return t2


def get_movepool(pokemon):
    """Returns up to 4 unique move types this Pokemon can use: its own
    type(s), a Normal-type filler, and one coverage type for variety."""
    types = [pokemon["type1"]]
    t2 = _clean_type2(pokemon)
    if t2:
        types.append(t2)
    if "normal" not in types:
        types.append("normal")
    coverage = COVERAGE_TYPE if COVERAGE_TYPE not in types else "fighting"
    if coverage not in types:
        types.append(coverage)
    return types


def move_damage_estimate(attacker, defender, move_type, level=LEVEL, power=MOVE_POWER):
    """Deterministic base damage (no crit / roll / accuracy) used for both
    feature building and as the core of the simulator's damage calc."""
    category = move_category(move_type)
    atk_stat = attacker["attack"] if category == "physical" else attacker["sp_atk"]
    def_stat = defender["defense"] if category == "physical" else defender["sp_def"]

    stab = 1.5 if move_type in (attacker["type1"], _clean_type2(attacker)) else 1.0
    defending_types = [defender["type1"], _clean_type2(defender)]
    mult = type_multiplier(move_type, defending_types)

    base = ((2 * level / 5 + 2) * power * atk_stat / max(def_stat, 1) / 50 + 2)
    dmg = base * stab * mult
    return dmg, mult, category


def all_moves(attacker, defender):
    """Every move in the movepool with its estimated damage vs THIS
    defender, sorted by damage descending. Used by the web UI to show
    the full movepool and highlight the chosen (best) move."""
    out = []
    for move_type in get_movepool(attacker):
        dmg, mult, category = move_damage_estimate(attacker, defender, move_type)
        out.append({
            "move_type": move_type,
            "damage": round(dmg, 2),
            "multiplier": mult,
            "category": category,
            "accuracy": MOVE_ACCURACY.get(move_type, 1.0),
        })
    out.sort(key=lambda m: m["damage"], reverse=True)
    return out


def best_move(attacker, defender):
    """Picks the highest-expected-damage move from attacker's movepool
    against this specific defender -- the 'optimal play' logic."""
    movepool = get_movepool(attacker)
    best = None
    for move_type in movepool:
        dmg, mult, category = move_damage_estimate(attacker, defender, move_type)
        if best is None or dmg > best["damage"]:
            best = {
                "move_type": move_type,
                "damage": dmg,
                "multiplier": mult,
                "category": category,
            }
    return best