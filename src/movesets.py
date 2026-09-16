"""
Simplified moveset model.

Real Pokemon have huge, hand-picked movepools (hundreds of possible moves
per species). For this project each Pokemon instead gets a small SYNTHETIC
movepool built from its own type(s) plus a couple of common coverage
types -- documented here as a deliberate simplification, not hidden.

Move category (physical vs special) follows the classic Gen 1-3 rule,
where category is determined by the move's TYPE rather than the
individual move. This is real, documented Pokemon game history, not a
made-up shortcut -- it's exactly how damage categories worked before
Gen 4 introduced the physical/special split per-move.
"""
from type_chart import type_multiplier

PHYSICAL_TYPES = {
    "normal", "fighting", "flying", "ground", "rock", "bug", "ghost", "poison", "steel",
}
# everything else (fire, water, grass, electric, psychic, ice, dragon, dark, fairy) is special

MOVE_POWER = 80
LEVEL = 50
COVERAGE_TYPE = "ice"  # generic 4th-slot filler for movepool variety


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
    category = move_category(move_type)
    atk_stat = attacker["attack"] if category == "physical" else attacker["sp_atk"]
    def_stat = defender["defense"] if category == "physical" else defender["sp_def"]

    stab = 1.5 if move_type in (attacker["type1"], _clean_type2(attacker)) else 1.0
    defending_types = [defender["type1"], _clean_type2(defender)]
    mult = type_multiplier(move_type, defending_types)

    base = ((2 * level / 5 + 2) * power * atk_stat / max(def_stat, 1) / 50 + 2)
    dmg = base * stab * mult
    return dmg, mult, category


def best_move(attacker, defender):
    """Picks the highest-expected-damage move from attacker's movepool
    against this specific defender -- this is the 'optimal play' logic."""
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
