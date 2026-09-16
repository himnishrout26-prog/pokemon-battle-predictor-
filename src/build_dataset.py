"""
Builds data/pokemon_stats.csv from the real dataset.

Reads data/pokemon_stats_raw.csv (the "Complete Pokemon Dataset with
Stats & Combat History"-style CSV, 1025 Pokemon across every generation)
and maps its columns onto the schema the rest of the pipeline expects:
name, type1, type2, hp, attack, defense, sp_atk, sp_def, speed, sprite_url.

If data/pokemon_stats_raw.csv isn't present, falls back to a smaller
151-Pokemon hardcoded roster (src/build_dataset_fallback.py) so the
pipeline still runs end-to-end without the real file.
"""
import os
import pandas as pd

RAW_PATH = "data/pokemon_stats_raw.csv"
OUT_PATH = "data/pokemon_stats.csv"

COLUMN_MAP = {
    "name": "name",
    "type_1": "type1",
    "type_2": "type2",
    "hp": "hp",
    "attack": "attack",
    "defense": "defense",
    "sp_attack": "sp_atk",
    "sp_defense": "sp_def",
    "speed": "speed",
    "sprite_url": "sprite_url",
}


def build_from_raw():
    df = pd.read_csv(RAW_PATH)
    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)
    df["name"] = df["name"].str.title()
    df = df.drop_duplicates(subset="name").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(df)} Pokemon (from real dataset) "
          f"covering {df['type1'].nunique()} primary types")
    return df


def build_fallback():
    from build_dataset_fallback import POKEMON
    df = pd.DataFrame(
        POKEMON,
        columns=["name", "type1", "type2", "hp", "attack", "defense", "sp_atk", "sp_def", "speed"],
    )
    df["sprite_url"] = None
    df.to_csv(OUT_PATH, index=False)
    print(f"[No {RAW_PATH} found -- using the smaller built-in 151-Pokemon roster instead.] "
          f"Wrote {OUT_PATH} with {len(df)} Pokemon")
    return df


def build():
    if os.path.exists(RAW_PATH):
        return build_from_raw()
    return build_fallback()


if __name__ == "__main__":
    build()
