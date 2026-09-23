# Pokemon battle outcome predictor

Predicts the winner of a 1v1 Pokemon battle from pre-battle stats, using
gradient-boosted models (XGBoost / LightGBM) trained on **simulated
multi-turn battles** rather than a raw stat comparison.

## Why multi-turn simulation instead of a stat lookup

A model that just compares base stat totals is trivial and doesn't
demonstrate much. Instead, `src/battle_simulator.py` actually plays out
each battle turn by turn:

1. Speed decides who attacks first each turn (with a small random factor,
   same as real Speed ties).
2. Each turn, the attacker picks the single most effective move from a
   small synthetic movepool against the CURRENT opponent (type
   effectiveness from the 18x18 chart in `src/type_chart.py`, plus STAB).
   This is real move-by-move analysis, not a one-shot stat comparison --
   see `src/movesets.py`.
3. HP depletes turn by turn until someone is knocked out.

The **label** (who won) comes from this simulation. The **model** only
ever sees pre-battle stats -- it has to learn to approximate the dynamic
outcome from static numbers, the same problem a real predictor would face.

## Move analysis (src/movesets.py)

Each Pokemon gets a small synthetic movepool: its own type(s), a Normal
filler, and one common coverage type. Move category (physical vs
special) follows the classic Gen 1-3 rule where category is determined
by the move's TYPE, not the individual move -- documented history, not a
shortcut. `best_move(attacker, defender)` checks every move in the
movepool against the specific opponent and returns the one that deals
the most damage, with its type-effectiveness multiplier ("Super
effective", "Not very effective", "No effect", or normal). This is used
both to generate battle labels (the simulator's attackers play
optimally) and as model features (`p1_best_move_multiplier`,
`effective_power_diff`, etc.) -- and it's surfaced directly in the web
app so you can see, per matchup, which move each side would pick and how
effective it is.

## Project structure

```
data/
  pokemon_stats.csv    # Pokemon roster: name, types, base stats
  battles.csv           # simulated battle outcomes (generated)
  features.csv           # engineered features + label (generated)
src/
  type_chart.py         # 18x18 type effectiveness chart
  movesets.py             # synthetic movepools + best-move-vs-opponent logic
  build_dataset.py       # builds pokemon_stats.csv (swap in real Kaggle data here)
  battle_simulator.py     # multi-turn battle engine + label generation
  features.py             # differential feature engineering
  train.py                 # baseline + XGBoost/LightGBM training
  explain.py                # SHAP global + per-prediction explanations
  simple_train.py            # plain logistic regression (used by webapp/)
models/                    # trained models + SHAP summary plot (generated)
app.py                      # Streamlit demo (XGBoost/LightGBM + SHAP)
webapp/                      # custom HTML/CSS/JS demo (logistic regression + move analysis)
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Uses the full dataset: **1025 Pokemon across every generation**, in
`data/pokemon_stats_raw.csv` (columns: pokedex_id, name, types, all six
base stats, sprite_url, plus extra metadata like legendary status and
abilities that this project doesn't currently use). `src/build_dataset.py`
maps its columns onto the schema the pipeline expects and writes
`data/pokemon_stats.csv`.

Note: the dataset includes named forme variants (e.g. `Toxtricity-Amped`,
not just `Toxtricity`) -- use the exact name shown in the dropdown / from
`/api/pokemon` when testing the API directly.

If `data/pokemon_stats_raw.csv` is ever missing, `build_dataset.py` falls
back automatically to a smaller hardcoded 151-Pokemon roster
(`src/build_dataset_fallback.py`) so the pipeline still runs end-to-end.

## Running the pipeline

```bash
cd src
python3 build_dataset.py        # generates data/pokemon_stats.csv
python3 battle_simulator.py     # simulates 20000 battles -> data/battles.csv
python3 features.py             # engineers features -> data/features.csv
python3 train.py                # trains baseline + XGBoost + LightGBM, saves best
python3 explain.py              # generates models/shap_summary.png
python3 simple_train.py         # trains the plain logistic regression model used by the web app
cd ..
```

## Two ways to view results

**1. Streamlit demo** (`streamlit run app.py`) -- uses the XGBoost/LightGBM
model and SHAP explanations. Quick to run, less visual control.

**2. Custom HTML/CSS/JS web app** (`webapp/`) -- uses the plain logistic
regression model (`src/simple_train.py`), with explanations coming
straight from the model's own coefficients (contribution = coefficient x
scaled feature value -- no SHAP needed for a linear model). Full control
over the look; Pokemon-themed UI with a self-contained CSS pokeball
background and live sprites pulled from PokeAPI in the browser.

```bash
python3 webapp/app.py
```

Then open http://127.0.0.1:5000 in your browser. The two demos are
independent -- run whichever one you want, or both (on different ports
isn't needed since only one runs at a time by default).

## Modeling notes

- **Baseline**: scaled logistic regression -- interpretable, sets the
  floor, good to cite in a writeup as "why gradient boosting wins here."
- **Main models**: XGBoost and LightGBM on engineered differential
  features (speed advantage, type matchup score, estimated effective
  damage output, rough "turns to KO", base stat total diff) rather than
  raw stats.
- **Evaluation**: accuracy + ROC-AUC on a held-out 20% split, stratified
  by label.
- **Explainability**: SHAP values, both globally (which features matter
  most overall) and per-prediction (why THIS matchup went the way it did)
  -- surfaced live in the Streamlit app.

## Stretch goals (mentioned, not required)

- Swap the synthetic movepools for a real per-species moveset dataset
  (exact moves, accuracy, priority) instead of type-based synthetic ones.
- Add status effects (paralysis/burn/poison/sleep) to the simulator.
- Multi-Pokemon team battles instead of 1v1.
- Deploy the Streamlit app publicly and link it from your resume.


## WHAT CHANGED IN THE UPDATE


- **Simulator is now genuinely stochastic.** Accuracy rolls (per type),
  6.25% crits at 1.5x, and the real level-50 HP formula
  (`floor((2*base + 31) * level / 100) + level + 10`). The feature builder
  deliberately does **not** see accuracy or crits, so the model has to
  learn stochastic effects instead of reading them off its inputs.
- **Feature set trimmed to remove leakage.** Dropped `effective_power_ratio`,
  `ttk_diff`, and `best_move_multiplier_diff` — all monotone transforms
  of features already present, which made SHAP unstable and made the
  model look stronger than it was.
- **Training reports 5-fold CV (mean ± std), Brier score, a calibration
  plot, permutation importance, and a stat-only baseline.** If the full
  feature set only beats the stat-only baseline by a hair, that's now
  visible in the output.
- **Web UI rebuilt around an animated battle replay, Chart.js analytics
  (stat radar, per-move damage bar chart, signed-coefficient waterfall,
  Monte Carlo turn histogram), and move recommendation cards.**
- **`/api/predict` is a single call** that returns the model prediction,
  feature contributions, all moves for both sides, a deterministic replay
  log, and an N-battle simulation.
- **Tests + Makefile added.**

## Tests

```bash
pip install -r requirements.txt
pytest -q
