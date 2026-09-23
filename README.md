
# Pokemon battle outcome predictor

Predicts the winner of a 1v1 Pokemon battle from pre-battle stats, using
a gradient-boosted model trained on **simulated multi-turn battles**
rather than a raw stat comparison.

## Why multi-turn simulation instead of a stat lookup

A model that just compares base stat totals is trivial. Instead,
`src/battle_simulator.py` actually plays out each battle turn by turn:

1. **Speed decides turn order** each turn, with a small random factor
   (±2) that breaks ties like real Speed rolls.
2. **Each attacker picks its best move** against the *current* opponent
   from a small synthetic movepool — type effectiveness from the 18×18
   chart in `src/type_chart.py`, plus STAB, category (physical/special,
   following the classic Gen 1-3 rule that category is determined by the
   move's TYPE), and per-type accuracy rolls.
3. **Damage uses the real Gen 3+ level-50 damage formula**, with the
   standard 0.85–1.0 random roll and a 6.25% crit chance at 1.5× power.
4. **HP depletes turn by turn** using the real level-50 HP formula
   (`floor((2*base + 31) * level / 100) + level + 10`, scaled 3× so
   battles run several turns).

The **label** (who won) comes from this simulation. The **model only
sees static pre-battle stats and type matchups** — it has to learn to
approximate the stochastic multi-turn outcome from those. It never sees
the simulator's damage formula, accuracy rolls, or crits.

## What the model actually looks at

Eight static features, all antisymmetric in `p1`/`p2`:

| Feature | Description |
|---|---|
| `hp_diff`, `attack_diff`, `defense_diff`, `sp_atk_diff`, `sp_def_diff`, `speed_diff` | raw stat differentials |
| `speed_advantage` | −1 / 0 / +1 |
| `type_advantage_diff` | type-chart multiplier of p1's typing vs. p2, minus the reverse |

**Deliberately excluded:** anything derived from the simulator's damage
formula (e.g. `best_move()["damage"]`, turns-to-KO estimates). Those
created a shortcut where the model just re-derived the simulator's own
math instead of learning to predict outcomes — see the "Design notes"
in `src/features.py` for details.

## Results

5-fold CV on 20,000 simulated battles, plus a held-out 20% test split:

| Model | CV acc | CV AUC | CV Brier | Test acc | Test AUC |
|---|---|---|---|---|---|
| stat-only logistic | 0.815 ± 0.005 | 0.903 | 0.127 | — | — |
| **full** logistic | 0.867 ± 0.006 | 0.944 | 0.095 | 0.862 | 0.941 |
| XGBoost | 0.869 ± 0.005 | 0.948 | 0.091 | — | — |
| LightGBM | **0.870 ± 0.005** | **0.948** | **0.091** | **0.865** | **0.946** |

The **+5.2 point gap** between the stat-only baseline and the full model
is entirely attributable to the type-matchup feature. `train.py` prints
these numbers, a calibration plot, and permutation importances so the
contribution of each feature is visible.

Permutation importance (drop in ROC-AUC) on the held-out test set:

```
type_advantage_diff    +0.077
sp_atk_diff            +0.041
attack_diff            +0.028
sp_def_diff            +0.028
defense_diff           +0.022
hp_diff                +0.022
speed_diff             +0.003
speed_advantage        +0.000
```

No single feature dominates, which is what you want — the model is
genuinely combining signals rather than reading one answer off.

## Project structure

```
data/
  pokemon_stats.csv      # Pokemon roster: name, types, base stats (built)
  battles.csv            # simulated battle outcomes (generated)
  features.csv           # engineered features + label (generated)
src/
  type_chart.py          # 18x18 type effectiveness chart
  movesets.py            # synthetic movepools + best-move-vs-opponent logic
  build_dataset.py       # builds pokemon_stats.csv from raw data or fallback
  build_dataset_fallback.py  # hardcoded 151-Pokemon roster if raw data absent
  battle_simulator.py    # multi-turn stochastic battle engine
  features.py            # differential feature engineering
  train.py               # CV + baselines + XGBoost/LightGBM + calibration
  explain.py             # SHAP global + per-prediction (Streamlit app)
  simple_train.py        # plain logistic regression (Flask app)
models/                  # trained models + calibration.png + SHAP summary (generated)
app.py                   # Streamlit demo
webapp/
  app.py                 # Flask backend
  templates/index.html   # self-contained UI (inline CSS + JS)
tests/
  test_pipeline.py       # type chart, movesets, features, simulator tests
Makefile                 # convenience targets
requirements.txt
```

## Setup

```
pip install -r requirements.txt
```

## Dataset

Uses **1025 Pokemon across every generation** from
`data/pokemon_stats_raw.csv`. `src/build_dataset.py` maps its columns
onto the schema the pipeline expects and writes
`data/pokemon_stats.csv`. If the raw file is missing, it falls back
automatically to a smaller hardcoded 151-Pokemon roster
(`src/build_dataset_fallback.py`).

Note: some entries are forme variants (e.g. `Toxtricity-Amped`, not
`Toxtricity`). Use the autocomplete dropdown in the web app, or the
`/api/pokemon` endpoint, when in doubt.

## Running the pipeline

```
make data        # build data/pokemon_stats.csv
make battles     # simulate 20000 battles -> data/battles.csv
make features    # engineer features -> data/features.csv
make train       # CV + train XGBoost/LightGBM -> models/best_model.joblib
make simple      # train logistic regression -> models/simple_model.joblib

# or all at once:
make train-all
```

## Two ways to view results

### 1. Flask web app (recommended)

```
make flask
# http://127.0.0.1:5000
```

A self-contained single-page UI with four tabs:

- **Overview** — stat radar chart + signed-coefficient waterfall (which
  features pushed the prediction toward each side)
- **Replay** — animated turn-by-turn battle log with HP bars draining,
  per-turn move/type/damage/crit/miss info, and a final winner banner
- **Moves** — the full movepool for each side, ranked by estimated damage
  against the *specific* opponent, with type badges and effectiveness
  labels; the simulator's chosen move is highlighted
- **Simulation** — 200 independent simulator runs with win rates, mean
  turns, and a histogram of turns-to-KO

The UI is entirely inline CSS + JS in `webapp/templates/index.html` —
no external file dependencies beyond the Chart.js CDN and Google Fonts.

### 2. Streamlit demo

```
make streamlit
```

Uses the XGBoost/LightGBM model with SHAP explanations. Faster to run,
less visual control.

## Tests

```
make tests
```

Covers the type chart, moveset construction, feature antisymmetry
(swapping p1/p2 must negate every signed feature), simulator termination,
and a directional sanity check (Mewtwo beats Magikarp ≥ 90% of the time).

## Modeling notes

- **Baseline**: scaled logistic regression on stat-only features — sets
  the floor at 81.5% / 0.903 AUC.
- **Main models**: XGBoost and LightGBM on the full 8-feature set. Both
  reach ~87% / 0.948.
- **Evaluation**: 5-fold stratified CV (mean ± std) plus a held-out 20%
  test split. Reports accuracy, ROC-AUC, and Brier score, plus a
  calibration curve saved to `models/calibration.png`.
- **Explainability**: the Flask app derives feature contributions
  directly from the logistic regression's coefficients
  (`contribution = coefficient × scaled_feature_value`). The Streamlit
  app uses SHAP for the tree models.

## Deliberate simplifications

Documented on purpose, not hidden:

- Synthetic 3–4 move movepools per Pokemon (its own type(s) + Normal
  filler + one coverage type), not real in-game movesets.
- No status effects, items, or abilities modeled.
- Level fixed at 50 for all Pokemon.
- HP scaled 3× after the level-50 formula, so most battles last 4–6
  turns instead of resolving in 1–2 (which would make the task trivial).

## Stretch goals

- Swap synthetic movepools for a real per-species moveset dataset
  (exact moves, power, accuracy, priority).
- Add status effects (paralysis/burn/poison/sleep) to the simulator.
- Multi-Pokemon team battles instead of 1v1.
- Deploy the Flask app publicly and link it from this README.
```

