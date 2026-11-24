# Predict2Act forecasting

This module estimates how variable sleep timing will be over the next seven days. It includes data preparation, calendar-window feature and label generation, retrospective model fitting and prediction, and separate experiment commands for group work.

## Run the shared Oura study

The bundle includes the actual cleaned nightly Oura inputs in `data/study/nightly.csv` and the frozen study splits in `data/study/splits.csv`. Run the following from this directory with Python 3.13:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m forecasting.experiments --study ablation --data data/study --output runs/ablation
```

Use `--study all` for the full experiment suite. See [the English report](docs/REPORT.md) for reference results.

### Model comparison

```sh
python -m forecasting.experiments --study models --seeds 5 --data data/study --output runs/models-initials
python -m forecasting.summarize --run runs/models-initials
```

- `--seeds 5` repeats every random model (ExtraTrees, Random Forest, MLP) with seeds 498–502 and reports mean ± SD. The default `--seeds 1` reproduces the report exactly.
- The run stops if XGBoost is missing. Install `requirements.txt`, or pass `--skip-xgboost` knowingly.
- `forecasting.summarize` picks one setting per model family by **validation** MAE and reports its **test** MAE, gain over the personal median, and per-person test MAE. It writes `summary_models.md/.csv` and `summary_other.md` into the run folder.
- Method names are `Family (setting)`, e.g. `Ridge (alpha=100)`. Baselines are the personal median, the recent 4-week median (new), and reuse of the last 7- or 14-day score.
- KNN does not accept sample weights, so it is the one model fitted without equal donor weighting. The fixture contains dated personal sleep records, so share this bundle through the private group repository with approved collaborators.

## Rebuild the cleaned inputs

The supplied snapshot is ready for training. To reproduce the data preparation stage from original exports, obtain the Oura sleep export directory, personal Oura trends workbook, harmonized `combined_nights.csv`, and original split inventory through the team's approved data-sharing channel:

```sh
python -m forecasting.prepare --sleep-dir /path/to/sleep \
  --own-xlsx /path/to/oura_trends.xlsx --combined-csv /path/to/combined_nights.csv \
  --splits /path/to/inventory_pacific.csv --output data/rebuilt
```

The frozen split manifest in this bundle can also be passed to `--splits`. The report's Fitbit and weekly direction figures come from separate research scripts. A generated demo is available with `python -m forecasting.prepare --demo --output data/demo`.

## Work in parallel

| Workstream | Command option or file | Shared starting point |
|---|---|---|
| Data processing | `forecasting.prepare`, `forecasting/oura.py` | Nightly schema, local clock conversion, deduplication |
| Model and baseline comparison | `--study models` | Original donor pool, 24 schedule features, frozen temporal splits |
| Feature ablation | `--study ablation` | Frozen ExtraTrees reference plus four feature variants, each refit on the same dates |
| Sensitivity analysis | `--study sensitivity` | History length, causal gap filling, tree settings, and donor pool checks |

Choose the `--study` option for your workstream and use a distinct output directory such as `runs/ablation-initials`. Each run includes the same frozen reference. To add an ablation, add a named feature subset to the ablation block in `forecasting/experiments.py`. Keep the ExtraTrees settings, target labels, and date boundaries fixed for that comparison.

## Input and output contracts

- `nightly.csv` has one row per `source,day`. The day is the assigned sleep date used throughout the study. Bedtime and wake time are local clock minutes in `[0,1440)`, duration is minutes, and `hrv,hr,resp,temp` may be blank. Oura timestamp exports use Pacific clock time under the original protocol. Preserve existing sleep-date conventions when adding new sources.
- `splits.csv` has `source,status,validation_start,test_start,is_target`. Use `included` for the original shared pool and `1` for held-out evaluation targets. Additional rows are candidates for the wider-pool analysis. Every fit and validation boundary uses the 13-day purge.
- The target is circular RMS dispersion of sleep midpoint in the seven calendar days after cutoff. Its label requires at least five genuine future nights. The fixed reference uses at least five and ten genuine input nights from the past 7 and 14 days, respectively.
- Each experiment writes `predictions.csv`, `metrics_by_target.csv`, `metrics.csv`, and `matched_metrics.csv`. Use the latter for paired accuracy versus ExtraTrees when a policy changes eligible dates. Compare window counts separately. The study-wide MAE gives each target record equal weight.

The prediction runner expects labeled historical windows for evaluation. A live reporting feature would need an inference entry point that constructs the same past-only features without requiring future labels.
