# Predict2Act | Forecasting next-week sleep timing variability

The selected ExtraTrees model reduced mean absolute error (MAE) from **14.63 to 12.93 minutes**, an **11.6% reduction** relative to each record’s historical median. It also outperformed forecasts that simply reused recent variability. The gain persisted when donor data were restricted to records available before the target validation period, though it fell to **6.8%**. These results support a weekly forecast for the three high-completeness Oura records studied.

## Task, data, and model

At each cutoff, the model uses the preceding 7 and 14 calendar days to predict the circular root-mean-square dispersion of sleep midpoint over the next seven calendar days. The outcome is measured in minutes. A higher value means sleep timing is less regular. Circular arithmetic handles the midnight boundary.

The three target records were selected using sleep data completeness: at least 170 valid nights, 90% calendar coverage, and a longest internal gap of three days or fewer. Each target was held out of shared fitting, and the other seven eligible Oura streams supplied donor data. The target’s own fit period provided its historical median and the normalization parameters used to personalize predictions. Donor streams received equal total training weight. The validation and test sets were chronological, with a 13-day purge at the split boundaries. The benchmark uses 123 validation windows and 265 held-out rolling test windows. MAE is calculated within each target first, then averaged equally across the three targets.

The model uses 24 features covering recent midpoint and wake-time variability, shifts in midpoint, circular summaries of clock time, duration levels and trends, and valid-night counts. The frozen ExtraTrees model averages 300 randomized shallow trees, with maximum depth 5 and at least 12 training windows per leaf. This allows the regression to represent interactions. For example, a shift in sleep midpoint could carry different information after a stable fortnight than after a variable one. That is a plausible mechanism for predictive performance. The feature comparisons below measure the effect of changing inputs on this task.

## 1. Baseline and model comparison

All rows use the same test dates and observed labels. Learned models in this benchmark use the same 24 features and donor pool. Ridge, Lasso, and XGBoost settings were selected within each family by lowest validation MAE among the settings tested. The previously frozen ExtraTrees configuration also ranked first among the models in this benchmark on validation.

| Method | Validation MAE (min) | Test MAE (min) |
|---|---:|---:|
| Personal historical median | 15.53 | 14.63 |
| Reuse recent 7-day score | 24.48 | 20.55 |
| Reuse recent 14-day score | 23.63 | 22.16 |
| Linear regression | 17.27 | 13.55 |
| Ridge regression (alpha 100) | 16.29 | 13.18 |
| Lasso regression (alpha 0.1) | 15.78 | 13.95 |
| Random Forest | 18.83 | 13.20 |
| Histogram gradient boosting | 15.67 | 13.11 |
| XGBoost (depth 2) | 17.48 | 13.12 |
| **ExtraTrees** | **15.29** | **12.93** |

| Target | Test windows | Personal median MAE | ExtraTrees MAE | Reduction |
|---|---:|---:|---:|---:|
| Own Oura | 158 | 13.64 | 11.28 | 17.3% |
| sleep-2.csv | 47 | 22.52 | 20.78 | 7.7% |
| sleep-5.csv | 60 | 7.72 | 6.73 | 12.8% |

Histogram gradient boosting and XGBoost were within 0.20 minutes of ExtraTrees on the test set. The evidence for forecasting value is stronger than the evidence for a unique advantage of ExtraTrees over the other tree ensembles.

## 2. Feature ablation

Each row refits the same ExtraTrees configuration on the same splits using the indicated feature set.

| Inputs | Validation MAE (min) | Test MAE (min) |
|---|---:|---:|
| Basic 9 schedule features | 15.59 | 13.83 |
| **Expanded 24 schedule features** | **15.29** | **12.93** |
| Expanded schedule without circular clock phase | 16.54 | 13.60 |
| Expanded schedule without duration features | 15.38 | 13.04 |
| Expanded schedule plus physiology | 18.13 | 14.91 |

Moving from 9 to 24 schedule features lowered test MAE by **0.90 minutes**. Removing clock phase increased error by **0.67 minutes**, compared with **0.10 minutes** when removing duration features. The tested block of HRV, sleeping heart rate, breathing rate, and temperature summaries increased error by **1.98 minutes**.

## 3. History length, missing values, and parameter sensitivity

The forecast horizon remains seven days. The original policy requires at least five observed nights in the last seven days and ten in the last fourteen. The single-lookback alternatives apply the corresponding minimum independently. The filling policy carries forward the previous observation for at most one consecutive missing night and permits one fewer observed night per input window, while retaining the original minimum counts after filling. All labels still require at least five actual future nights. The 265 shared dates isolate forecast error from changes in coverage.

| Input policy | Shared-date MAE (min) | All eligible windows | All-date MAE (min) |
|---|---:|---:|---:|
| **7- and 14-day history, observed nights** | **12.93** | 265 | **12.93** |
| 7-day history only | 13.26 | 265 | 13.26 |
| 14-day history only | 13.14 | 273 | 15.21 |
| 7- and 14-day history, past-only filling | 13.06 | 273 | 15.23 |

The longer-only and filling policies both added eight windows, about **3%** more coverage. Each was slightly worse on the shared dates and had higher MAE across all dates it served. For the weekly report, the observed-night policy sets a defensible minimum for issuing a forecast.

| ExtraTrees setting | Validation MAE (min) | Test MAE (min) |
|---|---:|---:|
| Frozen configuration, depth 5 and leaf minimum 12 | 15.287 | 12.932 |
| Depth 3 and leaf minimum 12 | 15.379 | 13.272 |
| Depth 8 and leaf minimum 12 | 15.454 | 12.933 |
| Depth 5 and leaf minimum 6 | 15.283 | 12.957 |
| Depth 5 and leaf minimum 24 | 15.379 | 13.175 |

The frozen model stayed within **0.34 minutes** of every tested setting. A six-window leaf minimum had a marginally lower validation error. All five settings beat the historical median on the same test dates.

## 4. Training pool and robustness

| Test or training condition | Test windows | ExtraTrees MAE (min) | Matched median MAE (min) |
|---|---:|---:|---:|
| Original rolling evaluation | 265 | 12.93 | 14.63 |
| Donor labels available before the target validation period and purge | 265 | 13.63 | 14.63 |
| Two additional Oura streams used for training only | 265 | 13.18 | 14.63 |
| Reports issued on Sundays | 39 | 11.50 | 14.07 |
| Future week has seven actual nights | 207 | 8.16 | 8.91 |

The wider Oura pool admitted two streams that could supply fit windows but had too few validation windows for the original full evaluation. This raised the donor count from seven to nine per target and increased MAE by **0.24 minutes**. In the earlier strictly chronological Fitbit experiment, only Own Oura could use the later Fitbit records. Its MAE rose from **11.28 to 11.67** after adding 132 night-filtered windows and to **12.49** after adding 1,613 windows with a wider time-of-day rule. The Fitbit timing fields came from the project’s preprocessed table and lacked timezone offsets.

An exploratory paired bootstrap resampled 14-day calendar blocks independently within each of the three fixed records. Its 95% interval for the mean gain over the personal median was **0.45 to 3.61 minutes**. The complete-future-week subset still improved on average, but the model was worse than the median for sleep-2.csv on that subset (8.19 vs 6.79 minutes, 16 windows). The bootstrap captures variation across time blocks within these records. It does not estimate how results vary across new participants.

## Weekly reporting

The primary weekly output is the predicted variability in minutes, accompanied by recent observed variability for context. The earlier direction replay had 28 weeks shared by all three comparators. ExtraTrees predicted the correct direction in **23/28**, the personal-median forecast in **21/28**, and prior-week momentum in **10/28**. On all 29 weeks in the direction analysis, counting changes within ±10 minutes as stable lowered ExtraTrees three-class accuracy to **51.7%**, and the model identified none of the ten observed increases above 10 minutes. The continuous forecast therefore has stronger support for a user-facing report than an alert for substantial worsening. The study measured forecast accuracy. An effect on sleep behavior would require evaluating the resulting intervention.

## Reproducibility

See the [module README](../README.md) for the shared data contract and commands. The model comparison, feature ablation, and sensitivity runs use the original Oura splits. The weekly direction and Fitbit figures summarize separate research analyses.
