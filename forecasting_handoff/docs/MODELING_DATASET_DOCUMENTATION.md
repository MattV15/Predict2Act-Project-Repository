# Modeling Dataset Documentation: `modeling_dataset.csv`

Built by `build_modeling_dataset.py` from `combined_oura_plus_personal.csv` (see `master_dataset_documentation.md`). The master dataset is read-only and was not modified (its SHA-256 is checked before and after the build). Quality-control details are in `modeling_qc_report.md`.

## 1. Dataset purpose

The modeling dataset supports machine-learning models that predict bedtime, wake-time, and duration deviations relative to each participant's recent sleep behavior. These deviations serve as the project's operational measure of sleep irregularity.

| | Master dataset | Modeling dataset |
|---|---|---|
| Purpose | Faithful consolidation of the sources | Ready-to-model feature and target table |
| Rows | 4,042 (original observations only) | 4,701 (complete daily timeline per participant) |
| Columns | 9 | 40 |
| Values | Never altered | Original values untouched; inserted dates are flagged; features and targets added |
| Clock times | UTC epoch milliseconds | Local minutes after midnight (original epoch columns kept) |
| Missing dates | Absent | Inserted, with gaps of up to 7 days interpolated |

## 2. Participant filtering

Participants are flagged, not deleted, so no rows were removed. Every row carries `low_history_flag` (1 = flagged).

| Participant | Nights | Reason | Recommendation |
|---|---|---|---|
| 14 | 5 | Fewer than 14 nights; a 14-night baseline cannot be formed (0 recommended modeling rows) | Exclude from modeling |
| 20 | 8 | Fewer than 14 nights (1 recommended modeling row) | Exclude from modeling |

The other 19 participants (including `own_oura`) have at least 24 nights.

## 3. Missing-date handling

- **Timeline.** For each participant the rows are sorted by date and a complete daily timeline is built between the first and last observation.
- **Dates inserted:** **659** dates absent from the master. The 7 `own_oura` days that were already in the master but contain no data are treated the same way, so 666 dates in total are non-observations. These rows have `imputed = True`.
- **Interpolation strategy.** Gaps of 1 to 7 consecutive missing days are filled by linear interpolation between the observations on either side, for `bedtime_min`, `wake_min`, `duration`, `hrv`, `hr`, `resp` and `temp`. `bedtime_min` and `wake_min` are interpolated circularly, so a gap between 23:50 and 00:10 passes through midnight instead of noon.
- **Maximum interpolation gap:** 7 days. Values are never extrapolated beyond the first or last observation.
- **Large gaps.** 11 gaps longer than 7 days (176 dates in total) keep their inserted dates with missing values. They are listed in the QC report.
- **Original observations are never changed.** A real night that is missing a single metric (3 for hrv and hr, 6 for temp in UC Irvine) is left missing.
- **Result:** 490 dates were filled (3,430 individual values).

## 4. Feature engineering

Units: clock times (`bedtime_min`, `wake_min` and everything derived from them) are in **minutes**; `duration` and everything derived from it is in **seconds**, the same unit as the master.

All features use **only the nights before the night being predicted** (`t-1`, `t-2`, ...). Because the timeline is complete, "previous night" always means the previous calendar day.

| Feature | Definition |
|---|---|
| `prev_bedtime`, `prev_wake`, `prev_duration` | Previous night's local bedtime (min), wake time (min) and time in bed (s) |
| `prev_hrv`, `prev_hr`, `prev_resp`, `prev_temp` | Previous night's physiology. These replace the same-night values as model inputs |
| `bedtime_avg_3d`, `wake_avg_3d` | Circular mean of the previous 3 nights (needs at least 2) |
| `duration_avg_3d` | Mean of the previous 3 nights (needs at least 2) |
| `bedtime_avg_7d`, `wake_avg_7d` | Circular mean of the previous 7 nights (needs at least 4) |
| `duration_avg_7d` | Mean of the previous 7 nights (needs at least 4) |
| `bedtime_std_7d`, `wake_std_7d` | Sample std of each night's circular deviation from the window's circular mean (min; needs at least 4) |
| `duration_std_7d` | Sample std of the previous 7 nights (needs at least 4) |
| `day_of_week` | Monday = 0 ... Sunday = 6, from `day` |
| `weekend_flag` | 1 for Friday and Saturday nights. `day` is the wake date, so these are the nights that end on a Saturday or Sunday |

Circular statistics are used for clock times because a plain average of 23:50 and 00:10 would give noon.


## 5. Target generation

**Baselines** (`bedtime_base`, `wake_base`, `duration_base`): for each participant, the median of the previous 14 nights (`t-14` ... `t-1`), requiring at least 7 of those nights. The current night is excluded.
- `duration_base` is the ordinary median, in seconds.
- `bedtime_base` and `wake_base` use a circular median: the window is unwrapped around its circular mean, the median is taken, and the result is wrapped back into [0, 1440).

**Targets:**

```python
def circular_diff(x, baseline):
    return ((x - baseline + 720) % 1440) - 720

bedtime_dev  = circular_diff(bedtime_min, bedtime_base)   # minutes, in [-720, 720)
wake_dev     = circular_diff(wake_min,    wake_base)      # minutes, in [-720, 720)
duration_dev = duration - duration_base                   # seconds
```

Positive values mean later (or longer) than the participant's recent normal. For example, bedtime 00:10 against a baseline of 23:50 gives +20 minutes.

## 6. Validation

| Item | Result |
|---|---|
| Final row count | **4,701** |
| Final participant count | **21** (20 UC Irvine plus `own_oura`) |
| Original observations | 4,035 |
| Interpolated rows | **490** dates (3,430 values) |
| Remaining missing values | **8,595** cells across all 40 columns (see below) |
| Rows with all three targets | 4,310 |
| Recommended modeling rows | **3,503** (original night, targets present, `interp_lookahead = False`) | 

**Recommended training subset:**

low_history_flag == 0 imputed == False interp_lookahead == False bedtime_dev, wake_dev, and duration_dev are all present

This yields approximately 3,503 rows.

**Remaining missing values** come from:
- the 176 dates in gaps longer than 7 days;
- the epoch `bedtime` and `wake` columns on the 666 inserted or blank dates (1,332 cells);
- the first nights of each participant, where there is too little history for a lag, rolling feature or baseline;
- the few real nights missing a single metric.

The by-column table is in the QC report.

**Checks run by the script (it stops if any fails):**
- **No future leakage.** 191 perturbation runs overwrote all values dated on or after a random night with garbage and rebuilt the participant. Features and baselines on 20,253 earlier rows were identical, so no feature uses the current or any later night. The same perturbation did change the flagged rows described below (30 of 30), so the test can detect leakage.
- **Window logic.** 300 random rows were recomputed from the dates `t-14` ... `t-1` only and matched.
- **Circular deviations.** All 4,310 rows of `bedtime_dev` and `wake_dev` match an independent unit-circle computation (max difference below 1e-12 minutes). 355 bedtime rows and 11 wake rows cross midnight, where plain subtraction would be wrong by 24 h.
- **Lagged physiology.** `prev_hrv`, `prev_hr`, `prev_resp` and `prev_temp` equal the previous calendar night's value on every row, are empty on each participant's first night, and never cross participants. The original same-night columns are retained.
- **Original observations.** All 4,035 are identical to the master.

**Caveat: interpolation looks ahead.** Linear interpolation uses the next real night, so 1,000 rows carry look-ahead information and are marked `interp_lookahead = True`: the 666 non-observation dates, plus 334 real nights directly after a filled gap (their previous night was interpolated using that very night). Exclude flagged rows from training and evaluation. All other features and baselines were confirmed leak-free by the perturbation test.

## 7. Column dictionary (40 columns)

| Column | Role | Unit | Description |
|---|---|---|---|
| `participant_id` | identifier | | 1 to 20 (UC Irvine), `own_oura` |
| `day` | identifier | date | Wake date of the night (`YYYY-MM-DD`) |
| `bedtime`, `wake` | original, not a model input | epoch ms (UTC) | Original timestamps, preserved. Blank on inserted dates |
| `duration` | original | s | Time in bed, same night |
| `hrv`, `hr`, `resp`, `temp` | original, not a model input | ms, bpm, breaths/min, deg C | These columns are retained for reference and exploratory analysis. Forecasting models should use prev_hrv, prev_hr, prev_resp, and prev_temp instead.
| `imputed` | flag | bool | False = original observation, True = inserted or blank date |
| `interpolated` | flag | bool | True = values filled by interpolation |
| `interp_lookahead` | flag | bool | True = lagged features contain interpolation that used the current or a later night. Exclude from training and evaluation |
| `low_history_flag` | flag | 0/1 | 1 = participant has fewer than 14 nights |
| `bedtime_min`, `wake_min` | derived (current night) | min in [0, 1440) | Local minutes after midnight, used to build the targets |
| `prev_bedtime`, `prev_wake`, `prev_duration` | feature | min, min, s | Previous night |
| `prev_hrv`, `prev_hr`, `prev_resp`, `prev_temp` | feature | as above | Previous night's physiology |
| `bedtime_avg_3d`, `wake_avg_3d`, `duration_avg_3d` | feature | min, min, s | 3-night average of the prior nights |
| `bedtime_avg_7d`, `wake_avg_7d`, `duration_avg_7d` | feature | min, min, s | 7-night average of the prior nights |
| `bedtime_std_7d`, `wake_std_7d`, `duration_std_7d` | feature | min, min, s | 7-night variability of the prior nights |
| `day_of_week`, `weekend_flag` | feature | 0 to 6, 0/1 | Calendar features |
| `bedtime_base`, `wake_base`, `duration_base` | baseline | min, min, s | 14-night rolling median of prior nights |
| `bedtime_dev`, `wake_dev` | target | min | Circular deviation from baseline |
| `duration_dev` | target | s | Deviation from baseline |

**New columns in this version (8):** `bedtime_min`, `wake_min`, `prev_hrv`, `prev_hr`, `prev_resp`, `prev_temp`, `interp_lookahead`, `low_history_flag`.
**Removed:** `bedtime_sec` and `wake_sec` from the previous version, replaced by `bedtime_min` and `wake_min`.
**Redefined:** the clock-time averages, std, baselines and deviations are now circular and in minutes. They were linear and in seconds before.

A note for modeling: `prev_bedtime`, `prev_wake` and the clock-time averages wrap at midnight, so a sine/cosine encoding may help tree-free models. That was not added here.

## 8. Reproduction

```bash
python build_modeling_dataset.py \
    --master combined_oura_plus_personal.csv \
    --uci-full combined_oura_sleep_dataset.csv \
    --personal oura_2025-03-23_2026-09-17_trends.csv \
    --out-csv modeling_dataset.csv --out-report modeling_qc_report.md
```

The master alone cannot give local time (it stores UTC epoch milliseconds with no time zone), so the script also reads the original 29-column UC Irvine file for local clock offsets and your Oura export for local timestamps. It checks that both align row by row with the master. Runtime is about 12 seconds.
