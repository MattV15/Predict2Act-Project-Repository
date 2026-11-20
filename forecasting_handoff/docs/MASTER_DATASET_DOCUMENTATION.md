# Master Dataset Documentation: `combined_oura_plus_personal.csv` 

## 1. Dataset Purpose 
This file serves as the project's Master Dataset.

The dataset combines the UC Irvine Oura sleep records and the personal Oura Ring export into a common schema that can be used for downstream preprocessing and modeling.

The purpose of the Master Dataset is to preserve all available observations in a single location prior to any participant filtering, missing-date handling, interpolation, feature engineering, baseline calculations, or target generation.

All modifications required for machine-learning experiments will be performed in a separate Modeling Dataset.

## 2. Dataset sources

| Source | File(s) | Description |
|---|---|---|
| UC Irvine Oura sleep data | 20 per-participant `sleep.csv` files (par_1 to par_20) | Nightly Oura sleep records. The per-participant files were first consolidated into `combined_oura_sleep_dataset.csv` (28 original columns plus `participant_id`). par_21 was excluded. |
| Personal Oura export | `oura_2025-03-23_2026-09-17_trends.csv` (also supplied as `.xlsx`) | One Oura Ring user's daily "trends" export, 2025-03-23 to 2026-09-17. 54 columns, including sleep, activity and readiness metrics. |

Build script: `build_combined_oura_plus_personal.py`

## 3. Row counts

| Stage | Rows |
|---|---|
| UC Irvine consolidated (before merge) | 3,498 |
| Personal Oura export | 544 |
| **Final merged dataset** | **4,042** |

Of the 544 personal rows, 537 have sleep data and 7 are dates with no sleep recorded (all values blank).

Rows per UC Irvine participant: 1: 320, 2: 24, 3: 136, 4: 85, 5: 172, 6: 172, 7: 334, 8: 318, 9: 316, 10: 156, 11: 150, 12: 156, 13: 178, 14: 5, 15: 55, 16: 69, 17: 355, 18: 284, 19: 205, 20: 8. Personal (`own_oura`): 544.

## 4. Participant counts

| Stage | Participants |
|---|---|
| Before merge | 20 (IDs 1 to 20) |
| After merge | 21 (IDs 1 to 20 plus `own_oura`) |

## 5. Column mappings

Final schema: `participant_id, day, bedtime, wake, duration, hrv, hr, resp, temp`

| Final column | UC Irvine column | Personal Oura column | Notes |
|---|---|---|---|
| participant_id | `participant_id` | constant `own_oura` | |
| day | `date` | first column (header corrupted in export) | ISO `YYYY-MM-DD` |
| bedtime | `bedtime_start_timestamp` | `Bedtime Start` | UTC epoch milliseconds |
| wake | `bedtime_end_timestamp` | `Bedtime End` | UTC epoch milliseconds |
| duration | `duration` | `Total Bedtime` | Seconds in bed (trailing space in the export header) |
| hrv | `rmssd` | `Average HRV` | |
| hr | `hr_average` | `Average Resting Heart Rate` | Average, not lowest |
| resp | `breath_average` | `Respiratory Rate` | |
| temp | `temperature_delta` | `Temperature Deviation (°C)` | |

All 8 value columns were matched. UC Irvine columns not carried into the final schema: the score columns, sleep stage durations, `awake`, `onset_latency`, `midpoint_time`, `efficiency`, `restless`, `hr_lowest` and the two `midnight_delta` columns. Personal export columns not carried over: activity, readiness, steps, recovery and the remaining score columns, which have no UC Irvine counterpart.

## 6. Assumptions

1. **Participant order.** The UC Irvine files were all named `sleep.csv`. IDs 1 to 20 were assigned in the order the files were attached. This should be checked against the original `par_N` folder names.
2. **Source schema.** The UC Irvine file supplied is the 29-column consolidated file, not a 9-column file. The final columns were derived from it using the mapping above.
3. **`duration` is time in bed**, not total sleep. UC Irvine `duration` matches the personal `Total Bedtime`. Total Sleep Duration was not used.
4. **`hr` is average heart rate** (`hr_average` and `Average Resting Heart Rate`). The lowest-heart-rate columns were not used.
5. **Equivalence of metrics.** `rmssd` is treated as equivalent to `Average HRV`, and `hr_average` to `Average Resting Heart Rate`. These come from different Oura export formats and have not been independently validated for comparability. In the data, the personal HRV (mean about 41) is lower than the UC Irvine average (mean about 65).
6. **Timestamps.** UC Irvine timestamps are UTC epoch milliseconds. Personal ISO timestamps carry a local UTC offset and were converted to the same format. The local-time offset is not kept in the final schema.
7. **Missing values.** UC Irvine used the text `null` and the personal export used `None`. Both are written as empty cells.
8. **Personal participant ID** is `own_oura`.
9. **Known data quirks left as-is.** 13 UC Irvine rows have a bedtime start later than the wake time (a timestamp wrap in the source). The UC Irvine files are not in date order within each participant. Neither was corrected.

## 7. Merge procedure

1. Load the UC Irvine consolidated CSV with every value read as text. Select and rename the mapped columns. Replace `null` with an empty cell.
2. Load the personal CSV the same way. Rename the corrupted first header to `date`. Replace `None` with an empty cell.
3. Convert personal dates from `M/D/YYYY` to `YYYY-MM-DD`.
4. Convert personal `Bedtime Start` and `Bedtime End` from ISO-8601 to UTC epoch milliseconds.
5. Select and rename the personal columns to the final schema and set `participant_id = own_oura`.
6. Append the personal rows below the UC Irvine rows with `pd.concat`. No filtering, interpolation, imputation, scaling, feature engineering or date filling was done, and no rows were added or removed.
7. Export to `combined_oura_plus_personal.csv`.

Validation: all 3,498 UC Irvine rows are unchanged from the source, apart from `null` becoming an empty cell. The build script is designed to reproduce the combined dataset deterministically from the original source files.
