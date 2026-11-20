#!/usr/bin/env python3
"""
Build combined_oura_plus_personal.csv

Appends a personal Oura Ring export to the consolidated UC Irvine Oura sleep
dataset (participants 1-20), harmonized to a 9-column schema:

    participant_id, day, bedtime, wake, duration, hrv, hr, resp, temp

No values are modified, imputed, scaled, filtered or interpolated, and no rows
are added or removed. The only changes are representation changes needed to
put both sources in one schema:
  * column renaming / selection
  * personal dates M/D/YYYY            -> YYYY-MM-DD
  * personal bedtime/wake ISO-8601     -> UTC epoch milliseconds (as in UC Irvine)
  * missing-value markers ("null" in UC Irvine, "None" in personal) -> empty cell

Usage:
    python build_combined_oura_plus_personal.py \
        --uci combined_oura_sleep_dataset.csv \
        --personal oura_2025-03-23_2026-09-17_trends.csv \
        --out combined_oura_plus_personal.csv
"""
import argparse

import pandas as pd

PERSONAL_ID = "own_oura"

SCHEMA = ["participant_id", "day", "bedtime", "wake", "duration", "hrv", "hr", "resp", "temp"]

# Schema column -> column in the UC Irvine consolidated file
UCI_MAP = {
    "day": "date",
    "bedtime": "bedtime_start_timestamp",   # UTC epoch ms
    "wake": "bedtime_end_timestamp",        # UTC epoch ms
    "duration": "duration",                 # seconds in bed
    "hrv": "rmssd",
    "hr": "hr_average",
    "resp": "breath_average",
    "temp": "temperature_delta",
}

# Schema column -> column in the personal Oura export
# NOTE: "Total Bedtime " has a trailing space in the export header.
PERSONAL_MAP = {
    "bedtime": "Bedtime Start",             # ISO-8601 with local UTC offset
    "wake": "Bedtime End",                  # ISO-8601 with local UTC offset
    "duration": "Total Bedtime ",           # seconds in bed
    "hrv": "Average HRV",
    "hr": "Average Resting Heart Rate",
    "resp": "Respiratory Rate",
    "temp": "Temperature Deviation (°C)",
}


def load_uci(path):
    # dtype=str + keep_default_na=False: read every value as its original text
    uci = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = pd.DataFrame({"participant_id": uci["participant_id"]})
    for new, old in UCI_MAP.items():
        out[new] = uci[old].replace("null", "")      # literal "null" -> empty
    return out[SCHEMA]


def load_personal(path):
    me = pd.read_csv(path, dtype=str, keep_default_na=False)
    # The first column (the date) has a corrupted header in the export
    # ("Compilation error: redefinition of 'int LED'"); rename it by position.
    me.columns = ["date"] + list(me.columns[1:])
    me = me.replace("None", "")                       # literal "None" -> empty

    out = pd.DataFrame({"participant_id": PERSONAL_ID}, index=me.index)

    # Date conversion: M/D/YYYY -> YYYY-MM-DD
    out["day"] = pd.to_datetime(me["date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")

    # Timestamp conversion: ISO-8601 with offset (e.g. 2025-03-23T03:17:32.000-05:00)
    # -> UTC epoch milliseconds, matching the UC Irvine format. Lossless: the
    # instant is unchanged. Empty strings stay empty.
    def iso_to_epoch_ms(s):
        if s == "":
            return ""
        return str(int(pd.Timestamp(s).timestamp() * 1000))

    out["bedtime"] = me[PERSONAL_MAP["bedtime"]].map(iso_to_epoch_ms)
    out["wake"] = me[PERSONAL_MAP["wake"]].map(iso_to_epoch_ms)

    for col in ["duration", "hrv", "hr", "resp", "temp"]:
        out[col] = me[PERSONAL_MAP[col]]
    return out[SCHEMA]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci", required=True)
    ap.add_argument("--personal", required=True)
    ap.add_argument("--out", default="combined_oura_plus_personal.csv")
    a = ap.parse_args()

    uci = load_uci(a.uci)
    personal = load_personal(a.personal)

    # Filtering: none. Every row from both sources is kept, including the
    # personal-export days that have a date but no sleep data (blank values).
    combined = pd.concat([uci, personal], ignore_index=True)

    print(f"Rows before merge:         {len(uci)}")
    print(f"Rows after merge:          {len(combined)}  (personal rows added: {len(personal)})")
    print(f"Participants before merge: {uci['participant_id'].nunique()}")
    print(f"Participants after merge:  {combined['participant_id'].nunique()}")

    combined.to_csv(a.out, index=False)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
