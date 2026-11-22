#!/usr/bin/env python3
"""
Build modeling_dataset.csv (+ modeling_qc_report.md) from combined_oura_plus_personal.csv.

The master dataset is READ ONLY: it is never written to or modified.

Pipeline
  0. Load master; derive LOCAL clock time for bedtime / wake (minutes after midnight)
  1. Participant summary (nights, date range, missing dates, largest gap, <14-night flag)
  2. Complete daily timeline per participant; flag inserted dates (`imputed`)
  3. Interpolate gaps of 1-7 consecutive missing days (circular-aware for clock times);
     leave longer gaps missing
  4. Sleep-history features + LAGGED physiology (previous night only)
  5. Participant-specific rolling 14-night median baselines (circular for clock times)
  6. Targets: bedtime_dev, wake_dev (circular, minutes), duration_dev (seconds)
  7. Validation (no future leakage, circular arithmetic, lags), export, QC report

Units
  bedtime_min, wake_min, *_dev for clock times : minutes (clock times in [0, 1440))
  duration and its features/baseline/dev         : seconds (same unit as the master)

Usage
  python build_modeling_dataset.py \
      --master combined_oura_plus_personal.csv \
      --uci-full combined_oura_sleep_dataset.csv \
      --personal oura_2025-03-23_2026-09-17_trends.csv \
      --out-csv modeling_dataset.csv --out-report modeling_qc_report.md
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

PERSONAL_ID = "own_oura"
NUM_COLS = ["duration", "hrv", "hr", "resp", "temp"]           # original numeric columns
CIRC_COLS = ["bedtime_min", "wake_min"]                        # circular (period 1440 min)
INTERP_COLS = CIRC_COLS + NUM_COLS                             # columns eligible for interpolation
PHYS_COLS = ["hrv", "hr", "resp", "temp"]                      # physiology -> lagged
MAX_INTERP_GAP = 7         # interpolate only gaps of 1..7 consecutive missing days
MIN_NIGHTS = 14            # participants below this cannot support a 14-night baseline
BASE_WINDOW, BASE_MIN = 14, 7       # 14-night median baseline, require >= 7 prior nights
ROLL3_MIN, ROLL7_MIN = 2, 4         # >= half of the window must be available
SLEEP_NEED_SEC = 8 * 3600           # nightly need used by sleep_debt (assumption, see docs)
PERIOD = 1440                       # minutes per day

FEATURE_COLS = (
    ["prev_bedtime", "prev_wake", "prev_duration", "prev_hrv", "prev_hr", "prev_resp", "prev_temp",
     "bedtime_avg_3d", "wake_avg_3d", "duration_avg_3d",
     "bedtime_avg_7d", "wake_avg_7d", "duration_avg_7d",
     "bedtime_std_7d", "wake_std_7d", "duration_std_7d",
     "sleep_debt", "day_of_week", "weekend_flag"]
)
BASE_COLS = ["bedtime_base", "wake_base", "duration_base"]
TARGET_COLS = ["bedtime_dev", "wake_dev", "duration_dev"]
FINAL_COLS = (
    ["participant_id", "day", "bedtime", "wake", "duration", "hrv", "hr", "resp", "temp",
     "imputed", "interpolated", "interp_lookahead", "low_history_flag",
     "bedtime_min", "wake_min"] + FEATURE_COLS + BASE_COLS + TARGET_COLS
)


# ----------------------------------------------------------------------------
# Circular helpers (period = 1440 minutes)
# ----------------------------------------------------------------------------
def circular_diff(x, baseline):
    """Signed shortest difference x - baseline on a 24 h clock, in [-720, 720)."""
    return ((x - baseline + 720) % 1440) - 720


def _finite(w):
    return w[np.isfinite(w)]


def circ_mean(w):
    """Circular mean (minutes in [0, 1440)) of the finite values in a window."""
    w = _finite(w)
    if w.size == 0:
        return np.nan
    ang = w * 2 * np.pi / PERIOD
    return (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * PERIOD / (2 * np.pi)) % PERIOD


def circ_median(w):
    """Circular median: unwrap the window around its circular mean, take the median, wrap."""
    w = _finite(w)
    if w.size == 0:
        return np.nan
    ref = circ_mean(w)
    return (ref + np.median(circular_diff(w, ref))) % PERIOD


def circ_std(w):
    """Spread (minutes): sample std of circular deviations from the circular mean."""
    w = _finite(w)
    if w.size < 2:
        return np.nan
    return np.std(circular_diff(w, circ_mean(w)), ddof=1)


def circular_interpolate(s):
    """Linear interpolation that goes the short way round midnight (23:50 -> 00:10 passes 00:00)."""
    valid = s.dropna()
    if len(valid) < 2:
        return s
    unwrapped = pd.Series(np.unwrap(valid.to_numpy(float), period=PERIOD), index=valid.index)
    return unwrapped.reindex(s.index).interpolate(method="linear", limit_area="inside") % PERIOD


# ----------------------------------------------------------------------------
# Step 0. Load master and derive LOCAL clock time (minutes after midnight)
# ----------------------------------------------------------------------------
def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_master(path):
    """Read the master with all values as text (nothing is altered)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def add_local_clock(m, uci_full_path, personal_path):
    """
    The master stores bedtime/wake as UTC epoch milliseconds and carries no time-zone
    information, so local clock time cannot be recovered from the master alone. We
    derive `bedtime_min` / `wake_min` = LOCAL MINUTES AFTER MIDNIGHT in [0, 1440):

      * UC Irvine: from bedtime/wake_midnight_delta (seconds after local midnight of
        `day`, negative before midnight) in the original 29-column UC Irvine file.
        13 rows have start > end (start wrapped by +24 h in the source); for those the
        start is shifted by -86400 s so end - start equals the recorded duration.
      * Personal: wall-clock time of the ISO timestamp (it carries the local UTC offset).

    The signed value is reduced modulo 1440, so a bedtime of 23:50 -> 1430 and 00:10 -> 10:
    sleep that crosses midnight is handled by circular arithmetic downstream.
    Rows are aligned by position and validated against the master's epoch values.
    """
    is_p = m["participant_id"].eq(PERSONAL_ID)
    n_uci = int((~is_p).sum())
    assert not is_p.iloc[:n_uci].any() and is_p.iloc[n_uci:].all(), "own_oura rows must be last"

    # ---- UC Irvine -----------------------------------------------------------
    uci = pd.read_csv(uci_full_path, dtype=str, keep_default_na=False)
    mu = m.iloc[:n_uci]
    assert len(uci) == n_uci
    assert (uci["participant_id"].values == mu["participant_id"].values).all()
    assert (uci["date"].values == mu["day"].values).all()
    assert (uci["bedtime_start_timestamp"].values == mu["bedtime"].values).all()
    assert (uci["bedtime_end_timestamp"].values == mu["wake"].values).all()
    start = uci["bedtime_start_midnight_delta"].astype(float).to_numpy()
    end = uci["bedtime_end_midnight_delta"].astype(float).to_numpy()
    wrapped = end < start
    start = np.where(wrapped, start - 86400, start)

    # ---- Personal --------------------------------------------------------------
    me = pd.read_csv(personal_path, dtype=str, keep_default_na=False)
    me.columns = ["date"] + list(me.columns[1:])   # first header is corrupted in the export
    me = me.replace("None", "")
    mp = m.iloc[n_uci:]
    day = pd.to_datetime(me["date"], format="%m/%d/%Y")
    assert (day.dt.strftime("%Y-%m-%d").values == mp["day"].values).all()
    for src, dst in [("Bedtime Start", "bedtime"), ("Bedtime End", "wake")]:
        epoch = me[src].map(lambda s: "" if s == "" else str(int(pd.Timestamp(s).timestamp() * 1000)))
        assert (epoch.values == mp[dst].values).all(), f"personal {dst} epoch mismatch"
    wall_s = pd.to_datetime(me["Bedtime Start"].str[:19], errors="coerce")
    wall_e = pd.to_datetime(me["Bedtime End"].str[:19], errors="coerce")
    start_p = (wall_s - day).dt.total_seconds().to_numpy()
    end_p = (wall_e - day).dt.total_seconds().to_numpy()

    m = m.copy()
    m["bedtime_min"] = np.r_[start, start_p] / 60 % PERIOD      # local minutes after midnight
    m["wake_min"] = np.r_[end, end_p] / 60 % PERIOD
    m["wrapped_fixed"] = np.r_[wrapped, np.zeros(len(mp), dtype=bool)]
    return m


# ----------------------------------------------------------------------------
# Steps 1-6 for one participant
# ----------------------------------------------------------------------------
def process_participant(g):
    g = g.sort_values("day")                                  # sort by date
    assert not g["day"].duplicated().any(), "duplicate dates within participant"

    has_obs = g["bedtime"] != ""                              # a blank row is not an observation
    obs = g[has_obs].set_index("day")
    first, last = obs.index.min(), obs.index.max()
    outside = g[(~has_obs) & ((g["day"] < first) | (g["day"] > last))]
    assert outside.empty, "blank rows outside observed range"

    # Step 2: complete daily timeline between first and last observation
    idx = pd.date_range(first, last, freq="D", name="day")
    t = obs[["bedtime", "wake"] + INTERP_COLS].reindex(idx)
    in_file = pd.Series(idx.isin(g["day"]), index=idx)        # date present in the master (even if blank)
    imputed = pd.Series(~idx.isin(obs.index), index=idx)      # False = original observation

    # Length of each run of consecutive missing dates
    run_id = (imputed != imputed.shift()).cumsum()
    gap_len = imputed.groupby(run_id).transform("sum").where(imputed, 0).astype(int)
    gaps = [
        {"participant_id": g["participant_id"].iloc[0], "start": s.index.min(), "end": s.index.max(), "days": len(s)}
        for _, s in imputed[imputed].groupby(run_id[imputed])
    ]

    # Step 3: interpolate small gaps only, and only inside imputed dates.
    # Original observations are never touched (even if one metric is missing).
    fillable = imputed & (gap_len <= MAX_INTERP_GAP)
    interpolated_row = pd.Series(False, index=idx)
    n_interp = 0
    for c in INTERP_COLS:
        if c in CIRC_COLS:
            full = circular_interpolate(t[c])                                 # circular-aware
        else:
            full = t[c].interpolate(method="linear", limit_area="inside")     # never extrapolates
        fill = fillable & t[c].isna() & full.notna()
        t.loc[fill, c] = full[fill]
        n_interp += int(fill.sum())
        interpolated_row |= fill

    # Step 4/5/6: everything below uses PRIOR nights only (shift(1)), so the night
    # being predicted never leaks into its own features or baseline.
    for c, name, circular in [("bedtime_min", "bedtime", True), ("wake_min", "wake", True),
                              ("duration", "duration", False)]:
        prior = t[c].shift(1)
        t[f"prev_{name}"] = prior
        r3 = prior.rolling(3, min_periods=ROLL3_MIN)
        r7 = prior.rolling(7, min_periods=ROLL7_MIN)
        r14 = prior.rolling(BASE_WINDOW, min_periods=BASE_MIN)
        if circular:
            t[f"{name}_avg_3d"] = r3.apply(circ_mean, raw=True)
            t[f"{name}_avg_7d"] = r7.apply(circ_mean, raw=True)
            t[f"{name}_std_7d"] = r7.apply(circ_std, raw=True)
            t[f"{name}_base"] = r14.apply(circ_median, raw=True)              # circular 14-night median
            t[f"{name}_dev"] = circular_diff(t[c], t[f"{name}_base"])         # circular deviation (min)
        else:
            t[f"{name}_avg_3d"] = r3.mean()
            t[f"{name}_avg_7d"] = r7.mean()
            t[f"{name}_std_7d"] = r7.std()
            t[f"{name}_base"] = r14.median()
            t[f"{name}_dev"] = t[c] - t[f"{name}_base"]                       # seconds

    # Lagged physiology: the previous night's values (same-night values are NOT features)
    for c in PHYS_COLS:
        t[f"prev_{c}"] = t[c].shift(1)
        
    # Calendar features. `day` is the WAKE date, so the night that ended on a
    # Saturday or Sunday began on a Friday or Saturday night -> weekend_flag = 1.
    t["day_of_week"] = t.index.dayofweek                       # Monday=0 ... Sunday=6
    t["weekend_flag"] = t["day_of_week"].isin([5, 6]).astype(int)

    # Flags
    t["imputed"] = imputed
    t["interpolated"] = interpolated_row
    # Linear interpolation looks AHEAD to the next observation. An imputed date, and the first
    # original night after a filled gap (its previous night was interpolated using that very
    # night), therefore carry look-ahead information in their lagged features / baselines.
    t["interp_lookahead"] = imputed | interpolated_row.shift(1, fill_value=False)
    t.insert(0, "participant_id", g["participant_id"].iloc[0])

    stats = {
        "participant_id": g["participant_id"].iloc[0],
        "number_of_nights": int(has_obs.sum()),
        "first_date": first.strftime("%Y-%m-%d"),
        "last_date": last.strftime("%Y-%m-%d"),
        "missing_dates": int(imputed.sum()),
        "largest_gap_days": int(gap_len.max()),
        "inserted_rows": int((~in_file).sum()),            # dates not in the master at all
        "blank_rows": int((in_file & imputed).sum()),      # dates in master but with no data
        "interpolated_values": n_interp,
        "rows_interpolated": int(interpolated_row.sum()),
        "wrapped_fixed": int(g["wrapped_fixed"].sum()),
    }
    return t.reset_index(), stats, gaps


# ----------------------------------------------------------------------------
# Step 7a. Validation
# ----------------------------------------------------------------------------
def _same(a, b, tol=1e-6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, atol=tol, equal_nan=True))


def validate(out, m, n_perturb=10, seed=0):
    """Returns a list of (check, result) strings; raises AssertionError on any failure."""
    res = []
    rng = np.random.default_rng(seed)

    # --- A. Circular deviations ------------------------------------------------
    assert circular_diff(10, 1430) == 20 and circular_diff(1430, 10) == -20   # 23:50 vs 00:10
    res.append(("Circular deviations: worked example",
                "23:50 (1430) vs 00:10 (10) -> circular_diff(10, 1430) = +20 min and (1430, 10) = -20 min"))
    for c in ["bedtime", "wake"]:
        x, b, d = out[f"{c}_min"].to_numpy(), out[f"{c}_base"].to_numpy(), out[f"{c}_dev"].to_numpy()
        ok = np.isfinite(d)
        # independent formulation: angle between the two points on the unit circle
        ang = np.angle(np.exp(2j * np.pi * x[ok] / PERIOD) * np.exp(-2j * np.pi * b[ok] / PERIOD)) * PERIOD / (2 * np.pi)
        gap = np.abs(((ang - d[ok] + 720) % PERIOD) - 720)       # distance modulo the +/-720 boundary
        assert gap.max() < 1e-6, f"{c}: circular deviation mismatch"
        assert d[ok].min() >= -720 and d[ok].max() < 720, f"{c}: deviation outside [-720, 720)"
        naive = x[ok] - b[ok]
        cross = np.abs(naive) > 720                              # rows where a naive subtraction is wrong
        assert np.allclose(d[ok][cross], naive[cross] - np.sign(naive[cross]) * PERIOD)
        res.append((f"Circular deviations: {c}_dev",
                    f"{ok.sum()} rows match an independent unit-circle computation (max diff {gap.max():.2e} min); "
                    f"range [{d[ok].min():.1f}, {d[ok].max():.1f}) min; {int(cross.sum())} rows cross midnight "
                    f"where a naive x - base would be off by 24 h"))

    # --- B. Lagged variables equal the previous calendar night ---------------------
    lag_map = {"prev_bedtime": "bedtime_min", "prev_wake": "wake_min", "prev_duration": "duration",
               "prev_hrv": "hrv", "prev_hr": "hr", "prev_resp": "resp", "prev_temp": "temp"}
    base_df = out[["participant_id", "day"] + list(lag_map.values())].copy()
    base_df["day"] = base_df["day"] + pd.Timedelta(days=1)       # value of night t-1 re-labelled as t
    joined = out[["participant_id", "day"] + list(lag_map)].merge(
        base_df, on=["participant_id", "day"], how="left", suffixes=("", "_src"))
    for p, c in lag_map.items():
        assert _same(joined[p], joined[c]), p
    first_rows = out.groupby("participant_id").head(1)
    assert first_rows[list(lag_map)].isna().all().all(), "first night of a participant must have no lag"
    res.append(("Lagged physiology",
                "prev_hrv / prev_hr / prev_resp / prev_temp (and prev_bedtime / prev_wake / prev_duration) equal the "
                f"previous calendar night's value on all {len(out)} rows; the first night of every participant is empty; "
                "lags never cross participants; the original hrv, hr, resp, temp columns are retained"))

    # --- C. Brute-force recomputation of features from past nights only -------------
    series = {pid: g.set_index("day") for pid, g in out.groupby("participant_id", sort=False)}
    cand = out[out["bedtime_base"].notna()]
    sample = cand.sample(min(300, len(cand)), random_state=seed)
    d1 = pd.Timedelta(days=1)

    def window(s, t, n):
        return s.loc[t - n * d1: t - d1].to_numpy(float)

    def cmean(w):  # circular mean via complex numbers (independent of circ_mean)
        w = w[np.isfinite(w)]
        return (np.angle(np.exp(2j * np.pi * w / PERIOD).mean()) * PERIOD / (2 * np.pi)) % PERIOD

    for r in sample.itertuples():
        s = series[r.participant_id]
        t0 = r.day
        for name, circular in [("bedtime", True), ("wake", True), ("duration", False)]:
            col = "duration" if name == "duration" else f"{name}_min"
            w3, w7, w14 = window(s[col], t0, 3), window(s[col], t0, 7), window(s[col], t0, 14)
            f3, f7, f14 = w3[np.isfinite(w3)], w7[np.isfinite(w7)], w14[np.isfinite(w14)]
            if circular:
                exp3 = cmean(f3) if len(f3) >= ROLL3_MIN else np.nan
                exp7 = cmean(f7) if len(f7) >= ROLL7_MIN else np.nan
                ref = cmean(f14)
                exp_base = (ref + np.median(((f14 - ref + 720) % PERIOD) - 720)) % PERIOD if len(f14) >= BASE_MIN else np.nan
                exp_std = np.std(((f7 - cmean(f7) + 720) % PERIOD) - 720, ddof=1) if len(f7) >= ROLL7_MIN else np.nan
            else:
                exp3 = f3.mean() if len(f3) >= ROLL3_MIN else np.nan
                exp7 = f7.mean() if len(f7) >= ROLL7_MIN else np.nan
                exp_base = np.median(f14) if len(f14) >= BASE_MIN else np.nan
                exp_std = f7.std(ddof=1) if len(f7) >= ROLL7_MIN else np.nan
            assert _same([exp3], [getattr(r, f"{name}_avg_3d")]), (r.participant_id, t0, name, "avg3")
            assert _same([exp7], [getattr(r, f"{name}_avg_7d")]), (r.participant_id, t0, name, "avg7")
            assert _same([exp_std], [getattr(r, f"{name}_std_7d")]), (r.participant_id, t0, name, "std7")
            assert _same([exp_base], [getattr(r, f"{name}_base")]), (r.participant_id, t0, name, "base")
        dur7 = window(s["duration"], t0, 7)
        sf = np.clip(SLEEP_NEED_SEC - dur7[np.isfinite(dur7)], 0, None)
        exp_debt = 7 * sf.mean() if len(sf) >= ROLL7_MIN else np.nan
        assert _same([exp_debt], [r.sleep_debt]), (r.participant_id, t0, "sleep_debt")
    res.append(("Window logic (brute force)",
                f"{len(sample)} random rows: prev_*, 3d/7d means, 7d std, 14-night baseline and sleep_debt recomputed "
                "from the dates t-14 ... t-1 only (current night excluded) and matched the pipeline output"))

    # --- D. Future-perturbation test (no future leakage) ---------------------------
    # Replace every value dated >= t0 with garbage, rebuild the participant, and require that all
    # features/baselines on rows dated <= t0 are unchanged. Rows with interp_lookahead are excluded
    # from the pass criterion (interpolation looks ahead by construction) and used as a control.
    feat_cols = FEATURE_COLS + BASE_COLS

    def perturb_and_rebuild(pid, t0):
        g = m[m["participant_id"] == pid].copy()
        hit = (g["day"] >= t0) & (g["bedtime"] != "")
        g.loc[hit, "duration"] = g.loc[hit, "duration"] * 1.7 + 4321
        g.loc[hit, "hrv"] = g.loc[hit, "hrv"] + 50
        g.loc[hit, "hr"] = g.loc[hit, "hr"] + 30
        g.loc[hit, "resp"] = g.loc[hit, "resp"] + 5
        g.loc[hit, "temp"] = g.loc[hit, "temp"] + 2
        g.loc[hit, "bedtime_min"] = (g.loc[hit, "bedtime_min"] + 311.7) % PERIOD
        g.loc[hit, "wake_min"] = (g.loc[hit, "wake_min"] + 197.3) % PERIOD
        return process_participant(g)[0]

    n_tests = n_rows_checked = 0
    for pid, o in out.groupby("participant_id", sort=False):
        ok = o[(~o["interp_lookahead"]) & o["bedtime_base"].notna()]
        if ok.empty:
            continue
        picks = ok["day"].iloc[rng.choice(len(ok), size=min(n_perturb, len(ok)), replace=False)]
        for t0 in picks:
            p = perturb_and_rebuild(pid, t0)
            a = o[(o["day"] <= t0) & (~o["interp_lookahead"])].reset_index(drop=True)
            b = p[p["day"].isin(a["day"])].reset_index(drop=True)     # same dates, rebuilt from perturbed data
            assert (a["day"].values == b["day"].values).all()
            for c in feat_cols:
                assert _same(a[c], b[c]), f"future leakage: {pid} t0={t0.date()} {c}"
            n_tests += 1
            n_rows_checked += len(a)

    # control: rows flagged interp_lookahead (original nights right after a filled gap) SHOULD change
    flagged = out[(~out["imputed"]) & out["interp_lookahead"]]
    changed = tested = 0
    for r in flagged.sample(min(30, len(flagged)), random_state=seed).itertuples():
        p = perturb_and_rebuild(r.participant_id, r.day)
        row = p[p["day"] == r.day].iloc[0]
        orig = out[(out["participant_id"] == r.participant_id) & (out["day"] == r.day)].iloc[0]
        tested += 1
        changed += int(not all(_same([orig[c]], [row[c]]) for c in feat_cols))
    res.append(("No future leakage (perturbation test)",
                f"{n_tests} perturbation runs: all raw values dated on/after a random night t0 were overwritten with "
                f"garbage and the participant was rebuilt; features and baselines on {n_rows_checked} rows dated <= t0 "
                "(excluding interp_lookahead rows) were identical -> no feature uses the current or any later night"))
    res.append(("Look-ahead through interpolation (control and caveat)",
                f"Linear interpolation of a gap uses the next observation, so {int(out['interp_lookahead'].sum())} rows "
                f"({int(out['imputed'].sum())} imputed dates + {int((~out['imputed'] & out['interp_lookahead']).sum())} "
                "original nights directly after a filled gap) are flagged interp_lookahead=True. As a control, "
                f"{changed} of {tested} sampled flagged original nights DID change under the same perturbation, "
                "so the test can detect leakage. Exclude flagged rows from training and evaluation"))

    # --- E. Original observations --------------------------------------------------
    orig = m[m["bedtime"] != ""].set_index(["participant_id", "day"]).sort_index()
    kept = out[~out["imputed"]].set_index(["participant_id", "day"]).sort_index()
    assert orig.index.equals(kept.index), "original observations lost or added"
    for c in NUM_COLS + CIRC_COLS:
        assert _same(orig[c], kept[c]), c
    for c in ["bedtime", "wake"]:
        assert (orig[c].to_numpy() == kept[c].fillna("").to_numpy()).all(), c
    res.append(("Original observations preserved",
                f"{len(kept)} original rows: bedtime/wake epoch timestamps, duration, hrv, hr, resp, temp identical to the master"))
    return res


# ----------------------------------------------------------------------------
# Report helpers
# ----------------------------------------------------------------------------
def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--uci-full", required=True, help="original 29-column UC Irvine file (for local clock offsets)")
    ap.add_argument("--personal", required=True, help="personal Oura CSV export (for local clock times)")
    ap.add_argument("--out-csv", default="modeling_dataset.csv")
    ap.add_argument("--out-report", default="modeling_qc_report.md")
    ap.add_argument("--n-perturb", type=int, default=10, help="leakage perturbation tests per participant")
    a = ap.parse_args()

    master_hash = sha256(a.master)                            # proves the master is untouched
    m = load_master(a.master)
    m = add_local_clock(m, a.uci_full, a.personal)
    m["day"] = pd.to_datetime(m["day"])
    for c in NUM_COLS:
        m[c] = pd.to_numeric(m[c].replace("", np.nan))
    order = list(dict.fromkeys(m["participant_id"]))          # keep master participant order

    # Steps 1-6 per participant
    parts, stats, gaps = [], [], []
    for pid in order:
        t, s, gp = process_participant(m[m["participant_id"] == pid])
        parts.append(t); stats.append(s); gaps += gp
    out = pd.concat(parts, ignore_index=True)
    st = pd.DataFrame(stats)
    gaps = pd.DataFrame(gaps)

    # Step 1: participant filtering flag (participants are flagged, never deleted)
    low = set(st.loc[st["number_of_nights"] < MIN_NIGHTS, "participant_id"])
    out["low_history_flag"] = out["participant_id"].isin(low).astype(int)

    # Step 7a: validation (raises on failure, so nothing is exported if a check fails)
    validation = validate(out, m, n_perturb=a.n_perturb)
    assert sha256(a.master) == master_hash, "master changed!"

    # Step 7b: export
    out["day"] = out["day"].dt.strftime("%Y-%m-%d")
    out["bedtime"] = out["bedtime"].fillna("")
    out["wake"] = out["wake"].fillna("")
    out[FINAL_COLS].to_csv(a.out_csv, index=False)

    # ---- QC report -------------------------------------------------------------
    flagged = st[st["number_of_nights"] < MIN_NIGHTS]
    out["has_targets"] = out[TARGET_COLS].notna().all(axis=1)
    out["usable"] = (~out["imputed"]) & (~out["interp_lookahead"]) & out["has_targets"]
    usable = out[out["usable"]].groupby("participant_id").size()
    st["usable_rows"] = st["participant_id"].map(usable).fillna(0).astype(int)
    excluded = st[(st["number_of_nights"] < MIN_NIGHTS) | (st["usable_rows"] == 0)]

    part_tbl = st[["participant_id", "number_of_nights", "first_date", "last_date", "missing_dates", "largest_gap_days"]].copy()
    part_tbl["flag_<14_nights"] = np.where(part_tbl["number_of_nights"] < MIN_NIGHTS, "FLAG", "")
    part_tbl = part_tbl.rename(columns={"number_of_nights": "nights"})
    imp_tbl = st[["participant_id", "inserted_rows", "blank_rows", "interpolated_values", "rows_interpolated"]]

    big = gaps[gaps["days"] > MAX_INTERP_GAP].copy()
    big["start"] = big["start"].dt.strftime("%Y-%m-%d"); big["end"] = big["end"].dt.strftime("%Y-%m-%d")
    left_missing_rows = int((out["imputed"] & ~out["interpolated"]).sum())
    miss = out[FINAL_COLS].replace("", np.nan).isna().sum()
    miss_by_col = miss[miss > 0].rename("missing").reset_index().rename(columns={"index": "column"})
    val_tbl = pd.DataFrame(validation, columns=["check", "result"])

    lines = ["# Modeling dataset: quality-control report", "",
             f"Master dataset: `combined_oura_plus_personal.csv` (unchanged; sha256 `{master_hash[:16]}...`)", "",
             "## Participant Summary", "",
             md_table(part_tbl), "",
             f"Participants with fewer than {MIN_NIGHTS} nights (flagged, not deleted): "
             + (", ".join(flagged["participant_id"]) or "none"), "",
             "## Imputation Summary", "",
             "`inserted_rows` = dates absent from the master; `blank_rows` = dates present but with no data "
             "(own_oura only); `interpolated_values` = individual cells filled; `rows_interpolated` = rows with at least one filled cell.", "",
             md_table(imp_tbl), "",
             f"### Gaps longer than {MAX_INTERP_GAP} days (rows inserted, values left missing)", "",
             f"{len(big)} gaps, {int(big['days'].sum())} dates in total.", "",
             md_table(big[["participant_id", "start", "end", "days"]]) if len(big) else "None", "",
             "## Final Dataset Summary", "",
             f"- Total participants: {out['participant_id'].nunique()}",
             f"- Total rows: {len(out)}",
             f"- Original observations: {int((~out['imputed']).sum())}",
             f"- Total inserted dates (absent from master): {int(st['inserted_rows'].sum())}",
             f"- Blank dates already in master (own_oura): {int(st['blank_rows'].sum())}",
             f"- Dates with interpolated values: {int(out['interpolated'].sum())} "
             f"({int(st['interpolated_values'].sum())} individual values)",
             f"- Dates left fully missing (gaps > {MAX_INTERP_GAP} days): {left_missing_rows}",
             f"- Rows with all three targets available: {int(out['has_targets'].sum())}",
             f"- Rows flagged interp_lookahead: {int(out['interp_lookahead'].sum())}",
             f"- Recommended modeling rows (original night, targets present, not interp_lookahead): {int(out['usable'].sum())}",
             f"- Rows with start > end in the source, start corrected by -24 h: {int(st['wrapped_fixed'].sum())}",
             f"- Remaining missing values (cells, across all output columns): {int(miss_by_col['missing'].sum())}", "",
             "Remaining missing values by column:", "", md_table(miss_by_col), "",
             "### Participants excluded from modeling recommendations", "",
             "Excluded if fewer than 14 nights or no recommended modeling row. Rows are kept in the file "
             "and marked `low_history_flag = 1`.", "",
             md_table(excluded[["participant_id", "number_of_nights", "usable_rows"]]
                      .rename(columns={"number_of_nights": "nights", "usable_rows": "recommended_rows"})), "",
             "### Recommended modeling rows per participant", "",
             md_table(st[["participant_id", "usable_rows"]]), "",
             "## Validation checks (all passed; the script stops otherwise)", "",
             md_table(val_tbl), ""]
    with open(a.out_report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(part_tbl.to_string(index=False))
    print(imp_tbl.to_string(index=False))
    print("rows", len(out), "| participants", out["participant_id"].nunique(), "| big gaps", len(big),
          "| fully-missing dates", left_missing_rows)
    print("recommended rows", int(out["usable"].sum()), "| lookahead", int(out["interp_lookahead"].sum()),
          "| excluded:", list(excluded["participant_id"]))
    for k, v in validation:
        print("-", k, ":", v)
    print("wrote", a.out_csv, a.out_report)


if __name__ == "__main__":
    main()
