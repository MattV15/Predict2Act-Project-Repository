"""Turn an experiments output folder into report-ready tables.

NEW FILE. Usage, after running forecasting.experiments:

    python -m forecasting.summarize --run runs/models-initials

Writes summary_models.csv / .md and summary_other.md into the run folder and prints them.

Selection rule (the important part): within each model family, the setting with the
lowest *validation* MAE is selected, and its *test* MAE is reported. Test MAE is never
used to choose anything.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASELINES = ['Personal median', 'Recent 4-week median', 'Reuse last 7-day score', 'Reuse last 14-day score']
REFERENCE = 'Personal median'   # the bar a learned model has to beat


def family(method):
    """'Ridge (alpha=10)' -> 'Ridge'. Methods without a setting are their own family."""
    return method.split(' (')[0]


def model_table(metrics, by_target):
    m = metrics[metrics.section.eq('models')]
    mae = m.pivot_table(index='method', columns='period', values='mae')
    sd = m[m.period.eq('test')].set_index('method').mae_sd
    seeds = m[m.period.eq('test')].set_index('method').seeds
    rows = []
    for fam, g in mae.groupby(family, sort=False):
        best = g.validation.idxmin()            # chosen on validation only
        rows.append(dict(family=fam, selected=best, settings_tried=len(g),
                         validation_mae=g.loc[best, 'validation'], test_mae=g.loc[best, 'test'],
                         test_sd_across_seeds=sd.get(best), seeds=seeds.get(best),
                         is_baseline=fam in BASELINES))
    table = pd.DataFrame(rows)
    ref = mae.loc[REFERENCE, 'test']
    table['test_gain_vs_personal_median_pct'] = 100*(ref-table.test_mae)/ref
    per_target = (by_target[by_target.section.eq('models') & by_target.period.eq('test')]
                  .pivot_table(index='method', columns='source', values='mae').add_prefix('test_mae_'))
    table = table.join(per_target, on='selected')
    return table.sort_values(['is_baseline', 'validation_mae'], ascending=[False, True]).reset_index(drop=True)


def other_table(matched, metrics):
    """Ablation / sensitivity / robustness rows vs. the frozen ExtraTrees reference on matched dates."""
    rows = matched[matched.section.ne('models') & matched.period.eq('test')].copy()
    rows['delta_vs_extratrees'] = rows.mae-rows.reference_mae
    all_dates = metrics[metrics.period.eq('test')].set_index(['section', 'method'])
    rows = rows.join(all_dates[['mae', 'windows']].rename(columns={'mae': 'all_date_mae', 'windows': 'all_date_windows'}),
                     on=['section', 'method'])
    return rows[['section', 'method', 'windows', 'mae', 'reference_mae', 'delta_vs_extratrees',
                 'all_date_windows', 'all_date_mae']].rename(columns={'windows': 'matched_windows',
                                                                      'mae': 'matched_mae'})


def markdown(df, digits=2):
    """Small Markdown table writer (avoids needing the optional `tabulate` package)."""
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return 'yes' if v else ''
        if isinstance(v, (float, np.floating)):
            return '' if np.isnan(v) else f'{v:.{digits}f}'
        return str(v)
    lines = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|'*len(df.columns)]
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, required=True, help='output folder of forecasting.experiments')
    args = parser.parse_args()
    metrics = pd.read_csv(args.run/'metrics.csv')
    by_target = pd.read_csv(args.run/'metrics_by_target.csv')
    matched = pd.read_csv(args.run/'matched_metrics.csv')
    if 'mae_sd' not in metrics:   # output from the pre-seed version of experiments.py
        metrics = metrics.assign(mae_sd=np.nan, seeds=1)

    models = model_table(metrics, by_target)
    models.to_csv(args.run/'summary_models.csv', index=False)
    shown = models.drop(columns=['is_baseline'])
    text = ('## Model comparison (setting chosen on validation MAE, test MAE reported; minutes)\n\n'
            + markdown(shown) + '\n')
    (args.run/'summary_models.md').write_text(text, encoding='utf-8')
    print(text)

    other = other_table(matched, metrics)
    if len(other):
        text = ('## Ablation, sensitivity and robustness vs. frozen ExtraTrees (test, minutes)\n\n'
                'Negative delta = better than the reference on the same dates.\n\n' + markdown(other) + '\n')
        (args.run/'summary_other.md').write_text(text, encoding='utf-8')
        print(text)


if __name__ == '__main__':
    main()
