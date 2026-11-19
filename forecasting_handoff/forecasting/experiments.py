from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .windows import GROUPS, make_expanded
from .models import normfit, srcweights, estimators
from .features import midpoint, dispersion, circ_mean

X = GROUPS['expanded_schedule']
Y = 'target_mid'
purge = pd.Timedelta(days=13)
cols7 = ['mid_sd7','wake_sd7','duration_mean7','duration_sd7','n7','duration_trend7',
         'mid_sin','mid_cos','bed_sin','bed_cos','wake_sin','wake_cos']
cols14 = [c for c in X if c not in cols7 and c != 'mid_drift']


def phase(row, prefix, values):
    a = np.asarray(values)*2*np.pi/1440
    row[prefix+'_sin'], row[prefix+'_cos'] = np.sin(a).mean(), np.cos(a).mean()


def flexible(nights, policy):
    grid = nights.set_index('day').sort_index().asfreq('D')
    schedule = grid[['bedtime','wake','duration']]
    filled = schedule.ffill(limit=1) if policy == 'fill' else schedule
    rows = []
    for t in range(6 if policy == '7' else 13, len(grid)-7):
        future = schedule.iloc[t+1:t+8].dropna()
        if len(future)<5:
            continue
        row = dict(cutoff=grid.index[t], target_end=grid.index[t+7], future_n=len(future),
                   target_mid=dispersion(midpoint(future.bedtime,future.wake)))
        history = {}
        for length in ([7] if policy=='7' else [14] if policy=='14' else [7,14]):
            start = t-length+1
            observed = schedule.iloc[start:t+1].dropna()
            used = filled.iloc[start:t+1].dropna()
            minimum = 5 if length==7 else 10
            if len(used)<minimum or len(observed)<minimum-(1 if policy=='fill' else 0):
                break
            mid = midpoint(used.bedtime,used.wake)
            history[length] = mid
            row.update({f'n{length}':len(observed),f'mid_sd{length}':dispersion(mid),
                        f'wake_sd{length}':dispersion(used.wake),
                        f'duration_mean{length}':used.duration.mean(),
                        f'duration_sd{length}':used.duration.std(ddof=0)})
            if length==7:
                row['duration_trend7'] = np.polyfit(np.arange(len(used)),used.duration,1)[0]
            for prefix, values in [('mid',mid),('bed',used.bedtime),('wake',used.wake)]:
                phase(row,prefix if length==7 else prefix+'14',values)
        else:
            if policy=='fill':
                older = filled.iloc[t-13:t-6].dropna()
                row['mid_drift'] = ((circ_mean(history[7])-circ_mean(midpoint(older.bedtime,older.wake))+720)%1440-720
                                    if len(older)>=3 else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def split(w, s, inv):
    v, t = (pd.Timestamp(inv.loc[s,k]) for k in ['validation_start','test_start'])
    return (w[w.target_end<v-purge], w[(w.cutoff>=v)&(w.target_end<t-purge)], w[w.cutoff>=t])


def et(depth=5, leaf=12):
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                         ExtraTreesRegressor(n_estimators=300,min_samples_leaf=leaf,max_depth=depth,
                                             max_features=.7,random_state=498,n_jobs=1))


def linear(model):
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                         StandardScaler(),model)


def evaluate(windows, cols, model_factory, name, section, inv, targets, original, pool='original'):
    rows=[]
    for s in targets:
        fit,val,test=split(windows[s],s,inv)
        c,k,b,scale=normfit(fit[cols],fit[Y])
        a, y, ids=[],[],[]
        if model_factory:
            for other in (sorted(windows) if pool=='wider' else sorted(original)):
                if other==s:
                    continue
                donor=split(windows[other],other,inv)[0]
                if pool=='asof':
                    donor=donor[donor.target_end<val.cutoff.min()-purge]
                if len(donor)<25:
                    continue
                dc,dk,db,dy=normfit(donor[cols],donor[Y])
                a.append((donor[cols]-dc)/dk)
                y.append((donor[Y]-db)/dy)
                ids.extend([other]*len(donor))
            model=model_factory()
            model.fit(pd.concat(a),pd.concat(y),**{model.steps[-1][0]+'__sample_weight':srcweights(ids)})
        for period, frame in [('validation',val),('test',test)]:
            pred=(b+scale*model.predict((frame[cols]-c)/k) if model_factory else
                  frame[name].to_numpy() if name in ['mid_sd7','mid_sd14'] else np.full(len(frame),b))
            q=frame[['cutoff','target_end','future_n',Y]].copy()
            q=q.assign(source=s,period=period,section=section,method=name,prediction=np.maximum(0,pred),
                       donor_windows=len(ids),donors=len(set(ids)))
            rows.append(q)
    return pd.concat(rows,ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate next-week sleep timing forecasts")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", choices=["models", "ablation", "sensitivity", "all"], default="all")
    args = parser.parse_args()
    nightly = pd.read_csv(args.data/"nightly.csv", parse_dates=["day"], float_precision="round_trip")
    inv = pd.read_csv(args.data/"splits.csv").set_index("source")
    if nightly.duplicated(["source","day"]).any() or not inv.index.is_unique:
        parser.error("Nightly source/day keys and split source keys must be unique.")
    targets = inv.index[inv.is_target.eq(1)].tolist()
    original = inv.index[inv.status.eq("included")].tolist()
    if not targets or not set(targets) <= set(original):
        parser.error("Select at least one target in the included pool.")
    raw = dict(tuple(nightly.groupby("source")))
    if missing := set(inv.index)-set(raw):
        parser.error(f"Missing nightly records for: {sorted(missing)}")
    windows = {src:make_expanded(raw[src]) for src in inv.index}
    result=[]
    def run(section,name,factory=None,cols=X,data=windows,pool='original'):
        result.append(evaluate(data,cols,factory,name,section,inv,targets,original,pool))
        print(section,name,flush=True)

    for name in ['Personal median','mid_sd7','mid_sd14']:
        run('models',name)
    run("models", "ExtraTrees", et)
    if args.study in ('models','all'):
        for name,factory in [('Linear',lambda:linear(LinearRegression())),
                             *[(f'Ridge {a}',lambda a=a:linear(Ridge(alpha=a))) for a in [1,10,100]],
                             *[(f'Lasso {a}',lambda a=a:linear(Lasso(alpha=a,max_iter=20000))) for a in [.01,.05,.1]],
                             ('Random Forest',estimators()['random_forest']),
                             ('Histogram gradient boosting',estimators()['hist_gradient_boost'])]:
            run('models',name,factory)
        try:
            from xgboost import XGBRegressor
        except (ImportError, OSError) as exc:
            print('XGBoost unavailable:',exc,flush=True)
        else:
            for depth in [2,4]:
                run('models', f'XGBoost depth {depth}',lambda depth=depth:linear(
                    XGBRegressor(n_estimators=200,max_depth=depth,learning_rate=.05,n_jobs=1,random_state=498)))
    if args.study in ('ablation','all'):
        for group in ['schedule','expanded_all']:
            run('ablation',group,et,GROUPS[group])
        run('ablation','Remove clock phase',et,[c for c in X if not c.endswith(('_sin','_cos'))])
        run('ablation','Remove duration',et,[c for c in X if not c.startswith('duration_')])
    if args.study in ('sensitivity','all'):
        for depth,leaf in [(3,12),(8,12),(5,6),(5,24)]:
            run('parameter',f'Depth {depth}, leaf {leaf}',lambda:et(depth,leaf))
        for policy, cols in [('7',cols7),('14',cols14),('fill',X)]:
            data={src:flexible(raw[src],policy) for src in original}
            run('history' if policy!='fill' else 'missing',policy,et,cols,data)
        for pool in ['asof','wider']:
            run('pool',pool,et,pool=pool)
    p=pd.concat(result,ignore_index=True)
    assert not p.duplicated(['section','method','source','period','cutoff']).any()
    assert (p.target_end-p.cutoff).dt.days.eq(7).all()
    args.output.mkdir(parents=True,exist_ok=True)
    p.to_csv(args.output/'predictions.csv',index=False)
    p['absolute_error'] = (p.prediction-p[Y]).abs()
    metrics = p.groupby(['section','method','period','source']).absolute_error.agg(mae='mean',windows='size')
    metrics.to_csv(args.output/'metrics_by_target.csv')
    metrics.groupby(level=[0,1,2]).agg(mae=('mae','mean'),windows=('windows','sum')).to_csv(args.output/'metrics.csv')
    reference = p.query("section=='models' and method=='ExtraTrees'")
    keys = ['source','period','cutoff','target_end']
    paired = p.merge(reference[keys+[Y,'absolute_error']], on=keys, suffixes=('','_reference'),validate='many_to_one')
    if not np.allclose(paired[Y],paired[Y+'_reference']):
        raise ValueError("Forecast variants disagree on observed labels.")
    paired.groupby(['section','method','period','source']).agg(
        mae=('absolute_error','mean'), reference_mae=('absolute_error_reference','mean'),
        windows=('absolute_error','size')).groupby(level=[0,1,2]).agg(
            mae=('mae','mean'),reference_mae=('reference_mae','mean'),windows=('windows','sum')
        ).to_csv(args.output/'matched_metrics.csv')


if __name__=='__main__':
    main()
