from pathlib import Path
import argparse
import inspect
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .windows import GROUPS, make_expanded
from .models import normfit, srcweights, estimators
from .features import midpoint, dispersion, circ_mean

X = GROUPS['expanded_schedule']
Y = 'target_mid'
purge = pd.Timedelta(days=13)
cols7 = ['mid_sd7','wake_sd7','duration_mean7','duration_sd7','n7','duration_trend7',
         'mid_sin','mid_cos','bed_sin','bed_cos','wake_sin','wake_cos']
cols14 = [c for c in X if c not in cols7 and c != 'mid_drift']
REFERENCE_SEED = 498   # seed of the frozen ExtraTrees reference; --seeds adds 499, 500, ...


def add_recent_baseline(w, weeks=4, minimum=2):
    """CHANGE (baseline): median of the person's last `weeks` completed weekly labels.

    A weekly label is known at cutoff t only if its 7-day target window has ended,
    i.e. it belongs to cutoff t-7, t-14, ... . Needs `minimum` such labels; otherwise
    NaN, and evaluate() falls back to the personal median.
    """
    by_cutoff = w.set_index('cutoff')[Y]
    values = []
    for t in w.cutoff:
        past = [by_cutoff.get(t-pd.Timedelta(days=7*k)) for k in range(1, weeks+1)]
        past = [v for v in past if v is not None and np.isfinite(v)]
        values.append(float(np.median(past)) if len(past) >= minimum else np.nan)
    return w.assign(recent4=values)


def fit_kwargs(model, weights):
    """CHANGE: pass donor sample weights only to estimators that accept them.

    KNN has no sample_weight, so it is fitted unweighted (stated in the docs).
    """
    name, final = model.steps[-1]
    if 'sample_weight' in inspect.signature(final.fit).parameters:
        return {name+'__sample_weight': weights}
    return {}


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


def et(depth=5, leaf=12, seed=REFERENCE_SEED):
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                         ExtraTreesRegressor(n_estimators=300,min_samples_leaf=leaf,max_depth=depth,
                                             max_features=.7,random_state=seed,n_jobs=1))


def rf(seed=REFERENCE_SEED):
    """Same settings as models.estimators()['random_forest'], with a settable seed."""
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                         RandomForestRegressor(n_estimators=300,min_samples_leaf=12,max_features=.7,
                                               max_depth=5,random_state=seed,n_jobs=1))


def linear(model):
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                         StandardScaler(),model)


def evaluate(windows, cols, model_factory, name, section, inv, targets, original, pool='original',
             baseline=None, seed=REFERENCE_SEED):
    """Fit per held-out target and predict its validation and test windows.

    CHANGE: model_factory now takes the random seed as its one argument.
    Baselines (model_factory=None) are chosen with `baseline`: 'median' for the
    personal historical median, or a window column such as 'mid_sd7' / 'recent4'
    (missing values fall back to the personal median).
    """
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
            model=model_factory(seed)
            model.fit(pd.concat(a),pd.concat(y),**fit_kwargs(model,srcweights(ids)))
        for period, frame in [('validation',val),('test',test)]:
            if model_factory:
                pred=b+scale*model.predict((frame[cols]-c)/k)
            elif baseline=='median':
                pred=np.full(len(frame),b)
            else:
                col=frame[baseline].to_numpy(dtype=float)
                pred=np.where(np.isfinite(col),col,b)
            q=frame[['cutoff','target_end','future_n',Y]].copy()
            q=q.assign(source=s,period=period,section=section,method=name,seed=seed,
                       prediction=np.maximum(0,pred),donor_windows=len(ids),donors=len(set(ids)))
            rows.append(q)
    return pd.concat(rows,ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate next-week sleep timing forecasts")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", choices=["models", "ablation", "sensitivity", "all"], default="all")
    # CHANGE: repeat every random model with several seeds (498, 499, ...). Default 1 = original behaviour.
    parser.add_argument("--seeds", type=int, default=1,
                        help="number of random seeds for stochastic models (default 1 reproduces the report)")
    # CHANGE: a missing XGBoost used to be skipped silently; now it stops the run unless this flag is given.
    parser.add_argument("--skip-xgboost", action="store_true",
                        help="run without XGBoost if it is not installed")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1.")
    seeds = [REFERENCE_SEED+i for i in range(args.seeds)]
    warnings.filterwarnings("ignore", category=ConvergenceWarning)   # MLP/Lasso iteration-limit noise
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
    windows = {src:add_recent_baseline(make_expanded(raw[src])) for src in inv.index}
    xgb = None
    if args.study in ('models','all'):
        try:
            from xgboost import XGBRegressor as xgb
        except (ImportError, OSError) as exc:
            if not args.skip_xgboost:
                parser.error(f"XGBoost is not installed ({exc}). Run `pip install -r requirements.txt`, "
                             "or pass --skip-xgboost to compare without it.")
            print('XGBoost unavailable, skipped:',exc,flush=True)
    result=[]
    def run(section,name,factory=None,cols=X,data=windows,pool='original',baseline=None,stochastic=False):
        """stochastic=True: repeat over every --seeds seed; otherwise run once with the reference seed."""
        for seed in (seeds if stochastic else [REFERENCE_SEED]):
            result.append(evaluate(data,cols,factory,name,section,inv,targets,original,pool,baseline,seed))
        print(section,name,flush=True)

    # Baselines. CHANGE: readable names, plus the new recent 4-week median baseline.
    run('models','Personal median',baseline='median')
    run('models','Recent 4-week median',baseline='recent4')
    run('models','Reuse last 7-day score',baseline='mid_sd7')
    run('models','Reuse last 14-day score',baseline='mid_sd14')
    run("models", "ExtraTrees", lambda s: et(seed=s), stochastic=True)
    if args.study in ('models','all'):
        # CHANGE: method names are now "Family (setting)". forecasting.summarize picks one setting
        # per family by validation MAE. SVR, kernel ridge, KNN and MLP are new families.
        candidates = [
            ('Linear regression', lambda s: linear(LinearRegression()), False),
            *[(f'Ridge (alpha={a})', lambda s,a=a: linear(Ridge(alpha=a)), False) for a in [1,10,100]],
            *[(f'Lasso (alpha={a})', lambda s,a=a: linear(Lasso(alpha=a,max_iter=20000)), False) for a in [.01,.05,.1]],
            *[(f'SVR RBF (C={c})', lambda s,c=c: linear(SVR(kernel='rbf',C=c,epsilon=.15,gamma='scale')), False)
              for c in [.3,1,3]],
            *[(f'Kernel ridge RBF (alpha={a})', lambda s,a=a: linear(KernelRidge(alpha=a,kernel='rbf')), False)
              for a in [1,10,100]],
            *[(f'KNN (k={k})', lambda s,k=k: linear(KNeighborsRegressor(n_neighbors=k)), False) for k in [15,30,60]],
            # MLP needs heavy L2 (alpha) here: at alpha=1 it overfits ~1,500 donor windows and is
            # unstable across seeds; alpha 10-30 makes it competitive.
            *[(f'MLP (hidden={label}, alpha={a})',
               lambda s,h=h,a=a: linear(MLPRegressor(hidden_layer_sizes=h,alpha=a,max_iter=2000,random_state=s)), True)
              for label,h in [('16',(16,)),('32-16',(32,16))] for a in [1,10,30]],
            ('Random Forest', lambda s: rf(s), True),
            ('Histogram gradient boosting', lambda s: estimators()['hist_gradient_boost'](), False),
        ]
        if xgb is not None:
            candidates += [(f'XGBoost (depth={d})', lambda s,d=d: linear(
                xgb(n_estimators=200,max_depth=d,learning_rate=.05,n_jobs=1,random_state=s)), False) for d in [2,4]]
        for name,factory,stochastic in candidates:
            run('models',name,factory,stochastic=stochastic)
    if args.study in ('ablation','all'):
        for group in ['schedule','expanded_all']:
            run('ablation',group,lambda s: et(seed=s),GROUPS[group])
        run('ablation','Remove clock phase',lambda s: et(seed=s),[c for c in X if not c.endswith(('_sin','_cos'))])
        run('ablation','Remove duration',lambda s: et(seed=s),[c for c in X if not c.startswith('duration_')])
    if args.study in ('sensitivity','all'):
        for depth,leaf in [(3,12),(8,12),(5,6),(5,24)]:
            run('parameter',f'Depth {depth}, leaf {leaf}',lambda s,depth=depth,leaf=leaf: et(depth,leaf,s))
        for policy, cols in [('7',cols7),('14',cols14),('fill',X)]:
            data={src:flexible(raw[src],policy) for src in original}
            run('history' if policy!='fill' else 'missing',policy,lambda s: et(seed=s),cols,data)
        for pool in ['asof','wider']:
            run('pool',pool,lambda s: et(seed=s),pool=pool)
    p=pd.concat(result,ignore_index=True)
    assert not p.duplicated(['section','method','seed','source','period','cutoff']).any()
    assert (p.target_end-p.cutoff).dt.days.eq(7).all()
    args.output.mkdir(parents=True,exist_ok=True)
    p.to_csv(args.output/'predictions.csv',index=False)
    p['absolute_error'] = (p.prediction-p[Y]).abs()
    # CHANGE: metrics are computed per seed, then averaged; mae_sd is the spread across seeds
    # (blank when only one seed ran). With --seeds 1 every mae equals the original output.
    by_seed = p.groupby(['section','method','seed','period','source']).absolute_error.agg(mae='mean',windows='size')
    by_seed.to_csv(args.output/'metrics_by_seed.csv')
    by_seed.groupby(level=['section','method','period','source']).agg(
        mae=('mae','mean'),mae_sd=('mae','std'),windows=('windows','first')).to_csv(args.output/'metrics_by_target.csv')
    study = by_seed.groupby(level=['section','method','seed','period']).agg(mae=('mae','mean'),windows=('windows','sum'))
    study.groupby(level=['section','method','period']).agg(
        mae=('mae','mean'),mae_sd=('mae','std'),seeds=('mae','size'),windows=('windows','first')
        ).to_csv(args.output/'metrics.csv')
    reference = p.query("section=='models' and method=='ExtraTrees' and seed==@REFERENCE_SEED")
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
