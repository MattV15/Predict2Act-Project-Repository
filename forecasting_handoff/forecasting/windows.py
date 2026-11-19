import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from .features import midpoint,dispersion,circ_mean,SCHEDULE,PHYS_COLS

def make_expanded(nights):
 g=nights.set_index('day').sort_index().asfreq('D').copy()
 g['mid']=midpoint(g.bedtime,g.wake);rows=[]
 for t in range(13,len(g)-7):
  h7=g.iloc[t-6:t+1];h14=g.iloc[t-13:t+1];f=g.iloc[t+1:t+8]
  a=h7.dropna(subset=['mid','wake','duration']);b=h14.dropna(subset=['mid','wake','duration']);future=f.dropna(subset=['mid','wake','duration'])
  if len(a)<5 or len(b)<10 or len(future)<5:continue
  row=dict(cutoff=g.index[t],target_end=g.index[t+7],future_n=len(future),target_mid=dispersion(future.mid),target_wake=dispersion(future.wake))
  # Original common timing features.
  row.update(mid_sd7=dispersion(a.mid),mid_sd14=dispersion(b.mid),wake_sd7=dispersion(a.wake),wake_sd14=dispersion(b.wake))
  prev=g.iloc[t-13:t-6].dropna(subset=['mid']);row['mid_drift']=((circ_mean(a.mid)-circ_mean(prev.mid)+720)%1440-720) if len(prev)>=3 else np.nan
  row.update(duration_mean7=a.duration.mean(),duration_sd7=a.duration.std(ddof=0),n7=len(a),n14=len(b))
  # Additional schedule phase features, represented as circular coordinates.
  for name,z in [('mid',a.mid),('mid14',b.mid),('bed',a.bedtime.dropna()),('bed14',b.bedtime.dropna()),('wake',a.wake),('wake14',b.wake)]:
   if name in ['mid','wake']: z=z.dropna()
   if len(z):
    ang=np.asarray(z)*2*np.pi/1440;row[name+'_sin']=float(np.sin(ang).mean());row[name+'_cos']=float(np.cos(ang).mean())
   else:row[name+'_sin']=np.nan;row[name+'_cos']=np.nan
  row['duration_mean14']=b.duration.mean();row['duration_sd14']=b.duration.std(ddof=0);row['duration_trend7']=np.polyfit(np.arange(len(a)),a.duration,1)[0] if len(a)>=4 else np.nan
  # Common physiology: level, variability, trend, latest value, and recent-vs-two-week shift.
  for ph in ['hrv','hr','resp','temp']:
   x7=pd.to_numeric(h7[ph],errors='coerce').dropna();x14=pd.to_numeric(h14[ph],errors='coerce').dropna()
   row[ph+'7']=float(x7.median()) if len(x7)>=3 else np.nan
   row[ph+'14']=float(x14.median()) if len(x14)>=5 else np.nan
   row[ph+'_sd7']=float(x7.std(ddof=0)) if len(x7)>=3 else np.nan
   row[ph+'_trend7']=float(np.polyfit(np.arange(len(x7)),x7,1)[0]) if len(x7)>=4 else np.nan
   row[ph+'_last']=float(x7.iloc[-1]) if len(x7)>=1 else np.nan
   row[ph+'_change']=row[ph+'7']-row[ph+'14'] if np.isfinite(row[ph+'7']) and np.isfinite(row[ph+'14']) else np.nan
   row[ph+'_n7']=len(x7);row[ph+'_n14']=len(x14)
  rows.append(row)
 return pd.DataFrame(rows)

SCHED_EXTRA=['mid_sin','mid_cos','mid14_sin','mid14_cos','bed_sin','bed_cos','bed14_sin','bed14_cos','wake_sin','wake_cos','wake14_sin','wake14_cos','duration_mean14','duration_sd14','duration_trend7']
PHYS_EXTRA=[f'{p}{x}' for p in ['hrv','hr','resp','temp'] for x in ['14','_sd7','_trend7','_last','_change','_n7','_n14']]
GROUPS={'schedule':SCHEDULE,'expanded_schedule':SCHEDULE+SCHED_EXTRA,'expanded_all':SCHEDULE+SCHED_EXTRA+PHYS_COLS+PHYS_EXTRA}

