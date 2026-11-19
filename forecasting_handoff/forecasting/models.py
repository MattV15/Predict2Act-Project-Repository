import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from .features import midpoint,dispersion,circ_mean,SCHEDULE,PHYS_COLS

def estimators():
 return {
  'ridge_alpha10':lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=10)),
  'extra_trees':lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),ExtraTreesRegressor(n_estimators=300,min_samples_leaf=12,max_features=.7,max_depth=5,random_state=498,n_jobs=1)),
  'random_forest':lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),RandomForestRegressor(n_estimators=300,min_samples_leaf=12,max_features=.7,max_depth=5,random_state=498,n_jobs=1)),
  'hist_gradient_boost':lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=5,min_samples_leaf=25,l2_regularization=20,learning_rate=.05,random_state=498)),
  'rbf_svr':lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),SVR(kernel='rbf',C=1.,epsilon=.15,gamma='scale')),
  'rbf_kernel_ridge':lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),KernelRidge(alpha=10.,kernel='rbf',gamma=None)),
 }
def fitmodel(x,y,w,name):
 model=estimators()[name]();steps=list(model.named_steps);kw={steps[-1]+'__sample_weight':w} if 'sample_weight' in __import__('inspect').signature(model.steps[-1][1].fit).parameters else {}
 model.fit(x,y,**kw);return model

def normfit(x,y):
 center=x.median().fillna(0);scale=x.quantile(.75)-x.quantile(.25);scale=scale.where(scale>1e-6,x.std(ddof=0));scale=scale.where(scale>1e-6,1.).fillna(1.)
 base=float(y.median());sy=max(5.,float(y.quantile(.75)-y.quantile(.25)))
 return center,scale,base,sy

def srcweights(ids):
 s=pd.Series(ids);return (len(s)/(s.nunique()*s.map(s.value_counts()))).to_numpy()
