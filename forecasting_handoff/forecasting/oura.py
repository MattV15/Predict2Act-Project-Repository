from pathlib import Path
import hashlib
import numpy as np,pandas as pd
BASE=['bedtime','wake','duration'];PHYS=['hrv','hr','resp','temp']
def local_minute(x):
 try:
  t=pd.Timestamp(x)
  return np.nan if pd.isna(t) else t.hour*60+t.minute+t.second/60
 except (TypeError,ValueError):return np.nan

def clean(x):
 x=x.dropna(subset=['day']+BASE).copy();x=x[((x.bedtime>=1080)|(x.bedtime<360))&(x.duration>=180)]
 return x.sort_values(['day','duration'],ascending=[True,False]).drop_duplicates('day').sort_values('day').reset_index(drop=True)

def load(sleep_dir: Path, own_xlsx: Path, combined_csv: Path):
 clock='pacific'
 data={};audit=[];seen={}
 for f in sorted(sleep_dir.glob('*.csv')):
  z=pd.read_csv(f);trajectory=z[['date','bedtime_start_timestamp','bedtime_end_timestamp','total']].sort_values('date').to_csv(index=False).encode();key=hashlib.sha256(trajectory).hexdigest()
  if key in seen:
   audit.append(dict(source=f.name,duplicate_of=seen[key],rows=len(z),sha256=hashlib.sha256(f.read_bytes()).hexdigest()));continue
  seen[key]=f.name
  if clock=='pacific':
   s=pd.to_datetime(z.bedtime_start_timestamp,unit='ms',utc=True).dt.tz_convert('America/Los_Angeles').dt.tz_localize(None);e=pd.to_datetime(z.bedtime_end_timestamp,unit='ms',utc=True).dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)
  else:
   s=pd.to_datetime(z.date)+pd.to_timedelta(z.bedtime_start_midnight_delta,unit='s');e=pd.to_datetime(z.date)+pd.to_timedelta(z.bedtime_end_midnight_delta,unit='s')
  x=pd.DataFrame(dict(day=e.dt.normalize(),bedtime=s.map(local_minute),wake=e.map(local_minute),duration=pd.to_numeric(z.total,errors='coerce')/60,hrv=pd.to_numeric(z.rmssd,errors='coerce'),hr=pd.to_numeric(z.hr_average,errors='coerce'),resp=pd.to_numeric(z.breath_average,errors='coerce'),temp=pd.to_numeric(z.temperature_delta,errors='coerce')))
  data[f.name]=clean(x);audit.append(dict(source=f.name,rows=len(z),valid_nights=len(data[f.name]),duplicate_of='',sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 z=pd.read_excel(own_xlsx);z=z.rename(columns={z.columns[0]:'day'});z.day=pd.to_datetime(z.day)
 d=pd.read_csv(combined_csv,parse_dates=['day']);d=d[d.dataset.eq('oura')&d.observed.eq(True)&d.excluded_reason.isna()]
 d=d[['day','duration_min','hrv_rmssd','nrem_hr','resp_rate','temp_dev_c']].merge(z[['day','Bedtime Start','Bedtime End']],on='day',validate='one_to_one')
 x=pd.DataFrame(dict(day=d.day,bedtime=d['Bedtime Start'].map(local_minute),wake=d['Bedtime End'].map(local_minute),duration=d.duration_min,hrv=d.hrv_rmssd,hr=d.nrem_hr,resp=d.resp_rate,temp=d.temp_dev_c))
 data['own_oura']=clean(x);audit.append(dict(source='own_oura',rows=len(z),valid_nights=len(data['own_oura']),duplicate_of='',sha256=hashlib.sha256(own_xlsx.read_bytes()).hexdigest()))
 return data,pd.DataFrame(audit)

