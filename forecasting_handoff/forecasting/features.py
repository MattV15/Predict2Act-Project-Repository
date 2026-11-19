import numpy as np
import pandas as pd
PHYS=['hrv','hr','resp','temp']
PHYS_COLS=[p+'7' for p in PHYS]
SCHEDULE=['mid_sd7','mid_sd14','wake_sd7','wake_sd14','mid_drift','duration_mean7','duration_sd7','n7','n14']
def midpoint(bed,wake):
 return (bed+((wake-bed)%1440)/2)%1440

def dispersion(v):
 v=np.asarray(v,dtype=float);a=v*2*np.pi/1440;mu=np.angle(np.exp(1j*a).mean());d=np.angle(np.exp(1j*(a-mu)))*1440/(2*np.pi)
 return float(np.sqrt(np.mean(d*d)))

def circ_mean(v):
 return np.angle(np.exp(1j*np.asarray(v)*2*np.pi/1440).mean())*1440/(2*np.pi)

