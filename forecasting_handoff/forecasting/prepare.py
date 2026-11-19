"""Create the common experiment inputs from Oura exports or synthetic records."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .oura import load

FIELDS = ['source','day','bedtime','wake','duration','hrv','hr','resp','temp']


def demo():
    rng = np.random.default_rng(498)
    days = pd.date_range('2020-01-01', periods=210)
    records, splits = [], []
    for i in range(4):
        source = f'demo-{i+1}'
        shift = rng.normal(0, 35, len(days)) + 35*np.sin(np.arange(len(days))/13+i)
        records.append(pd.DataFrame(dict(
            source=source, day=days, bedtime=(1380+shift)%1440,
            wake=(420+shift+rng.normal(0,20,len(days)))%1440,
            duration=rng.normal(420,25,len(days)), hrv=rng.normal(40,4,len(days)),
            hr=rng.normal(60,2,len(days)), resp=rng.normal(15,1,len(days)),
            temp=rng.normal(0,.1,len(days)))))
        splits.append(dict(source=source,status='included',validation_start=days[106],
                           test_start=days[156],is_target=int(i<3)))
    return pd.concat(records),pd.DataFrame(splits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--demo',action='store_true')
    for name in ['sleep-dir','own-xlsx','combined-csv','splits']:
        parser.add_argument('--'+name,type=Path)
    args = parser.parse_args()
    if args.demo:
        nightly,splits = demo()
    else:
        if not all([args.sleep_dir,args.own_xlsx,args.combined_csv,args.splits]):
            parser.error('Study mode requires --sleep-dir, --own-xlsx, --combined-csv and --splits.')
        splits = pd.read_csv(args.splits)
        if 'is_target' not in splits:
            splits['is_target'] = splits.source.isin(['own_oura','sleep-2.csv','sleep-5.csv']).astype(int)
        data,_ = load(args.sleep_dir,args.own_xlsx,args.combined_csv)
        nightly = pd.concat([data[source].assign(source=source) for source in splits.source])
    args.output.mkdir(parents=True,exist_ok=True)
    nightly[FIELDS].to_csv(args.output/'nightly.csv',index=False)
    splits[['source','status','validation_start','test_start','is_target']].to_csv(args.output/'splits.csv',index=False)


if __name__ == '__main__':
    main()
