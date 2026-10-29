"""Recompute the published application metrics from numeric test predictions.

This audits test-file evidence; it does not retrain models or prove training
isolation from test-only files. Uses the repository's NumPy/scikit-learn deps.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from sklearn.metrics import accuracy_score,f1_score

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'docs/results/application_scenarios_20261029/validation_manifest.json'

def classification(target,prediction,classes):
 y=np.asarray(target,dtype=np.int64);p=np.asarray(prediction,dtype=np.int64)
 if not (np.isin(y,range(classes)).all() and np.isin(p,range(classes)).all()):raise ValueError('Unknown class label')
 return dict(accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=list(range(classes)),average='macro',zero_division=0)),
             weighted_f1=float(f1_score(y,p,average='weighted',zero_division=0)))

def chsims(target,prediction):
 y=np.clip(np.asarray(target,dtype=np.float32),-1,1);p=np.clip(np.asarray(prediction,dtype=np.float32),-1,1)
 def bins(x,thresholds):return np.digitize(x,np.asarray(thresholds,dtype=x.dtype),right=True)
 return dict(MAE=float(np.abs(y-p).mean()),Corr=float(np.corrcoef(y.astype(np.float64),p.astype(np.float64))[0,1]),
  Acc2=float(accuracy_score(y>0,p>0)),Acc3=float(accuracy_score(bins(y,[-.1,.1]),bins(p,[-.1,.1]))),
  Acc5=float(accuracy_score(bins(y,[-.7,-.1,.1,.7]),bins(p,[-.7,-.1,.1,.7]))),
  binary_weighted_F1=float(f1_score(y>0,p>0,average='weighted',zero_division=0)),
  binary_macro_F1=float(f1_score(y>0,p>0,labels=[False,True],average='macro',zero_division=0)))

def mosei(target,prediction):
 y=np.asarray(target,dtype=np.float64);p=np.asarray(prediction,dtype=np.float64);nz=y!=0
 return dict(mmsa_mae=float(np.abs(y-p).mean()),mmsa_corr=float(np.corrcoef(y,p)[0,1]),
  mmsa_acc_7=float(np.mean(np.round(np.clip(y,-3,3))==np.round(np.clip(p,-3,3)))),
  mmsa_non0_acc_2=float(accuracy_score(y[nz]>0,p[nz]>0)),
  mmsa_non0_f1=float(f1_score(y[nz]>0,p[nz]>0,average='weighted',zero_division=0)))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--json-output',type=Path);args=parser.parse_args()
 manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));tolerance=manifest['metric_tolerance'];cache={}
 for entry in manifest['prediction_files']:
  path=(ROOT/entry['path']).resolve();path.relative_to(ROOT.resolve())
  # Git may convert CSV line endings between Windows and Linux checkouts.
  actual=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
  if actual!=entry['sha256']:raise AssertionError('Prediction file hash differs: '+entry['path'])
  with path.open(encoding='utf-8-sig',newline='') as f:cache[entry['path']]=list(csv.DictReader(f))
 checks=[]
 for case in manifest['cases']:
  groups=[]
  for name in case['files']:
   rows=cache[name]
   if case['group_by_seed']:
    seeds=sorted({r['seed'] for r in rows})
    if seeds!=['2026','2027','2028']:raise AssertionError('Unexpected seed coverage')
    groups.extend([[r for r in rows if r['seed']==seed] for seed in seeds])
   else:groups.append(rows)
  scores=[]
  for rows in groups:
   if len(rows)!=case['n_per_seed'] or len({r['sample_id'] for r in rows})!=len(rows):raise AssertionError('Wrong row/ID coverage: '+case['name'])
   target=[r['target'] for r in rows];pred=[r[case['prediction_column']] for r in rows]
   if case['metric_kind']=='classification':m=classification(target,pred,case['classes'])
   elif case['metric_kind']=='chsims_regression':m=chsims(target,pred)
   elif case['metric_kind']=='mosei_regression':m=mosei(target,pred)
   else:raise ValueError('Unknown metric convention')
   scores.append(m)
  errors={k:abs(float(np.mean([m[k] for m in scores]))-value) for k,value in case['expected_metrics'].items()}
  if not errors or max(errors.values())>tolerance:raise AssertionError((case['name'],errors))
  checks.append(dict(case=case['name'],groups=len(groups),rows_per_group=case['n_per_seed'],max_metric_error=max(errors.values())))
 result=dict(passed=True,prediction_files=len(cache),metric_cases=len(checks),checks=checks,
  scope='Test numeric evidence only; no retraining, raw-media validation, significance test or independent training-isolation claim.')
 if args.json_output:
  args.json_output.parent.mkdir(parents=True,exist_ok=True);args.json_output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print('Verified',len(cache),'prediction files and',len(checks),'metric cases. Maximum error:',max(r['max_metric_error'] for r in checks))

if __name__=='__main__':main()
