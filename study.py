"""CPU-sized FI-2010 study. Discriminative forecasting, not a trading strategy."""
import argparse
import copy
import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from torch import nn

ROOT = Path(__file__).resolve().parent
URL = 'https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip'
WINDOW, STRIDE, PURGE = 20, 40, 100

def windows(a, start=0, stop=None):
    """First 40 rows are book features, first of last 5 rows is k=10 label."""
    stop = a.shape[1] if stop is None else stop
    ends = np.arange(start + WINDOW - 1, stop - 10, STRIDE)
    x = np.stack([a[:40, t-WINDOW+1:t+1].T for t in ends]).astype('float32')
    y = a[-5, ends].astype('int64') - 1
    assert np.isfinite(x).all() and np.isin(y, [0, 1, 2]).all()
    return x, y, ends

def softmax(z, temperature=1.):
    z = z / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)

def metrics(y, p):
    pred = p.argmax(1)
    conf, correct = p.max(1), pred == y
    ece = 0.
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=[0,1,2], average='macro', zero_division=0)),
            'nll': float(log_loss(y, p, labels=[0,1,2])),
            'brier': float(np.mean(np.sum((p-np.eye(3)[y])**2, axis=1))),
            'ece_10_bins': float(ece), 'class_counts': np.bincount(y, minlength=3).tolist(),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0,1,2]).tolist()}

def select_temperature(z, y):
    grid = np.geomspace(.25, 8., 81)
    return float(min(grid, key=lambda t: log_loss(y, softmax(z, t), labels=[0,1,2])))

def select_threshold(p, coverage=.5):
    """Validation-only confidence quantile. Test coverage is not forced."""
    return float(np.quantile(p.max(1), 1-coverage))

def selective(y, p, threshold):
    keep = p.max(1) >= threshold
    return {'threshold': threshold, 'coverage': float(keep.mean()),
            'n_retained': int(keep.sum()),
            'accuracy_retained': float((p.argmax(1)[keep] == y[keep]).mean()) if keep.any() else None}

class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(40,32,3,padding=1), nn.ReLU(),
                                 nn.Conv1d(32,32,3,padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32,3))
    def forward(self, x):
        return self.net(x.transpose(1,2))

def logits(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+512])).numpy() for i in range(0,len(x),512)])

def train(x, y, xv, yv, seed, epochs, balanced=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TemporalCNN()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    weights = torch.tensor(len(y)/(3*np.bincount(y,minlength=3)),dtype=torch.float32) if balanced else None
    best, state, history = float('inf'), None, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/256)))):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(torch.from_numpy(x[ids])), torch.from_numpy(y[ids]),weight=weights)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        val = log_loss(yv, softmax(logits(model,xv)), labels=[0,1,2])
        history.append({'epoch':epoch+1,'train_ce':float(np.mean(losses)),'validation_nll':float(val)})
        if val < best:
            best, state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model, history

def read_zip():
    path = ROOT / 'data/data.zip'
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('.txt') and '__MACOSX' not in n]
        def load(name):
            match = [n for n in names if Path(n).name == name]
            if len(match) != 1:
                raise ValueError(f'Expected unique {name}: {match}')
            with z.open(match[0]) as f:
                return np.loadtxt(f, dtype='float32')
        a = load('Train_Dst_NoAuction_DecPre_CF_7.txt')
        tests = {f'CF_{i}':load(f'Test_Dst_NoAuction_DecPre_CF_{i}.txt') for i in [7,8,9]}
    return path, a, tests

def main(epochs=12, balanced=False):
    begun = time.time()
    torch.set_num_threads(2)
    path, a, test_arrays = read_zip()
    # Disjoint training, checkpoint-selection and calibration regions, with purges.
    b1, b2 = int(a.shape[1]*.6), int(a.shape[1]*.8)
    x,y,ids = windows(a,0,b1-PURGE)
    xv,yv,iv = windows(a,b1+PURGE,b2-PURGE)
    xc,yc,ic = windows(a,b2+PURGE,a.shape[1]-PURGE)
    # Any additional scaling is estimated ONLY from training features.
    mu, sd = x.mean((0,1)), x.std((0,1)).clip(.000001)
    normalize = lambda t: np.ascontiguousarray(np.clip((t-mu)/sd,-10,10),dtype='float32')
    x,xv,xc = map(normalize,[x,xv,xc])
    tests = {key:(normalize(windows(v,PURGE,v.shape[1]-PURGE)[0]), windows(v,PURGE,v.shape[1]-PURGE)[1]) for key,v in test_arrays.items()}
    results = {'protocol':{'window':WINDOW,'stride':STRIDE,'horizon_events':10,'purge':PURGE,
               'epochs':epochs,'class_weighted':balanced,'seeds':[7,17,27],'split_columns':[b1,b2],
               'n_train':len(y),'n_validation':len(yv),'n_calibration':len(yc),
               'train_class_counts':np.bincount(y,minlength=3).tolist(),
               'source_train_shape':list(a.shape),'source_test_shapes':{k:list(v.shape) for k,v in test_arrays.items()},
               'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
               'environment':{'python':platform.python_version(),'torch':torch.__version__,
                              'numpy':np.__version__,'sklearn':sklearn.__version__},'baselines':{},'runs':[]}
    prior = np.bincount(y,minlength=3)/len(y)
    lr = LogisticRegression(max_iter=500,random_state=7).fit(x[:,-1],y)
    for key,(xt,yt) in tests.items():
        results['baselines'][key] = {'train_prior':metrics(yt,np.tile(prior,(len(yt),1))),
                                     'last_snapshot_logistic':metrics(yt,lr.predict_proba(xt[:,-1]))}
    out = ROOT / 'results' / ('weighted_followup' if balanced else 'initial')
    out.mkdir(parents=True,exist_ok=True)
    for seed in [7,17,27]:
        model, history = train(x,y,xv,yv,seed,epochs,balanced)
        zc = logits(model,xc)
        temperature = select_temperature(zc,yc)
        threshold = select_threshold(softmax(zc,temperature))
        run = {'seed':seed,'temperature':temperature,'history':history,'test':{}}
        for key,(xt,yt) in tests.items():
            z = logits(model,xt)
            p = softmax(z,temperature)
            # Controlled stress intervention, not observed market regime change.
            stressed = xt.copy()
            stressed[:,:,1::2] = 0.  # volume slots in price/volume book ordering
            ps = softmax(logits(model,stressed),temperature)
            run['test'][key] = {'raw':metrics(yt,softmax(z)), 'calibrated':metrics(yt,p),
                                'selective':selective(yt,p,threshold),'volume_mask_stress':metrics(yt,ps)}
            np.savez_compressed(out/f'predictions_seed{seed}_{key}.npz',labels=yt,
                                logits=z,probabilities=p,stressed_probabilities=ps)
        results['runs'].append(run)
        print('completed seed',seed, 'temperature',round(temperature,3),flush=True)
        (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    results['elapsed_seconds'] = time.time()-begun
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    lines = ['# Recorded results','', 'AI-assisted introductory study. All numbers computed by study.py.', '',
             '| Test file | Prior F1 | Logistic F1 | CNN F1, mean ± SD | Raw NLL | Calibrated NLL | Retained coverage | Retained accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in tests:
        rr = [r['test'][key] for r in results['runs']]
        avg = lambda section,field: np.mean([r[section][field] for r in rr])
        f = [r['calibrated']['macro_f1'] for r in rr]
        lines.append(f"| {key} | {results['baselines'][key]['train_prior']['macro_f1']:.3f} | {results['baselines'][key]['last_snapshot_logistic']['macro_f1']:.3f} | {np.mean(f):.3f} ± {np.std(f,ddof=1):.3f} | {avg('raw','nll'):.3f} | {avg('calibrated','nll'):.3f} | {avg('selective','coverage'):.3f} | {avg('selective','accuracy_retained'):.3f} |")
    lines += ['', 'SD is across initialization seeds, not a statistical confidence interval.',
              'Temperature and abstention threshold are fitted on calibration data only.',
              'File names are the source convention; these are not independently identified market regimes.',
              'See results.json for all metrics, confusion matrices, stress checks and training histories.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--balanced',action='store_true')
    args = parser.parse_args()
    main(args.epochs,args.balanced)
