"""Independently recompute F1 and verify probability artifacts for both studies."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

def main():
    root=Path(__file__).resolve().parent/'results'
    count=0
    for name in ['initial','weighted_followup']:
        d=json.loads((root/name/'results.json').read_text())
        assert len(d['runs'])==3
        for run in d['runs']:
            for key,metrics in run['test'].items():
                p=np.load(root/name/f"predictions_seed{run['seed']}_{key}.npz")
                assert np.isfinite(p['probabilities']).all()
                np.testing.assert_allclose(p['probabilities'].sum(1),1,atol=1e-6)
                f=f1_score(p['labels'],p['probabilities'].argmax(1),labels=[0,1,2],average='macro',zero_division=0)
                assert abs(f-metrics['calibrated']['macro_f1'])<1e-10
                assert abs(f-metrics['raw']['macro_f1'])<1e-10
                count+=1
    print('Audited',count,'saved prediction sets')
if __name__=='__main__': main()
