"""Compute face/context ROC from complete scores; compare reconstructed hard negatives."""
import csv,json
from pathlib import Path
import numpy as np
from metrics import auc,operating_points,quantile_tpr
DATA=Path(__file__).resolve().parent/'data'


def transplant_results():
    z=np.load(DATA/'transplant_scores.npz',allow_pickle=False);result={}
    for tool in ['facefusion','blendface','canonswap']:
        for k in [1,5,10,25,0]:
            conds=['full','face_transplant','foreign_face'] if k else ['noswap_full','noswap_transplant','noswap_foreign']
            for cond in conds:
                m=(z['tool']==tool)&(z['condition']==cond)&(z['passes']==k)
                pos=z['target'][m];neg=z['nonmember'][m].ravel();tpr,fpr=quantile_tpr(pos,neg,.01)
                result[f'{tool}|{cond}|pass{k}']={'n':len(pos),'auc':auc(pos,neg),'tpr_at_1pct_fpr':tpr,'achieved_fpr':fpr}
    return result


def hard_results():
    with (DATA/'hard_nonmember_scores.csv').open() as f:rows=list(csv.DictReader(f))
    result={}
    for tool in sorted({r['tool'] for r in rows}):
        selected=[r for r in rows if r['tool']==tool]
        p=np.array([float(r['target']) for r in selected]);n=np.array([float(r['hard_nonmember']) for r in selected])
        result[tool]={'n':len(p),'auc':auc(p,n),'tpr_at_1pct_fpr':operating_points(p,n)['tpr_1pct'],'mean_nonmember':float(n.mean())}
    return result


def main():
    values=transplant_results()
    print('RECOMPUTED: full / face-only / context-only AUC from ten individual negative scores per chain')
    for tool in ['facefusion','blendface','canonswap']:
        print('\n'+tool)
        for k in [1,5,10,25,0]:
            cond=['full','face_transplant','foreign_face'] if k else ['noswap_full','noswap_transplant','noswap_foreign']
            r=[values[f'{tool}|{c}|pass{k}'] for c in cond]
            print(('No swap' if not k else str(k)),*[f"{v['auc']:.3f}" for v in r],f"N={r[0]['n']}")
    print('\nRECONSTRUCTED harder negatives (not exact historical assignments); paper recorded values alongside')
    old=json.loads((DATA/'recorded/hard_nonmember_paper_protocol.json').read_text())
    for tool,r in hard_results().items():
        print(f"{tool:<12} AUC {r['auc']:.3f} vs paper {old[tool]['auc_hard']:.3f}; TPR@1% {100*r['tpr_at_1pct_fpr']:.1f}% vs paper {old[tool]['tpr_hard']:.1f}%")
    print('Reconstruction uses max over all stored candidate photos, within the retained MAAD ranking. 84 candidate groups are truncated by the historical top-1000 file.')

if __name__=='__main__':main()
