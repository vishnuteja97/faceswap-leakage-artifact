"""Refit A, B, c from released aggregate normal equations and evaluate R2/b_u.

No input embedding vectors or pre-fitted matrices enter the solve. Stored operators
are used only for comparison. FaceFusion CV recomputes fold MSE; auxiliary CV
selects from released per-candidate scalar cosine components, a narrower check.
"""
import argparse,json
from pathlib import Path
import numpy as np

DATA=Path(__file__).resolve().parent/'data'
TOOLS=['facefusion','blendface','canonswap']

def solve(xx,xy,lam):
    m=xx.astype(float)+lam*np.eye(len(xx));m[-1,-1]-=lam
    return np.linalg.solve(m,xy.astype(float))

def squared_errors(w,z):
    return np.einsum('ij,ij->j',w,z['xx']@w)-2*np.einsum('ij,ij->j',w,z['xy'])+z['yy_diag']

def run(tool,cv=False):
    base=DATA/'operator_fit'/tool;cfg=json.loads((base/'config.json').read_text())
    lam=cfg['selected_lambda'];cv_scores={}
    if cv:
        if tool=='facefusion':
            accum={l:[] for l in cfg['lambdas']}
            for f in range(5):
                with np.load(base/f'cv_fold{f}.npz',allow_pickle=False) as z:
                    for l in cfg['lambdas']:
                        w=solve(z['train_xx'],z['train_xy'],l)
                        accum[l].append(float(squared_errors(w,z).mean()/int(z['n'])))
            cv_scores={str(l):float(np.mean(v)) for l,v in accum.items()}
            chosen=float(min(cv_scores,key=cv_scores.get))
        else:
            with np.load(base/'cv_holdout.npz',allow_pickle=False) as z:
                cv_scores={str(l):float(np.mean(z['cosine_dot'][i]/z['cosine_pred_norm'][i])) for i,l in enumerate(cfg['lambdas'])}
            chosen=float(max(cv_scores,key=cv_scores.get))
        assert np.isclose(chosen,lam,atol=1e-7,rtol=0),(tool,chosen,lam)
    with np.load(base/'train.npz',allow_pickle=False) as z:
        w=solve(z['xx'],z['xy'],lam)
    with np.load(base/'test.npz',allow_pickle=False) as z:
        mse=squared_errors(w,z)/int(z['n'])
        r2=float(np.mean(1-mse/np.maximum(z['variance_original_float32'],1e-12)))
        B=w[512:1024].T;bu=float(np.sum(B*z['target_unit_second_moment'].T))
    with np.load(base/'test_cosine_components.npz',allow_pickle=False) as z:
        cos=float(np.mean(z['dot']/z['pred_norm']))
    original=np.load(DATA/f'operator_{tool}.npz',allow_pickle=False)
    error=float(np.max(np.abs(B-original['B'])))
    result={'n_train':cfg['n_train'],'n_test':cfg['n_test'],'lambda':lam,'R2':r2,'b_u':bu,
            'rho_B':float(np.abs(np.linalg.eigvals(B)).max()),'sigma_max_B':float(np.linalg.svd(B,compute_uv=False)[0]),
            'fit_cosine_from_scalar_components':cos,'max_B_error_vs_original':error,'selection':cfg['selection'],'cv_scores':cv_scores}
    return result,w

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cv',action='store_true');ap.add_argument('--output',type=Path)
    a=ap.parse_args();results={}
    for tool in TOOLS:
        r,w=run(tool,a.cv);results[tool]=r;print(tool,json.dumps(r),flush=True)
        if a.output:
            a.output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(a.output/f'{tool}.npz',A=w[:512].T,B=w[512:1024].T,c=w[-1])
    if a.output:(a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('R2 and b_u are recomputed from aggregate test statistics. Cosine uses exported per-sample scalar components for the recovered fitted solution. See WALKTHROUGH.md for the limits.')

if __name__=='__main__':main()
