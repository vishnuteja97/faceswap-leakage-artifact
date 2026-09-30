"""Export aggregate normal equations and evaluation statistics; no face embeddings.

Uses the existing train/test splits and original float32 Gram multiplication. The
final solve uses float64. FaceFusion identity-fold CV can be rerun from aggregates.
Auxiliary selection uses its actual seed-11 80/20 row holdout and saved per-candidate
cosine components; it is not five-fold identity CV. All original outputs are read-only.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def norm(x): return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def design(d,t): return np.concatenate([d,t,np.ones((len(d),1),d.dtype)],axis=1)
def gram(x,y): return x.T@x,x.T@y

def evaluate_stats(x,y):
    x,y=x.astype(float),y.astype(float)
    return dict(xx=x.T@x,xy=x.T@y,yy_diag=(y*y).sum(0),y_sum=y.sum(0),n=np.array(len(y)))

def solve(xx,xy,lam):
    m=xx.astype(float)+lam*np.eye(len(xx));m[-1,-1]-=lam
    return np.linalg.solve(m,xy.astype(float))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--research-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();root=a.research_root.resolve();a.output.mkdir(parents=True,exist_ok=True)
    for tool in ['facefusion','blendface','canonswap']:
        ff=tool=='facefusion'
        src=root.parent/'outputs/embeddings/facefusion_vggface2_train_metadata_embeddings.npz' if ff else root/f'generality/outputs/aux_embeddings/{tool}_vggface2_ffpairs_buffalo_l_raw.npz'
        z=np.load(src,allow_pickle=True)
        d,t,s=[z[k].astype(np.float32) for k in ['donor_embeddings','target_embeddings','swap_embeddings']]
        did=np.array([Path(p).parent.name for p in z['donor_paths']]);tid=np.array([Path(p).parent.name for p in z['target_paths']])
        if ff:
            ids=np.unique(np.r_[did,tid]);perm=np.random.default_rng(0).permutation(len(ids));ntr=round(.7*len(ids))
            train=set(ids[perm[:ntr]]);test=set(ids[perm[ntr:]])
            tr=np.flatnonzero([u in train and v in train for u,v in zip(did,tid)])
            te=np.flatnonzero([u in test and v in test for u,v in zip(did,tid)])
        else:
            tr=np.flatnonzero(z['splits']=='train');te=np.flatnonzero(z['splits']=='test')
        assert not (set(did[tr])|set(tid[tr]))&(set(did[te])|set(tid[te]))
        out=a.output/tool;out.mkdir(exist_ok=True)
        x=design(d[tr],t[tr]);y=s[tr];xt=design(d[te],t[te]);yt=s[te]
        gx,gy=gram(x,y)
        np.savez_compressed(out/'train.npz',xx=gx,xy=gy,n=np.array(len(tr)))
        ev=evaluate_stats(xt,yt)
        ev['variance_original_float32']=yt.var(0)
        ut=norm(t[te]);ev['target_unit_second_moment']=ut.astype(float).T@ut.astype(float)/len(ut)
        np.savez_compressed(out/'test.npz',**ev)
        op=root/('fits/outputs/operator_raw_vgg.npz' if ff else f'generality/outputs/aux_operators/{tool}_operator_raw.npz')
        oz=np.load(op,allow_pickle=True);lam=float(oz['lam']);w=solve(gx,gy,lam)
        pred=xt@w
        np.savez_compressed(out/'test_cosine_components.npz',dot=np.einsum('ij,ij->i',pred,norm(yt)),pred_norm=np.linalg.norm(pred,axis=1))
        max_err=max(float(np.max(np.abs(w[:512].T-oz['A']))),float(np.max(np.abs(w[512:1024].T-oz['B']))))
        cfg={'source_sha256':sha(src),'operator_source_sha256':sha(op),'n_train':len(tr),'n_test':len(te),
             'identity_disjoint':True,'selected_lambda':lam,'max_coefficient_error_vs_stored_float32':max_err,
             'training_gram_precision':'float32 multiplication, float64 solve; bias unregularized',
             'test_cosine_note':'Per-sample predicted dot unit-swap and prediction norm, exported for this recovered fitted solution. Aggregate test equations independently support R2; these scalar components support cosine for the pinned solution only.'}
        folds=np.full(len(tr),-1,dtype=int)
        if ff:
            ids=np.unique(np.r_[did[tr],tid[tr]]);perm=np.random.default_rng(1).permutation(len(ids));mapping=dict(zip(ids[perm],perm%5))
            folds=np.array([mapping[u] if mapping[u]==mapping[v] else -1 for u,v in zip(did[tr],tid[tr])])
            cfg['selection']='five identity folds, mean validation MSE; cross-fold triplets excluded in CV only'
            cfg['lambdas']=[.0001,.001,.01,.1,1.,10.]
            for f in range(5):
                mask=(folds!=f)&(folds!=-1);xx,xy=gram(x[mask],y[mask]);v=evaluate_stats(x[folds==f],y[folds==f])
                np.savez_compressed(out/f'cv_fold{f}.npz',train_xx=xx,train_xy=xy,**v)
        else:
            perm=np.random.default_rng(11).permutation(len(x));nv=max(1,round(.2*len(x)));va=perm[:nv];ct=perm[nv:]
            folds[va]=0;folds[ct]=1
            xx,xy=gram(x[ct],y[ct]);cv=evaluate_stats(x[va],y[va]);dots=[];norms=[]
            cfg['lambdas']=[.03,.1,.3,1.,3.,10.]
            cfg['selection']='one seed-11 80/20 row holdout, highest mean validation cosine; not identity-disjoint CV'
            for l in cfg['lambdas']:
                wp=solve(xx,xy,l);p=x[va]@wp;dots.append(np.einsum('ij,ij->i',p,norm(y[va])));norms.append(np.linalg.norm(p,axis=1))
            np.savez_compressed(out/'cv_holdout.npz',train_xx=xx,train_xy=xy,cosine_dot=np.array(dots),cosine_pred_norm=np.array(norms),**cv)
        with (out/'split_manifest.csv').open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(['source_row','donor_id','target_id','split','validation_fold'])
            for j,i in enumerate(tr):wr.writerow([i,did[i],tid[i],'train',folds[j]])
            for i in te:wr.writerow([i,did[i],tid[i],'test',''])
        (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        print(tool,len(tr),len(te),'max coefficient error',max_err,flush=True)


if __name__=='__main__':main()
