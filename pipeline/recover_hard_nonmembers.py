"""Recover a fully specified harder-negative control from saved MAAD rankings.

For each target, use the highest MAAD-score candidate group in the saved rankings;
select the largest target-source cosine over all stored images per candidate
(or five equally spaced images with --selection-images five). Selection is
independent per target (negative identities may
repeat). Score all stored images of the selected identity. Results are compared
with the camera-ready summary, never adjusted to fit it. No recognition inference.
"""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import auc,operating_points


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def relative(p):return '/'.join(Path(str(p)).parts[-2:])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-data-root',type=Path,required=True)
    ap.add_argument('--artifact-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--selection-images',choices=['five','all'],default='all',help='All stored photos follows an uncapped maximum; five reproduces the older D2 candidate-sampling rule')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    z=np.load(a.source_data_root/'D2/embeddings/vggface2_raw_embeddings.npz',allow_pickle=True)
    raw=unit(z['embeddings'].astype(np.float32));paths=z['paths'];idx={relative(p):i for i,p in enumerate(paths)}
    ranges={str(u):(int(i),int(j)) for u,i,j in zip(z['unique_identity_ids'],z['identity_start_idx'],z['identity_end_idx'])}
    rankings={r['pair_id']:r for r in [json.loads(l) for l in (a.source_data_root/'D2/pairs/vggface2/maad_similarity_seed_42.jsonl').read_text().splitlines()]}
    pairs={r['pair_id']:r for r in [json.loads(l) for l in (a.source_data_root/'D1/pairs/vggface2/pairs_seed_42.jsonl').read_text().splitlines()]}
    with (a.artifact_root/'data/similarity_scores.csv').open() as f:
        original=[r for r in csv.DictReader(f) if r['embedding_model']=='buffalo_l' and r['status']=='ok']
    selected={};records=[]
    for pid in sorted({r['pair_id'] for r in original}):
        m=rankings[pid];p=pairs[pid];u=raw[idx[relative(p['target_image'])]]
        cand=[r['identity_id'] for r in m['top_matches'] if r['maad_score']==m['max_maad_score']]
        scored=[]
        for identity in cand:
            assert identity not in (p['target_id'],p['donor_id'])
            start,end=ranges[identity];sample=np.arange(start,end) if a.selection_images=='all' or end-start<=5 else np.linspace(start,end-1,5,dtype=np.int64)
            sims=raw[sample]@u;j=int(np.argmax(sims));scored.append((float(sims[j]),identity,int(sample[j])))
        best=max(scored,key=lambda q:q[0]);selected[pid]=best[1]
        records.append({'pair_id':pid,'target_image':relative(p['target_image']),'selected_identity':best[1],
                        'selected_anchor':relative(paths[best[2]]),'selected_cosine':best[0],
                        'maad_max_score':m['max_maad_score'],'n_candidates_stored':len(cand),
                        'n_candidates_original':m['n_identities_at_max_score'],
                        'candidate_ids':cand,'candidate_max_cosines':[s[0] for s in scored],
                        'gallery_images':[relative(p) for p in paths[slice(*ranges[best[1]])]]})
    result=[];summary={}
    for tool in sorted({r['tool'] for r in original}):
        cache=a.source_data_root/f'D1/analysis/d1sim/outputs/cache/swap_emb__vggface2__{tool}__buffalo_l.npz'
        zz=np.load(cache,allow_pickle=True);swap=unit(zz['matrix'].astype(np.float32));swapidx={Path(str(p)).name.split('__')[1]:i for i,p in enumerate(zz['paths'])}
        rr=[r for r in original if r['tool']==tool]
        for r in rr:
            pid=r['pair_id'];ident=selected[pid];lo,hi=ranges[ident]
            negative=float(np.median(raw[lo:hi]@swap[swapidx[pid]]))
            result.append({'tool':tool,'pair_id':pid,'target':float(r['target_sim_median']),
                           'random_nonmember':float(r['nonmember_sim_median']),'hard_nonmember':negative,'hard_identity':ident})
        q=[r for r in result if r['tool']==tool];pos=np.array([r['target'] for r in q]);neg=np.array([r['hard_nonmember'] for r in q])
        summary[tool]={'n':len(q),'auc_hard':auc(pos,neg),'tpr_hard':100*operating_points(pos,neg)['tpr_1pct'],'mean_nm_hard':float(neg.mean())}
        print(tool,summary[tool],flush=True)
    with (a.output/'hard_nonmember_scores.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(result[0]));wr.writeheader();wr.writerows(result)
    (a.output/'hard_nonmember_selection.json').write_text(json.dumps(records,indent=2)+'\n')
    (a.output/'hard_nonmember_recovered_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'protocol.json').write_text(json.dumps({'selection_images':a.selection_images,'negative_identity_reuse':'allowed','candidate_group':'stored maximum-MAAD-score tie group','n_truncated_groups':sum(r['n_candidates_stored']!=r['n_candidates_original'] for r in records),'source':'Recovered protocol; compare against recorded paper values before treating as exact'},indent=2)+'\n')


if __name__=='__main__':main()
