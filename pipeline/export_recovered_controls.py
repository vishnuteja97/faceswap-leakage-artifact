"""Bundle complete recovered transplant scores and a labelled reconstructed hard control."""
import argparse,csv,json,sys,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import auc,quantile_tpr


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir',type=Path,required=True);ap.add_argument('--artifact-root',type=Path,required=True)
    ap.add_argument('--hard-dir',type=Path,required=True);a=ap.parse_args();data=a.artifact_root/'data'
    fields={k:[] for k in ['tool','chain_id','condition','passes','target','nonmember']};prov=[]
    for tool,n in [('facefusion',478),('blendface',999),('canonswap',999)]:
        files=sorted((a.checkpoint_dir/tool).glob('*.npz'));assert len(files)==n,(tool,len(files))
        for p in files:
            z=np.load(p,allow_pickle=False);assert len(z['target'])==15
            for i in range(15):
                for k in ['condition','passes','target','nonmember']:fields[k].append(z[k][i])
                fields['tool'].append(tool);fields['chain_id'].append(str(z['chain_id']))
            prov.append({'tool':tool,'chain_id':str(z['chain_id']),'context_chain_id':str(z['context_chain_id']),
                         'nonmember_chain_ids':z['nonmember_chain_ids'].tolist(),'input_hashes':json.loads(str(z['input_hashes_json']))})
    arrays={k:np.array(v) for k,v in fields.items()};assert arrays['nonmember'].shape==(37140,10)
    np.savez_compressed(data/'transplant_scores.npz',**arrays)
    (data/'provenance/transplant_recovery.json').write_text(json.dumps({'runtime':json.loads((a.checkpoint_dir/'runtime.json').read_text()),'chains':prov},indent=2)+'\n')
    saved=json.loads((data/'recorded/transplant_deep.json').read_text());computed={};max_auc=0
    for key,old in saved.items():
        tool,cond,k=key.split('|');mask=(arrays['tool']==tool)&(arrays['condition']==cond)&(arrays['passes']==int(k[4:]))
        pos=arrays['target'][mask];neg=arrays['nonmember'][mask].ravel();assert len(pos)==old['n']
        tpr,fpr=quantile_tpr(pos,neg,.01)
        new={'n':len(pos),'auc':auc(pos,neg),'tpr_at_1pct_fpr':tpr,'achieved_fpr':fpr,'median_target':float(np.median(pos)),'median_nonmember':float(np.median(neg))}
        computed[key]=new;max_auc=max(max_auc,abs(new['auc']-old['auc']))
        assert f"{new['auc']:.3f}"==f"{old['auc']:.3f}",(key,new,old)
    with (data/'transplant_per_chain_medians.csv').open() as f:oldrows=list(csv.DictReader(f))
    ix={(str(t),str(c),str(d),int(k)):i for i,(t,c,d,k) in enumerate(zip(arrays['tool'],arrays['chain_id'],arrays['condition'],arrays['passes']))}
    errors=[abs(arrays['target'][ix[(r['tool'],r['chain'],r['condition'],int(r['pass']))]]-float(r['target_sim'])) for r in oldrows]
    report={'rows':len(arrays['target']),'negative_scores':arrays['nonmember'].size,'auc_groups':len(computed),'max_auc_error':max_auc,'max_original_target_score_error':max(errors),'results':computed}
    (data/'recorded/transplant_recovery_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in ['hard_nonmember_scores.csv','hard_nonmember_selection.json','hard_nonmember_recovered_summary.json','protocol.json']:
        destination=(data/'provenance/hard_nonmember_protocol.json' if name=='protocol.json' else data/('provenance/'+name if name.endswith('selection.json') else name))
        shutil.copyfile(a.hard_dir/name,destination)
    print('Recovered transplant scores:',{k:v for k,v in report.items() if k!='results'})
    print('Harder-negative reconstruction is separately labelled; it does not exactly recover the lost historical assignments.')

if __name__=='__main__':main()
