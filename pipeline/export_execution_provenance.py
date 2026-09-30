"""Export executed donor schedules from existing manifests and generation logs."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path


def lines(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def rel(p):return '/'.join(Path(str(p)).parts[-2:])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--research-root',type=Path,required=True);ap.add_argument('--source-data-root',type=Path,required=True);ap.add_argument('--local-home',type=Path,required=True);ap.add_argument('--artifact-root',type=Path,required=True);a=ap.parse_args()
    out=a.artifact_root/'data/provenance';records={};source_hashes={}
    d3={c['pair_id']:c for c in lines(a.source_data_root/'D3/pairs/vggface2/pairs_seed_42.jsonl')}
    for tool in ['facefusion','blendface','canonswap']:
        ff=tool=='facefusion';base=a.research_root/'cascade/outputs'
        mp=base/('chains_manifest_deep.jsonl' if ff else 'chains_manifest_deep_aux.jsonl')
        logs=sorted(base.glob('gen_deep_status_shard*.jsonl')) if ff else [base/f'gen_deep_aux_status_{tool}.jsonl']
        status={}
        for p in logs:
            source_hashes[str(p.relative_to(a.research_root))]=sha(p)
            for row in lines(p):
                key=row['chain_id' if ff else 'pair_id']
                if row['depth_reached']==25:status[key]=row
        rows=[];subs=0
        for c in lines(mp):
            key=c['chain_id' if ff else 'pair_id'];assert key in status
            if ff:donors=c['donor_images'].copy();target=c['t0_image']
            else:
                q=d3[key];donors=[q['donor_image'],q['donor2_image'],q['donor3_image']]+c['donor_images_deep'];target=c['target_image']
            substitutions=status[key]['donor_substitutions']
            for k,path in substitutions.items():donors[int(k)-1]=path;subs+=1
            rows.append({'chain_id':key,'target_image':rel(target),'donor_images':[rel(p) for p in donors],
                         'recorded_substitutions':{k:rel(v) for k,v in substitutions.items()},'depth':25})
        records[tool]={'chains':len(rows),'recorded_substitutions':subs}
        (out/f'executed_donors_{tool}.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'execution_log_hashes.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    versions={}
    for label,folder,checkpoint in [('facefusion','facefusion','.assets/models/hyperswap_1a_256.onnx'),('blendface','BlendFace','swapping/checkpoints/blendswap.pth'),('canonswap','CanonSwap','pretrained_weights/combined_weights.pth')]:
        repo=a.local_home/folder;p=repo/checkpoint
        rev=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(repo),'diff','HEAD','--name-only'],text=True).splitlines()
        patch=subprocess.check_output(['git','-C',str(repo),'diff','HEAD'],text=True).replace(str(a.local_home),'/opt/reproduction')
        patch_path=a.artifact_root/f'pipeline/source/{label}_current_checkout.patch'
        patch_path.parent.mkdir(parents=True,exist_ok=True)
        patch_path.write_text(patch)
        versions[label]={'current_checkout_commit':rev,'modified_tracked_files':dirty,'checkpoint':checkpoint,'checkpoint_sha256':sha(p),
                          'current_checkout_patch':str(patch_path.relative_to(a.artifact_root)),
                          'qualification':'Current local checkout fingerprint; not independently attested as the exact historical generation commit.'}
    (out/'model_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    # Preserve orchestration source, replacing private machine path defaults with a
    # documented neutral staging prefix. These adapters do not contain model code.
    dest=a.artifact_root/'pipeline/source';dest.mkdir(parents=True,exist_ok=True)
    sources={}
    roots=[(a.research_root,['cascade/*.py','fits/*.py','generality/fit_aux_operators.py']),
           (a.source_data_root.parent,['data/D1/swaps/*.py','data/D2/pairs/make_pairs_d2.py'])]
    for root,patterns in roots:
        for pattern in patterns:
            for p in sorted(root.glob(pattern)):
                path=dest/('research' if root==a.research_root else 'source-data')/p.relative_to(root)
                path.parent.mkdir(parents=True,exist_ok=True)
                text=p.read_text().replace(str(a.local_home),'/opt/reproduction')
                path.write_text(text)
                sources[str(path.relative_to(a.artifact_root))]={'original_sha256':sha(p),'change':'Machine path prefix replaced with /opt/reproduction; otherwise unchanged.'}
    (out/'orchestration_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(records)

if __name__=='__main__':main()
