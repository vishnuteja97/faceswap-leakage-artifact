"""Recover complete face/context scores from existing images, without generating swaps.

Requires the local research manifests, original raw embedding store, source images,
existing cascade images, OpenCV, InsightFace and ONNX Runtime. All paths are CLI
arguments. Per-chain checkpoints allow interruption and restart. The public release
uses exported scores; it does not require these image/model inputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-12)


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def blend(src, dst):
    if src.shape != dst.shape:
        src = cv2.resize(src, (dst.shape[1], dst.shape[0]))
    h, w = dst.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (int(.50*w), int(.54*h)), (int(.30*w), int(.34*h)), 0, 0, 360, 1., -1)
    feather = max(3, int(21*w/512))
    alpha = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)[..., None]
    return (alpha*src.astype(np.float32)+(1-alpha)*dst.astype(np.float32)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--research-root', type=Path, required=True)
    ap.add_argument('--source-data-root', type=Path, required=True)
    ap.add_argument('--image-root', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--provider', choices=['CUDAExecutionProvider', 'CPUExecutionProvider'], default='CUDAExecutionProvider')
    ap.add_argument('--limit', type=int, default=0, help='Smoke-test limit per tool; control assignment still uses the full panel')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'script_sha256': sha(__file__), 'model_sha256': sha(args.model),
                'provider': args.provider, 'python': sys.version,
                'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'opencv-python', 'insightface', 'onnxruntime-gpu']},
                'protocol': {'passes': [1,5,10,25], 'controls': 10, 'resize': [112,112],
                             'ellipse': [.50,.54,.30,.34], 'feather_px_at_512': 21}}
    mp = args.output/'runtime.json'
    if mp.exists():
        old = json.loads(mp.read_text())
        assert old == metadata, 'Runtime or script changed; use a new output directory'
    else:
        mp.write_text(json.dumps(metadata, indent=2)+'\n')
    import onnxruntime as ort
    from insightface.model_zoo import get_model
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    model = get_model(str(args.model), providers=[args.provider, 'CPUExecutionProvider'] if args.provider != 'CPUExecutionProvider' else [args.provider], sess_options=options)
    model.prepare(ctx_id=0 if args.provider == 'CUDAExecutionProvider' else -1)
    if args.provider == 'CUDAExecutionProvider':
        assert 'CUDAExecutionProvider' in model.session.get_providers(), 'CUDA unavailable; refusing silent CPU fallback'
    print('Loading original gallery embeddings', flush=True)
    rawz = np.load(args.source_data_root/'D2/embeddings/vggface2_raw_embeddings.npz', allow_pickle=True)
    raw = rawz['embeddings']
    ranges = {str(u):(int(a),int(b)) for u,a,b in zip(rawz['unique_identity_ids'],rawz['identity_start_idx'],rawz['identity_end_idx'])}
    paths = rawz['paths']
    def image_path(p):
        return args.image_root/Path(p).parent.name/Path(p).name
    def read_image(p):
        v = cv2.imread(str(p))
        if v is None:
            raise RuntimeError(f'Cannot read input image: {p}')
        return v
    completed = 0
    start = time.monotonic()
    total = 2476 if not args.limit else min(args.limit,478)+2*min(args.limit,999)
    for tool in ['facefusion','blendface','canonswap']:
        ff = tool == 'facefusion'
        manifest = args.research_root/'cascade/outputs'/('chains_manifest_deep.jsonl' if ff else 'chains_manifest_deep_aux.jsonl')
        rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        n = len(rows)
        ids = [c['chain_id' if ff else 'pair_id'] for c in rows]
        target_ids = [c['t0_id' if ff else 'target_id'] for c in rows]
        galleries = []
        for c in rows:
            if ff:
                ix = np.array(c['t0_gallery_idx'])
            else:
                a,b = ranges[c['target_id']]
                ix = np.arange(a,b); ix = ix[ix != c['target_image_idx']]
            galleries.append(ix)
        g = [norm(raw[ix]) for ix in galleries]
        for i,c in enumerate(rows):
            if args.limit and i >= args.limit:
                break
            out = args.output/tool/(ids[i]+'.npz')
            if out.exists():
                with np.load(out, allow_pickle=False) as done:
                    assert done['target'].shape==(15,) and done['nonmember'].shape==(15,10)
                continue
            controls = [(i+1+j*(n//11))%n for j in range(10)]
            excluded = {target_ids[i]}|{target_ids[j] for j in controls}
            ctx_index = None
            for step in range(7,7+n):
                j = (i+step)%n
                if target_ids[j] not in excluded:
                    q = image_path(rows[j]['t0_image' if ff else 'target_image'])
                    ctx = cv2.imread(str(q))
                    if ctx is not None:
                        ctx_index = j; break
            assert ctx_index is not None
            target_path = image_path(c['t0_image' if ff else 'target_image'])
            target_img = read_image(target_path)
            conds, passes, pos, neg = [], [], [], []
            inputs = {str(target_path.relative_to(args.image_root)): sha(target_path),
                      str(q.relative_to(args.image_root)): sha(q)}
            def score(img, condition, k):
                e = unit(np.asarray(model.get_feat(cv2.resize(img,(112,112))),np.float32).reshape(-1))
                assert np.isfinite(e).all()
                conds.append(condition); passes.append(k)
                pos.append(float(np.median(g[i]@e)))
                neg.append([float(np.median(g[j]@e)) for j in controls])
            score(target_img,'noswap_full',0)
            score(blend(target_img,ctx),'noswap_transplant',0)
            score(blend(ctx,target_img),'noswap_foreign',0)
            for k in [1,5,10,25]:
                if ff:
                    p=args.research_root/'cascade/outputs/chains'/ids[i]/f's{k}.jpg'
                elif k==1:
                    p=args.source_data_root/f'D1/swaps/{tool}/vggface2_seed_42'/c['swap_basenames']['1']
                else:
                    p=args.research_root/'cascade/outputs/chains_deep_aux'/tool/ids[i]/f's{k}.jpg'
                release=read_image(p)
                inputs[f'{tool}/{ids[i]}/s{k}.jpg']=sha(p)
                score(release,'full',k)
                score(blend(release,ctx),'face_transplant',k)
                score(blend(ctx,release),'foreign_face',k)
            out.parent.mkdir(parents=True,exist_ok=True)
            temp=out.with_suffix('.tmp.npz')
            np.savez_compressed(temp,condition=np.array(conds),passes=np.array(passes),
                                target=np.array(pos),nonmember=np.array(neg),
                                chain_id=np.array(ids[i]),context_chain_id=np.array(ids[ctx_index]),
                                nonmember_chain_ids=np.array([ids[j] for j in controls]),
                                input_hashes_json=np.array(json.dumps(inputs,sort_keys=True)))
            temp.replace(out)
            completed+=1
            if completed%10==0:
                elapsed=time.monotonic()-start
                rate=completed/elapsed
                remaining=sum(1 for t in ['facefusion','blendface','canonswap'] for _ in (args.output/t).glob('*.npz'))
                print(f'{tool} {i+1}/{n}; {completed} new chains; {rate:.3f} chains/s; ETA {(total-remaining)/rate/60:.1f} min',flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
