#!/usr/bin/env python3
r"""Does replacing the preserved context drive the deep cascade to chance?

The \Kdeep-pass cascades leave the identity-inference attack measurably above chance
(AUC 0.55-0.60). A cascade re-synthesizes only the INNER FACE at every pass; the hair,
face contour, forehead, ears, neck and background of the RELEASED image are still those
of the original target photograph, copied forward unchanged through all 25 passes. So the
residual signal has two candidate homes, and dilution can only have attacked one of them.

This experiment separates them at depth, with the attack exactly as in the paper (whole
aligned crop, natural unmasked galleries, median cosine, target vs pooled non-members):

  full             the pass-k release, unmodified                    (baseline)
  face_transplant  the release's inner-face ellipse blended into an
                   UNRELATED person's photo: face channel only, the
                   target's hair/contour/background all absent
  foreign_face     an unrelated person's inner face blended into the
                   release: context channel only, synthesized face absent

Plus two blend-fidelity controls on unswapped target photos, which bound how much identity
the transplant procedure itself adds or destroys:

  noswap_full        the target's original photo                     (upper anchor)
  noswap_transplant  the target's OWN face into an unrelated photo
  noswap_foreign     an unrelated face into the target's own photo

If face_transplant reaches chance while full does not, the residual leakage at depth is
carried by the preserved context, and the actionable fix is to replace the context rather
than to keep swapping. The blend uses the same fixed feathered ellipse as the single-swap
control (rebuttal/context_decouple.py); crops are pre-aligned so geometry transfers.

Run in the facefusion env:
  python cascade/transplant_deep.py --tools facefusion blendface canonswap --passes 1 5 25
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ART = Path("/opt/reproduction/S&P2027-Artifact/data")
RAW_NPZ = ART / "D2/embeddings/vggface2_raw_embeddings.npz"
MANIFEST_FF = OUT / "chains_manifest_deep.jsonl"
MANIFEST_AUX = OUT / "chains_manifest_deep_aux.jsonl"
CHAINS_FF = OUT / "chains"
CHAINS_AUX = OUT / "chains_deep_aux"

# fixed inner-face ellipse for aligned 512x512 crops, identical to the single-swap control
FACE_CX, FACE_CY, FACE_AX, FACE_AY = 0.50, 0.54, 0.30, 0.34
FEATHER_PX = 21
N_NONMEM = 10
CONDITIONS = ["full", "face_transplant", "foreign_face"]


def l2(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def unit(x):
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-12 else None


_ALPHAS: dict[tuple, np.ndarray] = {}


def alpha_ellipse(shape):
    h, w = shape[:2]
    if (h, w) not in _ALPHAS:
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (int(FACE_CX * w), int(FACE_CY * h)),
                    (int(FACE_AX * w), int(FACE_AY * h)), 0, 0, 360, 1.0, -1)
        f = max(3, int(FEATHER_PX * w / 512))
        m = cv2.GaussianBlur(m, (f * 2 + 1, f * 2 + 1), 0)
        _ALPHAS[(h, w)] = m[..., None]
    return _ALPHAS[(h, w)]


def blend_face(src, dst):
    """Paste src's inner-face ellipse onto dst with a feathered boundary."""
    if src.shape != dst.shape:
        src = cv2.resize(src, (dst.shape[1], dst.shape[0]))
    a = alpha_ellipse(dst.shape)
    return (a * src.astype(np.float32) + (1 - a) * dst.astype(np.float32)).astype(np.uint8)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    r = np.empty(len(allv), float)
    r[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at_fpr(pos, neg, fpr):
    neg = np.asarray(neg, float)
    if len(neg) == 0:
        return float("nan")
    return float(np.mean(np.asarray(pos, float) >= np.quantile(neg, 1.0 - fpr)))


def load_chains(tool: str, raw_meta: dict):
    """Return list of dicts: id, target_id, target_image, gallery_idx, pass_path(k)."""
    uid_range = raw_meta["uid_range"]
    if tool == "facefusion":
        rows = [json.loads(l) for l in MANIFEST_FF.read_text().splitlines() if l.strip()]
        return [{
            "id": c["chain_id"],
            "target_id": str(c["t0_id"]),
            "target_image": c["t0_image"],
            "gallery_idx": np.asarray(c["t0_gallery_idx"], np.int64),
            "path": (lambda cid: (lambda k: CHAINS_FF / cid / f"s{k}.jpg"))(c["chain_id"]),
        } for c in rows]

    rows = [json.loads(l) for l in MANIFEST_AUX.read_text().splitlines() if l.strip()]
    out = []
    for c in rows:
        a, b = uid_range[str(c["target_id"])]
        g = np.arange(a, b)
        g = g[g != int(c["target_image_idx"])]

        def mk(pid=c["pair_id"], bn=c["swap_basenames"], t=tool):
            def path(k):
                if k == 1:
                    return ART / f"D1/swaps/{t}/vggface2_seed_42" / bn["1"]
                if k == 2:
                    return ART / f"D3/swaps/{t}/swap2/vggface2_seed_42" / bn["2"]
                if k == 3:
                    return ART / f"D3/swaps/{t}/swap3/vggface2_seed_42" / bn["3"]
                return CHAINS_AUX / t / pid / f"s{k}.jpg"
            return path

        out.append({"id": c["pair_id"], "target_id": str(c["target_id"]),
                    "target_image": c["target_image"], "gallery_idx": g, "path": mk()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="+", default=["facefusion", "blendface", "canonswap"])
    ap.add_argument("--passes", nargs="+", type=int, default=[1, 5, 25])
    ap.add_argument("--max-chains", type=int, default=None)
    ap.add_argument("--dump-examples", type=int, default=3)
    args = ap.parse_args()

    z = np.load(RAW_NPZ, allow_pickle=True)
    raw = np.asarray(z["embeddings"], np.float32)
    uids = np.asarray(z["unique_identity_ids"], dtype=object)
    s_idx = np.asarray(z["identity_start_idx"], np.int64)
    e_idx = np.asarray(z["identity_end_idx"], np.int64)
    raw_meta = {"uid_range": {str(u): (int(a), int(b))
                              for u, a, b in zip(uids, s_idx, e_idx)}}

    from insightface.model_zoo import get_model
    rec = get_model("buffalo_l")
    try:
        rec.prepare(ctx_id=0)
    except Exception:
        rec.prepare(ctx_id=-1)

    def embed(img):
        return unit(np.asarray(rec.get_feat(cv2.resize(img, (112, 112))),
                               np.float32).reshape(-1))

    rows = []
    dumped = 0
    for tool in args.tools:
        chains = load_chains(tool, raw_meta)
        if args.max_chains:
            chains = chains[: args.max_chains]
        n = len(chains)
        # pooled non-member galleries: other chains' targets, same construction as the
        # deep-cascade measurement so `full` reproduces the already-reported AUC
        nonmem = {c["id"]: [chains[(i + 1 + j * (n // (N_NONMEM + 1))) % n]
                            for j in range(N_NONMEM)] for i, c in enumerate(chains)}

        for i, c in enumerate(chains):
            gal = l2(raw[c["gallery_idx"]])
            nmg = [l2(raw[q["gallery_idx"]]) for q in nonmem[c["id"]]]
            # context donor: an identity unrelated to this chain's target AND to every
            # non-member used for scoring, so the transplant cannot inflate either class
            excl = {c["target_id"]} | {q["target_id"] for q in nonmem[c["id"]]}
            ctx = None
            for step in range(7, 7 + n):
                q = chains[(i + step) % n]
                if q["target_id"] not in excl:
                    ctx = cv2.imread(str(q["target_image"]))
                    if ctx is not None:
                        break
            tgt_img = cv2.imread(str(c["target_image"]))
            if ctx is None or tgt_img is None:
                continue

            def score(img, cond, k):
                e = embed(img)
                if e is None:
                    return
                rows.append({
                    "tool": tool, "chain": c["id"], "pass": k, "condition": cond,
                    "target_sim": float(np.median(gal @ e)),
                    "nonmem_sims": [float(np.median(g @ e)) for g in nmg],
                })

            # blend-fidelity controls, once per chain (pass 0 = no swapping at all)
            score(tgt_img, "noswap_full", 0)
            score(blend_face(tgt_img, ctx), "noswap_transplant", 0)
            score(blend_face(ctx, tgt_img), "noswap_foreign", 0)

            for k in args.passes:
                f = c["path"](k)
                if not (f.is_file() and f.stat().st_size > 0):
                    continue
                rel = cv2.imread(str(f))
                if rel is None:
                    continue
                score(rel, "full", k)
                score(blend_face(rel, ctx), "face_transplant", k)
                score(blend_face(ctx, rel), "foreign_face", k)
                if dumped < args.dump_examples and k == max(args.passes):
                    cv2.imwrite(str(OUT / f"transplant_example_{tool}_{dumped}.jpg"),
                                np.hstack([tgt_img, rel, ctx, blend_face(rel, ctx),
                                           blend_face(ctx, rel)]))
                    dumped += 1
            if (i + 1) % 100 == 0:
                print(f"[transplant] {tool} {i + 1}/{n} chains, {len(rows)} rows",
                      flush=True)

    flat = [{k: v for k, v in r.items() if k != "nonmem_sims"} |
            {"nonmem_sim_median": float(np.median(r["nonmem_sims"]))} for r in rows]
    keys = sorted({k for r in flat for k in r})
    with (OUT / "transplant_deep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)

    summary: dict = {}
    for tool in args.tools:
        for cond in ["noswap_full", "noswap_transplant", "noswap_foreign"] + CONDITIONS:
            ks = [0] if cond.startswith("noswap") else args.passes
            for k in ks:
                rs = [r for r in rows if r["tool"] == tool
                      and r["condition"] == cond and r["pass"] == k]
                if not rs:
                    continue
                pos = [r["target_sim"] for r in rs]
                neg = [v for r in rs for v in r["nonmem_sims"]]
                key = f"{tool}|{cond}|pass{k}"
                summary[key] = {
                    "n": len(rs), "auc": auc(pos, neg),
                    "tpr_at_1pct_fpr": tpr_at_fpr(pos, neg, 0.01),
                    "median_target": float(np.median(pos)),
                    "median_nonmember": float(np.median(neg)),
                    "excess": float(np.median(pos) - np.median(neg)),
                }
    (OUT / "transplant_deep.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'condition':<34}{'n':>6}{'AUC':>8}{'TPR@1%':>9}{'excess':>9}")
    for key, s in summary.items():
        print(f"{key:<34}{s['n']:>6}{s['auc']:>8.3f}"
              f"{s['tpr_at_1pct_fpr'] * 100:>8.1f}%{s['excess']:>9.4f}")
    print(f"\nwrote {OUT / 'transplant_deep.json'}")


if __name__ == "__main__":
    main()
