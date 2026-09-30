#!/usr/bin/env python3
"""Generate the deep (K=25) cascade extension for BlendFace or CanonSwap.

Resumes the D3 three-pass chains: passes 1..3 are reused as-is from D1/D3, and passes
4..25 are generated with a fresh donor per pass from chains_manifest_deep_aux.jsonl
(one shared donor schedule for both tools). The model is loaded once per process.

Outputs: cascade/outputs/chains_deep_aux/<tool>/<pair_id>/s4.jpg .. s25.jpg

Run in the matching env, pinned to one GPU:
  CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/blendface/bin/python \
      cascade/gen_cascade_deep_aux.py --tool blendface
  CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/CanonSwap/bin/python \
      cascade/gen_cascade_deep_aux.py --tool canonswap
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ART = Path("/opt/reproduction/S&P2027-Artifact/data")
D1_SWAPS = ART / "D1" / "swaps"
MANIFEST = OUT / "chains_manifest_deep_aux.jsonl"
K_DEEP = 25

if str(D1_SWAPS) not in sys.path:
    sys.path.insert(0, str(D1_SWAPS))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_swapper(tool: str, work_dir: Path):
    """Return swap(donor, target, out) -> bool with a persistent model."""
    if tool == "blendface":
        import torch
        from run_blendface import BLENDFACE_ROOT_DEFAULT, load_blendswap_model, swap_pair
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        model, device = load_blendswap_model(BLENDFACE_ROOT_DEFAULT, device_str)

        def swap(donor: Path, target: Path, out: Path) -> bool:
            try:
                swap_pair(model, device, donor, target, out)
                return out.is_file() and out.stat().st_size > 0
            except Exception as exc:
                log(f"  blendface fail: {exc}")
                return False
        return swap

    if tool == "canonswap":
        from run_canonswap import (CANONSWAP_ROOT_DEFAULT, init_canonswap_pipeline,
                                   run_canonswap_pair)
        pipeline, arg_cls = init_canonswap_pipeline(
            CANONSWAP_ROOT_DEFAULT, dataset="vggface2", device_id=0)

        def swap(donor: Path, target: Path, out: Path) -> bool:
            try:
                run_canonswap_pair(donor, target, out, pipeline, arg_cls, work_dir,
                                   device_id=0)
                return out.is_file() and out.stat().st_size > 0
            except Exception as exc:
                log(f"  canonswap fail: {exc}")
                return False
        return swap

    raise ValueError(tool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, choices=["blendface", "canonswap"])
    ap.add_argument("--max-chains", type=int, default=None, help="debug/smoke")
    args = ap.parse_args()
    tool = args.tool

    chains = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    if args.max_chains:
        chains = chains[: args.max_chains]

    chains_dir = OUT / "chains_deep_aux" / tool
    chains_dir.mkdir(parents=True, exist_ok=True)
    work_dir = OUT / f".work_deep_aux_{tool}"
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    swap = make_swapper(tool, work_dir)
    log(f"{tool}: model ready in {time.perf_counter() - t0:.1f}s; chains={len(chains)}")

    status_path = OUT / f"gen_deep_aux_status_{tool}.jsonl"
    n_ok = n_fail = n_skip = n_sub = 0
    run0 = time.perf_counter()
    with status_path.open("a") as sf:
        for ci, c in enumerate(chains):
            pid = c["pair_id"]
            cdir = chains_dir / pid
            cdir.mkdir(parents=True, exist_ok=True)
            s3 = ART / f"D3/swaps/{tool}/swap3/vggface2_seed_42" / c["swap_basenames"]["3"]
            prev = s3
            last_ok = 3
            spares = list(c["spare_donor_images"])
            subs: dict[int, str] = {}
            for j, donor0 in enumerate(c["donor_images_deep"]):
                k = 4 + j
                out = cdir / f"s{k}.jpg"
                if out.is_file() and out.stat().st_size > 0:
                    n_skip += 1
                    prev = out
                    last_ok = k
                    continue
                done = False
                for attempt, donor in enumerate([donor0] + spares):
                    if swap(Path(donor), prev, out):
                        if attempt > 0:
                            spares.remove(donor)
                            subs[k] = donor
                            n_sub += 1
                        n_ok += 1
                        prev = out
                        last_ok = k
                        done = True
                        break
                    n_fail += 1
                if not done:
                    log(f"{pid} pass{k} FAILED after {1 + len(spares)} donors; "
                        f"chain stops at pass {last_ok}")
                    break
            sf.write(json.dumps({"pair_id": pid, "depth_reached": last_ok,
                                 "donor_substitutions": subs}) + "\n")
            sf.flush()
            if (ci + 1) % 10 == 0:
                el = time.perf_counter() - run0
                per = el / max(n_ok + n_skip, 1)
                eta_h = (len(chains) - ci - 1) * (K_DEEP - 3) * per / 3600
                log(f"[{ci + 1}/{len(chains)}] ok={n_ok} fail={n_fail} skip={n_skip} "
                    f"sub={n_sub} {60 / max(per, 1e-9):.1f} swaps/min eta={eta_h:.1f}h")
    log(f"DONE {tool}: ok={n_ok} fail={n_fail} skip={n_skip} sub={n_sub} "
        f"elapsed={(time.perf_counter() - run0) / 3600:.2f}h")


if __name__ == "__main__":
    main()
