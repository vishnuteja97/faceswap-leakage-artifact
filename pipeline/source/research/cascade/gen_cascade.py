#!/usr/bin/env python3
"""Generate FaceFusion deep-cascade chains (GPU). Loads FaceFusion ONCE (persistent
session) and produces s1..sK per chain, where s_k = FF(donor_k, s_{k-1}), s0 = t0 image.

Shardable across GPUs:  --shards 4 --shard-id {0,1,2,3} --device-id {0,1,2,3}

Run in the facefusion env:
  /opt/reproduction/miniconda3/envs/facefusion/bin/python cascade/gen_cascade.py \
      --shards 4 --shard-id 0 --device-id 0
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CHAINS_DIR = OUT / "chains"
MANIFEST = OUT / "chains_manifest.jsonl"
FACEFUSION_ROOT = Path("/opt/reproduction/facefusion")
SWAP_MODEL = "hyperswap_1a_256"


def log(m: str) -> None:
    print(f"[gen] {m}", flush=True)


def ensure_ff(root: Path) -> None:
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # FaceFusion pre_check needs ffmpeg on PATH (normally added by `conda activate`).
    env_bin = str(Path(sys.executable).parent)
    if env_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")


def init_session(device_id: int, jobs_path: Path, temp_path: Path, warm_donor: Path, warm_target: Path):
    ensure_ff(FACEFUSION_ROOT)
    from facefusion import logger, state_manager
    from facefusion.args import apply_args
    from facefusion.core import (common_pre_check, conditional_process, pre_check,
                                 processors_pre_check)
    from facefusion.processors.core import get_processors_modules
    from facefusion.program import create_program
    from facefusion.program_helper import validate_args

    if not pre_check():
        raise RuntimeError("FaceFusion pre_check failed")
    warm_out = jobs_path / "_warmup.jpg"
    argv = ["facefusion.py", "headless-run", "-s", str(warm_donor), "-t", str(warm_target),
            "-o", str(warm_out), "--jobs-path", str(jobs_path), "--temp-path", str(temp_path),
            "--processors", "face_swapper",
            "--face-swapper-model", SWAP_MODEL, "--execution-providers", "cuda",
            "--execution-device-ids", str(device_id), "--log-level", "warn"]
    old = sys.argv
    try:
        sys.argv = argv
        program = create_program()
        if not validate_args(program):
            raise RuntimeError("invalid FF args")
        apply_args(vars(program.parse_args()), state_manager.init_item)
        logger.init(state_manager.get_item("log_level"))
    finally:
        sys.argv = old
    if not common_pre_check() or not processors_pre_check():
        raise RuntimeError("FF module pre_check failed")
    for pm in get_processors_modules(state_manager.get_item("processors")):
        if not pm.pre_process("output"):
            raise RuntimeError("FF processor pre_process failed")
    if conditional_process() != 0:
        raise RuntimeError("FF bootstrap swap failed")
    return state_manager, conditional_process, get_processors_modules


def swap_once(sm, conditional_process, get_processors_modules, donor: Path, target: Path,
              output: Path) -> bool:
    from facefusion.core import common_pre_check, processors_pre_check
    output.parent.mkdir(parents=True, exist_ok=True)
    sm.set_item("source_paths", [str(donor.resolve())])
    sm.set_item("target_path", str(target.resolve()))
    sm.set_item("output_path", str(output.resolve()))
    if not common_pre_check() or not processors_pre_check():
        return False
    for pm in get_processors_modules(sm.get_item("processors")):
        if not pm.pre_process("output"):
            return False
    return conditional_process() == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    chains = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    chains = [c for i, c in enumerate(chains) if i % args.shards == args.shard_id]
    log(f"shard {args.shard_id}/{args.shards} device={args.device_id} chains={len(chains)}")

    jobs_path = (OUT / f".jobs_shard{args.shard_id}").resolve()
    jobs_path.mkdir(parents=True, exist_ok=True)
    temp_path = (OUT / f".temp_shard{args.shard_id}").resolve()
    temp_path.mkdir(parents=True, exist_ok=True)
    warm = chains[0]
    t0 = time.perf_counter()
    sm, cond, getmod = init_session(args.device_id, jobs_path, temp_path,
                                    Path(warm["donor_images"][0]), Path(warm["t0_image"]))
    log(f"models loaded in {time.perf_counter()-t0:.1f}s")

    status_path = OUT / f"gen_status_shard{args.shard_id}.jsonl"
    n_ok = n_fail = n_skip = 0
    run0 = time.perf_counter()
    with status_path.open("w") as sf:
        for ci, c in enumerate(chains):
            cdir = CHAINS_DIR / c["chain_id"]
            cdir.mkdir(parents=True, exist_ok=True)
            prev = Path(c["t0_image"])
            ok_chain = True
            for k in range(len(c["donor_images"])):
                out = cdir / f"s{k+1}.jpg"
                if out.is_file() and out.stat().st_size > 0:
                    n_skip += 1
                    prev = out
                    continue
                ok = swap_once(sm, cond, getmod, Path(c["donor_images"][k]), prev, out)
                if not ok or not out.is_file():
                    n_fail += 1
                    ok_chain = False
                    log(f"{c['chain_id']} pass{k+1} FAILED; aborting chain")
                    break
                n_ok += 1
                prev = out
            sf.write(json.dumps({"chain_id": c["chain_id"], "ok": ok_chain,
                                 "n_passes_done": k + 1 if ok_chain else k}) + "\n")
            sf.flush()
            if (ci + 1) % 10 == 0:
                el = time.perf_counter() - run0
                log(f"[{ci+1}/{len(chains)}] ok={n_ok} fail={n_fail} skip={n_skip} "
                    f"rate={n_ok/max(el,1)*60:.1f} swaps/min")
    log(f"DONE shard {args.shard_id}: ok={n_ok} fail={n_fail} skip={n_skip} "
        f"elapsed={time.perf_counter()-run0:.1f}s")


if __name__ == "__main__":
    main()
