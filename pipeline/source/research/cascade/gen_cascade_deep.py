#!/usr/bin/env python3
"""Generate the deep (K=25) FaceFusion cascade extension for the rebuttal.

Resumes the chains of the submitted 5-pass experiment: existing s1..s5 are reused as-is,
and s6..s25 are generated with a fresh donor per pass from chains_manifest_deep.jsonl.
FaceFusion is loaded once per shard (persistent session), as in gen_cascade.py.

Shardable across GPUs:  --shards 4 --shard-id {0,1,2,3} --device-id {0,1,2,3}

Run in the facefusion env:
  /opt/reproduction/miniconda3/envs/facefusion/bin/python cascade/gen_cascade_deep.py \
      --shards 4 --shard-id 0 --device-id 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gen_cascade import CHAINS_DIR, OUT, init_session, log, swap_once

MANIFEST_DEEP = OUT / "chains_manifest_deep.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    chains = [json.loads(l) for l in MANIFEST_DEEP.read_text().splitlines() if l.strip()]
    chains = [c for i, c in enumerate(chains) if i % args.shards == args.shard_id]
    log(f"deep shard {args.shard_id}/{args.shards} device={args.device_id} chains={len(chains)}")

    jobs_path = (OUT / f".jobs_deep_shard{args.shard_id}").resolve()
    jobs_path.mkdir(parents=True, exist_ok=True)
    temp_path = (OUT / f".temp_deep_shard{args.shard_id}").resolve()
    temp_path.mkdir(parents=True, exist_ok=True)

    warm = chains[0]
    t0 = time.perf_counter()
    sm, cond, getmod = init_session(args.device_id, jobs_path, temp_path,
                                    Path(warm["donor_images"][0]), Path(warm["t0_image"]))
    log(f"models loaded in {time.perf_counter() - t0:.1f}s")

    status_path = OUT / f"gen_deep_status_shard{args.shard_id}.jsonl"
    n_ok = n_fail = n_skip = n_sub = 0
    run0 = time.perf_counter()
    with status_path.open("a") as sf:
        for ci, c in enumerate(chains):
            cdir = CHAINS_DIR / c["chain_id"]
            cdir.mkdir(parents=True, exist_ok=True)
            prev = Path(c["t0_image"])
            last_ok = 0
            spares = list(c.get("spare_donor_images", []))
            subs: dict[int, str] = {}
            for k in range(len(c["donor_images"])):
                out = cdir / f"s{k + 1}.jpg"
                if out.is_file() and out.stat().st_size > 0:
                    n_skip += 1
                    prev = out
                    last_ok = k + 1
                    continue
                # A donor whose face the swapper cannot detect would otherwise truncate the
                # chain; retry with spare donors before giving up, so attrition at depth
                # does not silently bias the surviving sample.
                candidates = [c["donor_images"][k]] + spares
                done = False
                for attempt, donor in enumerate(candidates):
                    if swap_once(sm, cond, getmod, Path(donor), prev, out) and out.is_file():
                        if attempt > 0:
                            spares.remove(donor)
                            subs[k + 1] = donor
                            n_sub += 1
                        n_ok += 1
                        prev = out
                        last_ok = k + 1
                        done = True
                        break
                    n_fail += 1
                if not done:
                    log(f"{c['chain_id']} pass{k + 1} FAILED after {len(candidates)} donors; "
                        f"chain stops at pass {last_ok}")
                    break
            sf.write(json.dumps({"chain_id": c["chain_id"], "depth_reached": last_ok,
                                 "donor_substitutions": subs}) + "\n")
            sf.flush()
            if (ci + 1) % 5 == 0:
                el = time.perf_counter() - run0
                rate = n_ok / max(el, 1e-9)
                per_chain = n_ok / (ci + 1)
                eta_h = (len(chains) - ci - 1) * per_chain / max(rate, 1e-9) / 3600
                log(f"[{ci + 1}/{len(chains)}] ok={n_ok} fail={n_fail} skip={n_skip} "
                    f"sub={n_sub} rate={rate * 60:.1f} swaps/min eta={eta_h:.1f}h")
    log(f"DONE deep shard {args.shard_id}: ok={n_ok} fail={n_fail} skip={n_skip} "
        f"sub={n_sub} elapsed={time.perf_counter() - run0:.1f}s")


if __name__ == "__main__":
    main()
