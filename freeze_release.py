"""Maintainer command: refresh checksums after review; optionally make a ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
HASHES = ROOT / 'data/provenance/release_hashes.json'


def release_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
                  and '.git' not in p.relative_to(ROOT).parts
                  and '__pycache__' not in p.relative_to(ROOT).parts
                  and p.suffix != '.pyc')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, help='ZIP destination outside the artifact directory')
    args = ap.parse_args()
    if args.archive and ROOT in args.archive.resolve().parents:
        raise ValueError('Put the archive outside the artifact directory')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in release_files() if p != HASHES}
    HASHES.write_text(json.dumps(hashes, indent=2) + '\n')
    print(f'Wrote {len(hashes)} file hashes. Run validate.py after every freeze.')
    if args.archive:
        with zipfile.ZipFile(args.archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in release_files():
                z.write(p, Path('artifact') / p.relative_to(ROOT))
        print(f'Archive: {args.archive} ({args.archive.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
