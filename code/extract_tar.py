"""Extract the GSE184362 RAW tar into one directory per GSM sample.

Tar members look like GSM5585102_PTC1_T_matrix.mtx.gz; each sample becomes
  geo/GSE184362/GSM5585102/{matrix.mtx.gz,features.tsv.gz,barcodes.tsv.gz}
which is what the 10x-compatible readers expect.
"""

import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "geo"
TAR = ROOT / "GSE184362_RAW.tar"
OUT = ROOT / "GSE184362"
PATTERN = re.compile(r"^(GSM\d+)_[^_]+(?:_[^_]+)*?_(matrix\.mtx|features\.tsv|barcodes\.tsv)\.gz$")


def main() -> int:
    if not TAR.exists():
        print(f"missing {TAR}; run fetch_tar.py first", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    seen = {}
    with tarfile.open(TAR) as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            match = PATTERN.match(name)
            if not match:
                print(f"skip unmatched member: {name}", flush=True)
                continue
            gsm, leaf = match.group(1), match.group(2) + ".gz"
            target = OUT / gsm / leaf
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
            seen.setdefault(gsm, []).append(leaf)
    for gsm in sorted(seen):
        print(f"{gsm}: {sorted(seen[gsm])}")
    print(f"{len(seen)} samples -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
