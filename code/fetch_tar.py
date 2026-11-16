"""Resumable download of the GSE184362 RAW tar (all 23 10x samples).

NCBI drops PowerShell/curl connections here, so the transfer goes through
urllib with Range-based resume and progress output.

Usage: python fetch_tar.py [--url URL] [--dest PATH]
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE184nnn/GSE184362/suppl/GSE184362_RAW.tar"
CHUNK = 4 << 20


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    request = urllib.request.Request(url)
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=180) as response:
        total = int(response.headers.get("Content-Length", 0)) + have
        mode = "ab" if have and response.status == 206 else "wb"
        if mode == "wb":
            have = 0
        print(f"resume={have} total={total} status={response.status}", flush=True)
        start, last = time.perf_counter(), 0.0
        with dest.open(mode) as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                have += len(chunk)
                now = time.perf_counter()
                if now - last > 5:
                    last = now
                    rate = have / max(now - start, 1e-9) / 1e6
                    pct = 100 * have / total if total else 0
                    print(f"{have / 1e6:8.1f}/{total / 1e6:.1f} MB  {pct:5.1f}%  {rate:5.1f} MB/s", flush=True)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent / "geo" / "GSE184362_RAW.tar")
    args = parser.parse_args()
    path = download(args.url, args.dest)
    print(f"DONE {path} {path.stat().st_size / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
