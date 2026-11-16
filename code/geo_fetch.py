"""Download GEO supplementary files for a series, plus its SOFT metadata.

NCBI blocks some PowerShell/curl clients, but plain urllib works, so all
transfers go through this script.

Usage:
  python geo_fetch.py soft   GSE184362
  python geo_fetch.py list   GSE184362
  python geo_fetch.py sample GSE184362 GSM5585102 [more GSMs...]
"""

import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
ROOT = Path(__file__).resolve().parent / "geo"


def family_dir(accession: str) -> str:
    digits = accession[3:]
    return f"GSE{digits[:-3]}nnn" if len(digits) > 3 else "GSE"


def url_for(accession: str, leaf: str) -> str:
    return f"{BASE}/{family_dir(accession)}/{accession}/{leaf}"


def fetch(url: str, dest: Path, retries: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            start = time.perf_counter()
            with urllib.request.urlopen(url, timeout=180) as response, dest.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            mb = dest.stat().st_size / 1e6
            print(f"OK   {dest.name}  {mb:.1f} MB  {time.perf_counter() - start:.1f}s", flush=True)
            return dest
        except Exception as exc:  # noqa: BLE001 - retry then report
            print(f"retry {attempt}/{retries} {dest.name}: {type(exc).__name__} {exc}", flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed: {url}")


def soft(accession: str) -> Path:
    return fetch(url_for(accession, f"soft/{accession}_family.soft.gz"), ROOT / f"{accession}_family.soft.gz")


def listing(accession: str) -> list[str]:
    with urllib.request.urlopen(url_for(accession, "suppl/"), timeout=120) as response:
        html = response.read().decode("utf-8", "replace")
    import re

    return sorted({m for m in re.findall(r'href="([^"]+)"', html) if not m.startswith(("?", "/", "http"))})


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    command, accession = sys.argv[1], sys.argv[2]
    if command == "soft":
        soft(accession)
    elif command == "list":
        for name in listing(accession):
            print(name)
    elif command == "sample":
        for gsm in sys.argv[3:]:
            for suffix in ("matrix.mtx.gz", "features.tsv.gz", "barcodes.tsv.gz"):
                fetch(url_for(accession, f"suppl/{gsm}_{suffix}"), ROOT / accession / gsm / f"{suffix}")
    else:
        print(f"unknown command: {command}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
