"""Build data/samples.tsv for GSE184362 from the GEO SOFT metadata.

Patient and tissue fields come from !Sample_characteristics_ch1, never from
the sample title, so the manifest stays traceable to the archived record.
"""

import gzip
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOFT = ROOT / "geo" / "GSE184362_family.soft"
GEO = ROOT / "geo" / "GSE184362"
OUT = ROOT / "data" / "samples.tsv"

TISSUE = {
    "thyroid tumor": "primary_tumor",
    "thyroid paratumor": "adjacent",
    "lymph node metastase": "lymph_node_metastasis",
    "subcutaneous metastase": "subcutaneous_metastasis",
}


def parse_soft(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8") if path.exists() else gzip.decompress(
        path.with_suffix(".gz").read_bytes()
    ).decode("utf-8", "replace")
    records = {}
    for gsm, body in re.findall(r"\^SAMPLE = (GSM\d+)\n(.*?)(?=\n\^SAMPLE|\n\^SERIES|\Z)", text, re.S):
        chars = dict(
            line.split(": ", 1) for line in re.findall(r"!Sample_characteristics_ch1 = (.*)", body)
        )
        tissues = [value for key, value in chars.items() if key == "tissue"]
        records[gsm] = {"patient": chars.get("patient id", ""), "tissues": tissues}
    return records


def main() -> int:
    records = parse_soft(SOFT)
    if not records:
        print("no samples parsed from SOFT", file=sys.stderr)
        return 2
    rows = []
    for gsm in sorted(records):
        tissue_terms = records[gsm]["tissues"]
        mapped = [TISSUE[t] for t in tissue_terms if t in TISSUE]
        if not mapped:
            print(f"{gsm}: unmapped tissue terms {tissue_terms}", file=sys.stderr)
            return 1
        context = mapped[-1]  # most specific term is listed last in GEO
        patient = "P{:02d}".format(int(records[gsm]["patient"].split()[-1]))
        folder = GEO / gsm
        for leaf in ("matrix.mtx.gz", "features.tsv.gz", "barcodes.tsv.gz"):
            if not (folder / leaf).exists():
                print(f"{gsm}: missing {folder / leaf}", file=sys.stderr)
                return 1
        rows.append(
            {
                "sample_id": gsm,
                "patient_id": patient,
                "tissue_context": context,
                "matrix": f"../geo/GSE184362/{gsm}/matrix.mtx.gz",
                "features": f"../geo/GSE184362/{gsm}/features.tsv.gz",
                "barcodes": f"../geo/GSE184362/{gsm}/barcodes.tsv.gz",
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = ["sample_id", "patient_id", "tissue_context", "matrix", "features", "barcodes"]
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row[column] for column in header) + "\n")
    print(f"{len(rows)} samples -> {OUT}")
    for row in rows:
        print(f"  {row['sample_id']} {row['patient_id']:>4} {row['tissue_context']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
