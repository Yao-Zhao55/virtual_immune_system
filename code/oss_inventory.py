"""Inventory OSS buckets: object count, bytes, storage classes, top prefixes.

Read-only. Uses ListObjects (V1) with marker pagination, which keeps the
canonical resource unsigned and therefore avoids the continuation-token
signature mismatch seen with ListObjectsV2 here.

Usage: python oss_inventory.py [bucket ...]
"""

import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from email.utils import formatdate
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oss_upload import sign  # noqa: E402

ENDPOINT = os.environ.get("OSS_ENDPOINT", "oss-cn-shanghai.aliyuncs.com")


def list_page(bucket, marker, ak_id, ak_secret):
    query = {"max-keys": "1000"}
    if marker:
        query["marker"] = marker
    url = f"https://{bucket}.{ENDPOINT}/?{urllib.parse.urlencode(query)}"
    date = formatdate(usegmt=True)
    string_to_sign = f"GET\n\n\n{date}\n/{bucket}/"
    request = urllib.request.Request(url, method="GET")
    request.add_header("Date", date)
    request.add_header("Authorization", f"OSS {ak_id}:{sign(ak_secret, string_to_sign)}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return ET.fromstring(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"HTTP {exc.code} on {bucket}: {body}") from exc


def inventory(bucket, ak_id, ak_secret, max_pages=400):
    total, size, classes, top = 0, 0, Counter(), Counter()
    marker, pages, last_key = None, 0, None
    while pages < max_pages:
        root = list_page(bucket, marker, ak_id, ak_secret)
        keys = []
        for item in root.findall("Contents"):
            key = item.findtext("Key")
            keys.append(key)
            total += 1
            size += int(item.findtext("Size") or 0)
            classes[item.findtext("StorageClass") or "Standard"] += 1
            top[key.split("/")[0] if "/" in key else "(root files)"] += 1
        pages += 1
        truncated = (root.findtext("IsTruncated") or "false").lower() == "true"
        next_marker = root.findtext("NextMarker") or (keys[-1] if keys else None)
        if not truncated or not next_marker or next_marker == last_key:
            break
        last_key, marker = next_marker, next_marker
    return total, size, classes, top, pages, truncated


def main() -> int:
    ak_id = os.environ.get("OSS_ACCESS_KEY_ID", "")
    ak_secret = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
    if not ak_id or not ak_secret:
        print("ERROR: set OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET", file=sys.stderr)
        return 2
    buckets = sys.argv[1:] or ["dsh-immune-system-data", "dsh-fvcell-data", "dsh-kidney-data"]
    grand = 0
    for bucket in buckets:
        total, size, classes, top, pages, truncated = inventory(bucket, ak_id, ak_secret)
        grand += size
        print(f"\n== {bucket}: {total:,} objects, {size / 2**30:.2f} GiB, {pages} page(s)"
              f"{' (TRUNCATED)' if truncated else ''}")
        print("   storage classes:", dict(classes))
        for name, count in top.most_common(6):
            print(f"   {count:6d}  {name}/")
    print(f"\nTOTAL across {len(buckets)} bucket(s): {grand / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
