"""Download official ZuCo 2.0 Matlab files from OSF.

The repository's get_data.sh downloads benchmark feature files. This script
downloads the larger official ZuCo 2.0 task Matlab files that contain
sentence/word/fixation structures needed for word/fixation-level work.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


API_ROOT = "https://api.osf.io/v2/nodes/2urht/files/osfstorage/"
DEFAULT_SUBJECTS = [
    "YAC", "YAG", "YAK", "YDG", "YDR", "YFR", "YFS", "YHS",
    "YIS", "YLS", "YMD", "YRK", "YRP", "YSD", "YSL", "YTL",
]


def make_session() -> requests.Session:
    session = requests.Session()
    # The local Python environment has a broken proxy configuration. OSF is
    # publicly reachable directly, so ignore proxy env vars for this downloader.
    session.trust_env = False
    return session


def get_json(session: requests.Session, url: str) -> dict:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def walk_osf(session: requests.Session, url: str):
    while url:
        payload = get_json(session, url)
        for item in payload["data"]:
            attrs = item["attributes"]
            if attrs["kind"] == "folder":
                child_url = item["relationships"]["files"]["links"]["related"]["href"]
                yield from walk_osf(session, child_url)
            else:
                yield {
                    "name": attrs["name"],
                    "path": attrs["materialized_path"],
                    "size": attrs.get("size") or 0,
                    "download": item["links"]["download"],
                }
        url = payload.get("links", {}).get("next")


def select_matlab_files(session: requests.Session, subjects: set[str]) -> list[dict]:
    selected = []
    pattern = re.compile(
        r"/task[12] - (NR|TSR)/Matlab files/results(Y[A-Z]+)_(NR|TSR)\.mat$"
    )
    for item in walk_osf(session, API_ROOT):
        match = pattern.search(item["path"] or "")
        if match and match.group(2) in subjects:
            selected.append(item)
    return sorted(selected, key=lambda x: x["path"])


def download_file(item: dict, out_dir: Path) -> None:
    session = make_session()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / item["name"]
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    expected_size = int(item["size"] or 0)

    if out_path.exists() and expected_size and out_path.stat().st_size == expected_size:
        print(f"SKIP complete {out_path.name}", flush=True)
        return

    if part_path.exists() and expected_size and part_path.stat().st_size > expected_size:
        part_path.unlink()

    downloaded = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    print(f"DOWNLOAD {item['name']} ({expected_size / 1024 ** 3:.2f} GB)", flush=True)
    if downloaded:
        print(f"  {item['name']}: resume at {downloaded / 1024 ** 2:.1f} MB", flush=True)

    with session.get(
        item["download"],
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=60,
    ) as response:
        if response.status_code == 200 and downloaded:
            downloaded = 0
            part_path.unlink(missing_ok=True)
        response.raise_for_status()

        mode = "ab" if downloaded else "wb"
        start = time.time()
        last_report = start
        with part_path.open(mode + "") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= 10:
                    mb = downloaded / 1024 ** 2
                    total_mb = expected_size / 1024 ** 2 if expected_size else 0
                    rate = mb / max(now - start, 1)
                    if total_mb:
                        print(
                            f"  {item['name']}: {mb:.1f}/{total_mb:.1f} MB "
                            f"({rate:.1f} MB/s)",
                            flush=True,
                        )
                    else:
                        print(f"  {item['name']}: {mb:.1f} MB ({rate:.1f} MB/s)", flush=True)
                    last_report = now

    final_size = part_path.stat().st_size
    if expected_size and final_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {item['name']}: got {final_size}, expected {expected_size}"
        )
    part_path.replace(out_path)
    print(f"OK {out_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="data/train",
        help="Directory for flattened resultsY*_NR/TSR.mat files.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=DEFAULT_SUBJECTS,
        help="Subject IDs to download.",
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of files to download in parallel.",
    )
    args = parser.parse_args()

    subjects = {s.upper() for s in args.subjects}
    session = make_session()
    files = select_matlab_files(session, subjects)

    total_size = sum(item["size"] for item in files)
    print(f"Selected {len(files)} files, {total_size / 1024 ** 3:.2f} GB total")
    for item in files:
        print(f"{item['size'] / 1024 ** 3:5.2f} GB  {item['path']}")

    if args.list_only:
        return 0

    out_dir = Path(args.out_dir)
    workers = max(1, args.workers)
    if workers == 1:
        for item in files:
            download_file(item, out_dir)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download_file, item, out_dir) for item in files]
            for future in as_completed(futures):
                future.result()
    return 0


if __name__ == "__main__":
    sys.exit(main())
