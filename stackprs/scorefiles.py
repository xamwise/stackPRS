"""Local cache of PGS Catalog harmonised scoring files (parallel, md5-verified).

Inside every run, pgsc_calc downloads its scoring files one at a time (measured
on a Mac: 200 files / 301 MB in 137 s, ~2 MB/s) and does so again whenever a
chunk is recomputed. Prefetching them once, in parallel, into
``data/scorefiles/<build>/`` and handing each chunk its files via
``--scorefile`` feeds pgsc_calc's formatting step exactly the same files:
pgsc_calc mixes ``--scorefile`` paths and downloaded files into one channel, and
harmonised files in the target build need no liftover. It also lets cluster
compute nodes without internet access score from the cache.
"""
from __future__ import annotations

import concurrent.futures as cf
import gzip
import hashlib
import json
import os
import pathlib
import shutil

import requests

from . import catalog


def harmonized_urls(records, build: str) -> dict[str, str | None]:
    """{pgs_id: URL of the harmonised scoring file in `build`} from raw Catalog records."""
    return {r["id"]: (((r.get("ftp_harmonized_scoring_files") or {}).get(build) or {}).get("positions"))
            for r in records}


def load_harmonized_urls(cache_dir: pathlib.Path, build: str) -> dict[str, str | None]:
    """URLs from the raw records step 01 caches (data/cache/pgs_scores.json.gz)."""
    path = cache_dir / "pgs_scores.json.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/01_fetch_metadata.py")
    with gzip.open(path, "rt") as fh:
        return harmonized_urls(json.load(fh), build)


def _md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_one(url: str, dest: pathlib.Path, session: requests.Session) -> pathlib.Path:
    """Download `url` to `dest` unless a verified copy exists; verify against `<url>.md5`."""
    ok = dest.with_name(dest.name + ".md5ok")
    if dest.exists() and ok.exists() and ok.read_text().strip() == _md5(dest):
        return dest
    r = session.get(url + ".md5", timeout=120)
    r.raise_for_status()
    expected = r.text.split()[0].lower()
    tmp = dest.with_name(dest.name + ".part")
    with session.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(1 << 20):
                fh.write(block)
    got = _md5(tmp)
    if got != expected:
        tmp.unlink(missing_ok=True)
        raise IOError(f"md5 mismatch for {url}: expected {expected}, got {got}")
    tmp.replace(dest)
    ok.write_text(got)
    return dest


def fetch(ids, urls: dict[str, str | None], dest_dir: pathlib.Path, workers: int = 8,
          session: requests.Session | None = None, log=print) -> dict[str, pathlib.Path]:
    """Download the harmonised files for `ids` in parallel; returns {pgs_id: path}.

    Ids without a harmonised file in this build are left out (pgsc_calc then
    fetches them itself via --pgs_id).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    sess = session or catalog.make_session()
    todo = {pid: urls.get(pid) for pid in ids}
    out: dict[str, pathlib.Path] = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(_fetch_one, u, dest_dir / u.rsplit("/", 1)[-1], sess): pid
                   for pid, u in todo.items() if u}
        for fut in cf.as_completed(futures):
            out[futures[fut]] = fut.result()
    missing = sorted(pid for pid, u in todo.items() if not u)
    if missing and log:
        log(f"  {len(missing)} score(s) have no harmonised file in this build; pgsc_calc will fetch "
            f"them via --pgs_id: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return out


def stage_chunk(chunk_id: str, ids, files: dict[str, pathlib.Path], stage_root: pathlib.Path) -> pathlib.Path:
    """Directory with exactly this chunk's scoring files (hard links into the cache).

    Ids without a cached file are listed in ``missing_pgs_ids.txt``.
    """
    d = stage_root / chunk_id
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    missing = []
    for pid in ids:
        src = files.get(pid)
        if src is None:
            missing.append(pid)
            continue
        try:
            os.link(src, d / src.name)
        except OSError:                # different filesystem: copy instead
            shutil.copy2(src, d / src.name)
    (d / "missing_pgs_ids.txt").write_text(",".join(missing))
    return d


def staged_inputs(stage_dir: pathlib.Path) -> tuple[str | None, str]:
    """(--scorefile glob or None, comma-separated ids for --pgs_id) for a staged chunk."""
    has_files = any(stage_dir.glob("*.txt.gz"))
    missing = (stage_dir / "missing_pgs_ids.txt").read_text().strip()
    return (str(stage_dir / "*.txt.gz") if has_files else None), missing
