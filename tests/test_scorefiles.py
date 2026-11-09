"""Scoring-file prefetch cache (no network: a fake HTTP session)."""
import gzip
import hashlib
import json

import pytest

from stackprs import scorefiles


class FakeResponse:
    def __init__(self, body: bytes):
        self.body, self.text, self.status_code = body, body.decode(errors="replace"), 200
    def raise_for_status(self): pass
    def iter_content(self, n): yield self.body
    def __enter__(self): return self
    def __exit__(self, *a): pass


class FakeSession:
    """Serves files from a dict; `.md5` URLs get the real checksum unless corrupted."""
    def __init__(self, files, corrupt=()):
        self.files, self.corrupt, self.calls = files, set(corrupt), []
    def get(self, url, **_):
        self.calls.append(url)
        if url.endswith(".md5"):
            name = url[:-4].rsplit("/", 1)[-1]
            digest = "0" * 32 if name in self.corrupt else hashlib.md5(self.files[url[:-4]]).hexdigest()
            return FakeResponse(f"{digest}  {name}\n".encode())
        return FakeResponse(self.files[url])


URLS = {"PGS000001": "https://ftp/PGS000001_hmPOS_GRCh38.txt.gz",
        "PGS000002": "https://ftp/PGS000002_hmPOS_GRCh38.txt.gz",
        "PGS000003": None}                                   # no harmonised file in this build
FILES = {URLS["PGS000001"]: b"one", URLS["PGS000002"]: b"two"}


def test_urls_from_raw_records(tmp_path):
    rec = [{"id": "PGS000001", "ftp_harmonized_scoring_files": {"GRCh38": {"positions": "u38"}, "GRCh37": {"positions": "u37"}}},
           {"id": "PGS000002", "ftp_harmonized_scoring_files": None}]
    with gzip.open(tmp_path / "pgs_scores.json.gz", "wt") as fh:
        json.dump(rec, fh)
    assert scorefiles.load_harmonized_urls(tmp_path, "GRCh38") == {"PGS000001": "u38", "PGS000002": None}


def test_fetch_verifies_and_caches(tmp_path):
    sess = FakeSession(FILES)
    got = scorefiles.fetch(["PGS000001", "PGS000002", "PGS000003"], URLS, tmp_path, workers=2, session=sess, log=None)
    assert set(got) == {"PGS000001", "PGS000002"} and got["PGS000001"].read_bytes() == b"one"
    n = len(sess.calls)
    scorefiles.fetch(["PGS000001", "PGS000002"], URLS, tmp_path, workers=2, session=sess, log=None)
    assert len(sess.calls) == n                               # verified copies are not downloaded again


def test_fetch_rejects_corrupt_download(tmp_path):
    with pytest.raises(IOError, match="md5 mismatch"):
        scorefiles.fetch(["PGS000001"], URLS, tmp_path, session=FakeSession(FILES, corrupt={"PGS000001_hmPOS_GRCh38.txt.gz"}), log=None)
    assert not (tmp_path / "PGS000001_hmPOS_GRCh38.txt.gz").exists()


def test_stage_chunk_links_files_and_lists_missing(tmp_path):
    files = scorefiles.fetch(["PGS000001", "PGS000002"], URLS, tmp_path / "cache", session=FakeSession(FILES), log=None)
    d = scorefiles.stage_chunk("chunk_000", ["PGS000001", "PGS000003"], files, tmp_path / "chunks")
    assert sorted(p.name for p in d.glob("*.txt.gz")) == ["PGS000001_hmPOS_GRCh38.txt.gz"]
    assert (d / "PGS000001_hmPOS_GRCh38.txt.gz").stat().st_ino == files["PGS000001"].stat().st_ino   # hard link
    glob, missing = scorefiles.staged_inputs(d)
    assert glob == str(d / "*.txt.gz") and missing == "PGS000003"
    d2 = scorefiles.stage_chunk("chunk_001", ["PGS000003"], files, tmp_path / "chunks")
    assert scorefiles.staged_inputs(d2) == (None, "PGS000003")
