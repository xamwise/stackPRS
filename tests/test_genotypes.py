"""Genome-build detection and the step-00 guard (no network: reference positions faked)."""
import pytest

from conftest import load_script
from stackprs import genotypes

REF = {   # (pgs, build) -> positions; the same variants at their GRCh37 / GRCh38 coordinates
    "GRCh37": {("2", 106120020 + i) for i in range(50)},
    "GRCh38": {("2", 105107792 + i) for i in range(50)},
}


@pytest.fixture
def fake_reference(monkeypatch):
    monkeypatch.setattr(genotypes, "reference_positions", lambda pgs, build, cache_dir, session=None: REF[build])


def write_bim(path, positions):
    path.write_text("".join(f"{c}\trs{i}\t0\t{p}\tA\tG\n" for i, (c, p) in enumerate(positions)))


def test_target_positions_bim_and_pvar(tmp_path):
    write_bim(tmp_path / "g.bim", [("chr2", 10), ("2", 20)])
    assert genotypes.target_positions(tmp_path / "g", "bfile") == {("2", 10), ("2", 20)}
    (tmp_path / "p.pvar").write_text("##fileformat\n#CHROM\tPOS\tID\tREF\tALT\n2\t30\trs1\tA\tG\n")
    assert genotypes.target_positions(tmp_path / "p", "pfile") == {("2", 30)}


@pytest.mark.parametrize("build", ["GRCh37", "GRCh38"])
def test_detect_build(tmp_path, fake_reference, build):
    write_bim(tmp_path / "g.bim", sorted(REF[build])[:40] + [("1", 5)])
    got, hits = genotypes.detect_build(tmp_path / "g", "bfile", tmp_path)
    assert got == build and hits[build] == 40


def test_detect_build_undecided_with_few_overlaps(tmp_path, fake_reference):
    write_bim(tmp_path / "g.bim", sorted(REF["GRCh38"])[:5])
    assert genotypes.detect_build(tmp_path / "g", "bfile", tmp_path)[0] is None


def test_step00_refuses_wrong_build(make_cfg, tmp_path, fake_reference, monkeypatch, capsys):
    cfg = make_cfg()                                            # target.build: GRCh37
    for ext in (".bed", ".fam"):
        (tmp_path / f"geno{ext}").write_text("")
    write_bim(tmp_path / "geno.bim", sorted(REF["GRCh38"]))
    monkeypatch.setattr("sys.argv", ["00_make_samplesheet.py"])
    with pytest.raises(SystemExit, match="Your genotypes are in GRCh38, but target.build is GRCh37"):
        load_script("00_make_samplesheet").main()
    assert not (cfg.work_dir / "pgsc_params.env").exists()     # nothing written with the wrong build
    cfg.raw["target"]["build"] = "GRCh38"
    load_script("00_make_samplesheet").main()
    assert "TARGET_BUILD=GRCh38" in (cfg.work_dir / "pgsc_params.env").read_text()
    assert "genotypes are GRCh38" in capsys.readouterr().out


def test_centimorgan_positions_are_zeroed_in_a_copy(tmp_path):
    (tmp_path / "g.bim").write_text("1\trs1\t0.490722\t756604\tA\tG\n1\trs2\t0\t768448\tC\tT\n")
    (tmp_path / "g.bed").write_bytes(b"\x6c\x1b\x01")
    (tmp_path / "g.fam").write_text("a a 0 0 1 -9\n")
    assert genotypes.bim_has_genetic_distances(tmp_path / "g.bim")
    out = genotypes.zero_genetic_distances(tmp_path / "g", tmp_path / "data" / "genotypes" / "g")
    rows = [l.split("\t") for l in (tmp_path / "data/genotypes/g.bim").read_text().splitlines()]
    assert [r[2] for r in rows] == ["0", "0"] and rows[0][3] == "756604"
    assert not genotypes.bim_has_genetic_distances(tmp_path / "data/genotypes/g.bim")
    assert (tmp_path / "g.bim").read_text().startswith("1\trs1\t0.490722")          # input untouched
    assert (tmp_path / "data/genotypes/g.bed").stat().st_ino == (tmp_path / "g.bed").stat().st_ino   # hard link
    assert out == tmp_path / "data" / "genotypes" / "g"


@pytest.mark.parametrize("cm, sanitized", [("0.490722", True), ("0", False)])
def test_step00_gives_pgsc_calc_a_cm_free_bim_only_when_needed(make_cfg, tmp_path, monkeypatch, cm, sanitized):
    cfg = make_cfg()
    (tmp_path / "geno.bim").write_text(f"1\trs1\t{cm}\t756604\tA\tG\n")
    (tmp_path / "geno.bed").write_bytes(b"\x6c\x1b\x01")
    (tmp_path / "geno.fam").write_text("a a 0 0 1 -9\n")
    monkeypatch.setattr("sys.argv", ["00_make_samplesheet.py", "--no-build-check"])
    load_script("00_make_samplesheet").main()
    sheet = (cfg.work_dir / "samplesheet.csv").read_text()
    expected = cfg.work_dir / "genotypes" / "geno" if sanitized else tmp_path / "geno"
    assert f",{expected},," in sheet


def test_fingerprint_drops_stale_prepared_genotypes(tmp_path):
    """pgsc_calc's storeDir reuses prepared genotypes by file name; changed inputs must invalidate them."""
    import os
    (tmp_path / "g.bed").write_bytes(b"\x6c\x1b\x01")
    (tmp_path / "g.bim").write_text("1\trs1\t0.4\t756604\tA\tG\n")
    (tmp_path / "g.fam").write_text("a a 0 0 1 -9\n")
    sheet = tmp_path / "samplesheet.csv"
    sheet.write_text(f"sampleset,path_prefix,chrom,format\nCOH,{tmp_path / 'g'},,bfile\n")
    fp1 = genotypes.fingerprint(sheet)
    work = tmp_path / "work"
    (work / "genomes" / "relabelled").mkdir(parents=True)
    assert genotypes.drop_stale_genotypes(work, fp1)                 # no stamp (older run) -> stale
    assert not (work / "genomes").exists()
    (work / "genomes").mkdir()
    assert not genotypes.drop_stale_genotypes(work, fp1)             # unchanged -> cache kept
    assert (work / "genomes").exists()
    (tmp_path / "g.bim").write_text("1\trs1\t0\t756604\tA\tG\n")    # genotype files change
    os.utime(tmp_path / "g.bim", ns=(1, 1))
    fp2 = genotypes.fingerprint(sheet)
    assert fp2 != fp1
    assert genotypes.drop_stale_genotypes(work, fp2) and not (work / "genomes").exists()


def test_cm_free_copy_is_only_rewritten_when_the_source_changes(tmp_path):
    import os
    (tmp_path / "g.bim").write_text("1\trs1\t0.49\t756604\tA\tG\n")
    (tmp_path / "g.bed").write_bytes(b"\x6c\x1b\x01")
    (tmp_path / "g.fam").write_text("a a 0 0 1 -9\n")
    out = tmp_path / "copy" / "g"
    genotypes.zero_genetic_distances(tmp_path / "g", out)
    before = (tmp_path / "copy/g.bim").stat().st_mtime_ns
    genotypes.zero_genetic_distances(tmp_path / "g", out)               # nothing changed
    assert (tmp_path / "copy/g.bim").stat().st_mtime_ns == before
    (tmp_path / "g.bim").write_text("1\trs1\t0.49\t999\tA\tG\n")        # source changes
    os.utime(tmp_path / "g.bim", ns=(before + 10**9, before + 10**9))
    genotypes.zero_genetic_distances(tmp_path / "g", out)
    assert "\t999\t" in (tmp_path / "copy/g.bim").read_text()
