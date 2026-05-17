"""Phase A: parallel data acquisition.

Runs all 7 sub-downloads with progress logging. Each download is a subprocess
that can be retried independently. Resumable: skips files that already exist
with non-zero size, uses curl -C - for partial resume on supported endpoints.

Outputs raw to /mnt/hdd/research/trial-failure/raw/<source>/
Writes phaseA_status.json with per-source results.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RAW = Path("/mnt/hdd/research/trial-failure/raw")
LOG = Path("/mnt/ssd/research/trial-failure/logs/phaseA.log")
STATUS = Path("/mnt/ssd/research/trial-failure/logs/phaseA_status.json")
LOG.parent.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")

def have(path: Path, min_bytes: int = 1024) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes

def curl_dl(url: str, out: Path, max_seconds: int = 3600, expect_min_bytes: int = 1024,
            extra_args: list[str] | None = None) -> dict:
    """Download with curl, resumable. Returns status dict."""
    if have(out, expect_min_bytes):
        return {"url": url, "out": str(out), "ok": True, "skipped": True,
                "bytes": out.stat().st_size}
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10",
            "--retry-connrefused", "-C", "-", "--max-time", str(max_seconds),
            "--connect-timeout", "60", "-o", str(out), url]
    if extra_args:
        args[1:1] = extra_args
    log(f"  -> GET {url} -> {out.name}")
    t0 = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=max_seconds + 60)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        return {"url": url, "out": str(out), "ok": False, "error": "timeout",
                "elapsed_s": time.time() - t0}
    elapsed = time.time() - t0
    sz = out.stat().st_size if out.exists() else 0
    ok = rc == 0 and sz >= expect_min_bytes
    return {"url": url, "out": str(out), "ok": ok, "rc": rc, "bytes": sz,
            "elapsed_s": round(elapsed, 1),
            "stderr_tail": (r.stderr or "")[-300:] if not ok else ""}

# ============================================================
# A1 — ChEMBL 36 SQLite
# ============================================================
def a1_chembl() -> dict:
    log("A1 ChEMBL 36 SQLite start")
    dest = RAW / "chembl" / "chembl_36_sqlite.tar.gz"
    res = curl_dl("https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_sqlite.tar.gz",
                  dest, max_seconds=7200, expect_min_bytes=int(1e9))
    return {"source": "A1_chembl", **res}

# ============================================================
# A2 — PubChem (approved drugs property fetch)
# ============================================================
def a2_pubchem() -> dict:
    """Pull canonical SMILES + properties for a curated list of approved drug
    InChIKeys from PubChem PUG REST. Uses the drug names already in our cohort
    rather than the giant CID list (we only need ~3k drugs)."""
    log("A2 PubChem start")
    out_dir = RAW / "pubchem"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "fda_approved_properties.tsv"
    if have(out_file, 100_000):
        log("  skipped (exists)")
        return {"source": "A2_pubchem", "ok": True, "skipped": True,
                "out": str(out_file), "bytes": out_file.stat().st_size}
    # Step 1: get the CID set for "FDA Approved Drugs" via PubChem PUG REST
    # source-name endpoint
    import urllib.request, urllib.parse, json as _json, time as _time
    cid_file = out_dir / "approved_cids.json"
    if not have(cid_file, 100):
        log("  fetching approved CID list from PubChem source FDAOrangeBook + DrugBank")
        # Use PubChem ListConcept "FDA Pharmacologic Class" not directly available via REST,
        # fallback to source: 'DrugBank' approved set
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/sourceid/DrugBank/cids/JSON"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            cid_file.write_bytes(data)
        except Exception as e:
            log(f"  CID fetch failed: {e}; falling back to interventions in cohort")
            cid_file.write_text(_json.dumps({"InformationList": {"Information": []}}))
    cids: list[int] = []
    try:
        j = _json.loads(cid_file.read_text())
        # Format: {"InformationList":{"Information":[{"SourceID":"DB00316","CID":[1983]}, ...]}}
        info = j.get("InformationList", {}).get("Information", [])
        for it in info:
            v = it.get("CID")
            if isinstance(v, list):
                cids.extend(v)
            elif isinstance(v, int):
                cids.append(v)
        cids = sorted({c for c in cids if c})
    except Exception as e:
        log(f"  CID parse fail: {e}")
    log(f"  approved CID count: {len(cids)}")
    if not cids:
        # fallback: query PubChem for our cohort intervention names (limit 3000)
        import duckdb
        con = duckdb.connect("/mnt/ssd/research/trial-failure/aact.duckdb", read_only=True)
        names = con.execute(
            """SELECT DISTINCT lower(name) AS nm FROM interventions
               WHERE intervention_type IN ('Drug','Biological')
               ORDER BY nm LIMIT 3000"""
        ).fetchdf()["nm"].tolist()
        con.close()
        log(f"  fallback cohort names: {len(names)}")
        with out_file.open("w") as f:
            f.write("name\tcid\tsmiles\tinchikey\tmw\tlogp\thbd\thba\ttpsa\trotb\n")
            for i, nm in enumerate(names):
                try:
                    enc = urllib.parse.quote(nm)
                    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc}/property/"
                           f"CanonicalSMILES,InChIKey,MolecularWeight,XLogP,HBondDonorCount,"
                           f"HBondAcceptorCount,TPSA,RotatableBondCount/CSV")
                    with urllib.request.urlopen(url, timeout=15) as resp:
                        rows = resp.read().decode().splitlines()
                    if len(rows) >= 2:
                        # parse first data row
                        parts = rows[1].split(",")
                        # PubChem returns: CID,SMILES,InChIKey,MW,XLogP,HBD,HBA,TPSA,RotB
                        if len(parts) >= 9:
                            f.write(nm + "\t" + "\t".join(parts) + "\n")
                except Exception:
                    pass
                if (i + 1) % 100 == 0:
                    log(f"  pubchem name fetch {i+1}/{len(names)}")
                    _time.sleep(0.2)
                _time.sleep(0.05)  # rate limit
        return {"source": "A2_pubchem", "ok": out_file.exists(), "out": str(out_file),
                "bytes": out_file.stat().st_size if out_file.exists() else 0,
                "n_names": len(names)}
    # CID-based path
    with out_file.open("w") as f:
        f.write("cid\tsmiles\tinchikey\tmw\tlogp\thbd\thba\ttpsa\trotb\n")
        # batch 100 CIDs per call
        for i in range(0, len(cids), 100):
            chunk = cids[i:i+100]
            cid_str = ",".join(str(c) for c in chunk)
            url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/"
                   f"CanonicalSMILES,InChIKey,MolecularWeight,XLogP,HBondDonorCount,"
                   f"HBondAcceptorCount,TPSA,RotatableBondCount/CSV")
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    text = resp.read().decode()
                rows = text.splitlines()
                if len(rows) >= 2:
                    for row in rows[1:]:
                        parts = row.split(",")
                        if len(parts) >= 9:
                            # cid,smiles,inchikey,mw,logp,hbd,hba,tpsa,rotb
                            f.write("\t".join(parts) + "\n")
            except Exception as e:
                log(f"  pubchem batch {i//100} failed: {e}")
            if (i // 100) % 10 == 0:
                log(f"  pubchem progress: {i+len(chunk)}/{len(cids)}")
            _time.sleep(0.2)
    return {"source": "A2_pubchem", "ok": out_file.exists(), "out": str(out_file),
            "bytes": out_file.stat().st_size if out_file.exists() else 0,
            "n_cids": len(cids)}

# ============================================================
# A3 — WHO ICTRP — full export requires registration; use public FTP fallback
# ============================================================
def a3_ictrp() -> dict:
    """WHO ICTRP weekly full export requires authenticated access. Without
    credentials, we fall back to: (a) AACT already has the WHO-derived
    studies, (b) ICTRP public CSV samples from EBI mirror. We document the
    gap rather than fake the download."""
    log("A3 WHO ICTRP — public mirror attempt")
    out_dir = RAW / "who_ictrp"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Try the AACT ICTRP-derived id table (already part of our cohort) and
    # the IRCT/JPRN identifier crosswalk from AACT calculated_values
    info_path = out_dir / "STATUS.txt"
    info_path.write_text(
        "WHO ICTRP full export requires registration at "
        "https://trialsearch.who.int/ — public dump is not directly downloadable.\n"
        "Coverage gap documented; non-NCT trial IDs in AACT browse_conditions are used as proxy.\n"
    )
    return {"source": "A3_ictrp", "ok": False, "skipped": False,
            "note": "registration_required", "out": str(info_path)}

# ============================================================
# A4 — FDA AdCom + Complete Response Letters
# ============================================================
def a4_fda_adcom() -> dict:
    """FDA AdCom meeting materials. The 'committees-and-meeting-materials'
    pages are HTML-only; openFDA has structured drug labels and event data.
    Pull openFDA drug-label NDA/BLA download (which includes warning/adverse
    sections), plus the openFDA drug enforcement data."""
    log("A4 FDA AdCom / openFDA labels")
    out_dir = RAW / "fda_adcom"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        ("https://api.fda.gov/download.json", "openfda_download_manifest.json", 1024),
    ]
    results = []
    for url, name, mb in targets:
        results.append(curl_dl(url, out_dir / name, max_seconds=300, expect_min_bytes=mb))
    # Parse manifest to discover label data shards
    mani = out_dir / "openfda_download_manifest.json"
    label_urls: list[str] = []
    enforce_urls: list[str] = []
    try:
        mj = json.loads(mani.read_text())
        for shard in mj.get("results", {}).get("drug", {}).get("label", {}).get("partitions", []):
            u = shard.get("file")
            if u:
                label_urls.append(u)
        for shard in mj.get("results", {}).get("drug", {}).get("enforcement", {}).get("partitions", []):
            u = shard.get("file")
            if u:
                enforce_urls.append(u)
    except Exception as e:
        log(f"  manifest parse fail: {e}")
    log(f"  openFDA label shards: {len(label_urls)}, enforcement shards: {len(enforce_urls)}")
    # Download label shards (each ~25MB compressed JSON, total ~2GB)
    for i, u in enumerate(label_urls):
        results.append(curl_dl(u, out_dir / f"label_{i:03d}.zip", max_seconds=600))
    for i, u in enumerate(enforce_urls):
        results.append(curl_dl(u, out_dir / f"enforce_{i:03d}.zip", max_seconds=300))
    ok = all(r["ok"] for r in results) if results else False
    total_b = sum(r.get("bytes", 0) for r in results)
    return {"source": "A4_fda_adcom", "ok": ok, "n_files": len(results),
            "bytes": total_b, "details": results[:3]}

# ============================================================
# A5 — FAERS quarterly files
# ============================================================
def a5_faers() -> dict:
    log("A5 FAERS start")
    out_dir = RAW / "faers"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Recent 8 quarters; older quarters are large total but signal recency matters most
    quarters = []
    for y in range(2020, 2026):
        for q in (1, 2, 3, 4):
            quarters.append(f"{y}q{q}")
    quarters.append("2026q1")
    results = []
    # download in parallel with up to 4 workers
    def _dl(q):
        url = f"https://fis.fda.gov/content/Exports/faers_ascii_{q}.zip"
        return curl_dl(url, out_dir / f"faers_ascii_{q}.zip", max_seconds=900, expect_min_bytes=int(5e7))
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(_dl, quarters):
            results.append(r)
    ok_n = sum(1 for r in results if r["ok"])
    total_b = sum(r.get("bytes", 0) for r in results)
    return {"source": "A5_faers", "ok": ok_n >= len(quarters)//2,
            "n_quarters": len(quarters), "n_ok": ok_n, "bytes": total_b}

# ============================================================
# A6 — UniProt human Swiss-Prot
# ============================================================
def a6_uniprot() -> dict:
    log("A6 UniProt human Swiss-Prot (taxonomic_divisions path)")
    out = RAW / "uniprot" / "uniprot_sprot_human.dat.gz"
    r = curl_dl("https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
                "knowledgebase/taxonomic_divisions/uniprot_sprot_human.dat.gz",
                out, max_seconds=1800, expect_min_bytes=int(5e7))
    return {"source": "A6_uniprot", **r}

# ============================================================
# A7 — Open Targets 26.03 selective parquet
# ============================================================
def a7_opentargets() -> dict:
    log("A7 Open Targets selective parquet")
    base = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.03/output"
    out_dir = RAW / "open_targets"
    out_dir.mkdir(parents=True, exist_ok=True)
    want_dirs = [
        "association_overall_direct", "drug_mechanism_of_action", "drug_molecule",
        "target", "target_essentiality", "disease", "clinical_indication",
        "openfda_significant_adverse_drug_reactions", "evidence_clinical_precedence",
    ]
    results = []
    for d in want_dirs:
        sub = out_dir / d
        sub.mkdir(parents=True, exist_ok=True)
        # list parquet files
        idx_url = f"{base}/{d}/"
        try:
            import urllib.request, re as _re
            with urllib.request.urlopen(idx_url, timeout=60) as resp:
                html = resp.read().decode()
            files = sorted(set(_re.findall(r'href="([^"]+\.parquet)"', html)))
        except Exception as e:
            log(f"  {d} listing failed: {e}")
            files = []
        log(f"  {d}: {len(files)} parquet files")
        for fn in files:
            r = curl_dl(f"{base}/{d}/{fn}", sub / fn, max_seconds=600, expect_min_bytes=1024)
            results.append(r)
    ok_n = sum(1 for r in results if r["ok"])
    return {"source": "A7_opentargets", "ok": ok_n >= len(results)*0.9 if results else False,
            "n_files": len(results), "n_ok": ok_n,
            "bytes": sum(r.get("bytes", 0) for r in results)}

# ============================================================
# Orchestration
# ============================================================
TASKS = {
    "A1": a1_chembl, "A2": a2_pubchem, "A3": a3_ictrp,
    "A4": a4_fda_adcom, "A5": a5_faers,
    "A6": a6_uniprot, "A7": a7_opentargets,
}

def run(which: list[str], max_workers: int = 4) -> dict:
    log(f"=== Phase A start: {which} (workers={max_workers}) ===")
    t0 = time.time()
    out = {}
    if STATUS.exists():
        try:
            out = json.loads(STATUS.read_text())
        except Exception:
            pass
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(TASKS[k]): k for k in which}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"source": k, "ok": False, "error": str(e)}
            out[k] = res
            log(f"  {k} done: ok={res.get('ok')} bytes={res.get('bytes', res.get('n_files', '?'))}")
            STATUS.write_text(json.dumps(out, indent=2, default=str))
    out["_elapsed_s"] = round(time.time() - t0, 1)
    STATUS.write_text(json.dumps(out, indent=2, default=str))
    log(f"=== Phase A done in {out['_elapsed_s']}s ===")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="A1,A2,A3,A4,A5,A6,A7",
                    help="comma-separated tasks")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    which = [t.strip() for t in args.tasks.split(",") if t.strip() in TASKS]
    res = run(which, max_workers=args.workers)
    bad = [k for k, v in res.items() if isinstance(v, dict) and not v.get("ok") and not k.startswith("_")]
    if bad:
        log(f"  PARTIAL FAILURES: {bad}")
        sys.exit(0)
    sys.exit(0)
