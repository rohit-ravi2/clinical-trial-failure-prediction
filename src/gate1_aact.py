"""Gate 1: AACT pipe-delimited static export download + DuckDB ingest.

Resumable:
  - skips download if zip exists with expected size
  - skips extraction if extracted dir already populated
  - skips DuckDB load for any table already present with rowcount > 0

Outputs:
  - /mnt/hdd/research/trial-failure/raw/aact/aact.zip
  - /mnt/hdd/research/trial-failure/raw/aact/extracted/*.txt
  - /mnt/ssd/research/trial-failure/aact.duckdb
  - /mnt/ssd/research/trial-failure/logs/gates.log (append)
  - /mnt/ssd/research/trial-failure/logs/gate1_aact.log

Pass criterion: studies row count > 400,000.
Hard stop if studies < 350,000.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import requests
from tqdm import tqdm

# ---- paths -----------------------------------------------------------------
RAW_DIR = Path("/mnt/hdd/research/trial-failure/raw/aact")
EXTRACT_DIR = RAW_DIR / "extracted"
ZIP_PATH = RAW_DIR / "aact.zip"
DUCKDB_PATH = Path("/mnt/ssd/research/trial-failure/aact.duckdb")
LOGS_DIR = Path("/mnt/ssd/research/trial-failure/logs")
GATES_LOG = LOGS_DIR / "gates.log"
GATE1_LOG = LOGS_DIR / "gate1_aact.log"

REQUIRED_TABLES = [
    "studies",
    "interventions",
    "intervention_other_names",
    "outcomes",
    "drop_withdrawals",
    "reported_events",
    "sponsors",
    "browse_conditions",
    "browse_interventions",
    "calculated_values",
]

# NOTE: The bucket path `static/static_db_copies/` serves a *PostgreSQL custom
# dump* (postgres.dmp v1.14-0) regardless of the requested filename — it's a
# misnomer. The genuine pipe-delimited export lives under
# `static/exported_files/`. We hit that URL directly.
URL_TEMPLATES = [
    "https://aact.ctti-clinicaltrials.org/static/exported_files/monthly/{date}_pipe-delimited-export.zip",
    "https://aact.ctti-clinicaltrials.org/static/exported_files/daily/{date}_pipe-delimited-export.zip",
]

# ---- logging ---------------------------------------------------------------
RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(GATE1_LOG, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("gate1")


def append_gate_line(line: str) -> None:
    with GATES_LOG.open("a") as f:
        f.write(line.rstrip() + "\n")


# ---- snapshot discovery ----------------------------------------------------
def discover_snapshot_url() -> tuple[str, str]:
    """Return (url, snapshot_date_str) for the most recent available snapshot.

    Strategy: try monthly snapshots from the current month back 6 months, then
    daily snapshots from today back 14 days. Returns the first HEAD-200 hit.
    """
    today = date.today()
    candidates: list[tuple[str, date]] = []
    for m_back in range(0, 6):
        y = today.year
        m = today.month - m_back
        while m <= 0:
            m += 12
            y -= 1
        candidates.append(("monthly", date(y, m, 1)))
    for d_back in range(0, 14):
        candidates.append(("daily", today - timedelta(days=d_back)))

    for kind, d in candidates:
        ds = d.strftime("%Y%m%d")
        for tmpl in URL_TEMPLATES:
            if kind not in tmpl:
                continue
            url = tmpl.format(date=ds)
            try:
                r = requests.head(url, timeout=15, allow_redirects=True)
                if r.status_code == 200:
                    size = r.headers.get("Content-Length")
                    log.info("snapshot found: kind=%s date=%s size=%s url=%s",
                             kind, ds, size, url)
                    return url, ds
            except Exception as e:
                log.debug("HEAD failed for %s: %s", url, e)
    raise RuntimeError("Could not locate any AACT pipe-delimited snapshot URL")


# ---- download --------------------------------------------------------------
def download_zip(url: str) -> int:
    """Stream-download with progress bar. Returns expected size in bytes.

    Resumes if file already exists and matches Content-Length.
    """
    head = requests.head(url, timeout=30, allow_redirects=True)
    head.raise_for_status()
    expected = int(head.headers.get("Content-Length", "0"))

    if ZIP_PATH.exists():
        have = ZIP_PATH.stat().st_size
        if expected and have == expected:
            log.info("zip already present and matches size (%d bytes); skipping download", have)
            return expected
        log.warning("existing zip size=%d != expected=%d; re-downloading", have, expected)
        ZIP_PATH.unlink()

    log.info("downloading %s (%.2f GB) -> %s", url, expected / 1e9, ZIP_PATH)
    chunk = 1 << 20  # 1 MiB
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with ZIP_PATH.open("wb") as f, tqdm(
            total=expected, unit="B", unit_scale=True, unit_divisor=1024,
            desc="aact.zip"
        ) as bar:
            for piece in r.iter_content(chunk_size=chunk):
                if not piece:
                    continue
                f.write(piece)
                bar.update(len(piece))
    got = ZIP_PATH.stat().st_size
    if expected and got != expected:
        raise RuntimeError(f"download size mismatch: got {got}, expected {expected}")
    log.info("download complete: %d bytes", got)
    return got


# ---- extract ---------------------------------------------------------------
def extract_zip() -> list[Path]:
    """Extract pipe files. Skip if extracted dir already has >= 30 .txt files."""
    existing = sorted(EXTRACT_DIR.glob("*.txt"))
    if len(existing) >= 30:
        log.info("extracted dir already has %d .txt files; skipping extraction",
                 len(existing))
        return existing

    log.info("extracting %s -> %s", ZIP_PATH, EXTRACT_DIR)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = zf.namelist()
        log.info("zip contains %d members", len(members))
        for m in tqdm(members, desc="extract"):
            zf.extract(m, EXTRACT_DIR)
    # Flatten if extraction created a single subdir
    contents = list(EXTRACT_DIR.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        sub = contents[0]
        log.info("flattening single subdir %s", sub.name)
        for p in sub.iterdir():
            p.rename(EXTRACT_DIR / p.name)
        sub.rmdir()
    files = sorted(EXTRACT_DIR.glob("*.txt"))
    log.info("extraction produced %d .txt files", len(files))
    return files


# ---- DuckDB load -----------------------------------------------------------
def existing_tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    return {r[0] for r in rows}


def load_table(con: duckdb.DuckDBPyConnection, table: str, txt_path: Path) -> int:
    """Drop+recreate the table from the pipe file. Returns rowcount."""
    log.info("loading %s from %s (%.1f MB)", table, txt_path.name,
             txt_path.stat().st_size / 1e6)
    con.execute(f'DROP TABLE IF EXISTS "{table}"')
    # AACT uses '|' delim, header row, double-quote literal '\\N' or empty for null.
    # sample_size=-1 forces full scan for robust type inference; AACT files have
    # awkward free-text columns so we let DuckDB fall back to VARCHAR where unsure.
    sql = f"""
        CREATE TABLE "{table}" AS
        SELECT * FROM read_csv_auto(
            '{txt_path.as_posix()}',
            delim='|',
            header=True,
            quote='"',
            escape='"',
            sample_size=-1,
            ignore_errors=True,
            null_padding=True,
            strict_mode=False
        )
    """
    t0 = time.time()
    con.execute(sql)
    n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    log.info("  -> %s: %d rows in %.1fs", table, n, time.time() - t0)
    return n


def load_into_duckdb(files: list[Path]) -> dict[str, int]:
    name_to_path: dict[str, Path] = {p.stem: p for p in files}
    present_required = [t for t in REQUIRED_TABLES if t in name_to_path]
    missing = [t for t in REQUIRED_TABLES if t not in name_to_path]
    log.info("required tables present in snapshot: %d/%d (missing: %s)",
             len(present_required), len(REQUIRED_TABLES), missing)

    con = duckdb.connect(DUCKDB_PATH.as_posix())
    # Pragma: keep memory bounded so other agents can coexist
    con.execute("PRAGMA memory_limit='24GB'")
    con.execute("PRAGMA threads=8")

    have_tables = existing_tables(con)
    counts: dict[str, int] = {}
    for t in present_required:
        if t in have_tables:
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            if n > 0:
                log.info("  table %s already loaded (%d rows); skipping", t, n)
                counts[t] = n
                continue
        try:
            counts[t] = load_table(con, t, name_to_path[t])
        except Exception as e:
            log.exception("failed to load %s: %s", t, e)
            raise
    for t in missing:
        counts[t] = -1  # sentinel: not present in snapshot
    con.close()
    return counts


# ---- main ------------------------------------------------------------------
def main() -> int:
    t_start = time.time()
    log.info("=" * 70)
    log.info("GATE 1 start")

    url, snapshot_date = discover_snapshot_url()
    download_zip(url)
    files = extract_zip()
    if not files:
        raise RuntimeError("no .txt files in extracted dir")

    counts = load_into_duckdb(files)

    n_studies = counts.get("studies", 0)
    log.info("studies row count: %d", n_studies)

    extracted_size = sum(p.stat().st_size for p in EXTRACT_DIR.glob("*.txt"))
    duckdb_size = DUCKDB_PATH.stat().st_size if DUCKDB_PATH.exists() else 0
    log.info("extracted size: %.2f GB; duckdb size: %.2f GB",
             extracted_size / 1e9, duckdb_size / 1e9)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    counts_str = ",".join(f"{t}={counts.get(t, -1)}" for t in REQUIRED_TABLES)

    if n_studies < 350_000:
        line = (f"[{ts}] GATE_1: FAIL (HARD_STOP) snapshot={snapshot_date} "
                f"studies={n_studies} (<350k) counts={counts_str}")
        append_gate_line(line)
        log.error(line)
        return 2

    if n_studies < 400_000:
        line = (f"[{ts}] GATE_1: WARN snapshot={snapshot_date} "
                f"studies={n_studies} (<400k, >=350k) counts={counts_str}")
        append_gate_line(line)
        log.warning(line)
        return 1

    line = (f"[{ts}] GATE_1: PASS snapshot={snapshot_date} "
            f"studies={n_studies} counts={counts_str} "
            f"extracted_gb={extracted_size/1e9:.2f} duckdb_gb={duckdb_size/1e9:.2f} "
            f"elapsed_s={int(time.time()-t_start)}")
    append_gate_line(line)
    log.info(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
