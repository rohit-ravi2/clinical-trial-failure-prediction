"""Gate 0: Environment setup and import verification.

Verifies the trial-failure venv has all required packages installed and importable.
Writes a PASS/FAIL line to work/logs/gates.log.

Usage:
    source ~/venvs/trial-failure/bin/activate
    python work/src/gate0_setup.py
"""
from __future__ import annotations

import datetime as _dt
import importlib
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SSD_ROOT = Path("work")
HDD_ROOT = Path("work")
LOG_DIR = SSD_ROOT / "logs"
GATES_LOG = LOG_DIR / "gates.log"
REQUIREMENTS_PATH = SSD_ROOT / "requirements.txt"

REQUIRED_SUBDIRS_SSD = ["src", "processed", "logs", "results", "models", "embeddings"]
REQUIRED_SUBDIRS_HDD = ["raw"]

REQUIRED_PACKAGES = [
    # (import name, pretty name)
    ("duckdb", "duckdb"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("requests", "requests"),
    ("tqdm", "tqdm"),
    ("rdkit", "rdkit"),
    ("chembl_webresource_client", "chembl-webresource-client"),
    ("pyarrow", "pyarrow"),
    ("matplotlib", "matplotlib"),
    ("optuna", "optuna"),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gate0")


def ensure_dirs() -> None:
    for sd in REQUIRED_SUBDIRS_SSD:
        (SSD_ROOT / sd).mkdir(parents=True, exist_ok=True)
    for sd in REQUIRED_SUBDIRS_HDD:
        (HDD_ROOT / sd).mkdir(parents=True, exist_ok=True)
    log.info("Verified directory tree under %s and %s", SSD_ROOT, HDD_ROOT)


def verify_imports() -> list[tuple[str, str, str]]:
    """Return list of (pretty_name, version, status). Raises on any failure."""
    results = []
    for mod_name, pretty in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "unknown")
            log.info("import OK: %-30s version=%s", pretty, version)
            results.append((pretty, version, "OK"))
        except Exception as exc:  # noqa: BLE001
            log.error("import FAIL: %s -- %s", pretty, exc)
            results.append((pretty, "n/a", f"FAIL: {exc}"))
    # rdkit sanity
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    if mol is None or mol.GetNumAtoms() != 3:
        raise RuntimeError("rdkit sanity failed: CCO did not parse to 3 atoms")
    log.info("rdkit sanity: parsed CCO -> 3 atoms OK")
    failures = [r for r in results if not r[2].startswith("OK")]
    if failures:
        raise RuntimeError(f"Import failures: {failures}")
    return results


def append_gate_log(line: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    with GATES_LOG.open("a") as f:
        f.write(f"[{ts}] {line}\n")


def main() -> int:
    log.info("=== Gate 0: Environment setup ===")
    log.info("Python: %s", sys.version.split()[0])
    log.info("Executable: %s", sys.executable)
    try:
        ensure_dirs()
        results = verify_imports()
    except Exception as exc:  # noqa: BLE001
        log.exception("Gate 0 FAILED")
        append_gate_log(f"GATE_0: FAIL ({exc})")
        return 1
    n_ok = sum(1 for r in results if r[2] == "OK")
    append_gate_log(
        f"GATE_0: PASS ({n_ok}/{len(results)} packages imported; python={sys.version.split()[0]}; venv={sys.executable})"
    )
    log.info("Gate 0 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
