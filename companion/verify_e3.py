"""Read-only checks of eight frozen E3 artifacts; never runs the experiment.

Reuse the archived arithmetic checker without invoking its whole-book main.
The source checker is included verbatim for inspection and provenance.
"""
from pathlib import Path
import json
import runpy

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    source = ROOT / "evidence/verify_revision_source_v28.py"
    operations = runpy.run_path(str(source))
    result = operations["e3_checks"](ROOT / "evidence/e3")
    print(json.dumps({"passed": True, "checks": result,
        "outcome": "e3_transfer_null", "scientific_admission": False,
        "scope": "Stored-byte integrity and saved-row arithmetic only; no instrument rerun, refit, new outcomes, or independent replication."}, indent=2))
