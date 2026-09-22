"""Read-only consistency check of an inherited framework forward-check record.

This verifies receipt, environment, saved-array and printed-snippet bindings.
It does not import frameworks, execute their snippets, or reproduce training.
The companion builder copies this file as verify_framework_record.py.
"""
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math
import re

RECEIPT = "reviews/publication-v34/framework-verification.json"
LOCK = "reviews/publication-v34/environment-lock.txt"
ARCHIVE = "outputs/chapter-one-draft/v3/experiment/data-and-model.npz"
SNIPPETS = "evidence/framework-forward-snippets.json"
FRAMEWORKS = ("torch", "tensorflow", "jax", "keras")
LABELS = {"torch": "PyTorch", "tensorflow": "TensorFlow", "jax": "JAX", "keras": "Keras"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return sha256(content).hexdigest()


def read_record(root):
    record = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
    require(set(record["frameworks"]) == set(FRAMEWORKS), "Framework receipt must name all four frameworks")
    require(record["archive_unchanged"] is True, "Historical receipt does not attest unchanged saved arrays")
    require(digest((root / ARCHIVE).read_bytes()) == record["archive_sha256"], "Framework saved-array hash mismatch")
    locked = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s]+)$", (root / LOCK).read_text(encoding="utf-8"), re.M))
    for name in FRAMEWORKS:
        row = record["frameworks"][name]
        require(row["status"] == "executed", name + " lacks an executed forward check")
        require(row["backend"] == "CPU" and row["test_rows"] == 600 and row["score_shape"] == [600, 1], name + " scope mismatch")
        expected_dtype = "float32" if name == "keras" else "float64"
        require(row["dtype"] == expected_dtype, name + " unexpected score dtype")
        error = row["max_absolute_error_against_numpy"]
        require(isinstance(error, (int, float)) and math.isfinite(error) and 0 <= error < (1e-5 if name == "keras" else 1e-10), name + " forward error outside recorded acceptance bound")
        lock_name = "tensorflow_cpu" if name == "tensorflow" else name
        require(locked.get(lock_name) == row["version"].split("+")[0], name + " version differs from historical environment lock")
        require("Forward score equivalence only" in row["scope"], name + " missing forward-only limit")
        require(re.fullmatch(r"[0-9a-f]{64}", row["snippet_sha256"]) is not None, name + " invalid snippet hash")
    return record


def summary(record):
    lines = [
        "The inherited v34 receipt records CPU execution of all four printed forward examples on 600 saved test inputs with copied parameters:",
        "",
        "| Framework | Recorded version | Score dtype | Maximum absolute score error against NumPy |",
        "|---|---|---|---:|",
    ]
    for name in FRAMEWORKS:
        row = record["frameworks"][name]
        lines.append(f"| {LABELS[name]} | {row['version']} | {row['dtype']} | {row['max_absolute_error_against_numpy']:.17g} |")
    lines.extend([
        "",
        f"The unchanged execution receipt and its environment lock are included at `{RECEIPT}` and `{LOCK}`. Packaging checks the saved-array hash and verifies that the v40 printed snippets have the receipt's exact hashes. `verify_framework_record.py` rechecks those bindings and this table after extraction. These are inherited execution results; v40 packaging does not execute frameworks. They establish the reported forward-score agreement, not training or gradient equivalence. The portable reader requirements do not install frameworks.",
    ])
    return "\n".join(lines)


def snippet_record(root, source):
    record = read_record(root)
    text = source.read_text(encoding="utf-8")
    snippets = re.findall(r"\\begin\{lstlisting\}\[language=Python\]\n(.*?)\\end\{lstlisting\}", text, re.S)
    selected = {}
    for name in FRAMEWORKS:
        matches = [snippet for snippet in snippets if f"import {name}" in snippet]
        require(len(matches) == 1, name + " snippet must be unique in final notes")
        selected[name] = matches[0]
        require(digest(matches[0].encode()) == record["frameworks"][name]["snippet_sha256"], name + " v40 snippet differs from inherited forward check")
    return {
        "edition": "v40",
        "source": source.relative_to(root).as_posix(),
        "source_sha256": digest(source.read_bytes()),
        "execution_receipt": RECEIPT,
        "execution_receipt_sha256": digest((root / RECEIPT).read_bytes()),
        "environment_lock_sha256": digest((root / LOCK).read_bytes()),
        "frameworks_executed_by_v40_builder": False,
        "snippets": selected,
        "scope": "Exact v40 snippet and saved-array compatibility with an unchanged historical forward-only execution receipt.",
    }


def verify(root):
    record = read_record(root)
    binding = json.loads((root / SNIPPETS).read_text(encoding="utf-8"))
    require(binding["edition"] == "v40" and binding["frameworks_executed_by_v40_builder"] is False, "Incorrect edition or execution attribution")
    require(binding["execution_receipt"] == RECEIPT, "Wrong historical receipt reference")
    require(binding["execution_receipt_sha256"] == digest((root / RECEIPT).read_bytes()), "Historical receipt hash mismatch")
    require(binding["environment_lock_sha256"] == digest((root / LOCK).read_bytes()), "Environment lock hash mismatch")
    require(set(binding["snippets"]) == set(FRAMEWORKS), "Missing packaged snippet")
    for name in FRAMEWORKS:
        require(digest(binding["snippets"][name].encode()) == record["frameworks"][name]["snippet_sha256"], name + " packaged snippet differs from historical check")
    readme = (root / "README.md").read_text(encoding="utf-8")
    require(summary(record) in readme, "README framework table is not generated from the included receipt")
    require("TensorFlow and Keras were unavailable" not in readme, "Stale framework availability claim")
    manifest = json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))
    require(manifest["compatible_edition"]["edition"] == "v40", "Companion PDF edition is not v40")
    return {
        "passed": True,
        "framework_count": len(FRAMEWORKS),
        "frameworks_executed_in_this_check": False,
        "receipt": RECEIPT,
        "receipt_sha256": binding["execution_receipt_sha256"],
        "archive_sha256": record["archive_sha256"],
        "snippets_match_recorded_execution": True,
        "readme_matches_receipt": True,
        "environment_versions_match_receipt": True,
        "scope": "Record and source compatibility only; inherited CPU forward-score checks, no fresh framework run or training.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2))
