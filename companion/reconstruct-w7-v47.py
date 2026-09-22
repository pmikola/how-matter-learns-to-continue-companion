"""Read-only reconstruction of the already exposed E3 W7 response block.

Python 3.10+; standard library only. From a delivered companion, run:
    python -B reconstruct-w7-v47.py
The default emits JSON to stdout and writes nothing. --output is opt-in.
No training, refitting, new confirmation block, or historical campaign execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys
import zipfile


PROTOCOL = "omega.xfam.e3.shared_response_law.v0_1"
SOURCE_COMMIT = "cb0de6317622aa66813801d8534d6d9373536a92"
MANIFEST_SHA256 = "3750b736c09efd0aaa7bc512a74de06383f4b53458e287b6bfefd348eb81556a"
SOURCE_BINDINGS = {
    "config": {
        "path": "configs/experiments/xfam_e3_shared_response_law_v0_1.json",
        "sha256": "ad3923b5438c3bf17e70a9d72d9044e3db32d10191c73bb301b3299d7317ba69",
    },
    "implementation": {
        "path": "src/omega_protocol/experiments/xfam_e3_shared_response_law_v1.py",
        "sha256": "b8ae2db507509e915f202414f22073ee0e7c5a8231dd30ab9123b93d5216a146",
        "functions": ["balanced_recoding_closed_probes", "_reflect_state", "_w7_response_count", "_fit_matrix", "_fit_disjoint_intercepts"],
    },
    "experiment_common": {
        "path": "src/omega_protocol/experiments/universe_targets_common_v1.py",
        "sha256": "f35bcfcde063ecc626aae3fd4c141bb304d0bb6edaaa7385780e448931f2e0fa",
        "functions": ["canonical_json_bytes", "stable_uint64"],
    },
}
CLASSES = (140, 24, 44, 73, 34, 8, 6, 13, 128, 178, 5, 142, 32, 57, 74,
           15, 150, 164, 38, 14, 58, 106, 132, 152, 33, 12, 126, 108, 146, 3)
EXPECTED_PROBES = (933770067, 1390568821, 1565298325, 1785189414,
                   2509777881, 2729668970, 2904398474, 3361197228)
PROBE_SEED = 2026082621
SIZE = 32
HORIZON = 32
DESCRIPTORS = ("mean_local_influence", "normalized_neighborhood_size", "determinism_margin")
EXPECTED_MSE = {
    "primary": 0.04651764075889651,
    "constant": 0.022214819625990624,
    "memory": 0.011103334688740193,
    "family_aware": 0.01693321813870276,
}
PREDICTION_KEYS = {"primary": "law_prediction", "constant": "floor_prediction",
                   "memory": "memory_prediction", "family_aware": "disjoint_union_prediction"}


def canonical_json(value):
    """Historical framing: ASCII JSON, sorted keys, compact separators, no LF."""
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("ascii")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def stable_key(*parts):
    return int.from_bytes(hashlib.sha256(b"\0".join(canonical_json(p) for p in parts)).digest()[:8], "big")


def reflect(state, size):
    return sum(((state >> site) & 1) << ((-site) % size) for site in range(size))


def generate_probes():
    """Translate the inspected historical generator; do not search for fit."""
    mask = (1 << SIZE) - 1
    probes = set()
    for candidate_index in range(128):
        ordered = sorted(range(SIZE), key=lambda site: (
            stable_key(PROTOCOL, PROBE_SEED, candidate_index, site, "balanced_probe"), site))
        state = sum(1 << site for site in ordered[:SIZE // 2])
        reversed_state = reflect(state, SIZE)
        orbit = {state, state ^ mask, reversed_state, reversed_state ^ mask}
        if len(orbit) != 4 or probes.intersection(orbit):
            continue
        probes.update(orbit)
        if len(probes) == 8:
            return tuple(sorted(probes))
    raise ValueError("Probe generation failed within its declared candidate bound")


def evolve_cells(rule, state, size=SIZE, horizon=HORIZON):
    """Implementation A: explicit cell array and rule truth table."""
    cells = [(state >> site) & 1 for site in range(size)]
    for _ in range(horizon):
        cells = [(rule >> (4 * cells[(site - 1) % size] + 2 * cells[site]
                           + cells[(site + 1) % size])) & 1 for site in range(size)]
    return sum(value << site for site, value in enumerate(cells))


def evolve_bits(rule, state, size=SIZE, horizon=HORIZON):
    """Implementation B: rotate packed bits and form Boolean minterms."""
    mask = (1 << size) - 1
    for _ in range(horizon):
        left = ((state << 1) | (state >> (size - 1))) & mask
        right = ((state >> 1) | (state << (size - 1))) & mask
        result = 0
        for pattern in range(8):
            if (rule >> pattern) & 1:
                result |= ((left if pattern & 4 else left ^ mask)
                           & (state if pattern & 2 else state ^ mask)
                           & (right if pattern & 1 else right ^ mask))
        state = result
    return state


def response_count(rule, probes, evolve):
    changed = 0
    mask = (1 << SIZE) - 1
    for probe in probes:
        baseline = evolve(rule, probe)
        for site in range(SIZE):
            changed += ((baseline ^ evolve(rule, probe ^ (1 << site)))
                        & (mask ^ (1 << site))).bit_count()
    return changed


def descriptor_values(rule):
    influence = sum(((rule >> pattern) & 1) != ((rule >> (pattern ^ flip)) & 1)
                    for pattern in range(8) for flip in (4, 1)) / 16
    return (influence, 2 / (SIZE - 1), 0.0)


def frozen_predictions(models, values):
    vector = (1.0, *values)
    centers = models["quantized_centroid_memory"]["centroids"]
    # min returns the first centre on a distance tie, as the archived procedure.
    nearest = min(centers, key=lambda center: sum((a - b) ** 2 for a, b in zip(values, center[:3])))
    return {
        "primary": math.fsum(a * b for a, b in zip(models["primary_law"]["coefficients"], vector)),
        "constant": models["constant_floor"]["prediction"],
        "memory": nearest[3],
        "family_aware": math.fsum(a * b for a, b in zip(
            models["disjoint_family_intercepts"]["coefficients"], (*vector, 0.0, 0.0))),
    }


def rule150_diagnostic():
    """Later exact diagnostic; never substitute its horizons into E3 errors."""
    table_xor = all(((150 >> value) & 1) == (value.bit_count() % 2) for value in range(8))
    identity = all(evolve(150, 1 << site, horizon=16) == 1 << site
                   for evolve in (evolve_cells, evolve_bits) for site in range(SIZE))
    records = []
    for time, expected in ((31, 20), (32, 0), (33, 2)):
        bits = evolve_bits(150, 1, horizon=time)
        cells = evolve_cells(150, 1, horizon=time)
        other_count = (bits & (((1 << SIZE) - 1) ^ 1)).bit_count()
        if bits != cells or other_count != expected:
            raise ValueError("Rule 150 neighboring-time calculation disagrees")
        records.append({"update": time, "difference_mask_site0_lsb": bits,
                        "different_sites": [site for site in range(SIZE) if (bits >> site) & 1],
                        "different_other_sites": other_count,
                        "response_fraction": f"{other_count}/31",
                        "response": other_count / 31})
    return {"status": "later_exposed_rule_diagnostic", "ring_size": SIZE,
            "rule": 150, "initial_difference_site": 0,
            "truth_table_is_xor": table_xor, "both_implementations_F16_identity_on_32_basis": identity,
            "algebra": "F=S^-1+I+S over F2; F^16=S^-16+I+S^16=I on a 32-ring; all state periods divide 16.",
            "records": records,
            "registered_endpoint_unchanged": 32,
            "pass": table_xor and identity}


class Evidence:
    """Read only from a supplied evidence directory or companion ZIP."""

    def __init__(self, path):
        self.path = path.resolve()
        self.archive = None
        self.prefix = ""
        if self.path.is_file():
            self.archive = zipfile.ZipFile(self.path, "r")
            matches = [name for name in self.archive.namelist() if name.endswith("evidence/e3/manifest.json")]
            if len(matches) != 1:
                self.archive.close()
                raise ValueError("ZIP must contain exactly one evidence/e3/manifest.json")
            self.prefix = matches[0][:-len("manifest.json")]

    def read(self, relative):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError("Unsafe evidence-relative path")
        if self.archive:
            return self.archive.read(self.prefix + relative)
        return self.path.joinpath(*path.parts).read_bytes()

    def close(self):
        if self.archive:
            self.archive.close()


def reconstruct(evidence_path):
    evidence = Evidence(evidence_path)
    try:
        manifest_raw = evidence.read("manifest.json")
        if sha256(manifest_raw) != MANIFEST_SHA256:
            raise ValueError("Historical E3 manifest bytes do not match their pinned digest")
        manifest = json.loads(manifest_raw)
        if manifest["protocol_id"] != PROTOCOL or manifest["git_commit"] != SOURCE_COMMIT:
            raise ValueError("Historical identity mismatch")
        if manifest["outcome"] != "e3_transfer_null" or manifest["scientific_admission"] is not False:
            raise ValueError("Historical negative outcome changed")
        artifact_checks = []
        for item in manifest["artifacts"]:
            data = evidence.read(item["relative_path"])
            ok = len(data) == item["byte_count"] and sha256(data) == item["sha256"]
            artifact_checks.append({**item, "pass": ok})
            if not ok:
                raise ValueError(f"Frozen evidence failed integrity check: {item['relative_path']}")
        if len(artifact_checks) != 8:
            raise ValueError("Expected eight historical artifacts")
        for key, binding in SOURCE_BINDINGS.items():
            if manifest["freeze_byte_hashes"][key] != binding["sha256"]:
                raise ValueError(f"Historical source hash mismatch: {key}")
        commitment = json.loads(evidence.read("controls/probe_commitment.json"))
        coefficient_raw = evidence.read("controls/coefficient_freeze.json")
        models = json.loads(coefficient_raw)["models"]
        rows = [json.loads(line) for line in evidence.read("analysis/per_class.jsonl").splitlines()]
        development = json.loads(evidence.read("controls/development_receipt.json"))["analysis"]
        aggregate = json.loads(evidence.read("analysis/aggregate.json"))
    finally:
        evidence.close()

    probes = generate_probes()
    probe_digest = sha256(canonical_json(list(probes)))
    probe_match = probe_digest == commitment["probe_set_sha256"] == development["controls"]["probe_set_sha256"]
    closed = {image for state in probes for image in (state ^ ((1 << SIZE) - 1), reflect(state, SIZE))} == set(probes)
    if probes != EXPECTED_PROBES or not probe_match or not closed or any(p.bit_count() != 16 for p in probes):
        raise ValueError("Generated probes fail original commitment or balance/closure")
    if tuple(row["class_representative"] for row in rows) != CLASSES:
        raise ValueError("Named class block or order differs from the historical config")
    for name in ("primary_law", "disjoint_family_intercepts"):
        model = models[name]
        if struct.pack("<" + "d" * len(model["coefficients"]), *model["coefficients"]).hex() != model["float64_le_hex"]:
            raise ValueError("Frozen coefficient decimal and IEEE payload disagree")
    family_model = models["disjoint_family_intercepts"]
    if (len(family_model["coefficients"]), family_model["rank"], family_model["coefficients_identified"]) != (6, 5, False):
        raise ValueError("Family-aware design disclosure does not match the archive")
    if (models["primary_law"]["rank"], models["primary_law"]["coefficients_identified"]) != (4, True):
        raise ValueError("Primary design disclosure does not match the archive")

    comparison_count = len(probes) * SIZE * (SIZE - 1)
    reconstructed = []
    squared_errors = {name: [] for name in EXPECTED_MSE}
    for row in rows:
        rule = row["class_representative"]
        cell_count = response_count(rule, probes, evolve_cells)
        packed_count = response_count(rule, probes, evolve_bits)
        values = descriptor_values(rule)
        response = cell_count / comparison_count
        if cell_count != packed_count or cell_count != row["response_changed_count"]:
            raise ValueError(f"Terminal count disagreement for class {rule}")
        if (row["horizon"], row["ring_size"], row["probe_count"], row["response_comparison_count"]
                ) != (HORIZON, SIZE, len(probes), comparison_count):
            raise ValueError(f"Protocol dimensions differ for class {rule}")
        if values != tuple(row[key] for key in DESCRIPTORS) or response != row["response"]:
            raise ValueError(f"Descriptor or response mismatch for class {rule}")
        predictions = frozen_predictions(models, values)
        for name, prediction in predictions.items():
            if not math.isclose(prediction, row[PREDICTION_KEYS[name]], rel_tol=0, abs_tol=2e-15):
                raise ValueError(f"Frozen {name} prediction differs for class {rule}")
            squared_errors[name].append((prediction - response) ** 2)
        reconstructed.append({"class_representative": rule, "array_changed_count": cell_count,
                              "bit_parallel_changed_count": packed_count,
                              "archived_changed_count": row["response_changed_count"],
                              "comparison_count": comparison_count, "response": response,
                              "descriptors": list(values), "predictions": predictions})
    mse = {name: math.fsum(values) / len(rows) for name, values in squared_errors.items()}
    aggregate_keys = {"primary": "mse_law", "constant": "mse_floor", "memory": "mse_memory", "family_aware": "mse_disjoint_union"}
    for name, value in mse.items():
        if not math.isclose(value, EXPECTED_MSE[name], rel_tol=0, abs_tol=2e-15):
            raise ValueError(f"Headline MSE disagrees: {name}")
        if not math.isclose(value, aggregate["metrics"][aggregate_keys[name]], rel_tol=0, abs_tol=2e-15):
            raise ValueError(f"Archived MSE disagrees: {name}")
    diagnostic = rule150_diagnostic()
    if not diagnostic["pass"]:
        raise ValueError("Rule 150 proof checks failed")
    return {
        "schema": "hmltc.w7_reconstruction.v47.1", "status": "passed",
        "evidence": str(evidence_path.resolve()), "script_sha256": sha256(Path(__file__).read_bytes()),
        "historical_outcome": "e3_transfer_null", "scientific_admission": False,
        "historical_manifest_sha256": sha256(manifest_raw), "artifact_checks": artifact_checks,
        "source_specification": {
            "commit": SOURCE_COMMIT, "protocol_id": PROTOCOL, "source_bindings": SOURCE_BINDINGS,
            "source_check_scope": "Embedded source hashes agree with the archived freeze; upstream source bytes are not fetched by this reader command.",
            "probe_seed": PROBE_SEED, "ring_size": SIZE, "horizon": HORIZON,
            "class_order": list(CLASSES),
            "stable_key": "SHA256 of null-separated ASCII canonical JSON parts; first eight digest bytes as big-endian unsigned integer",
            "key_parts": ["protocol_id", "probe_seed", "candidate_index", "site", "balanced_probe"],
            "probe_generator": "Rank sites by (stable_key,site); set first sixteen; reflect site i to -i mod32; form complement/reflection orbit; skip non-four or intersecting orbits; stop at eight; sort integers.",
            "candidate_indices": "0 upwards; historical finite rejection bound 128",
            "perturbation": "Flip each of all32 initial sites in each probe; synchronous periodic updates; exclude that site's terminal disagreement.",
        },
        "probes": list(probes), "probe_set_sha256": probe_digest,
        "original_probe_commitment_matched": probe_match, "probes_balanced_and_recoding_closed": closed,
        "frozen_models": models, "mse": mse, "reconstructed_rows": reconstructed,
        "family_aware_disclosure": {
            "coefficient_count": 6, "rank": 5, "coefficients_identified": False,
            "selection_convention": "numpy.linalg.lstsq(weighted_matrix, weighted_response, rcond=None); minimum Euclidean coefficient norm",
            "training_family_neighborhood_support": development["commensurability_diagnostics"]["per_family_descriptor_support"],
            "primary_rank": 4, "primary_coefficients_identified": True,
        },
        "rule150_diagnostic": diagnostic,
        "checks": {"all_pass": True, "all_30_counts_match_both_implementations_and_archive": True,
                   "all_frozen_predictions_match": True, "all_eight_frozen_artifacts_match": True},
        "limits": ["No original training materialization or fitting run reproduced.",
                   "No bootstrap, descriptor-sham or full recoding-control campaign rerun.",
                   "Class-selection derivation remains unrecoverable; no random population coverage claimed.",
                   "This reconstructs already exposed cases and is not a new confirmatory result.",
                   "Frozen files are read unchanged; a matching probe digest binds probes, not all historical execution bytes."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parent / "evidence" / "e3",
                        help="Frozen evidence/e3 directory, or an existing companion ZIP")
    parser.add_argument("--output", type=Path, help="Optional new JSON report path; default prints only")
    args = parser.parse_args(argv)
    try:
        report = reconstruct(args.evidence)
        content = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        if args.output:
            output = args.output.resolve()
            source = args.evidence.resolve()
            if output == source or (source.is_dir() and source in output.parents):
                raise ValueError("Output must be outside the frozen evidence directory")
            output.parent.mkdir(parents=True, exist_ok=True)
            # Explicit output is still non-overwriting, preserving every earlier receipt.
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
            print(json.dumps({"status": "passed", "output": str(output), "checks": report["checks"]}))
        else:
            print(content, end="")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
