"""Bounded, read-only v28 regression audit; writes JSON to stdout only.

No model fitting, campaign imports, outcome generation, bootstrap, manuscript
editing, or visual certification. E3 checks independently reconstruct stored
arithmetic using the frozen bundle; they do not independently replicate E3.
The historical preservation list is a baseline, not an assertion that it covers
every historical file. Run with Python 3.11+; standard library only.
"""

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from math import fsum, isclose
from pathlib import Path
import argparse
import json
import struct


BOOK = Path(__file__).resolve().parents[2]
DEFAULT_E3 = Path("D:/OmegaProtocol/results/omega_universe_targets_v1/xfam_e3_shared_response_20260826_v0_1")
PRESERVATION_SHA256 = "9803028e3642db395c7cbd37efea968bd7e283ef5c8be3f5ecde4124185eac97"
E3_BASELINE = {
    "analysis/aggregate.json": (1656, "d0d37e3a0016ad2225829070a2c7fa6c3f0e0c06b11ada4624899cc0101af5ed"),
    "analysis/per_class.jsonl": (21024, "a3df4ef9eceb693f821cf1886bd76d7db6e2d3170ea3702f1309fad6f73106b6"),
    "controls/coefficient_freeze.json": (7496, "50c889e01d6a0edb8bbfa9addc2b8d6fcef864d427ed1fae8f77ff69ff8e78b5"),
    "controls/controls.json": (377, "a0a8621f67da976d825d9467c58cfcfea874858d83c932a75cdfcfe2eee24800"),
    "controls/development_receipt.json": (10408, "90539577e88942d328a44061008c26923cd2f328f80eaf74132b6c6fef36a4a7"),
    "controls/post_coefficient_pilot_receipt.json": (4169, "6ad93e3947f89d3fbcf854dc9be6826bdab878d9d56410913bf348b2405eadd0"),
    "controls/probe_commitment.json": (185, "5a2e90dccdcfa694615a3347d4169f52310dd3caed90b942680649dab858c800"),
    "public/classes.jsonl": (7565, "4b6257109681f051e8871f2730a80b87487c5069aa8c8bf312367b1334e523cf"),
}
THETA = [-0.49414752789909744, 0.7191418084524686, 0.6260880627343716, 0.48161909928328844]
DESCRIPTORS = ("mean_local_influence", "normalized_neighborhood_size", "determinism_margin")
EXPECTED_MSES = {
    "mse_law": 0.04651764075889651,
    "mse_floor": 0.022214819625990624,
    "mse_memory": 0.011103334688740193,
    "mse_disjoint_union": 0.01693321813870276,
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label):
    require(isclose(actual, expected, rel_tol=0, abs_tol=1e-14),
            f"{label}: {actual!r} != {expected!r}")


def digest(path):
    result = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def contained(root, relative):
    target = (root / relative).resolve()
    require(target.is_relative_to(root.resolve()), f"Path leaves scoped root: {relative}")
    return target


def pool_checks():
    # Exact antiderivatives for polynomials, low-order coefficient first.
    def integral(coefficients, start, stop):
        return sum((Fraction(c, degree + 1) *
                    (stop ** (degree + 1) - start ** (degree + 1))
                    for degree, c in enumerate(coefficients)), Fraction(0))

    q_threshold = Fraction(1, 2)
    densities = {"uniform_1": [1], "rising_2q": [0, 2]}
    normalization = {name: integral(poly, Fraction(0), Fraction(1))
                     for name, poly in densities.items()}
    probabilities = {name: integral(poly, q_threshold, Fraction(1))
                     for name, poly in densities.items()}
    require(set(normalization.values()) == {Fraction(1)}, "Density normalization")
    require(probabilities == {"uniform_1": Fraction(1, 2), "rising_2q": Fraction(3, 4)},
            "Supported probabilities")
    # With u=J/Jc, the J densities times dJ become (1/2)du and (u/2)du.
    # This checks the displayed dimensional densities after the change of units.
    dimensional_densities = {"uniform_1": [Fraction(1, 2)], "rising_2q": [0, Fraction(1, 2)]}
    for name, polynomial in dimensional_densities.items():
        require(integral(polynomial, Fraction(0), Fraction(2)) == 1,
                f"Original J density normalization: {name}")
        require(integral(polynomial, Fraction(1), Fraction(2)) == probabilities[name],
                f"Original J density passing probability: {name}")
    # Both densities are nonnegative on [0,1]; no observational inference.
    with localcontext() as context:
        context.prec = 80
        k, pause, minimum = Decimal("0.1"), Decimal(5), Decimal(20)
        cutoff = k * minimum * (k * pause).exp()
        upper = 2 * cutoff
        terminal_at_cutoff = (cutoff / k) * (-k * pause).exp()
        require(abs(terminal_at_cutoff - minimum) < Decimal("1e-70"), "Pool threshold")
        require(upper < Decimal(12), "Selected interval must fit earlier J plot range")
        samples = []
        for q in [Fraction(0), Fraction(1, 4), q_threshold, Fraction(3, 4), Fraction(1)]:
            # J = 2*k*Mmin*exp(k*pause)*q implies M(pause) = 2*Mmin*q exactly.
            exact_terminal = 40 * q
            require((exact_terminal >= 20) == (q >= q_threshold), "Threshold equivalence")
            samples.append({"q": str(q), "terminal_amount_exact": str(exact_terminal)})
        return {
            "k": str(k), "pause": str(pause), "minimum": str(minimum),
            "Jc_exact": "2*exp(1/2)", "Jc_decimal80": str(cutoff),
            "selected_J_upper_decimal80": str(upper), "q_threshold_exact": str(q_threshold),
            "normalization_exact": {key: str(value) for key, value in normalization.items()},
            "supported_probabilities_exact": {key: str(value) for key, value in probabilities.items()},
            "original_J_density_change_of_variable_checked": True,
            "terminal_amount_checks": samples,
            "scope": "Declared one-dimensional J slice with fixed k and pause; range and densities are assumptions, not inferred measures.",
        }


def influence(rule):
    return Fraction(sum(((rule >> x) & 1) != ((rule >> (x ^ bit)) & 1)
                        for x in range(8) for bit in (1, 4)), 16)


def e3_checks(root):
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    declared = {row["relative_path"]: (row["byte_count"], row["sha256"])
                for row in manifest["artifacts"]}
    require(len(manifest["artifacts"]) == 8 and declared == E3_BASELINE,
            "E3 artifact declarations differ from accepted baseline")
    for name, (size, expected_hash) in E3_BASELINE.items():
        path = contained(root, name)
        require(path.stat().st_size == size and digest(path) == expected_hash,
                f"E3 immutable artifact mismatch: {name}")
    freeze = read_json(root / "controls/coefficient_freeze.json")
    primary = freeze["models"]["primary_law"]
    theta = primary["coefficients"]
    require(theta == THETA, "Frozen coefficients changed")
    require(struct.pack("<4d", *theta).hex() == primary["float64_le_hex"],
            "Frozen coefficient payload mismatch")
    require(freeze["coefficient_vector_committed_before_w7_response"] is True,
            "Coefficient-freeze receipt assertion changed")
    rows = [json.loads(line) for line in (root / "analysis/per_class.jsonl").read_text(encoding="utf-8").splitlines()]
    public = [json.loads(line) for line in (root / "public/classes.jsonl").read_text(encoding="utf-8").splitlines()]
    aggregate = read_json(root / "analysis/aggregate.json")
    by_class = {row["class_representative"]: row for row in rows}
    public_by_class = {row["class_representative"]: row for row in public}
    require(len(rows) == len(by_class) == len(public) == len(public_by_class) == 30,
            "Unique row count")
    require(by_class.keys() == public_by_class.keys(), "Public/analysis class set")
    require(aggregate["class_count"] == 30 and aggregate["outcome"] == "e3_transfer_null",
            "Aggregate outcome/count changed")
    require(manifest["scientific_admission"] is False and aggregate["scientific_admission"] is False,
            "Scientific admission flag changed")
    centroids = freeze["models"]["quantized_centroid_memory"]["centroids"]
    disjoint = freeze["models"]["disjoint_family_intercepts"]["coefficients"]
    floor = freeze["models"]["constant_floor"]["prediction"]
    for representative, row in by_class.items():
        for key, value in public_by_class[representative].items():
            require(row[key] == value, f"Public/analysis mismatch: {representative}/{key}")
        desc = tuple(row[key] for key in DESCRIPTORS)
        require(desc == (float(influence(representative)), 2 / 31, 0),
                f"Truth-table descriptor mismatch: {representative}")
        require(all(influence(member) == influence(representative) for member in row["class_orbit"]),
                f"Orbit descriptor mismatch: {representative}")
        require(row["probe_count"] == 8 and row["ring_size"] == row["horizon"] == 32,
                f"Declared probes, sites and terminal horizon: {representative}")
        require(row["response_comparison_count"] == row["probe_count"] * row["ring_size"] * (row["ring_size"] - 1) == 7936,
                f"Response denominator: {representative}")
        require(0 <= row["response_changed_count"] <= 7936, "Bounded response count")
        close(row["response"], row["response_changed_count"] / 7936, f"Response {representative}")
        close(row["law_prediction"], fsum(a * b for a, b in zip(theta, [1, *desc])), f"Prediction {representative}")
        closest = min(range(len(centroids)), key=lambda i: (sum((centroids[i][j] - desc[j]) ** 2 for j in range(3)), i))
        close(row["memory_prediction"], centroids[closest][3], f"Memory {representative}")
        close(row["disjoint_union_prediction"], fsum(a * b for a, b in zip(disjoint[:4], [1, *desc])), f"Disjoint {representative}")
        close(row["floor_prediction"], floor, f"Constant {representative}")
    all_rule_levels = Counter(influence(rule) for rule in range(256))
    require(all_rule_levels == {Fraction(0): 4, Fraction(1, 4): 48, Fraction(1, 2): 152, Fraction(3, 4): 48, Fraction(1): 4},
            "Exhaustive truth-table influence levels")
    heldout_levels = Counter(influence(rule) for rule in by_class)
    require(heldout_levels == {Fraction(1, 4): 7, Fraction(1, 2): 17, Fraction(3, 4): 5, Fraction(1): 1},
            "Actual held-out influence multiplicities")
    predictions = Counter(row["law_prediction"] for row in rows)
    negative_classes = sorted(rule for rule, row in by_class.items() if row["law_prediction"] < 0)
    require(len(predictions) == 4 and len(negative_classes) == 24, "Four predictions / 24 negative")
    collision = [by_class[rule] for rule in (24, 44)]
    require([row["response_changed_count"] for row in collision] == [292, 237], "Class 24/44 counts")
    require(tuple(collision[0][key] for key in DESCRIPTORS) == tuple(collision[1][key] for key in DESCRIPTORS), "Collision descriptors")
    require(collision[0]["law_prediction"] == collision[1]["law_prediction"], "Collision predictions")
    close(collision[0]["law_prediction"], -0.09418384543193596, "Class 24 worked prediction")
    mses = {}
    for field, metric in (("law_prediction", "mse_law"), ("floor_prediction", "mse_floor"),
                          ("memory_prediction", "mse_memory"), ("disjoint_union_prediction", "mse_disjoint_union")):
        value = fsum((row[field] - row["response"]) ** 2 for row in rows) / 30
        close(value, aggregate["metrics"][metric], metric)
        close(value, EXPECTED_MSES[metric], f"Accepted {metric}")
        mses[metric] = value
    mean_response_exact = sum((Fraction(row["response_changed_count"], 7936) for row in rows), Fraction(0)) / 30
    require(mean_response_exact == Fraction(157, 1984), "Exact mean response")
    mean_prediction = fsum(row["law_prediction"] for row in rows) / 30
    mean_residual = mean_prediction - float(mean_response_exact)
    centered_mse = fsum((row["law_prediction"] - row["response"] - mean_residual) ** 2 for row in rows) / 30
    close(mean_residual ** 2 + centered_mse, mses["mse_law"], "Descriptive MSE decomposition")
    exact_theta = [Fraction.from_float(value) for value in theta]
    require(all(slope > 0 for slope in exact_theta[1:]), "Positive slopes for bounding-box argument")
    box_lower = exact_theta[0]
    box_upper = sum(exact_theta[:3]) + exact_theta[3] / 2
    squared_error_bound = max((1 - box_lower) ** 2, box_upper ** 2)
    require(squared_error_bound < Fraction(2233, 1000), "Displayed MSE bound <2.233")
    return {
        "root": str(root), "manifest_sha256": digest(manifest_path), "baseline_artifacts_verified": len(E3_BASELINE),
        "coefficients": theta, "coefficient_float64_le_hex": primary["float64_le_hex"],
        "analysis_public_rows_agree": 30, "all_256_rule_influence_counts": {str(k): v for k, v in sorted(all_rule_levels.items())},
        "actual_30_class_influence_counts": {str(k): v for k, v in sorted(heldout_levels.items())},
        "actual_predictions": [{"prediction": k, "class_count": v} for k, v in sorted(predictions.items())],
        "negative_prediction_count": len(negative_classes), "negative_classes": negative_classes,
        "worked_collision": [{key: row[key] for key in ("class_representative", *DESCRIPTORS, "law_prediction", "response_changed_count", "response_comparison_count", "response")} for row in collision],
        "collision_response_difference_exact": str(Fraction(55, 7936)), "recomputed_mse": mses,
        "mean_response_exact": str(mean_response_exact), "mean_prediction": mean_prediction,
        "mean_residual": mean_residual, "mean_residual_squared": mean_residual ** 2,
        "centered_residual_mse": centered_mse,
        "bounding_box_prediction_range": [float(box_lower), float(box_upper)],
        "bounding_box_squared_error_upper_bound": float(squared_error_bound),
        "scope": "Immutable stored outcomes, descriptors and predictions only; no refitting, clipping, new outcomes, bootstrap, or causal attribution.",
    }


def selection_checks():
    train = [Fraction(-1), Fraction(0), Fraction(1)]
    test = [Fraction(-1, 2), Fraction(1, 2)]
    intercept = sum((x * x for x in train), Fraction(0)) / 3
    require(intercept == Fraction(2, 3), "Least-squares constant")
    require(sum((x * (intercept - x * x) for x in train), Fraction(0)) == 0,
            "Zero slope satisfies linear least-squares normal equation")
    require(all((intercept - x * x) ** 2 == Fraction(25, 144) for x in test),
            "Constant/line test error")
    correction = lambda x: x * (x * x - 1) * (x * x - Fraction(1, 4))
    require(all(correction(x) == 0 for x in train + test), "Five-point non-identification construction")
    require(correction(Fraction(2)) != 0, "Nontrivial alternative away from observed points")
    return {"constant_and_linear_intercept_exact": str(intercept), "linear_slope_exact": "0",
            "constant_and_linear_test_mse_exact": "25/144", "quadratic_test_mse_exact": "0",
            "alternative_function_agrees_at_all_five_points": True,
            "scope": "Exact constructed selection example, not empirical generalization."}


def preservation_checks():
    path = BOOK / "reviews/publication-v28/preservation.json"
    require(digest(path) == PRESERVATION_SHA256, "Preservation baseline itself changed")
    baseline = read_json(path)
    require(len(baseline) == 2778, "Protected baseline file count changed")
    mismatches, missing = [], []
    categories = Counter()
    total_bytes = 0
    for name, expected in baseline.items():
        target = contained(BOOK, name)
        categories[name.split("/")[0]] += 1
        if not target.is_file():
            missing.append(name)
            continue
        total_bytes += target.stat().st_size
        actual = digest(target)
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    require(not missing and not mismatches,
            json.dumps({"missing_protected_files": missing, "changed_protected_files": mismatches}))
    return {"baseline_sha256": PRESERVATION_SHA256, "protected_files_verified": len(baseline),
            "total_bytes_hashed": total_bytes, "by_top_level_directory": dict(sorted(categories.items())),
            "missing": missing, "changed": mismatches,
            "scope": "Exactly the files listed in preservation.json; no assertion about unlisted history."}


def manuscript_manifest_checks():
    path = BOOK / "reviews/publication-v28/manifest.json"
    manifest = read_json(path)
    previous = read_json(BOOK / "reviews/publication-v27/manifest.json")
    chapters = manifest["chapters"]
    require(manifest["version"] == "v28" and manifest["source_version"] == "v27", "Manifest versions")
    require([chapter["reading_number"] for chapter in chapters] == list(range(1, 38)), "37 ordered chapters")
    require([chapter["unit"] for chapter in chapters] == [chapter["unit"] for chapter in previous["chapters"]], "Chapter order changed")
    for chapter in chapters:
        for key in ("main", "notes"):
            require(contained(BOOK, chapter[key]).is_file(), f"Missing chapter artifact: {chapter[key]}")
    release = manifest.get("release", {})
    warnings = []
    if "v27" in release.get("pdf", "") or "v27" in release.get("record", ""):
        warnings.append("Manifest still contains inherited v27 release metadata; update before claiming a v28 build.")
    return {"manifest_sha256": digest(path), "ordered_chapters": 37, "main_and_notes_files_present": 74,
            "warnings": warnings, "scope": "Manifest structure and file existence, not prose, compilation, navigation, figures or page layout."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3-root", type=Path, default=DEFAULT_E3)
    args = parser.parse_args()
    result = {"audit": "v28 bounded revision regression", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": digest(Path(__file__)), "book": str(BOOK), "checks": {}, "failures": []}
    for label, operation in (("pool", pool_checks), ("selection", selection_checks), ("e3", lambda: e3_checks(args.e3_root)),
                             ("preservation", preservation_checks), ("manuscript_manifest", manuscript_manifest_checks)):
        try:
            result["checks"][label] = operation()
        except Exception as error:
            result["failures"].append({"section": label, "error": f"{type(error).__name__}: {error}"})
    result["passed"] = not result["failures"]
    result["limits"] = "This is not whole-book scientific, experimental, editorial, visual or publication certification. Dialogue provenance and ending progression require a separate human-readable source audit."
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
