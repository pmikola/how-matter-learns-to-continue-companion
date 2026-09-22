#!/usr/bin/env python3
"""Independent v44 numerical lab. Python standard library only; no old calculators.

Default: read inputs, run nine groups, print JSON, create no files. --output PATH
writes the same JSON to that explicit path (its parent must already exist).
--companion accepts a companion ZIP, its extracted root, or evidence/e3 itself.
Without it, nearby evidence/e3 or the repository's v43 companion is discovered.
E3 response fractions are recomputed from frozen integer counts, not regenerated
from missing probe trajectories. All new diagnostics are post-test analyses.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import platform
import struct
import sys
import time
import zipfile


PINNED_E3 = {
    "analysis/aggregate.json": (1656, "d0d37e3a0016ad2225829070a2c7fa6c3f0e0c06b11ada4624899cc0101af5ed"),
    "analysis/per_class.jsonl": (21024, "a3df4ef9eceb693f821cf1886bd76d7db6e2d3170ea3702f1309fad6f73106b6"),
    "controls/coefficient_freeze.json": (7496, "50c889e01d6a0edb8bbfa9addc2b8d6fcef864d427ed1fae8f77ff69ff8e78b5"),
    "controls/controls.json": (377, "a0a8621f67da976d825d9467c58cfcfea874858d83c932a75cdfcfe2eee24800"),
    "controls/development_receipt.json": (10408, "90539577e88942d328a44061008c26923cd2f328f80eaf74132b6c6fef36a4a7"),
    "controls/post_coefficient_pilot_receipt.json": (4169, "6ad93e3947f89d3fbcf854dc9be6826bdab878d9d56410913bf348b2405eadd0"),
    "controls/probe_commitment.json": (185, "5a2e90dccdcfa694615a3347d4169f52310dd3caed90b942680649dab858c800"),
    "public/classes.jsonl": (7565, "4b6257109681f051e8871f2730a80b87487c5069aa8c8bf312367b1334e523cf"),
}


class Checks:
    """Always-on assertions, including when Python is run with -O."""

    def __init__(self):
        self.counts = Counter()

    def require(self, condition, label):
        self.counts[label] += 1
        if not condition:
            raise AssertionError(label)

    def close(self, left, right, label, tolerance=2e-14):
        self.require(math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance), label)

    def record(self):
        return {"total": sum(self.counts.values()), "by_assertion": dict(sorted(self.counts.items()))}


def number(value):
    value = F(value)
    return {"exact": str(value), "decimal": float(value)}


def rule_output(rule, cells):
    left, center, right = cells
    return (rule >> (4 * left + 2 * center + right)) & 1


def two_steps(rule, cells):
    intermediate = [rule_output(rule, cells[i:i + 3]) for i in range(3)]
    return rule_output(rule, intermediate)


def influences(rule, width):
    evaluate = rule_output if width == 3 else two_steps
    counts = [0] * width
    for cells in product((0, 1), repeat=width):
        original = evaluate(rule, cells)
        for position in range(width):
            flipped = list(cells)
            flipped[position] ^= 1
            counts[position] += original != evaluate(rule, flipped)
    return tuple(F(count, 2 ** width) for count in counts)


def recode_rule(rule, reflection=False, complement=False):
    result = 0
    for left, center, right in product((0, 1), repeat=3):
        cells = (left, center, right)
        if reflection:
            cells = cells[::-1]
        if complement:
            cells = tuple(1 - bit for bit in cells)
        output = rule_output(rule, cells) ^ int(complement)
        result |= output << (4 * left + 2 * center + right)
    return result


def group_eca(check):
    profiles = {rule: (influences(rule, 3), influences(rule, 5)) for rule in range(256)}
    for rule, formula in ((24, lambda l, c, r: (l and not c and not r) or (not l and c and r)),
                          (44, lambda l, c, r: (not l and c) or (l and not c and r))):
        for cells in product((0, 1), repeat=3):
            check.require(rule_output(rule, cells) == int(formula(*cells)), "Boolean rules 24 and 44 agree with numbered truth tables")
    for rule, (one, two) in profiles.items():
        for reflection, complement in ((True, False), (False, True), (True, True)):
            transformed = recode_rule(rule, reflection, complement)
            expected = (one[::-1], two[::-1]) if reflection else (one, two)
            check.require(profiles[transformed] == expected, "one- and two-step profiles transform under all 768 recodings")
            check.require(abs(one[0] - one[-1]) == abs(profiles[transformed][0][0] - profiles[transformed][0][-1]),
                          "external imbalance invariant under all 768 recodings")
    check.require(profiles[24] == ((F(1, 2),) * 3, (F(3, 8), F(3, 8), F(3, 8), F(1, 8), F(1, 8))), "rule 24 exact profiles")
    check.require(profiles[44] == ((F(3, 4), F(3, 4), F(1, 4)), (F(7, 16), F(5, 16), F(7, 16), F(3, 16), F(1, 16))), "rule 44 exact profiles")
    rows = []
    for rule, (one, two) in profiles.items():
        rows.append({"rule": rule, "one_step": [number(v) for v in one], "two_step": [number(v) for v in two],
                     "external_mean": number((one[0] + one[2]) / 2), "external_imbalance": number(abs(one[0] - one[2])),
                     "two_step_noncentral_mean": number((sum(two) - two[2]) / 4)})
    return {"status": "passed", "classification": "later exploratory descriptor diagnostic", "rule_count": 256,
            "recoding_comparisons": 768, "numbering": "(rule >> (4*left + 2*center + right)) & 1",
            "measure": "uniform over all 8 one-step or 32 two-step input neighborhoods",
            "scope": "Separates the exposed pair; no held-out prediction or global sufficiency claim.", "profiles": rows}


def locate_companion(explicit):
    if explicit is not None:
        return Path(explicit).resolve()
    script = Path(__file__).resolve()
    bases = [Path.cwd(), script.parent, *list(script.parents)[:3]]
    for base in bases:
        for candidate in (base / "evidence/e3", base / "output/companion/how-matter-learns-to-continue-v43-companion.zip"):
            if candidate.exists():
                return candidate.resolve()
    return None


def read_e3(path, check):
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            roots = [name[:-len("manifest.json")] for name in archive.namelist() if name.endswith("/evidence/e3/manifest.json")]
            check.require(len(roots) == 1, "ZIP has exactly one frozen E3 manifest")
            raw = {name: archive.read(roots[0] + name) for name in ["manifest.json", *PINNED_E3]}
        source = {"type": "ZIP", "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    else:
        root = path if (path / "manifest.json").is_file() and (path / "analysis").is_dir() else path / "evidence/e3"
        raw = {name: (root / name).read_bytes() for name in ["manifest.json", *PINNED_E3]}
        source = {"type": "directory", "path": str(root)}
    manifest = json.loads(raw["manifest.json"])
    declared = {item["relative_path"]: item for item in manifest["artifacts"]}
    check.require(set(declared) == set(PINNED_E3), "manifest lists exactly the eight pinned E3 artifacts")
    hashes = []
    for name, (expected_bytes, expected_hash) in PINNED_E3.items():
        actual_hash = sha256(raw[name]).hexdigest()
        check.require(len(raw[name]) == expected_bytes == declared[name]["byte_count"], "frozen artifact byte counts agree")
        check.require(actual_hash == expected_hash == declared[name]["sha256"], "frozen artifact hashes agree with pinned bytes and manifest")
        hashes.append({"path": name, "bytes": len(raw[name]), "sha256": actual_hash})
    source.update({"manifest_sha256": sha256(raw["manifest.json"]).hexdigest(), "artifacts": hashes})
    parsed = {name: [json.loads(line) for line in data.decode("utf-8").splitlines()] if name.endswith("jsonl") else json.loads(data)
              for name, data in raw.items()}
    return parsed, source


def dot(coefficients, features):
    return sum(coefficient * feature for coefficient, feature in zip(coefficients, features))


def frozen_e3_diagnostics(path, check):
    data, source = read_e3(path, check)
    rows = data["analysis/per_class.jsonl"]
    public = data["public/classes.jsonl"]
    models = data["controls/coefficient_freeze.json"]["models"]
    aggregate = data["analysis/aggregate.json"]
    manifest = data["manifest.json"]
    check.require(len(rows) == len(public) == aggregate["class_count"] == 30, "exactly 30 frozen E3 rows")
    check.require(aggregate["outcome"] == manifest["outcome"] == "e3_transfer_null", "original failed E3 outcome retained")
    check.require(aggregate["scientific_admission"] is False, "original scientific-admission flag remains false")
    coefficients = models["primary_law"]["coefficients"]
    for model in [models["primary_law"], models["adapter_shuffle_law"], models["disjoint_family_intercepts"], *models["per_family_laws"].values()]:
        packed = struct.pack("<" + "d" * len(model["coefficients"]), *model["coefficients"])
        check.require(packed.hex() == model["float64_le_hex"] and len(packed) == model["byte_count"], "frozen coefficient decimals match committed IEEE bytes")
    centroids = models["quantized_centroid_memory"]["centroids"]
    memory_payload = struct.pack("<16f", *(v for centroid in centroids for v in centroid))
    check.require(memory_payload.hex() == models["quantized_centroid_memory"]["payload_hex"], "frozen memory values match committed IEEE bytes")
    by_rule = {row["class_representative"]: row for row in rows}
    grouped = defaultdict(list)
    computed_rows = []
    error_sums = defaultdict(float)
    for row, public_row in zip(rows, public):
        check.require(all(row[key] == value for key, value in public_row.items()), "public E3 row agrees with full record")
        rule = row["class_representative"]
        one = influences(rule, 3)
        external = (one[0] + one[2]) / 2
        check.require(float(external) == row["mean_local_influence"], "independent ECA descriptor equals frozen descriptor")
        check.require(row["normalized_neighborhood_size"] == 2 / (row["ring_size"] - 1), "frozen external-neighborhood normalization")
        check.require(row["determinism_margin"] == 0, "deterministic rules have zero margin descriptor")
        orbit = sorted({recode_rule(rule, reflected, complemented) for reflected, complemented in product((False, True), repeat=2)})
        check.require(orbit == row["class_orbit"], "independently reconstructed recoding orbit equals frozen orbit")
        response = F(row["response_changed_count"], row["response_comparison_count"])
        check.require(row["response_comparison_count"] == 7936, "frozen denominator is 7936 comparisons")
        check.close(float(response), row["response"], "response reproduced from frozen integer counts")
        features = (1.0, row["mean_local_influence"], row["normalized_neighborhood_size"], row["determinism_margin"])
        predicted = dot(coefficients, features)
        projected = min(1.0, max(0.0, predicted))
        memory_index = min(range(len(centroids)), key=lambda i: sum((features[j + 1] - centroids[i][j]) ** 2 for j in range(3)))
        disjoint = models["disjoint_family_intercepts"]["coefficients"]
        permuted_row = by_rule[models["sealed_descriptor_permutation"][str(rule)]]
        permuted_features = (1.0, permuted_row["mean_local_influence"], permuted_row["normalized_neighborhood_size"], permuted_row["determinism_margin"])
        predictions = {"law": predicted, "floor": models["constant_floor"]["prediction"],
                       "memory": centroids[memory_index][3], "disjoint_union": dot(disjoint[:4], features),
                       "adapter_sham": dot(models["adapter_shuffle_law"]["coefficients"], features),
                       "sealed_descriptor_sham": dot(coefficients, permuted_features)}
        for name, value in predictions.items():
            check.close(value, row[name + "_prediction"], "frozen predictions reconstructed from committed models")
            error_sums[name] += (float(response) - value) ** 2
        for family, model in models["per_family_laws"].items():
            check.close(dot(model["coefficients"], features), row["per_family_predictions"][family], "frozen per-family prediction reproduced")
        check.require((projected - float(response)) ** 2 <= (predicted - float(response)) ** 2 + 1e-16, "projection cannot increase any E3 row error")
        grouped[tuple(features[1:])].append((rule, response, predicted, projected))
        computed_rows.append({"class_representative": rule, "response": number(response), "original_frozen_prediction": predicted,
                              "later_clipped_prediction": projected, "descriptor_tuple": list(features[1:]), "later_external_imbalance": number(abs(one[0] - one[-1]))})
    metrics = {name: value / len(rows) for name, value in error_sums.items()}
    for name, value in metrics.items():
        check.close(value, aggregate["metrics"]["mse_" + name], "rowwise original MSE reproduces frozen aggregate")
    check.require(sorted(len(group) for group in grouped.values()) == [1, 5, 7, 17], "actual E3 descriptor bucket counts")
    within = F(0)
    mismatch = 0.0
    clipped_mismatch = 0.0
    clipped_mse = sum((float(row["response"]["decimal"]) - row["later_clipped_prediction"]) ** 2 for row in computed_rows) / len(rows)
    groups = []
    for descriptors, group in sorted(grouped.items()):
        mean = sum((entry[1] for entry in group), F(0)) / len(group)
        squared_sum = sum(((entry[1] - mean) ** 2 for entry in group), F(0))
        within += squared_sum / len(rows)
        predicted, projected = group[0][2:]
        mismatch += len(group) * (predicted - float(mean)) ** 2 / len(rows)
        clipped_mismatch += len(group) * (projected - float(mean)) ** 2 / len(rows)
        groups.append({"descriptors": list(descriptors), "row_count": len(group), "rules": [entry[0] for entry in group],
                       "exposed_sample_group_mean": number(mean), "within_group_contribution_to_30_row_MSE": number(squared_sum / len(rows)),
                       "original_frozen_prediction": predicted, "later_clipped_prediction": projected})
    check.close(metrics["law"], float(within) + mismatch, "actual E3 original within-plus-between identity")
    check.close(clipped_mse, float(within) + clipped_mismatch, "actual E3 clipped within-plus-between identity")
    check.close(metrics["law"], 0.04651764075889651, "original MSE reference")
    check.close(clipped_mse, 0.014347580116304564, "later clipped MSE reference")
    check.close(float(within), 0.006511098182057845, "actual exposed-sample descriptor floor reference")
    return {"status": "passed", "source": source, "original_record": {"classification": "original computed frozen E3 results, arithmetically reproduced",
            "outcome": aggregate["outcome"], "scientific_admission": False, "mse": metrics, "coefficients": coefficients},
            "later_diagnostics": {"classification": "post-test diagnostics on all 30 exposed rows; not registered tests or a deployable fitted predictor",
                "clipped_mse": clipped_mse, "error_removed": metrics["law"] - clipped_mse,
                "percent_error_removed": 100 * (1 - clipped_mse / metrics["law"]),
                "exposed_sample_descriptor_floor": number(within), "original_group_prediction_mismatch": mismatch,
                "clipped_group_prediction_mismatch": clipped_mismatch, "groups": groups, "rows": computed_rows,
                "comparisons": {"clipped_below_constant_comparator": clipped_mse < metrics["floor"],
                                "clipped_below_disjoint_union_comparator": clipped_mse < metrics["disjoint_union"],
                                "clipped_below_memory_comparator": clipped_mse < metrics["memory"]}},
            "scope": ["Eight frozen artifacts are checked against pinned SHA-256 hashes and their manifest.",
                      "Responses are reconstructed from frozen integer counts; original probe trajectories are not regenerated.",
                      "The archive has a probe commitment hash, not the initial probe bit arrays or historical training materializer.",
                      "The descriptor floor minimizes empirical error on these exposed rows only; it is not a population floor.",
                      "Frozen comparators and the failed original experiment remain unchanged."]}


def group_error(check, companion):
    for prediction in [F(i, 10) for i in range(-20, 31)]:
        clipped = min(F(1), max(F(0), prediction))
        for response in [F(i, 20) for i in range(21)]:
            reduction = (prediction - response) ** 2 - (clipped - response) ** 2
            check.require(reduction >= 0, "squared-error projection inequality on rational grid")
            if prediction < 0:
                check.require(reduction >= prediction ** 2, "negative-prediction guaranteed reduction on rational grid")
    levels = (F("-0.2739692975"), F("-0.0941838454"), F("0.0856016067"), F("0.2653870588"))
    lower = sum(count * prediction ** 2 for count, prediction in zip((7, 17, 5, 1), levels) if prediction < 0) / 30
    d = F(55, 7936)
    strengthened = lower - 2 * levels[1] * (F(292, 7936) + F(237, 7936)) / 30
    # Synthetic arithmetic check is explicitly separate from the actual E3 data.
    synthetic = ((F(1, 5), F(3, 5)), (F(1, 10), F(4, 10), F(9, 10)))
    predictions = (F(-1, 5), F(4, 5))
    total = sum((y - p) ** 2 for ys, p in zip(synthetic, predictions) for y in ys)
    within = between = F(0)
    for ys, p in zip(synthetic, predictions):
        mean = sum(ys) / len(ys)
        within += sum((y - mean) ** 2 for y in ys)
        between += len(ys) * (p - mean) ** 2
    check.require(total == within + between, "synthetic exact within-plus-between decomposition")
    result = {"status": "passed", "classification": "later post-test error diagnostics",
              "projection_identity": "negative p: (p-y)^2-y^2=p^2-2*p*y >= p^2 for 0<=y<=1",
              "published_rounded_levels_bound": number(lower), "bound_percent_of_rounded_original_MSE": float(100 * lower / F("0.0465176")),
              "bound_strengthened_by_exposed_pair": number(strengthened), "pair_average_floor": number(d ** 2 / 4),
              "pair_contribution_to_30_row_MSE": number(d ** 2 / 60), "synthetic_decomposition": number(total / 5)}
    result["frozen_e3"] = frozen_e3_diagnostics(companion, check) if companion else {
        "status": "not_run", "reason": "Supply --companion ZIP or extracted root to verify the eight frozen E3 files and all 30 rows."}
    return result


def group_ring(check):
    lower, center, upper = F(9, 20), F(13, 20), F(17, 20)
    radii = [F(i, 1000) for i in range(1401)] + [point + delta for point in (lower, center, upper) for delta in (-F(1, 10 ** 9), F(1, 10 ** 9))]
    check.require(len(set(radii)) == 1407, "all 1407 rational radius cases are distinct")
    for radius in radii:
        score = F(1, 5) - max(F(0), radius - center) - max(F(0), center - radius)
        check.require(score == F(1, 5) - abs(radius - center), "two-ReLU score equals absolute-deviation score exactly")
        check.require((score >= 0) == (lower <= radius <= upper), "inclusive ring classification on exact rational cases")
    return {"status": "passed", "classification": "hand-built supplied-radius construction; no training claim", "rational_cases": len(radii),
            "score": "1/5 - ReLU(r-13/20) - ReLU(13/20-r)", "accepted_interval": "9/20 <= r <= 17/20",
            "algebraic_argument": "ReLU(x)+ReLU(-x)=abs(x); sigma(K*z)>=1/2 iff z>=0 for K>0.",
            "boundaries": [{"radius": number(r), "score": number(0), "sigmoid_probability": number(F(1, 2)), "accepted": True} for r in (lower, upper)],
            "scope": "Real-arithmetic classification; radius supplied; two units do not reproduce the compact-support three-unit tent. Floating-point ties require an implementation convention."}


@lru_cache(maxsize=None)
def backward(horizon, amount, success):
    if amount <= 0:
        return F(0)
    if horizon == 0:
        return F(1)
    return success * backward(horizon - 1, amount + 1, success) + (1 - success) * backward(horizon - 1, amount - 1, success)


def forward(horizon, amount, success):
    alive = {amount: F(1)} if amount > 0 else {}
    for _ in range(horizon):
        next_alive = defaultdict(F)
        for resource, mass in alive.items():
            next_alive[resource + 1] += mass * success
            if resource > 1:
                next_alive[resource - 1] += mass * (1 - success)
        alive = next_alive
    return sum(alive.values(), F(0))


def enumerate_survival(horizon, amount, success):
    total = F(0)
    for path in product((0, 1), repeat=horizon):
        resource = amount
        alive = amount > 0
        for correct in path:
            resource += 2 * correct - 1
            alive = alive and resource > 0
        if alive:
            correct_count = sum(path)
            total += success ** correct_count * (1 - success) ** (horizon - correct_count)
    return total


def calibration_error():
    return sum((F(9, 10) ** sum(history) * F(1, 10) ** (5 - sum(history))
                for history in product((0, 1), repeat=5) if sum(history) < 3), F(0))


def group_resource(check):
    success_values = (F(1, 10), F(1, 2), F(9, 10))
    survival = {}
    for success in success_values:
        values = (backward(12, 3, success), forward(12, 3, success), enumerate_survival(12, 3, success))
        check.require(len(set(values)) == 1, "12-step backward, forward and all 4096 paths agree exactly")
        survival[success] = values[0]
    for horizon in range(9):
        for amount in range(1, 5):
            for success in (F(0), F(1, 5), F(1, 2), F(4, 5), F(1)):
                check.require(backward(horizon, amount, success) == forward(horizon, amount, success) == enumerate_survival(horizon, amount, success),
                              "small-horizon exhaustive resource cross-check")
    error = calibration_error()
    check.require(error == F(107, 12500), "32-history calibration error equals 0.00856")
    correct, wrong, ignored = survival[F(9, 10)], survival[F(1, 10)], survival[F(1, 2)]
    stationary = (1 - error) * correct + error * wrong
    reversed_probability = error * correct + (1 - error) * wrong
    crossing = (stationary - ignored) / (stationary - reversed_probability)
    check.require(ignored == F(627, 1024), "original ignore probability equals 627/1024")
    check.require(crossing == F(5903024594197, 15308200938394), "exact reversal-mixture crossing")
    check.require((1 - crossing) * stationary + crossing * reversed_probability == ignored, "crossing gives exact equality")
    duplicate = F(9, 10) * correct + F(1, 10) * wrong
    return {"status": "passed", "classification": "original finite model reproduced plus separately stipulated extensions",
            "assumptions": ["Start at amount 3; independent +1/-1 resource increments; ruin at first zero.",
                            "Two equally weighted sensor orientations, fixed for each deployment episode; fair resource sides.",
                            "Five independent labelled calibration trials; majority selects follow or reverse.",
                            "Optional alpha reversal is one episode-level event independent of calibration, then orientation stays fixed."],
            "original_horizon": 12, "conditional_survival": {str(p): number(v) for p, v in survival.items()},
            "calibration_error": number(error), "stationary_learned": number(stationary), "certain_reversal_learned": number(reversed_probability),
            "always_follow": number((correct + wrong) / 2), "always_ignore": number(ignored),
            "later_alpha_crossing": number(crossing), "later_alpha_rows": [{"alpha": number(alpha), "learned": number((1 - alpha) * stationary + alpha * reversed_probability)}
                                                                         for alpha in (F(0), F(1, 10), F(1, 4), F(1, 2), F(1))],
            "later_duplicate_label_stress_test": {"calibration_error": number(F(1, 10)), "stationary_continuation": number(duplicate),
                                                    "assumption": "Five copies of one correctness event, not five independent labels."},
            "scope": "No measured natural reversal frequency; no controller adapting again during deployment."}


def ruin_before_upper(success, amount, barrier):
    if success == F(1, 2):
        return F(barrier - amount, barrier)
    ratio = (1 - success) / success
    return (ratio ** amount - ratio ** barrier) / (1 - ratio ** barrier)


def indefinite(success, amount):
    return F(0) if success <= F(1, 2) else 1 - ((1 - success) / success) ** amount


def group_horizon(check):
    rows = []
    for horizon in range(25):
        values = {}
        for success in (F(1, 10), F(1, 2), F(9, 10)):
            value = backward(horizon, 3, success)
            check.require(value == forward(horizon, 3, success), "two exact finite-horizon implementations agree through 24")
            values[success] = value
        ignore = values[F(1, 2)]
        follow = (values[F(1, 10)] + values[F(9, 10)]) / 2
        rows.append({"horizon": horizon, "ignore": number(ignore), "follow": number(follow), "follow_greater": follow > ignore})
    first = next(row["horizon"] for row in rows if row["follow_greater"])
    check.require(first == 19, "first integer horizon ranking crossover is 19")
    for barrier in range(4, 21):
        for success in (F(1, 10), F(1, 2), F(9, 10)):
            check.require(ruin_before_upper(success, 0, barrier) == 1 and ruin_before_upper(success, barrier, barrier) == 0,
                          "finite upper-barrier ruin boundary values")
            for amount in range(1, barrier):
                probability = ruin_before_upper(success, amount, barrier)
                recurrence = success * ruin_before_upper(success, amount + 1, barrier) + (1 - success) * ruin_before_upper(success, amount - 1, barrier)
                check.require(probability == recurrence, "finite upper-barrier ruin recurrence")
    limit_learned = (1 - calibration_error()) * indefinite(F(9, 10), 3)
    limit_follow = indefinite(F(9, 10), 3) / 2
    check.require(limit_learned == F(3094, 3125), "ideal infinite learned probability")
    check.require(limit_follow == F(364, 729), "ideal infinite follow probability")
    for p in (F(1, 10), F(1, 2), F(9, 10)):
        check.require(backward(24, 3, p) >= indefinite(p, 3), "finite survival bounds ideal infinite continuation")
    return {"status": "passed", "classification": "later horizon and ideal infinite-time extensions", "first_crossover": first, "finite_rows": rows,
            "ideal_infinite": {"learned": number(limit_learned), "follow": number(limit_follow), "ignore": number(0),
                               "formula": "0 for p<=1/2; 1-((1-p)/p)^m for p>1/2",
                               "limit_argument": "Finite-upper-barrier ruin is (r^m-r^B)/(1-r^B), or (B-m)/B when p=1/2; take B to infinity.",
                               "assumptions": "Unbounded storage, independent stationary steps and opportunities forever, functioning sensor/memory/actuator forever.",
                               "apparatus_caveat": "An independent fatal hazard h>0 each update multiplies finite survival by (1-h)^T, whose limit is zero. No immortality claim."}}


def group_detector(check):
    # Work in q=k/log(2), so endpoint comparisons are exact rational arithmetic.
    for q in [F(i, 1000) for i in range(401)] + [F(1, 10), F(1, 5)]:
        reports = (5 * q <= 1, 10 * q <= 1)
        check.require((reports == (True, False)) == (F(1, 10) < q <= F(1, 5)), "detector reports equivalent to strict/inclusive rate interval")
    low, high = math.log(2) / 10, math.log(2) / 5
    check.require(low < 0.08 <= high, "generating loss rate belongs to identified interval")
    check.require(100 * math.exp(-5 * 0.08) >= 50 > 100 * math.exp(-10 * 0.08), "generating rate produces both reported detections")
    return {"status": "passed", "classification": "later inverse task with generating rate and prior supply hidden",
            "assumptions": "Known amount 100 at shutoff time 20, no inflow, positive constant k, exact clock and threshold D=1 iff amount>=50. Prior supply J is not independently supplied.",
            "reports": {"D(25)": 1, "D(30)": 0}, "identified_interval": {"lower": low, "upper": high, "lower_inclusive": False, "upper_inclusive": True,
                                                                                "exact": "log(2)/10 < k <= log(2)/5"},
            "reader_transfer_case": "80 exp(-kt), threshold 20, positive at 3 and negative at 4: log(4)/4 < k <= log(4)/3",
            "scope": "Partial identification under the decay model, not a uniquely recovered rate or new measurement."}


def group_state(check):
    rows = []
    for history in product((0, 1), repeat=5):
        policy_parameter = "follow" if sum(history) >= 3 else "reverse"
        writable_count = 0
        for observed_correctness in history:
            writable_count += observed_correctness
        actions = []
        for report in (0, 1):
            parameter_action = report if policy_parameter == "follow" else 1 - report
            fixed_interpreter_action = report ^ int(writable_count < 3)
            check.require(parameter_action == fixed_interpreter_action, "parameter and state implementations give identical actions")
            actions.append(parameter_action)
        rows.append({"history": list(history), "parameter": policy_parameter, "state_count": writable_count, "actions_for_reports_0_1": actions})
    return {"status": "passed", "classification": "later exhaustive implementation-equivalence construction", "histories": 32, "action_comparisons": 64,
            "scope": "Retained experience-selected response rules can be implemented as writable state. This does not make every state change learning.", "rows": rows}


def group_coupling(check):
    result = {}
    for name, model in (("A", lambda intervention, noise: noise), ("B", lambda intervention, noise: noise ^ intervention)):
        marginal = [F(sum(model(intervention, noise) for noise in (0, 1)), 2) for intervention in (0, 1)]
        paired = F(sum(model(0, noise) != model(1, noise) for noise in (0, 1)), 2)
        independent = F(sum(model(0, left) != model(1, right) for left, right in product((0, 1), repeat=2)), 4)
        check.require(marginal == [F(1, 2), F(1, 2)], "counterfactual models have identical fair interventional marginals")
        check.require(independent == F(1, 2), "independent-draw pairing yields one-half for each model")
        check.require(paired == (0 if name == "A" else 1), "matched-noise pairing distinguishes the two models")
        result[name] = {"marginal_probabilities_of_one": [number(p) for p in marginal], "matched_noise_disagreement": number(paired), "independent_noise_disagreement": number(independent)}
    return {"status": "passed", "classification": "later protocol stress test, not a discovered deterministic W7 bug", "models": result,
            "scope": "A paired disagreement rate additionally depends on cross-intervention coupling; marginal distributions alone do not determine it."}


def group_information(check):
    rows = []
    for probability in [F(i, 100) for i in range(101)]:
        # Enumerate all four deterministic maps report -> action under a
        # symmetric binary channel and balanced true side.
        policy_scores = []
        for policy in product((0, 1), repeat=2):
            score = F(0)
            for side, report in product((0, 1), repeat=2):
                mass = F(1, 2) * (probability if report == side else 1 - probability)
                score += mass * (policy[report] == side)
            policy_scores.append(score)
        optimum = max(policy_scores)
        check.require(policy_scores == [F(1, 2), probability, 1 - probability, F(1, 2)], "four binary-channel policy scores")
        check.require(optimum == max(probability, 1 - probability, F(1, 2)), "optional information includes ignore and best follow/reverse")
        check.require(optimum >= F(1, 2), "optional cost-free report cannot reduce best feasible success")
        rows.append({"report_correct_probability": number(probability), "optional_best": number(optimum), "forced_follow": number(probability)})
    return {"status": "passed", "classification": "later exhaustive binary-channel decision construction", "channel_grid_size": 101,
            "assumptions": "Known symmetric channel, balanced sides, free report, controller may ignore, follow, or reverse.",
            "scope": "A policy-set inclusion result for the declared task, not general optimal control or a claim that sensing has no physical cost.", "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--companion", type=Path, help="Companion ZIP, extracted root, or evidence/e3 directory")
    parser.add_argument("--output", type=Path, help="Explicit JSON output path; parent must exist; otherwise stdout only")
    args = parser.parse_args()
    companion = locate_companion(args.companion)
    started = time.perf_counter()
    check = Checks()
    groups = {}
    for name, function in (("eca_spatial_resolution_and_recoding", group_eca),
                           ("e3_quantitative_error_diagnosis", lambda c: group_error(c, companion)),
                           ("inclusive_radius_classifier", group_ring), ("resource_protocol_sensitivity", group_resource),
                           ("horizon_and_ideal_infinite_extension", group_horizon), ("detector_partial_identification", group_detector),
                           ("parameter_state_implementation_equivalence", group_state), ("counterfactual_coupling", group_coupling),
                           ("optional_information", group_information)):
        before = sum(check.counts.values())
        result = function(check)
        result["assertion_count"] = sum(check.counts.values()) - before
        groups[name] = result
    source_path = Path(__file__).resolve()
    report = {"schema": "how-matter-learns-to-continue.v44.orthogonal-lab.v1", "status": "passed", "groups_passed": 9,
              "frozen_e3_status": groups["e3_quantitative_error_diagnosis"]["frozen_e3"]["status"],
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "runtime": {"python": sys.version, "implementation": platform.python_implementation(),
                  "platform": platform.platform(), "executable": sys.executable, "elapsed_seconds": time.perf_counter() - started, "dependencies": "Python standard library only"},
              "source": {"script": str(source_path), "script_sha256": sha256(source_path.read_bytes()).hexdigest(),
                         "review": "how-matter-learns-to-continue-v43-orthogonal-validation.md",
                         "review_sha256": "dc2d7ce61f217eb039a4945ac80dd0213057ff2e64c18d11db2c81b1a5ce76a6",
                         "review_identified_manuscript_sha256": "33b3417eab32b527af2364f338a5d494abc74c9087ec9dd3bf597ff6e6059ef5"},
              "evidence_classes": {"original": "Frozen E3 bytes and original computed results remain original records.",
                                   "later": "Independent v44 reproductions, explanatory constructions and post-test diagnostics are labelled within each group.",
                                   "not_performed": "Historical E3 training/materialization, raw probe trajectory regeneration, population error estimation, or human reader testing."},
              "assertions": check.record(), "groups": groups}
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        output = args.output.resolve()
        protected = source_path == output or (companion is not None and (output == companion or (companion.is_dir() and companion in output.parents)))
        if protected:
            parser.error("--output must not overwrite the script, companion, or a file inside the companion input directory")
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
