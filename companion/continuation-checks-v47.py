#!/usr/bin/env python3
"""Standalone, standard-library checks for the v47 constructive calculations.

By default only JSON is printed. --output explicitly requests a saved report.
No historical training, physical experiment or human-reader test is performed.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
from functools import lru_cache
from itertools import product
import json
from math import comb, exp, isclose, log
from pathlib import Path


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise AssertionError(explanation)


def exact(value: Q) -> dict:
    return {"fraction": str(value), "probability": float(value),
            "percent": float(100 * value)}


@lru_cache(maxsize=None)
def survival(horizon: int, p: Q, start: int, capacity: int | None = None) -> Q:
    """Backward recurrence, with zero absorbing and optional saturating cap."""
    if start == 0:
        return Q(0)
    if horizon == 0:
        return Q(1)
    upper = start + 1 if capacity is None else min(capacity, start + 1)
    return (p * survival(horizon - 1, p, upper, capacity)
            + (1 - p) * survival(horizon - 1, p, start - 1, capacity))


def enumerate_survival(horizon: int, p: Q, start: int,
                       capacity: int | None = None) -> Q:
    """Enumerate complete strings, absorbing immediately on a failed boundary."""
    counts = [0] * (horizon + 1)
    for outcomes in product((0, 1), repeat=horizon):
        amount = start
        for outcome in outcomes:
            amount += 2 * outcome - 1
            if capacity is not None:
                amount = min(capacity, amount)
            if amount <= 0:
                break
        else:
            counts[sum(outcomes)] += 1
    return sum((Q(count) * p ** wins * (1 - p) ** (horizon - wins)
                for wins, count in enumerate(counts)), Q(0))


def calibration_error(labels: int) -> Q:
    return sum((Q(comb(labels, j)) * Q(9, 10) ** j * Q(1, 10) ** (labels - j)
                for j in range((labels - 1) // 2 + 1)), Q(0))


def enumerate_calibration_error(labels: int, correctness: Q) -> Q:
    error = Q(0)
    for record in product((0, 1), repeat=labels):
        follow = sum(record) > labels // 2
        if follow != (correctness > Q(1, 2)):
            k = sum(record)
            error += correctness ** k * (1 - correctness) ** (labels - k)
    return error


def original_check() -> dict:
    conditional = {}
    for p in (Q(1, 10), Q(1, 2), Q(9, 10)):
        result = survival(12, p, 3)
        require(result == enumerate_survival(12, p, 3), "Original enumeration")
        conditional[str(p)] = exact(result)
    e = calibration_error(5)
    require(e == Q(107, 12500), "Original calibration error")
    good, bad = survival(12, Q(9, 10), 3), survival(12, Q(1, 10), 3)
    learned, reversed_result = (1 - e) * good + e * bad, e * good + (1 - e) * bad
    require(learned == Q(30940570672769, 31250000000000), "Original learned value")
    require(reversed_result == Q(324168795981, 31250000000000), "Original reversal")
    return {"status": "passed", "conditional_survival": conditional,
            "learned": exact(learned), "reversed": exact(reversed_result),
            "deployment_strings_per_probability": 4096}


def rule150_check() -> dict:
    width = 32
    mask = (1 << width) - 1

    def bit_step(state: int) -> int:
        left = ((state << 1) & mask) | (state >> (width - 1))
        right = (state >> 1) | ((state & 1) << (width - 1))
        return left ^ state ^ right

    def table_step(cells: list[int]) -> list[int]:
        return [(150 >> (4 * cells[(i - 1) % width] + 2 * cells[i]
                         + cells[(i + 1) % width])) & 1 for i in range(width)]

    counts = {}
    for site in range(width):
        start = 1 << site
        bits, cells = start, [int(i == site) for i in range(width)]
        for time in range(1, 34):
            bits, cells = bit_step(bits), table_step(cells)
            require(bits == sum(v << i for i, v in enumerate(cells)),
                    "Rule 150 truth-table/bitwise agreement")
            if time in (16, 32):
                require(bits == start, "Rule 150 return for every basis vector")
            if time in (31, 32, 33):
                count = (bits & (mask ^ start)).bit_count()
                require(count == {31: 20, 32: 0, 33: 2}[time], "Neighboring readout")
                counts[str(time)] = {"differing_other_sites": count,
                                     "response": exact(Q(count, 31))}
    return {"status": "passed", "ring_size": width, "basis_vectors_checked": width,
            "independent_implementations": ["truth-table cells", "bitwise XOR"],
            "readouts": counts,
            "proof": "Over F2, F^(16)=S^(-16)+I+S^(16)=I on the 32-ring. "
                     "The basis checks supplement this operator identity.",
            "scope": "Later diagnostic, no altered registered endpoint or fitted model."}


def infinite_survival(p: Q, start: int) -> Q:
    if p <= Q(1, 2):
        return Q(0)
    return 1 - ((1 - p) / p) ** start


def barrier_ruin(p: Q, start: int, upper: int) -> Q:
    if p == Q(1, 2):
        return 1 - Q(start, upper)
    ratio = (1 - p) / p
    return (ratio ** start - ratio ** upper) / (1 - ratio ** upper)


def infinite_and_cap_check() -> dict:
    cap15_path_checks = 0
    for outcomes in product((0, 1), repeat=12):
        uncapped = capped = 3
        for outcome in outcomes:
            if uncapped == 0:
                break
            uncapped += 2 * outcome - 1
            capped = min(15, capped + 2 * outcome - 1)
            require(uncapped == capped, "Capacity 15 preserves every live path boundary")
        cap15_path_checks += 1
    harmonic_rows = 0
    for p in (Q(1, 10), Q(1, 2), Q(9, 10)):
        for upper in (4, 7, 12):
            require(barrier_ruin(p, 0, upper) == 1, "Lower hitting boundary")
            require(barrier_ruin(p, upper, upper) == 0, "Upper hitting boundary")
            for m in range(1, upper):
                require(barrier_ruin(p, m, upper)
                        == p * barrier_ruin(p, m + 1, upper)
                        + (1 - p) * barrier_ruin(p, m - 1, upper),
                        "Finite-barrier harmonic equation")
                harmonic_rows += 1
        require(survival(12, p, 3, 15) == survival(12, p, 3), "Capacity 15 equality")
        require(enumerate_survival(12, p, 3, 15) == survival(12, p, 3),
                "Capacity 15 independent path check")
        for m in (1, 2, 3):
            require(survival(8, p, m, 3) == enumerate_survival(8, p, m, 3),
                    "Active cap recurrence/enumeration")
        for capacity in (3, 5):
            for blocks in (1, 2, 4):
                bound = (1 - (1 - p) ** capacity) ** blocks
                require(survival(blocks * capacity, p, 3, capacity) <= bound,
                        "Disjoint-failure-block bound")
    good = infinite_survival(Q(9, 10), 3)
    e = calibration_error(5)
    learned = (1 - e) * good
    require(good == Q(728, 729), "Infinite favorable value")
    require(learned == Q(3094, 3125), "Infinite learned value")
    require(infinite_survival(Q(0), 3) == 0 and infinite_survival(Q(1), 3) == 1,
            "Deterministic endpoints")
    return {"status": "passed", "finite_barrier_harmonic_rows": harmonic_rows,
            "unlimited": {"given_orientation": exact(good), "learned": exact(learned),
                          "always_follow": exact(good / 2), "ignore": exact(Q(0)),
                          "certain_reversal": exact(e * good)},
            "capacity_15_preserves_all_twelve_step_paths": True,
            "capacity_15_path_strings_checked": cap15_path_checks,
            "cap_proof": "Any B consecutive failures empty a buffer at most B. "
                         "Independent disjoint B-trial blocks give the bound "
                         "[1-(1-p)^B]^k, tending to zero for p<1.",
            "limit_proof": "Events {tau_0<tau_N} increase to eventual ruin, "
                           "since every finite-time ruin path has a finite prior maximum.",
            "scope": "Analytic limit proofs, finite calculations are supporting checks."}


def paid_calibration_check() -> dict:
    good, bad = Q(9, 10), Q(1, 10)
    enumerated = {}
    for m in (1, 3, 5, 6):
        for p in (bad, good):
            enumerated[p, m] = enumerate_survival(12, p, m)
            require(enumerated[p, m] == survival(12, p, m), "Paid deployment enumeration")
    ignore = survival(12, Q(1, 2), 6)
    require(ignore == enumerate_survival(12, Q(1, 2), 6), "Paid ignore enumeration")
    rows = [{"policy": "ignore", "labels": 0, "deployment_start": 6,
             "correct_orientation": None, "continuation": exact(ignore)},
            {"policy": "always_follow", "labels": 0, "deployment_start": 6,
             "correct_orientation": exact(Q(1, 2)),
             "continuation": exact((survival(12, good, 6) + survival(12, bad, 6)) / 2)}]
    expected = {1: Q(18031390157, 20000000000),
                3: Q(1213400832263, 1250000000000),
                5: Q(137701662242869, 156250000000000)}
    for n in (1, 3, 5):
        e = calibration_error(n)
        for q in (bad, good):
            require(e == enumerate_calibration_error(n, q), "Paid calibration enumeration")
        m = 6 - n
        result = (1 - e) * survival(12, good, m) + e * survival(12, bad, m)
        independent_result = ((1 - enumerate_calibration_error(n, good)) * enumerated[good, m]
                              + enumerate_calibration_error(n, good) * enumerated[bad, m])
        require(result == independent_result == expected[n], "Paid exact result")
        rows.append({"policy": f"learn_{n}", "labels": n, "deployment_start": m,
                     "correct_orientation": exact(1 - e), "continuation": exact(result)})
    require(expected[3] > expected[5] and ignore > expected[5], "Cost reverses comparison")
    require(calibration_error(5) < calibration_error(3), "Evidence improves orientation")
    return {"status": "passed", "pre_calibration_budget": 6, "price_per_label": 1,
            "deployment_horizon": 12, "rows": rows,
            "scope": "Stipulated cost, fixed label counts, no apparatus or adaptive optimum."}


def natural_gradient_check() -> dict:
    for a in (Q(0), Q(1, 5), Q(3, 5), Q(1)):
        for p in (Q(1, 10), Q(1, 2), Q(9, 10)):
            metric, derivative = 1 / (p * (1 - p)), (p - a) / (p * (1 - p))
            velocity = -derivative / metric
            require(velocity == a - p, "Natural-gradient identity")
            require(derivative * velocity == -(p - a) ** 2 / (p * (1 - p)),
                    "Exact nonpositive loss derivative")
            logit_velocity = (a - p) / (p * (1 - p))
            require(p * (1 - p) * logit_velocity == velocity, "Coordinate flow identity")

    records_checked = 0
    for record in product((0, 1), repeat=5):
        k, a = sum(record), Q(sum(record), 5)
        for decay in (Q(1, 100), Q(1, 3), Q(1, 2), Q(99, 100)):
            fitted = a + (Q(1, 2) - a) * decay
            require(fitted - Q(1, 2) == Q(2 * k - 5, 10) * (1 - decay),
                    "Exact threshold factorization")
            require((fitted > Q(1, 2)) == (k >= 3), "Same retained orientation")
            require(0 < fitted < 1, "Interior preserved at positive finite time")
        records_checked += 1

    for a in (0.0, 0.2, 0.6, 1.0):
        for p0 in (0.1, 0.5, 0.9):
            loss = lambda p: -a * log(p) - (1 - a) * log(1 - p)
            for duration in (0.1, 1.0, 4.0):
                fitted = a + (p0 - a) * exp(-duration)
                require(loss(fitted) <= loss(p0) + 1e-14, "Exact-flow log loss")

    e = Q(0)
    for q in (Q(1, 10), Q(9, 10)):
        good_probability = Q(0)
        for record in product((0, 1), repeat=5):
            k = sum(record)
            fitted = Q(k, 5) + (Q(1, 2) - Q(k, 5)) / 2
            if (fitted > Q(1, 2)) == (q > Q(1, 2)):
                good_probability += q ** k * (1 - q) ** (5 - k)
        require(1 - good_probability == calibration_error(5), "Same episode mixture")
        e += (1 - good_probability) / 2
    deployment = (1 - e) * survival(12, Q(9, 10), 3) + e * survival(12, Q(1, 10), 3)
    require(deployment == Q(30940570672769, 31250000000000), "Same deployment result")
    likelihood_good = Q(9, 10) ** 3 * Q(1, 10) ** 2
    likelihood_bad = Q(1, 10) ** 3 * Q(9, 10) ** 2
    posterior = likelihood_good / (likelihood_good + likelihood_bad)
    fitted_example = Q(3, 5) + (Q(1, 2) - Q(3, 5)) / 2
    require(posterior == Q(9, 10) and fitted_example == Q(11, 20),
            "Fitted correctness differs from posterior orientation probability")
    p0, a, step = 0.2, 0.8, 0.5
    p_euler = p0 + step * (a - p0)
    phi_euler = log(p0 / (1 - p0)) + step * (a - p0) / (p0 * (1 - p0))
    p_from_phi = 1 / (1 + exp(-phi_euler))
    require(not isclose(p_euler, p_from_phi), "Finite Euler steps are not invariant")
    return {"status": "passed", "calibration_records_checked": records_checked,
            "bridge": "p(tau)-1/2=((2K-5)/10)*(1-exp(-tau)). For tau>0 this "
                      "has the sign of 2K-5, giving the original majority setting.",
            "tau_zero": "Tie for every record, not covered by strict-threshold equivalence.",
            "same_deployment_continuation": exact(deployment),
            "example_at_tau_log_2": {"K": 3, "fitted_p": exact(Q(11, 20)),
                                      "posterior_q_equals_0_9": exact(posterior)},
            "euler_counterexample": {"p_coordinate_step": p_euler,
                                      "logit_coordinate_step_mapped_back": p_from_phi},
            "scope": "Standard natural-gradient method with a chosen loss and optimization "
                     "clock. Fitted p is not posterior orientation probability. "
                     "No architecture validation or physical implementation."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="Explicitly save the JSON report at this path as well as stdout.")
    args = parser.parse_args()
    checks = {"original_result": original_check(), "F04_rule150": rule150_check(),
              "F05_infinite_and_capped": infinite_and_cap_check(),
              "F06_paid_calibration": paid_calibration_check(),
              "F07_natural_gradient_bridge": natural_gradient_check()}
    report = {"schema": "v47-constructive-mathematics-checks-v1", "version": "v47",
              "all_checks_passed": True, "check_group_count": len(checks),
              "dependencies": "Python standard library only", "checks": checks,
              "evidence_status": "Analytic teaching extensions and later diagnostics. "
                                 "Not historical reruns, physical experiments or human-reader tests."}
    serialized = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
