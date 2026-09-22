#!/usr/bin/env python3
"""Independent, standard-library checks of four v46 reader operations.

These are analytic teaching constructions, not observed learner trajectories,
human-reader results, or a historical E3 rerun. No prior calculator is imported.
Default: print JSON, create no files. --output PATH writes to that explicit path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
from math import comb
from pathlib import Path
import platform


def require(condition, message):
    """Keep every acceptance check active under Python -O as well."""
    if not condition:
        raise AssertionError(message)


def number(value):
    value = F(value)
    return {"exact": str(value), "decimal": float(value)}


def selected_action(calibration, report):
    # Inputs are labeled correctness bits and the currently reported side.
    # The next hidden resource side is deliberately not an input.
    return report if sum(calibration) >= 3 else 1 - report


def next_amount(amount, action, resource_side):
    return 0 if amount == 0 else amount - 1 + 2 * (action == resource_side)


def surviving_path_counts(horizon=12, start=3):
    """Enumerate all binary outcome paths; discard any zero-reaching prefix."""
    counts = Counter()
    for outcomes in product((0, 1), repeat=horizon):
        amount = start
        for success in outcomes:
            amount += 2 * success - 1
            if amount <= 0:
                break
        else:
            counts[sum(outcomes)] += 1
    return counts


def path_probability(counts, success, horizon=12):
    return sum((count * success**wins * (1-success)**(horizon-wins)
                for wins, count in counts.items()), F(0))


def boundary_mass(success, horizon=12, start=3):
    """Independent forward propagation, retaining only positive amounts."""
    masses = {start: F(1)}
    for _ in range(horizon):
        following = {}
        for amount, mass in masses.items():
            for delta, chance in ((1, success), (-1, 1-success)):
                if amount + delta > 0:
                    following[amount+delta] = following.get(amount+delta, F(0)) + mass*chance
        masses = following
    return sum(masses.values(), F(0))


def n01():
    checked = 0
    for calibration, report, resource in product(product((0, 1), repeat=5), (0, 1), (0, 1)):
        action = selected_action(calibration, report)
        amount = next_amount(3, action, resource)
        require(action == (report if sum(calibration) >= 3 else 1-report), "retained action map")
        require(amount == (4 if action == resource else 2), "one-step resource accounting")
        require(selected_action(calibration, 1-report) == 1-action, "left-right symmetry")
        checked += 1
    pair = ((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))
    examples = []
    for report, resource in product((0, 1), repeat=2):
        actions = [selected_action(history, report) for history in pair]
        amounts = [next_amount(3, action, resource) for action in actions]
        examples.append({"report": ("LEFT", "RIGHT")[report],
                         "hidden_resource_for_scoring": ("LEFT", "RIGHT")[resource],
                         "chosen_sides": [("LEFT", "RIGHT")[action] for action in actions],
                         "next_amounts": amounts})
    require(examples[0]["next_amounts"] == [4, 2], "proposed left-report worked pair")
    require(examples[2]["next_amounts"] == [2, 4], "right report with resource held left")
    require(next_amount(0, 0, 0) == 0, "zero remains absorbing")
    return {"status": "passed", "case_count": checked, "calibration_histories": 32,
            "start_amount": 3, "example_histories": ["11100", "00011"],
            "example_policies": ["FOLLOW", "REVERSE"], "examples": examples,
            "scope": "Constructed single actions; hidden truth scores the action and is not a learner input. No typical trajectory or aggregate continuation is inferred."}


def n02_n03():
    paths = surviving_path_counts()
    high, low = (path_probability(paths, q) for q in (F(9, 10), F(1, 10)))
    for q in (F(0), F(1, 10), F(1, 2), F(9, 10), F(1)):
        require(path_probability(paths, q) == boundary_mass(q), "enumeration agrees with absorbing forward mass")
    require(high > low, "higher success has higher continuation")
    rows, weighted, count_mass = [], F(0), F(0)
    for count in range(6):
        like_high = comb(5, count)*F(9, 10)**count*F(1, 10)**(5-count)
        like_low = comb(5, count)*F(1, 10)**count*F(9, 10)**(5-count)
        posterior = like_high/(like_high+like_low)
        require(like_high/like_low == F(9)**(2*count-5), "count likelihood ratio")
        follow = posterior*high+(1-posterior)*low
        reverse = posterior*low+(1-posterior)*high
        require(follow-reverse == (2*posterior-1)*(high-low), "posterior policy-value identity")
        require((follow > reverse) == (count >= 3), "majority maximizes the two frozen choices")
        selected = follow if count >= 3 else reverse
        mass = (like_high+like_low)/2
        count_mass += mass
        weighted += mass*selected
        rows.append({"correct_count": count, "policy": "FOLLOW" if count >= 3 else "REVERSE",
                     "posterior_high_orientation": number(posterior),
                     "count_probability": number(mass), "selected_continuation": number(selected)})
    require(count_mass == 1, "count mixture is normalized")
    for count, expected in ((3, F(9, 10)), (4, F(729, 730)), (5, F(59049, 59050))):
        require(F(rows[count]["posterior_high_orientation"]["exact"]) == expected, "exact posterior reference")
    follow_high_joint = sum((F(row["count_probability"]["exact"])*F(row["posterior_high_orientation"]["exact"])
                             for row in rows if row["policy"] == "FOLLOW"), F(0))
    follow_mass = sum((F(row["count_probability"]["exact"]) for row in rows if row["policy"] == "FOLLOW"), F(0))
    bit_posterior = follow_high_joint/follow_mass
    require(follow_mass == F(1, 2), "equal-prior FOLLOW mass")
    require(bit_posterior == F(12393, 12500), "posterior conditional on FOLLOW is 0.99144")
    require(len({row["posterior_high_orientation"]["exact"] for row in rows[3:]}) == 3,
            "one FOLLOW bit merges three distinct original-count posteriors")
    bit_continuation = bit_posterior*high+(1-bit_posterior)*low
    n02 = {"status": "passed", "count_rows": rows,
           "follow_bit_probability": number(follow_mass),
           "posterior_high_given_follow_bit": number(bit_posterior),
           "continuation_given_follow_bit": number(bit_continuation),
           "scope": "The bit suffices for this two-policy decision, but cannot reconstruct confidence or continuation conditional on the original count. A posterior conditional on the bit still exists."}
    # Compute independently by ordered calibration histories and environment,
    # rather than reusing count-binomial probabilities.
    history_total = reversed_total = history_mass = F(0)
    for calibration in product((0, 1), repeat=5):
        for q in (F(1, 10), F(9, 10)):
            mass = F(1, 2)
            for correct in calibration:
                mass *= q if correct else 1-q
            success = q if sum(calibration) >= 3 else 1-q
            history_mass += mass
            history_total += mass*path_probability(paths, success)
            reversed_total += mass*path_probability(paths, 1-success)
    error = sum((comb(5, count)*F(9, 10)**count*F(1, 10)**(5-count) for count in range(3)), F(0))
    expected = F(30940570672769, 31250000000000)
    require(history_mass == 1, "ordered history mixture is normalized")
    require(weighted == history_total == bit_continuation == (1-error)*high+error*low == expected,
            "total expectation agrees across count/history/bit/calibration-error representations")
    require(reversed_total == F(324168795981, 31250000000000), "certain reversal preserves historical value")
    require(error == F(107, 12500), "calibration error is 0.00856")
    fair = path_probability(paths, F(1, 2))
    require(fair == F(627, 1024), "report-ignoring reference")
    n03 = {"status": "passed", "deployment_paths_enumerated": 4096,
           "positive_prefix_path_counts_by_successes": dict(sorted(paths.items())),
           "calibration_environment_cases": 64, "survival_high": number(high), "survival_low": number(low),
           "learned_continuation": number(weighted), "certain_reversal": number(reversed_total),
           "report_ignoring_continuation": number(fair), "calibration_error": number(error),
           "scope": "Twelve deployment updates from three, positive at every boundary, equal fixed environment weights, conditionally independent labels and deployment, frozen policy; no apparatus cost or global controller optimum."}
    return n02, n03


def n04():
    cases = 0
    example = None
    for position, velocity, acceleration, interval in product((F(-2), F(0), F(7, 3)),
            (F(-3), F(0), F(5, 2)), (F(-2), F(0), F(3)), (F(1, 2), F(1), F(2))):
        # Current time is zero; position and velocity are its state variables.
        previous_constant = position-velocity*interval+acceleration*interval**2/2
        following = position+velocity*interval+acceleration*interval**2/2
        forecast_constant = 2*position-previous_constant
        previous_coast = position-velocity*interval
        forecast_after = 2*position-previous_coast
        error_constant = following-forecast_constant
        error_after = following-forecast_after
        require(error_constant == acceleration*interval**2, "constant acceleration across both intervals")
        require(error_after == acceleration*interval**2/2, "acceleration begins after current frame")
        require(error_constant == 2*error_after, "protocol-specific factor of two")
        if acceleration == 0:
            require(error_constant == error_after == 0, "constant-velocity boundary")
        if (position, velocity, acceleration, interval) == (F(0), F(0), F(3), F(2)):
            example = {"current_position": number(position), "current_velocity": number(velocity),
                       "acceleration": number(acceleration), "interval": number(interval),
                       "true_next_minus_forecast_constant_two_intervals": number(error_constant),
                       "true_next_minus_forecast_acceleration_after_current": number(error_after)}
        cases += 1
    return {"status": "passed", "paired_protocol_cases": cases, "example": example,
            "scope": "Exact analytic counterfactuals in one-dimensional motion; no trained predictor was tested here. Each error means true next position minus the linear two-frame extrapolation. Position, velocity, acceleration and interval use consistent arbitrary length/time units."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="explicit JSON output; parent directory must exist")
    args = parser.parse_args()
    second, third = n02_n03()
    script = Path(__file__).resolve()
    result = {"schema_version": "reader-operations-v46/1", "status": "passed",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "script": {"path": str(script), "sha256": sha256(script.read_bytes()).hexdigest()},
              "environment": {"python": platform.python_version(), "implementation": platform.python_implementation()},
              "scope": "Four overlapping analytic acceptance groups, independently implemented for v46. Not empirical experiments, reader testing, or E3 trajectory reproduction.",
              "groups": {"N01": n01(), "N02": second, "N03": third, "N04": n04()}}
    output = json.dumps(result, indent=2, sort_keys=True)+"\n"
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
