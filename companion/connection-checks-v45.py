#!/usr/bin/env python3
"""Independent v45 connection checks; Python standard library only.

Print JSON without writing by default. --output writes the same report to an
explicit path with an existing parent. --e3 accepts the frozen per_class.jsonl
for optional read-only P06 arithmetic. No training, E3 trajectory regeneration,
PDF modification, external data, or human-reader validation is performed.
The implementation was written from the declared models, not copied from R2.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import sys
import zipfile

E3_ROWS_SHA256 = "a3df4ef9eceb693f821cf1886bd76d7db6e2d3170ea3702f1309fad6f73106b6"
REVIEW_SHA256 = "7ca9215df96f942a75af01097dc4ce26979139f32f33363c6d518032cf74a923"
ASSERTIONS = Counter()


def require(condition, label):
    ASSERTIONS[label] += 1
    if not condition:
        raise AssertionError(label)


def close(actual, expected, label, tolerance=1e-12):
    require(math.isclose(float(actual), float(expected), rel_tol=tolerance,
                         abs_tol=tolerance), label)


def number(value):
    return {"exact": str(value), "decimal": float(value)}


def bernoulli(record, probability):
    return math.prod(probability if bit else 1-probability for bit in record)


def alive_mass(probability, horizon, amount=3):
    """Forward absorbing probability transport, with no free paths after ruin."""
    masses = {amount: F(1)} if amount > 0 else {}
    for _ in range(horizon):
        following = defaultdict(F)
        for resource, mass in masses.items():
            following[resource+1] += mass*probability
            if resource > 1:
                following[resource-1] += mass*(1-probability)
        masses = following
    return sum(masses.values(), F(0))


def optimizer_closure():
    rows = []
    for w, v in ((F(1), F(1)), (F(2), F(1, 2))):
        # Positive w implies v ReLU(w*x)=(v*w) ReLU(x) on the entire real line.
        require(w > 0 and w*v == 1, "global present ReLU function equality")
        score, eta = v*w, F(1, 10)
        dw, dv = score*v, score*w  # gradient of (v*ReLU(w)-0)^2/2
        wn, vn = w-eta*dw, v-eta*dv
        require(wn > 0, "updated ReLU coefficient describes the entire function")
        rows.append({"parameters": [str(w), str(v)], "updated_parameters": [str(wn), str(vn)],
                     "next_function_coefficient": number(wn*vn)})
    require([row["next_function_coefficient"]["exact"] for row in rows] == ["81/100", "117/200"],
            "different next functions after the specified simultaneous gradient step")
    return {"rows": rows, "factorization": "H(U(theta))=G(H(theta)) cannot hold on this domain",
            "scope": "H is the entire score function, not a contribution summary or one input score. Fixed example x=1,y=0; half-squared loss; eta=1/10. No claim about every optimizer."}


def calibration_bayes():
    count_posteriors = {}
    error = {F(1, 10): F(0), F(9, 10): F(0)}
    for record in product((0, 1), repeat=5):
        k = sum(record)
        high, low = bernoulli(record, F(9, 10)), bernoulli(record, F(1, 10))
        posterior = high/(high+low)
        require(high/low == F(9)**(2*k-5), "sequence likelihood ratio equals count likelihood ratio")
        require((posterior > F(1, 2)) == (k >= 3), "equal-prior Bayes decision is majority")
        require(k not in count_posteriors or count_posteriors[k] == posterior, "count retains all two-source evidence")
        count_posteriors[k] = posterior
        for q in error:
            if (k >= 3) != (q == F(9, 10)):
                error[q] += bernoulli(record, q)
    require(set(error.values()) == {F(107, 12500)}, "symmetric exact five-label error")
    return {"records_checked": 32, "error": number(error[F(9, 10)]),
            "posterior_high_by_count": {k: number(v) for k, v in sorted(count_posteriors.items())},
            "scope": "Two Bernoulli candidates, equal priors, independent labels conditional on one fixed environment."}


def path_not_endpoint():
    positive_prefixes, positive_endpoint = Counter(), Counter()
    for path in product((-1, 1), repeat=12):
        resources, current = [3], 3
        for step in path:
            current += step
            resources.append(current)
        wins = path.count(1)
        if min(resources) > 0:
            positive_prefixes[wins] += 1
        if resources[-1] > 0:
            positive_endpoint[wins] += 1
    survival = {}
    for p in (F(1, 10), F(1, 2), F(9, 10)):
        enumerated = sum((n*p**k*(1-p)**(12-k) for k, n in positive_prefixes.items()), F(0))
        require(enumerated == alive_mass(p, 12), "forward absorbing transport equals exhaustive path counting")
        survival[str(p)] = number(enumerated)
    fair_endpoint = F(sum(positive_endpoint.values()), 4096)
    require(survival["1/2"]["exact"] == "627/1024", "fair survival fraction")
    require(fair_endpoint == F(1651, 2048), "free endpoint positivity fraction")
    require(F(627, 1024) < fair_endpoint, "the two events differ")
    examples = []
    for path in ((-1, -1, -1, 1, 1, 1), (1, 1, 1, -1, -1, -1)):
        values = [3]
        for step in path:
            values.append(values[-1]+step)
        examples.append(values)
    require(examples[0][-1] == examples[1][-1] and min(examples[0]) == 0 < min(examples[1]), "same free endpoint but distinct survival")
    # The continuous supply-free pool has derivative -k*M<=0, hence its minimum
    # on [0,Delta] is M(Delta). Check actual declared boundary and both sides.
    bound = .1*20*math.exp(.1*5)
    close(bound/.1*math.exp(-.1*5), 20, "pool boundary endpoint equals threshold")
    require(2/.1*math.exp(-.5) < 20 < 5/.1*math.exp(-.5), "pool boundary separates stated candidates")
    return {"paths_checked": 4096, "survival": survival, "free_endpoint_fair": number(fair_endpoint),
            "free_sum_examples": examples, "prefix_counts_by_successes": dict(sorted(positive_prefixes.items())),
            "scope": "Free sums are arithmetic comparisons, never resurrected absorbing trajectories. Pool endpoint sufficiency follows from monotonicity; the discrete process needs every positive prefix."}


def policy_composition():
    high, low = alive_mass(F(9, 10), 12), alive_mass(F(1, 10), 12)
    require(high > low, "higher success has higher checked continuation")
    stationary = reversed_result = F(0)
    for calibration in product((0, 1), repeat=5):
        a, b = bernoulli(calibration, F(9, 10)), bernoulli(calibration, F(1, 10))
        posterior = a/(a+b)
        follow = posterior*high+(1-posterior)*low
        reverse = (1-posterior)*high+posterior*low
        require(follow-reverse == (2*posterior-1)*(high-low), "policy-value difference factorization")
        require((follow > reverse) == (sum(calibration) >= 3), "majority maximizes posterior two-policy continuation")
        for q, likelihood in ((F(9, 10), a), (F(1, 10), b)):
            p = q if sum(calibration) >= 3 else 1-q
            stationary += likelihood*alive_mass(p, 12)/2
            reversed_result += likelihood*alive_mass(1-p, 12)/2
    require(stationary == F(30940570672769, 31250000000000), "stationary composition exact")
    require(reversed_result == F(324168795981, 31250000000000), "reversed composition exact")
    horizon_rows = []
    for horizon in range(25):
        ignored = alive_mass(F(1, 2), horizon)
        followed = (alive_mass(F(1, 10), horizon)+alive_mass(F(9, 10), horizon))/2
        horizon_rows.append({"horizon": horizon, "ignore": number(ignored), "follow": number(followed)})
    first = next(row["horizon"] for row in horizon_rows if F(row["follow"]["exact"]) > F(row["ignore"]["exact"]))
    require(first == 19, "legacy first integer crossover retained")
    return {"stationary": number(stationary), "reversed": number(reversed_result),
            "first_integer_crossover": first, "horizon_rows": horizon_rows,
            "scope": "Conditional optimality only among two frozen follow/reverse policies, given the specified prior, five labels, twelve updates and no apparatus or calibration cost."}


def stage_marginals():
    calibrations, deployments = list(product((0, 1), repeat=5)), list(product((0, 1), repeat=3))
    tables = []
    for changed in (False, True):
        table = {}
        for c in calibrations:
            for d in deployments:
                table[c, d] = sum((bernoulli(c, q)*bernoulli(d, 1-q if changed else q)/2 for q in (F(1, 10), F(9, 10))), F(0))
        require(sum(table.values()) == 1, "joint stage law normalized")
        tables.append(table)
    for c in calibrations:
        require(sum(tables[0][c, d] for d in deployments) == sum(tables[1][c, d] for d in deployments), "calibration marginal preserved")
    for d in deployments:
        require(sum(tables[0][c, d] for c in calibrations) == sum(tables[1][c, d] for c in calibrations), "deployment marginal preserved")
    difference = max(abs(tables[0][cell]-tables[1][cell]) for cell in tables[0])
    require(difference > 0, "cross-stage coupling changed")
    return {"calibration_bits": 5, "deployment_bits": 3, "cells_per_joint_table": 256,
            "maximum_joint_cell_difference": number(difference),
            "scope": "Finite exact stage-law check. Swapping the equal-prior alternatives proves marginal invariance at any finite length. This does not change the twelve-update resource result."}


def conway_closure():
    boards = ({(2, 3), (3, 4), (4, 2), (4, 3), (4, 4)}, {(2, 2), (2, 4), (4, 2), (4, 3), (4, 4)})
    def coarse(board):
        return [sum((r, c) in board for r in range(2*i, 2*i+2) for c in range(2*j, 2*j+2)) for i in range(4) for j in range(4)]
    def step(board):
        out = set()
        # Finite 8x8 inspection window with exterior dead; witnesses and all
        # one-step successors remain strictly inside, so no wrap is used.
        for r in range(8):
            for c in range(8):
                neighbors = sum((i, j) in board for i in range(r-1, r+2) for j in range(c-1, c+2) if (i, j) != (r, c))
                if neighbors == 3 or neighbors == 2 and (r, c) in board:
                    out.add((r, c))
        require(all(0 < r < 7 and 0 < c < 7 for r, c in out), "Conway successor unaffected by window boundary")
        return out
    before, after = [coarse(board) for board in boards], [coarse(step(board)) for board in boards]
    require(before[0] == before[1], "Conway current coarse arrays agree")
    require([(i, a, b) for i, (a, b) in enumerate(zip(*after)) if a != b] == [(10, 1, 0)], "only named successor block differs")
    return {"present_counts": before[0], "successor_counts": after,
            "scope": "Declared five-cell closure witness only; independent dense enumeration, no E3 historical trajectories."}


def linear_propagator():
    for a, b, r0 in product((-2., 0., 3.), repeat=3):
        for t in (0., .125, 1., 3.):
            transient = (r0-a/2+b/4)*math.exp(-2*t)
            r = b*t/2+a/2-b/4+transient
            derivative = b/2-2*transient
            close(derivative, (a+b*t)-2*r, "polynomial forced solution satisfies hidden equation")
            if t == 0:
                close(r, r0, "forced solution retains initial state")
    for r0, t in product((-2., 0., 3.), (0., .125, 1., 3.)):
        r = math.exp(-t)+(r0-1)*math.exp(-2*t)
        derivative = -math.exp(-t)-2*(r0-1)*math.exp(-2*t)
        close(derivative, math.exp(-t)-2*r, "exponential probe satisfies hidden equation")
    return {"identity": "r(t)=exp(-2t)r0+integral_0^t exp(-2(t-s))q(s) ds",
            "scope": "Forced hidden equation only. A supplied probe input need not solve the full feedback pair, which also requires qdot=-q+r."}


def curvature():
    rows = []
    for name, w, exponent, scale in (("matter", F(0), F(2, 3), F(3, 2)), ("radiation", F(1, 3), F(1, 2), F(2))):
        first, second = exponent*scale, exponent*(exponent-1)*scale**2
        require(first == 1 and second == -(1+3*w)/2, "power-law Taylor coefficients agree with Friedmann ODE")
        rows.append({"component": name, "value": "1", "first_derivative": str(first), "second_derivative": str(second)})
    require(-(1+3*F(-1))/2 == 1, "cosmological-constant ODE agrees with exponential curvature")
    rows.append({"component": "positive cosmological constant", "value": "1", "first_derivative": "1", "second_derivative": "1"})
    for q, velocity, acceleration, dt in product((F(-2), F(0), F(3)), (F(-1), F(2)), (F(-1, 2), F(1, 3)), (F(1, 2), F(2))):
        previous = q-velocity*dt+acceleration*dt**2/2
        following = q+velocity*dt+acceleration*dt**2/2
        require(following-(2*q-previous) == acceleration*dt**2, "two-frame constant-velocity forecast misses acceleration term")
    return {"normalized_expansion": rows,
            "scope": "Same scale and rate, different specified contents and hence different model equations. Not nonuniqueness for identical complete data under one fixed cosmology; the motion comparison is not a derivation of cosmology."}


def clock_interval():
    scale, offset = F(6, 5), F(-3)
    require((scale*25+offset)-(scale*20+offset) == 6, "offset cancels in first clock duration")
    require((scale*30+offset)-(scale*20+offset) == 12, "offset cancels in second clock duration")
    for normalized_k in [F(i, 1000) for i in range(401)] + [F(1, 10), F(1, 5)]:
        kb = normalized_k/scale
        original = 5*normalized_k <= 1 and 10*normalized_k > 1
        transported = 6*kb <= 1 and 12*kb > 1
        require(original == transported == (F(1, 12) < kb <= F(1, 6)), "open/closed detector interval transported exactly")
    return {"interval_reference": "log(2)/10 < k <= log(2)/5", "interval_B": "log(2)/12 < k_B <= log(2)/6",
            "scope": "Known reset-free map B=1.2t-3; exponent and interval preserved, with lower endpoint open and upper closed. No unique rate is identified."}


def constraint_not_accuracy():
    target = (1., .5)
    length = math.sqrt(sum(x*x for x in target))
    normalized = tuple(x/length for x in target)
    loss = sum((a-b)**2 for a, b in zip(normalized, target))
    close(sum(x*x for x in normalized), 1., "unit normalization restores length constraint")
    require(loss > 0, "normalization worsens chosen target loss from zero")
    return {"loss_before": 0, "loss_after": loss,
            "scope": "Explicit reviewer target equal to the unnormalized column; demonstrates no generic task-loss guarantee, without disputing norm preservation."}


def metric_transport():
    centres = ((F(0), F(1)), (F(2), F(0)))
    original = [x*x+y*y for x, y in centres]
    raw = [(x/10)**2+y*y for x, y in centres]
    transported = [100*(x/10)**2+y*y for x, y in centres]
    require(original[0] < original[1] and raw[1] < raw[0], "raw rescaling changes nearest centre")
    require(original == transported, "transported metric restores all distances exactly")
    return {"original_squared_distances": list(map(str, original)), "raw_rescaled_squared_distances": list(map(str, raw)),
            "transported_squared_distances": list(map(str, transported)), "scope": "Reviewer toy only, not an E3 memorizer rerun or claim of improved E3 accuracy."}


def decomposition(e3):
    for q, prediction in product((F(-2), F(0), F(3, 2)), (F(-3), F(1, 5), F(4))):
        actual = ((prediction-(q-1))**2+(prediction-(q+1))**2)/2
        require(actual == 1+(prediction-q)**2, "Chapter 3 exact balanced-pair identity")
    for y, prediction in product((F(i, 10) for i in range(11)), (F(i, 10) for i in range(-20, 31))):
        clipped = max(F(0), min(F(1), prediction))
        require((clipped-y)**2 <= (prediction-y)**2, "bounded scalar projection cannot worsen squared error")
    synthetic = [([F(1, 5), F(3, 5)], F(-1, 5)), ([F(1, 10), F(4, 10), F(9, 10)], F(4, 5))]
    def split(groups):
        total = within = between = F(0)
        count = sum(len(ys) for ys, _ in groups)
        for ys, prediction in groups:
            mean = sum(ys, F(0))/len(ys)
            require(sum((y-mean for y in ys), F(0)) == 0, "group residual cross term vanishes exactly")
            total += sum(((y-prediction)**2 for y in ys), F(0))
            within += sum(((y-mean)**2 for y in ys), F(0))
            between += len(ys)*(mean-prediction)**2
        require(total == within+between, "exact within-plus-assigned-prediction identity")
        return {"total": number(total/count), "within": number(within/count), "between": number(between/count)}
    result = {"synthetic_identity": split(synthetic), "frozen_rows": {"status": "not_requested"},
              "scope": "Algebra plus optional post-test arithmetic on exposed saved rows. Group floor is an empirical property of those rows, not a population bound or a newly validated predictor."}
    if e3:
        if zipfile.is_zipfile(e3):
            with zipfile.ZipFile(e3) as archive:
                candidates = [name for name in archive.namelist() if name.endswith("/evidence/e3/analysis/per_class.jsonl")]
                require(len(candidates) == 1, "companion contains exactly one E3 saved-row artifact")
                raw = archive.read(candidates[0])
        else:
            raw = e3.read_bytes()
        require(sha256(raw).hexdigest() == E3_ROWS_SHA256, "P06 input exactly matches pinned frozen 30-row artifact")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        require(len(rows) == 30, "P06 has all thirty saved rows")
        buckets = defaultdict(list)
        for row in rows:
            key = tuple(row[k] for k in ("mean_local_influence", "normalized_neighborhood_size", "determinism_margin"))
            response = F(row["response_changed_count"], row["response_comparison_count"])
            prediction = F(str(row["law_prediction"]))
            buckets[key].append((response, prediction, row["class_representative"]))
        require(sorted(map(len, buckets.values())) == [1, 5, 7, 17], "P06 descriptor group counts retained")
        groups = []
        for group in buckets.values():
            require(len({p for _, p, _ in group}) == 1, "one saved prediction per identical-input group")
            groups.append(([y for y, _, _ in group], group[0][1]))
        original, clipped = split(groups), split([(ys, min(F(1), max(F(0), p))) for ys, p in groups])
        close(original["total"]["decimal"], .04651764075889651, "P06 original MSE retained")
        close(clipped["total"]["decimal"], .014347580116304564, "P06 clipped MSE retained")
        close(original["within"]["decimal"], .006511098182057845, "P06 sample floor retained")
        collided = {rule: (y, p) for group in buckets.values() for y, p, rule in group if rule in (24, 44)}
        require(collided[24][1] == collided[44][1] < 0, "collision still shares negative prediction")
        require(abs(collided[24][0]-collided[44][0]) == F(55, 7936), "saved collision response difference retained")
        result["frozen_rows"] = {"status": "passed", "path": str(e3), "sha256": E3_ROWS_SHA256,
                                  "original": original, "projected": clipped,
                                  "interpretation": "Saved decimal predictions and integer response counts, with exact rational decomposition; no model fitting or trajectory reproduction."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--e3", type=Path, help="Optional frozen analysis/per_class.jsonl or companion ZIP; exact row hash required")
    args = parser.parse_args()
    source = Path(__file__).resolve()
    if args.output and args.output.resolve() in {source, args.e3.resolve() if args.e3 else source}:
        parser.error("--output must not overwrite an input or this script")
    groups = {}
    functions = (optimizer_closure, calibration_bayes, path_not_endpoint, policy_composition, stage_marginals,
                 conway_closure, linear_propagator, curvature, clock_interval, constraint_not_accuracy, metric_transport)
    for index, function in enumerate(functions, 1):
        before = sum(ASSERTIONS.values())
        group = function()
        groups[f"M{index:02d}_{function.__name__}"] = {"status": "passed", "assertion_count": sum(ASSERTIONS.values())-before, **group}
    supplemental = decomposition(args.e3.resolve() if args.e3 else None)
    report = {"schema": "how-matter-learns-to-continue.v45.connections.v1", "status": "passed", "check_groups": len(groups),
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": {"script": source.name, "script_sha256": sha256(source.read_bytes()).hexdigest(),
                         "review": "how-matter-learns-to-continue-v44-graph-validation-R2.md", "review_sha256": REVIEW_SHA256},
              "runtime": {"python": sys.version, "dependencies": "standard library only"},
              "checks": groups, "supplemental_P06": supplemental,
              "assertions": {"total": sum(ASSERTIONS.values()), "by_label": dict(sorted(ASSERTIONS.items()))},
              "evidence_limit": "Independent calculations of declared teaching models and explicitly labelled reviewer examples. No E3 experiment or training rerun, missing trajectory regeneration, new natural evidence, or reader test."}
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
