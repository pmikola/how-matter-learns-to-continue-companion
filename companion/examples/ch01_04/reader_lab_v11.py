"""Read-only reader experiments for the connected chapters 2-4.
Run from the book directory: python examples/ch01_04/reader_lab_v11.py --help
Archives are opened read-only. No predictions overwrite recorded experiments.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np

BOOK = Path(__file__).resolve().parents[2]
RING = BOOK / "outputs/chapter-one-draft/v5/ring/data-and-models.npz"
MEMORY = BOOK / "outputs/chapters-three-six/v7/experiments-memory/data.npz"

def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("Use a finite number.")
    return number

def representation(index=0):
    with np.load(RING, allow_pickle=False) as data:
        if not 0 <= index < len(data["test_x"]):
            raise ValueError("Point index must be between 0 and 4999.")
        x = data["test_x"][index].copy()
        W, b, v, c = [data["width256_step5000_" + key].copy() for key in ("W", "b", "v", "c")]
        h = np.maximum(x @ W + b, 0)
        terms = h * v
        positive = float(terms[terms > 0].sum())
        negative = float(-terms[terms < 0].sum())
        score = float(h @ v + c)
        reconstructed = positive - negative + float(c)
        probability = float(np.exp(-np.logaddexp(0, -score)))
        radius = float(np.linalg.norm(x))
        return dict(point_index=index, input=x.tolist(), hidden_units=len(h),
                    positive_total=positive, negative_magnitude_total=negative,
                    bias=float(c), original_score=score, reconstructed_score=reconstructed,
                    score_error=abs(score-reconstructed), probability_ring=probability,
                    chosen_label=int(score >= 0), true_label=int(data["test_y"][index]),
                    radius=radius, radius_rule_label=int(.45 <= radius <= .85),
                    parameters_changed=False, source="archived 256-unit model after 5000 updates")

def motion(current=0., previous=None):
    if not math.isfinite(current) or (previous is not None and not math.isfinite(previous)):
        raise ValueError("Positions must be finite.")
    with np.load(MEMORY, allow_pickle=False) as data:
        prefix = "current_model" if previous is None else "history_model"
        x = np.array([current] if previous is None else [previous, current]) / 3
        W, b, v, c = [data[prefix + "_" + k].copy() for k in ("w1", "b1", "w2", "b2")]
        predicted = float((np.maximum(x @ W + b, 0) @ v + c).item())
    result = dict(current=current, previous=previous, learned_prediction=predicted,
                  source="archived feedforward model", parameters_changed=False,
                  assumptions="constant velocity, equal observation intervals; no new training")
    if previous is None:
        result.update(possible_next_positions=[current-1, current+1],
                      exact_squared_error_optimum=current,
                      balanced_pair_minimum_rmse=1.)
    else:
        result.update(hand_built_prediction=2*current-previous,
                      learned_minus_hand=predicted-(2*current-previous),
                      warning="Exact hand prediction needs constant velocity and exact observations.")
    return result

def entropy(probability):
    return -sum(p*math.log2(p) for p in (probability,1-probability) if p > 0)

def sensor(readings=(-.6,), sigma=.75, prior=.5):
    readings = tuple(readings)
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("Sigma must be positive and finite, and prior strictly between 0 and 1.")
    if not all(math.isfinite(y) for y in readings):
        raise ValueError("Readings must be finite.")
    variance = sigma * sigma
    if not math.isfinite(variance) or variance == 0:
        raise ValueError("Sigma exceeds the numerical range of this small demonstration.")
    odds_log = math.log(prior/(1-prior))
    stages = [dict(observations=0, reading=None, posterior_right=prior,
                   posterior_left=1-prior, mean_next_position=2*prior-1,
                   entropy_bits=entropy(prior))]
    for i, reading in enumerate(readings, 1):
        # Y = -V + Gaussian(0, sigma^2). Measurements repeat the SAME past position.
        odds_log += -2*reading/variance
        if not math.isfinite(odds_log):
            raise ValueError("Inputs exceed the numerical range of this small demonstration.")
        p = float(np.exp(-np.logaddexp(0, -odds_log)))
        stages.append(dict(observations=i, reading=reading, posterior_right=p,
                           posterior_left=1-p, mean_next_position=2*p-1,
                           entropy_bits=entropy(p)))
    return dict(sigma=sigma, prior_right=prior, stages=stages,
                source="analytic Bayesian model, not a trained network",
                assumptions="current position exactly 0; V in {-1,+1}; repeated errors independent given V",
                readings_are="chosen illustrative measurements of the same past position")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    r = sub.add_parser("representation", help="Follow one saved ring input into contribution coordinates.")
    r.add_argument("--index", type=int, default=0)
    m = sub.add_parser("motion", help="Compare the saved model with a hand-built constant-velocity predictor.")
    m.add_argument("--current", type=finite, default=0.)
    m.add_argument("--previous", type=finite)
    s = sub.add_parser("sensor", help="Change readings, prior or sensor noise without training.")
    s.add_argument("--readings", nargs="+", type=finite, default=[-.6])
    s.add_argument("--sigma", type=finite, default=.75)
    s.add_argument("--prior", type=finite, default=.5)
    args = parser.parse_args()
    try:
        if args.task == "representation":
            result = representation(args.index)
        elif args.task == "motion":
            result = motion(args.current, args.previous)
        else:
            result = sensor(args.readings, args.sigma, args.prior)
    except (ValueError, FloatingPointError, OverflowError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
