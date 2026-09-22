"""Exact resource-learning reader for Chapter30. Standard library, stdout only.

The new v31 teaching construction is separate from the continuous pool and the
historical evidence. No training instrument, archived producer or file write.
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import comb
import json

def continuation_dp(p, horizon=12, start=3):
    """Backward recurrence; absorption at zero overrides the zero-horizon base."""
    @lru_cache(None)
    def s(h, m):
        if m == 0:
            return F(0)
        if h == 0:
            return F(1)
        return p*s(h-1, m+1)+(1-p)*s(h-1, m-1)
    return s(horizon, start)

def continuation_enumeration(p, horizon=12, start=3):
    """Independent check: examine all 2**horizon sequences, no recurrence."""
    total = F(0)
    valid_by_successes = [0]*(horizon+1)
    for outcomes in product((0, 1), repeat=horizon):
        amount = start
        for success in outcomes:
            amount += 2*success-1
            if amount == 0:
                break
        else:
            n = sum(outcomes)
            total += p**n*(1-p)**(horizon-n)
            valid_by_successes[n] += 1
    return total, valid_by_successes

def exact_record(value):
    return {"rational": str(value), "decimal": float(value), "ten_decimals": f"{float(value):.10f}"}

def calculate():
    hi, lo = F(9, 10), F(1, 10)
    e = sum(F(comb(5, j))*hi**j*lo**(5-j) for j in range(3))
    # Independent enumeration of calibration labels in each environment.
    for correct_probability, right_decision in [(hi, True), (lo, False)]:
        error = F(0)
        for labels in product((0, 1), repeat=5):
            if (sum(labels) >= 3) != right_decision:
                j = sum(labels)
                error += correct_probability**j*(1-correct_probability)**(5-j)
        assert error == e
    values, counts = {}, None
    for p in (lo, F(1, 2), hi):
        dp = continuation_dp(p)
        enumerated, counts = continuation_enumeration(p)
        assert dp == enumerated
        values[str(p)] = exact_record(dp)
    H, L, N = (continuation_dp(p) for p in (hi, lo, F(1, 2)))
    strategies = {
        "always_follow": (H+L)/2,
        "always_left": N,
        "learn_five_labels": (1-e)*H+e*L,
        "given_deployment_orientation": H,
        "learn_then_unannounced_reversal": e*H+(1-e)*L,
    }
    assert strategies["learn_five_labels"] > N > strategies["always_follow"]
    assert strategies["learn_then_unannounced_reversal"] < N
    avg_p = (1-e)*hi+e*lo
    assert continuation_dp(avg_p) != strategies["learn_five_labels"]
    # A path with terminal amount 3 can already have failed at an earlier boundary.
    bad = (0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1)
    assert 3+sum(2*x-1 for x in bad) == 3
    assert 3+sum(2*x-1 for x in bad[:3]) == 0
    model = {
        "model_id": "v31-learning-resource-continuation",
        "status": "new analytic teaching construction, not a historical or biological experiment",
        "environment": "Equal stipulated mixture of sensor correctness 0.9 and 0.1. Resource sides iid fair. Correctness draws iid and independent of side within environment. Calibration and deployment independent conditional on environment.",
        "calibration": "Five labeled trials, same opportunity for all four strategies. Follow iff at least three sensor reports were correct, otherwise reverse. One retained binary policy; frozen during deployment.",
        "deployment": {"start": 3, "horizon": 12, "update": "M(t+1)=M(t)-1+2*X(t)", "failure": "Absorbing on first update boundary at zero", "outcome": "Positive at every boundary 1 through 12"},
        "cost_scope": "Equal deployment starts after calibration; does not debit separate calibration, sensing, memory or actuator costs; abstract trial accounting only.",
        "boundary": "Sensor orientation reverses once after calibration, before deployment; no warning or relearning. Resource-side distribution and all other conditions stay fixed.",
        "calibration_error": exact_record(e),
        "conditional_continuation": values,
        "strategies": {k: exact_record(v) for k, v in strategies.items()},
        "independent_verification": {"calibration_sequences_per_environment": 32, "deployment_sequences_per_probability": 4096, "probabilities_checked": ["1/10", "1/2", "9/10"], "surviving_sequences_by_number_of_successes": counts, "exact_fraction_agreement": True},
        "wrong_average_probability_result_for_diagnostic_only": exact_record(continuation_dp(avg_p)),
        "source": "examples/resource_learning_v31.py",
        "generation_source": "examples/publication_v31_late.py",
    }
    return model

if __name__ == "__main__":
    print(json.dumps({"verified": True, "writes": False, "model": calculate()}, indent=2))

