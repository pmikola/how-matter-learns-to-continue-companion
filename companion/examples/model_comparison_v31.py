"""Exact observation-relative comparison of Chapter 14's two lamp models.

Standard library only. No files are written. Every probability is a declared
model prediction, not a new empirical result or probability over universes.
"""
from fractions import Fraction as F
from itertools import product
import json


def total_variation(p, q):
    assert len(p) == len(q) and sum(p) == sum(q) == 1
    assert all(x >= 0 for x in (*p, *q))
    return sum(abs(a-b) for a, b in zip(p, q)) / 2


def best_single_draw_decoder(p, q):
    # Equal prior probabilities for A and B, selected for this exercise.
    scores = []
    for choices in product((0, 1), repeat=len(p)):
        scores.append(sum(p[i] if choice == 0 else q[i]
                          for i, choice in enumerate(choices)) / 2)
    return max(scores)


def calculate():
    # Outcome order: Y=0, Y=1. The same fair U in both models/protocols.
    # Passive A: X=U,Y=X. Passive B: X=U,Y=U.
    # Action replaces only the assignment for X by X=1.
    protocols = {
        'passive_Y': ((F(1, 2), F(1, 2)), (F(1, 2), F(1, 2))),
        'force_X_on_then_read_Y': ((F(0), F(1)), (F(1, 2), F(1, 2))),
    }
    result = {}
    for protocol, (p, q) in protocols.items():
        distance = total_variation(p, q)
        decoder = best_single_draw_decoder(p, q)
        assert decoder == (1+distance)/2
        result[protocol] = {'A': list(map(float, p)), 'B': list(map(float, q)),
                            'total_variation': float(distance),
                            'best_equal_prior_single_draw_success': float(decoder)}
    assert result['passive_Y']['total_variation'] == 0
    assert result['force_X_on_then_read_Y']['total_variation'] == .5
    assert result['force_X_on_then_read_Y']['best_equal_prior_single_draw_success'] == .75
    return result


if __name__ == '__main__':
    print(json.dumps(calculate(), indent=2))
