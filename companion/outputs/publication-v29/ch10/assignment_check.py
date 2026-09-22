"""Read-only exact reproduction of Chapter 10's constructed error channel.

The standard assignment table and designed sequence match the archived v8
experiment. No archived module is imported and no output file is written.
Table 1: https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi
"""

from fractions import Fraction
from itertools import product
from math import prod

BASES = "UCAG"
CODONS = ["".join(letters) for letters in product(BASES, repeat=3)]
ASSIGNMENTS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODE = dict(zip(CODONS, ASSIGNMENTS))
SEQUENCE = ["AUG", "UUU", "GGU", "AAA", "CUC", "UAC", "AAC", "GCU", "CCU", "UGG"] * 4


def check(epsilon=Fraction(1, 100)):
    """Return exact-string and all-assignment probabilities for this sequence."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be a probability")
    assert len(CODE) == len(ASSIGNMENTS) == 64
    retained = {}
    for original in CODONS:
        row = {}
        for changed in CODONS:
            distance = sum(a != b for a, b in zip(original, changed))
            row[changed] = (1 - epsilon) ** (3 - distance) * (epsilon / 3) ** distance
        assert sum(row.values()) == 1
        retained[original] = sum(p for changed, p in row.items() if CODE[changed] == CODE[original])
        assert retained[original] >= (1 - epsilon) ** 3
    exact = (1 - epsilon) ** (3 * len(SEQUENCE))
    assigned = prod(retained[codon] for codon in SEQUENCE)
    assert assigned >= exact
    return exact, assigned


if __name__ == "__main__":
    exact, assigned = check()
    print("All 64 conditional rows sum exactly to one.")
    print(f"Same nucleotide string: {float(exact):.10f}")
    print(f"All assignments retained: {float(assigned):.10f}")
