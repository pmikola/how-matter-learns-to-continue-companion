"""Read-only checks for Chapter 11's illustrative lag and turnover models."""

from math import exp, isclose

for elapsed in (1, 6):
    response = exp(-elapsed / 3)
    reference = 2 - response
    print(f"time={10 + elapsed}, m={reference:.4f}, y={response:.4f}")
assert isclose(exp(-3 / 3), 1 / exp(1))
at_return = 100 * exp(-0.08 * 15)
after_ten = 100 + (at_return - 100) * exp(-0.08 * 10)
assert 30 < at_return < 31
assert at_return < after_ten < 100
print(f"M(35)={at_return:.4f}, M(45)={after_ten:.4f}")
