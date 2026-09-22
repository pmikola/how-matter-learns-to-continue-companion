# Answer key: decision and confidence

Read this after answering the [exercise](reader-exercise.md). These are computed consequences of the stated model, not reader-study results.

Both K = 3 and K = 5 retain FOLLOW. Both choose RIGHT after a RIGHT report. The bit preserves that decision, while it cannot identify which original count selected it.

With w = P(q = 0.9 given K), equal prior weights give w = 9^(2K−5)/(1 + 9^(2K−5)). For a retained FOLLOW policy, conditional twelve-update continuation is w S(0.9) + (1−w) S(0.1), where S(p) tests every update boundary from the initial amount three.

| Correct count | Policy | P(q = 0.9 given count) | Twelve-update continuation given count |
|---|---|---|---|
| 3 | FOLLOW | 9/10 = 0.9 | 0.89895179278 |
| 4 | FOLLOW | 729/730 = 0.998630136986 | 0.997265317479178 |
| 5 | FOLLOW | 59049/59050 = 0.999983065199 | 0.998613902661334 |

The bit still supports a posterior: P(q = 0.9 given FOLLOW) = 12393/12500 = 0.99144. That one posterior averages over the retained counts; it does not reconstruct each original-count posterior.

Averaging all six count-conditioned continuation values with their stipulated mixture probabilities gives 30940570672769/31250000000000 = 0.990098261528608. This is the existing aggregate result, approximately 99.01%. By symmetry, continuation conditioned only on FOLLOW also equals that aggregate here. It is different from continuation conditioned on a particular count.

The bit is sufficient for choosing the better of these two frozen orientation policies under these assumptions. It is insufficient for the different output of reconstructing confidence conditional on the original count. No claim of optimality over every possible controller follows.

N01 also checks the constructed action trace over 128 calibration/report/resource cases. With records 11100 and 00011, the policies are FOLLOW and REVERSE. A RIGHT report leads to RIGHT and LEFT; a hidden RIGHT resource gives next amounts four and two. If the hidden resource is LEFT, the amounts are two and four. One choice does not establish a twelve-update aggregate probability.

N03 independently checks total expectation against recurrence and path enumeration. N04 distinguishes constant acceleration across two intervals, with extrapolation error a Δt², from acceleration beginning just after the current frame, with error a Δt²/2. These are analytic counterfactuals, not tests of a trained predictor.

Executable evidence: `python -B reader-operations-v46.py`. Exact fractions and all four group results are in `evidence/v46/reader-operations-v46.json` in the delivered companion. That script's SHA-256 is `26a8dc8b496d9c3015a1906e76a2fc2aa394020a5acc9de4cbcc0a726e2e78fc`. The fresh delivery receipt separately records execution in the extracted package.
