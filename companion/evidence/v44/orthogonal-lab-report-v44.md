# v44 independent numerical lab

The new `scripts/orthogonal-checks-v44.py` implements all nine numerical groups from the v43 orthogonal review using only Python's standard library. It imports no earlier calculator. Exact fractions support the finite enumerations, radius construction, resource calculations and sample descriptor floor. Floating-point arithmetic is limited to reproducing the frozen fitted predictions and their errors, and displaying logarithmic bounds.

The recorded run passed all nine groups and 8,069 always-on assertions. Isolated Python execution from an empty working directory produced valid stdout JSON and created no files. Separate runs accepted the extracted companion root and its E3 directory, reproduced the same E3 results, and left every input byte unchanged.

The frozen E3 experiment remains unchanged. The lab reads the eight original artifact byte streams from the v43 companion ZIP, verifies each against its pinned SHA-256 hash and manifest, reconstructs all 30 responses from integer changed/comparison counts, reconstructs the six frozen prediction types and three family-specific predictions, and checks the original aggregate errors. It does not regenerate the original probe trajectories or training: the selected bundle contains a probe-set commitment hash but not the initial probe arrays or complete historical materializer.

## Results and interpretation

The original predictor MSE is `0.04651764075889651`. The later clipped-prediction diagnostic has MSE `0.014347580116304564`, removing about `69.1566900594%` of that squared error. On these same 30 rows, clipping scores below the frozen constant and disjoint-union comparators, and above the frozen memory comparator. This is a post-test comparison; the registered unclipped test still failed.

Grouping the actual 30 exposed rows by their original descriptor triples gives group sizes 7, 17, 5 and 1. Their within-group contribution is exactly computed from the count ratios, approximately `0.006511098182057845`. This is an empirical minimum for these observed responses and these descriptor groups. It is neither a population floor nor a trained or deployable predictor. The complete original and clipped within-plus-between decompositions appear in the JSON.

All 256 ECA rules receive independent one-step and two-step influence profiles. Each profile is checked under reflection, joint state/rule complement and their combination: 768 recoding comparisons, with both time depths checked in every comparison. Explicit Boolean formulas independently cross-check rules 24 and 44. Their one-step external imbalances are 0 and 1/2, while their two-step noncentral means both remain 1/4. These are later diagnostics on exposed rules, not held-out predictive validation.

The supplied-radius two-ReLU classifier is checked on 1,407 exact rational cases including exact boundaries and points on either side. Its general classification equivalence also follows from the absolute-value identity; the finite grid is not substituted for that argument.

Backward recursion, forward propagation and complete 4,096-path enumeration agree on the original twelve-update resource model. Additional small-horizon enumeration checks cover several buffers and probabilities. The first integer follow/ignore ranking crossover is 19. The stipulated episode-level reversal crossing is exactly `5903024594197/15308200938394`, approximately `0.3856119095`. Duplicate calibration labels and the ideal infinite-resource extension are separately labelled stress tests. The infinite extension grants unbounded storage and functioning apparatus indefinitely; adding a positive independent fatal apparatus hazard gives zero indefinite apparatus survival.

The remaining groups check detector interval endpoint conventions, all 32 calibration histories under both report values for parameter/state implementations, the complete coupled-counterfactual bit examples, and every deterministic policy on a 101-point binary-channel probability grid.

## Rerun and data interface

From the book directory:

```text
python -B scripts/orthogonal-checks-v44.py --companion output/companion/how-matter-learns-to-continue-v43-companion.zip
python -B scripts/orthogonal-checks-v44.py --companion output/companion/how-matter-learns-to-continue-v43-companion.zip --output reviews/publication-v44/evidence/orthogonal-results-rerun.json
```

The first command writes only JSON to stdout. `--output` writes only the explicitly named path; its parent must exist. `--companion` also accepts an extracted companion root or its `evidence/e3` directory. With no explicit input, the script looks for nearby `evidence/e3` and the repository's v43 ZIP. If neither exists, the mathematical checks still run, but the frozen-data section explicitly reports `not_run`; it must not be represented as frozen-data verification.

The main result file is `orthogonal-results-v44.json`. Its stable top-level group keys identify the nine groups. The E3 record is `groups.e3_quantitative_error_diagnosis.frozen_e3`, whose `original_record` and `later_diagnostics` fields separate evidence classes. Runtime, script hash, source ZIP hash, all eight artifact hashes, exact values, assumptions and named assertion counts are included. A rerun can differ in timestamps, execution path, platform or duration; numerical results and frozen-data hashes should agree.

No human transfer assessment was administered. These checks establish the specified mathematics and frozen-record arithmetic, not independent empirical confirmation of the wider theoretical proposal.
