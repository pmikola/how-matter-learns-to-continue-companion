# Optional calculations for the v44 reader

These are later analytical extensions of the book's declared examples. The original figures, fitted predictions and frozen experimental outcome remain unchanged. The independent v44 lab calculates these extensions using Python's standard library. The inherited scientific files retain their original authorship and edition records.

From the extracted companion's root directory, run:

```text
python -B orthogonal-checks-v44.py
```

The default prints JSON and writes no files. It discovers the adjacent `evidence/e3` directory. To save a fresh report outside the extracted bundle, use:

```text
python -B orthogonal-checks-v44.py --output ../v44-rerun.json
```

The output parent must already exist. An explicit `--companion` argument accepts a ZIP, extracted companion root or `evidence/e3` directory. The bundled `evidence/v44/orthogonal-results-v44.json` is a repository-run snapshot with its original execution paths and v43 input-archive hash. The new default invocation uses the frozen artifacts in this extraction. Its nine groups include both original arithmetic checks and later diagnostics, labelled separately in the JSON.

## One horizon is not every horizon

In Chapter 30, the amount starts at three and gains or loses one per update. Zero ends the episode. Write `S_h(p)` for the probability of remaining positive at every update through horizon `h`, with independent success probability `p` conditional on the episode's fixed setting.

Ignoring the report always gives `S_h(0.5)`. Always following gives `[S_h(0.9) + S_h(0.1)]/2` because the environment is selected once per episode with equal weights.

| Updates | Ignore report | Always follow |
|---:|---:|---:|
| 12 | 61.230469% | 50.023583% |
| 18 | 51.931763% | 49.934163% |
| 19 | 49.655533% | 49.932287% |
| 24 | 45.874381% | 49.931504% |

Nineteen is the first integer horizon where always following wins this pairwise comparison. Averaging the two episode probabilities differs from inserting the average success probability into one survival function. The fixed orientation creates dependence between the unconditional outcomes within an episode. The always-follow episodes that continue for a long time become concentrated in the favorable environment. Backward recurrence and forward probability propagation agree exactly through horizon 24 in the lab.

## One possible reversal before deployment

Keep the original twelve-update test. After calibration, flip the sensor orientation once with probability alpha, independently of the calibration outcomes. With probability `1-alpha`, leave it unchanged. The selected orientation then remains fixed throughout deployment, and the controller retains its frozen learned setting.

The original stationary and certainly reversed learned probabilities are:

```text
P0 = 0.990098261528608
P1 = 0.010373401471392
P(alpha) = (1-alpha) P0 + alpha P1
```

Always choosing left stays at `627/1024`. The crossing is

```text
alpha* = (P0 - 627/1024) / (P0 - P1)
       = 5903024594197 / 15308200938394
       ≈ 0.3856119095
```

Below this value the frozen learner has the higher probability of continuation, and above it always choosing left does. Alpha is a stipulated probability across episodes. It is not an observed environmental reversal rate. The crossing depends on the chosen horizon, buffer, payoffs and calibration procedure. Allowing deployment feedback to change the retained setting would ask a different question.

Five copies of one correctness event provide a related stress test. Their majority has orientation error `0.1`, rather than the `0.00856` from five independent events. With the otherwise unchanged stationary deployment rule, continuation becomes `0.89895179278`. This changes the evidence structure while keeping the apparent count of labels at five.

## An explicitly ideal infinite walk

Extend the same resource process to indefinitely many updates. This extension grants unbounded storage capacity, resource opportunities forever, a fixed success probability and independent increments conditional on that probability. It also grants indefinitely functioning sensing, memory and actuation, with no additional apparatus failure, damage, energetic cost or interruption. These assumptions define the mathematical question.

Starting from positive integer amount `m`, let `R_B(m)` be the probability of hitting zero before reaching an upper amount `B>m`. For `0<p<1`, put `r=(1-p)/p`. The finite-boundary recurrence and its solution are:

```text
R_B(m) = p R_B(m+1) + (1-p) R_B(m-1)
R_B(0) = 1, R_B(B) = 0

R_B(m) = (r^m-r^B)/(1-r^B)        when p != 1/2
R_B(m) = (B-m)/B                  when p = 1/2
```

The lab checks these boundary values and recurrence by exact rational arithmetic. This is the standard gambler's-ruin calculation, treated in Lehman, Leighton and Meyer's [MIT Mathematics for Computer Science, section 20.1](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/mit6_042js15_textbook.pdf#page=847).

Taking `B` upward makes the zero-before-`B` events increase to eventual ruin: every finite ruin path has a finite maximum. Their limiting probability therefore gives

```text
S_infinity(p; m) = 0                         for p <= 1/2
S_infinity(p; m) = 1 - ((1-p)/p)^m           for p > 1/2
```

The endpoint cases are immediate: `p=0` loses one unit every update and `p=1` never loses a unit. The lab's finite checks support the implemented formula. They do not replace the limiting argument.

Substituting the book's amount three, fixed episode mixture and independent-calibration error gives:

| Strategy in the ideal stationary model | Probability of never reaching zero |
|---|---:|
| Ignore report | 0 |
| Always follow | `364/729`, about 49.931413% |
| Learn from five independent labels | `3094/3125`, exactly 99.008% |

These probabilities concern the ideal resource walk under the granted conditions. They do not establish immortality of an implemented learner. For example, add an independent fatal apparatus hazard `h` at each update, where `0<h<=1`. Apparatus survival through `T` updates contributes the factor `(1-h)^T`, whose limit is zero. A positive probability of resource continuation under the original ideal assumptions does not remove that additional failure mechanism.

## The pairing is part of a paired response

Let `U` be a fair bit, `I` a binary intervention, and `xor` exclusive OR. Define two constructed response models:

```text
Model A: Y(I,U) = U
Model B: Y(I,U) = U xor I
```

For each fixed intervention, both models give probability one half for `Y=1`. Their intervention-specific distributions agree. The joint comparison across interventions depends on which noise values are paired:

| Noise correspondence between runs | Model A disagreement | Model B disagreement |
|---|---:|---:|
| Same `U` in both interventions | 0 | 1 |
| Independent `U` in each intervention | 1/2 | 1/2 |

The lab checks all bit cases. Separate terminal distributions do not determine this paired disagreement rate. The declared correspondence between runs supplies additional structure. Pearl's [Causal Inference](https://proceedings.mlr.press/v6/pearl10a.html), especially its structural counterfactual account, provides the broader framework for distinguishing intervention and counterfactual questions. The two-bit example here is an independently enumerated teaching construction.

Chapter 36 already declares matched randomness, and the held-out elementary rules in W7 are deterministic. This optional example is a stress test for transferring a paired-response protocol to another setting. It is not a newly discovered stochastic fault in the original W7 calculation.

## Where the new E3 calculation obtains its evidence

The eight frozen E3 artifacts retain their original hashes and provenance. The v44 lab reconstructs all 30 responses from the archived integer counts and reproduces the original fitted predictions and aggregate errors. It then calculates a separately labelled clipping diagnostic with MSE `0.014347580116304564` and a descriptor-group sample floor of approximately `0.006511098182057845`.

Those are analyses of exposed rows. The original predictor remains unclipped and its registered test remains a failure. The sample floor is not a population error bound or a trained predictor. The companion contains a commitment to the historical probes, not their complete initial bit arrays or training materializer, so reconstructing their archived arithmetic is distinct from regenerating the experiment.
