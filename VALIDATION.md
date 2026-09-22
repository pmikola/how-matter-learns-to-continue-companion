# Validation of the initial public companion

The v48 archive was rechecked locally on 22 September 2026 before publication. Publishing it does not turn historical experiments into newly performed experiments.

## Fresh execution

The following six commands completed successfully. Their JSON results were checked, not only their process exit codes:

```text
python -B verify_companion.py
python -B orthogonal-checks-v44.py
python -B connection-checks-v45.py --e3 evidence/e3/analysis/per_class.jsonl
python -B reader-operations-v46.py
python -B continuation-checks-v47.py
python -B reconstruct-w7-v47.py
```

The first command exercises 35 portable runs. The next four cover 9, 11, 4 and 5 numerical groups respectively. The final command independently reconstructs the specified W7 probes and compares their resulting counts with the saved evidence. All 152 extracted files remained unchanged.

Additional analytic checks covered the paired-motion squared-error identity, a Gaussian posterior and entropy, the lazy-walk return probability, the reduced phase-alignment equation, finite Bernoulli discrimination, a coordinate-dependent learning step, and the probability mass of the illustrated Gaussian ellipse.

## Preserved limitations

- This is not a fresh run of every historical neural-framework training experiment.
- A successful evidence check does not make a negative scientific result positive. The archived E3 transfer-null result and its false scientific-admission flag remain intact.
- Reader exercises and blank feedback forms are prepared materials, not reported human-subject results.
- The checks do not independently replicate external experimental papers or establish the book's wider speculative possibilities.
- The PDF navigation guides are bound to v48, as recorded in `companion/evidence/v48-compatibility.json`.

## Publication boundary

Only the checked reader archive and the small documents at this repository's root are published. No history from the private manuscript repository is copied. The `.git` directory, full book PDF, chapter drafts, account credentials and unrelated research directories are excluded.
