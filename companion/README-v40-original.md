# How Matter Learns to Continue: optional v40 reader companion

This is a **local companion bundle**, not a public release or a complete historical experiment repository. It makes the selected reader commands self-contained. Frozen data and E3 artifacts are copied, never regenerated. File provenance and SHA-256 values are in `bundle-manifest.json`.

The exact compatible v40 PDF hashes and page counts are in `bundle-manifest.json` under `compatible_edition`. A bundle marked `provisional_not_for_delivery` is a test build, not the final companion. Retained v39 and earlier filenames describe historical calculations and do not change the compatible book edition.

The v40 revision retains these reader routes and their assumptions. Its Chapter 8 deletion check is additionally recomputed by `verify_companion.py`: deleting the topmost glider site (1,2) produces a six-site still life by update three, unchanged at update four. This is a consequence of the existing rule, not repair, learning, or a new observed experiment.

## Setup

Extract the entire ZIP. Open a terminal in its `how-matter-learns-to-continue-v40-companion` folder: the folder containing this README, `requirements.txt`, `examples`, and `outputs`. Keep the nested directories intact. Start each practical route from this extracted root, then follow any explicit `cd` in that route. The commands in the table below already use paths relative to the extracted root.

Use Python 3.11 or newer. A separate virtual environment is recommended:

```text
python -m venv .venv
```

Activate it on Windows PowerShell with `.\.venv\Scripts\Activate.ps1`, or on macOS/Linux with `source .venv/bin/activate`. If activation is restricted, use the environment's Python directly: `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` elsewhere. Then:

```text
python -m pip install -r requirements.txt
python -B verify_companion.py
```

The v40 portable reader checks were rerun from the extracted ZIP in a fresh isolated environment with Python 3.11.0 and NumPy 2.4.4. The original stroke training reported NumPy 2.3.5; readers load its saved values. requirements.txt pins NumPy 2.4.4 for these reader modes. Python and dependency installation may require a download; no runtime is bundled.

The inherited v34 receipt records CPU execution of all four printed forward examples on 600 saved test inputs with copied parameters:

| Framework | Recorded version | Score dtype | Maximum absolute score error against NumPy |
|---|---|---|---:|
| PyTorch | 2.14.0+cpu | float64 | 1.7763568394002505e-15 |
| TensorFlow | 2.21.0 | float64 | 3.5527136788005009e-15 |
| JAX | 0.10.2 | float64 | 0 |
| Keras | 3.15.1 | float32 | 2.1617272860652292e-06 |

The unchanged execution receipt and its environment lock are included at `reviews/publication-v34/framework-verification.json` and `reviews/publication-v34/environment-lock.txt`. Packaging checks the saved-array hash and verifies that the v40 printed snippets have the receipt's exact hashes. `verify_framework_record.py` rechecks those bindings and this table after extraction. These are inherited execution results; v40 packaging does not execute frameworks. They establish the reported forward-score agreement, not training or gradient equivalence. The portable reader requirements do not install frameworks.

`verify_companion.py` prints JSON, including `passed: true`, every command's output, and a before/after integrity check. It does not write a report or alter the data. To save its output, redirect stdout deliberately to a new file outside the bundle if desired.

## Reader routes and expected results

| Reading chapters | Entry point from extracted root | Checked result or purpose |
|---|---|---|
| 1 | `python examples/ch01/investigation_v9.py` | Middle vertical stroke probability rounds to 0.99809. |
| 1 | `python examples/ch01/investigation_v9.py --pixel 5 0.5` | Changes the probe image only. Parameters stay copied from the archive. |
| 1 | `python examples/ch01/investigation_v9.py --weight-delta 0.01` | Manually changes one copied weight, not the saved model. |
| 1 | `python examples/ch01/investigation_v9.py --stage 300 --step` | Reconstructs 300 updates and performs 301 in memory. Mean loss changes from approximately 0.35798014 to 0.35671639. The original training run stays unchanged. |
| 2 | `python examples/ch01_04/reader_lab_v11.py representation --index 50` | Point B score approximately -0.20065843. The saved misclassification is retained. |
| 3 | `python examples/ch01_04/reader_lab_v11.py motion --previous -1 --current 0` | Compares saved-model prediction with exact hand-built next position 1. |
| 4 | `python examples/ch01_04/reader_lab_v11.py sensor --readings -0.6 0.6` | Final right-direction posterior 0.5 under the symmetric model. |
| 5 | `python reader_lab_v13.py walk --steps 16` | Best accuracy 0.9186221713. At 96 steps, 0.7092692376. |
| 6 | `python reader_lab_v13.py phase --gap 1.8 --coupling 0` | Gap stays 1.8 under the uncoupled rule. |
| 7 | `python reader_lab_v13.py sensor --mode x` | Equal-prior mirror-angle best accuracy 0.5. |
| 7 | `python reader_lab_v13.py bernoulli --p 0.45 --q 0.5 --readings 100` | Best accuracy 0.6922359155. The 0.01/0.06 comparison gives 0.9320070105. |
| 10 | `python outputs/publication-v29/ch10/assignment_check.py` | Exact-string probability 0.2993803913 and all-assignment probability 0.3710703619. Conditional rows sum exactly to one. |
| 11 | `python outputs/publication-v29/ch11/continuation_checks.py` | At 11, (m,y)=(1.2835,0.7165). At 16, (1.8647,0.1353). M(35)=30.1194. |
| 13 | `python outputs/publication-v29/ch13/survival_checks.py` | Fixed-risk P100=0.3660323 and P1000=0.00004317. Repaired state at 1000 approximately (0.334269,0.033427,0.632305). |
| 17 | `python -B verify_companion.py` | Reads all 1,001 saved oscillator samples and independently checks the component energies and conserved 0.75 J total. |
| 30 | `python -B examples/resource_learning_v31.py` | Exact calibration and all 4,096 deployment paths: learned continuation 0.9900982615, unannounced reversal 0.0103734015. |
| 36 | `python -B verify_e3.py` | Eight immutable artifact checks and four MSEs below. Failed transfer remains failed. |
| 37 | `python -B examples/model_comparison_v31.py` | Passive distance 0 and best decision 0.5, intervention distance 0.5 and best decision 0.75. |

The short `reader_lab_v13.py` filename is a byte-identical convenience alias of `examples/ch05_07/reader_lab_v13.py`. Both work in this extracted root. The book's Chapters 5–7 already precede their short commands with `cd examples/ch05_07`. That original route is valid without an alias. For Chapters 10, 11 and 13, use the actual `outputs/publication-v29/...` entrypoints shown above. Their retained version identifies compatible historical calculations.

The default Chapter 10/11/13 checks need only Python's standard library. The Chapter 13 source also contains an optional `--figure` producer for the full repository. That option requires plotting dependencies and a figure source archive outside this reduced companion. The documented default reader command works without them.

## Retained v31 teaching comparisons in v40

The resource reader uses the standard library only. Five labeled calibration trials set one retained follow/reverse decision, frozen for twelve deployment updates. Resource amount starts at 3, success supplies 2 and each update consumes 1. First reaching zero ends the episode. It verifies exact rational recurrence results by enumerating all 4,096 success sequences and both sets of 32 calibration sequences. Its record and selected new figures are included under `reviews/publication-v31` and `outputs/publication-v31`.

The four stationary comparisons are always follow (0.5002358315), always left (0.6123046875), learn from five labels (0.9900982615), and given orientation (0.9986307831). Reversing the environment after calibration gives the frozen learner 0.0103734015. This is a designed accounting model, distinct from the continuous passive pool and all biological evidence. The error is shared across one deployment episode, so continuation probabilities are mixed after calibration, not evaluated at an averaged one-step success probability.

The model-comparison reader enumerates the best single-observation decision over the same two reported outcomes. The common protocol and equal candidate priors are chosen for the calculation. They are not frequencies of physical worlds. Both new readers print JSON and write no files.

## Frozen E3 evidence: what verification means

`evidence/e3` contains the original manifest plus exactly its eight listed artifacts: aggregate results, thirty saved class rows, coefficient freeze, controls, development receipt, post-coefficient pilot receipt, probe commitment and public class catalogue. The original manifest's internal paths and source hashes are retained as historical text; they are not required local paths in this bundle.

`verify_e3.py` calls only the `e3_checks` function from the verbatim archived arithmetic-checker source included as `evidence/verify_revision_source_v28.py`. It passes the bundled evidence directory explicitly, bypassing that source file's machine-specific default and whole-book main routine. Do not run the archived checker directly: its other checks refer to the full source repository, which is not included.

The verifier checks byte counts and SHA-256 values, coefficient representation, public/analysis row agreement, descriptors, saved predictions, the class-24/class-44 collision, and the four mean squared errors:

| Frozen comparator | Recomputed MSE |
|---|---:|
| Shared-law candidate | 0.04651764075889651 |
| Constant floor | 0.022214819625990624 |
| Memory comparator | 0.011103334688740193 |
| Disjoint-family comparator | 0.01693321813870276 |

The recorded outcome is `e3_transfer_null`, with `scientific_admission: false`. These checks reconstruct saved arithmetic. They do not independently replicate the experiment, validate all historical execution claims, rerun the materializer, refit coefficients, invent confidence intervals, or change the failed transfer result.

## Deliberately excluded

This is not a complete regeneration package for every book figure. The historical `examples/investigation_v24.py` source and its byte-identical `outputs/investigation-v24/figure-data.json` are included to make the Chapter 17 provenance and archived arrays obtainable at the printed paths. **Do not execute or import that producer in this bundle.** It has top-level generation code that writes archival output paths and expects other repository material. `verify_companion.py` instead reads its saved `conserved-energy` arrays and independently checks all 1,001 samples against the printed normal-mode formulas, including the 0.75 J conserved total. This read-only check does not rerun the generator or claim verification of every other panel in the shared JSON.

Other historical programs such as `strokes_v3.py`, `ring_v5.py`, the v18/v21/v23 figure producers and publication figure generators write to archival output paths and may need additional data or plotting dependencies. They are not copied or run here. The new `publication_v31_late.py` and `figure-ch11-v31.py` producers also belong to the full repository. Full regeneration requires a separately preserved working copy of the full repository and its original dependencies. Other chapters' self-contained inline snippets can still be copied into a fresh Python session as stated in the book. Those snippets are not all smoke-tested by this bundle. The E3 execution code and development generators are not included.

No public URL, DOI, public-release availability, or redistribution clearance is claimed by creating this local archive.


## Provenance of the retained v39 precision table

The following table and its wording are retained from the original v39 evidence verifier. Here, "Fresh v39" identifies CPU executions performed for v39; none is a fresh v40 framework execution. The original v39 records, checker and hash-bound v39 notes are copied byte-for-byte. The separate v40 notes bind the currently printed snippets to the unchanged recorded snippets. The v40 portable suite reruns reader calculations and checks evidence consistency without importing frameworks.


## Fresh v39 forward precision checks

These are fresh CPU executions with copies of the author's saved 600 test inputs and weights. They are separate from the inherited v34 receipt above. No training or gradient equivalence is claimed.

| Configuration | Framework / backend version | Input / first product / score dtype | Maximum absolute logit error | Strict 1e-11 | Practical 1e-5 |
|---|---|---|---:|---|---|
| torch64 | 2.14.0+cpu / 2.14.0+cpu | torch.float64 / torch.float64 / torch.float64 | 1.7763568e-15 | pass | pass |
| tensorflow64 | 2.21.0 / 2.21.0 | float64 / float64 / float64 | 3.5527137e-15 | pass | pass |
| jax64 | 0.10.2 / 0.10.2 | float64 / float64 / float64 | 0 | pass | pass |
| keras-tensorflow-default | 3.15.1 / 2.21.0 | <dtype: 'float32'> / <dtype: 'float32'> / <dtype: 'float32'> | 2.1617273e-06 | FAIL | pass |
| keras-tensorflow-float64 | 3.15.1 / 2.21.0 | <dtype: 'float64'> / <dtype: 'float64'> / <dtype: 'float64'> | 2.5936205e-07 | FAIL | pass |
| keras-tensorflow-full64 | 3.15.1 / 2.21.0 | <dtype: 'float64'> / <dtype: 'float64'> / <dtype: 'float64'> | 3.5527137e-15 | pass | pass |
| keras-jax-default | 3.15.1 / 0.10.2 | float32 / float32 / float32 | 1.6848901e-06 | FAIL | pass |
| keras-jax-float64 | 3.15.1 / 0.10.2 | float64 / float64 / float64 | 2.5936205e-07 | FAIL | pass |
| keras-jax-full64 | 3.15.1 / 0.10.2 | float64 / float64 / float64 | 0 | pass | pass |
| keras-torch-default | 3.15.1 / 2.14.0+cpu | torch.float32 / torch.float32 / torch.float32 | 1.6945625e-06 | FAIL | pass |
| keras-torch-float64 | 3.15.1 / 2.14.0+cpu | torch.float64 / torch.float32 / torch.float64 | 1.586914e-06 | FAIL | pass |
| keras-torch-full64 | 3.15.1 / 2.14.0+cpu | torch.float64 / torch.float32 / torch.float64 | 1.586914e-06 | FAIL | pass |

Every configuration preserved all 600 threshold classifications. That does not make the scores bit-identical. The strict failures above remain failures even though the separately declared practical comparison passes.

`default` retains the printed Keras input and layer defaults. `float64` sets only the layer policy, retaining the default Input specification. `full64` explicitly sets both. In this installed Keras/PyTorch configuration, the directly probed first matrix product still returned float32 with float64 inputs and weights. This records observed behavior, not a diagnosis of every internal kernel or a claim about every library release.

The exact versions, intermediate probes, snippet hashes, original/final checks and initial warning record are included under `reviews/publication-v39`. `verify_v39_evidence.py` checks record consistency, not fresh framework execution. Framework packages are not installed by the portable requirements. With a separately prepared matching environment, run one read-only configuration with `python -B scripts/verify-framework-precision-v39.py --configuration torch64` (or another listed configuration). Do not run the no-argument production orchestrator inside the frozen reader bundle.

## Retained v39 programs and fresh v40 portable checks

The v40 reading edition retains the original `memory_reproduction_v39.py` and `local_rule_profiles_v39.py` programs, their v39 result records, and the original v39 precision checker and evidence verifier. Their filenames and execution records identify their actual provenance. Fresh v40 validation runs the same 35 portable commands from the extracted v40 ZIP in the isolated NumPy 2.4.4 environment. It does not rerun neural-network frameworks or the E3 experiment.

The twelve CPU framework configurations remain the executions recorded in `reviews/publication-v39/framework-precision.json`. The bundle retains `manuscript/chapters/v39-01-neural-networks/notes-v39.tex` to satisfy that record's original whole-file hash, alongside `manuscript/chapters/v40-01-neural-networks/notes-v40.tex` for the current printed snippets. The portable verifier checks both bindings, including the printed-snippet hash for every recorded configuration. The original `scripts/verify-extra-companion-v39.py` is also available as the byte-identical root command `verify_v39_evidence.py`; use that root alias for its relative data paths.

## Chapter 3: check the complete saved comparison

From the extracted companion root:

```text
python -B examples/memory_reproduction_v39.py
python -B examples/memory_reproduction_v39.py --retrain
```

The first command checks all training and held-out positions, the paired data recipe, the initialization recipe, every saved prediction, and both global RMSEs. It does not train. The second additionally performs two fixed 10,000-update training reproductions in memory and compares final parameters and 401 learning-curve checkpoints per model. Neither command writes files. The v39 execution record is retained unchanged; the v40 portable receipt separately records the new reader run. In the recorded v39 NumPy 2.4.4 execution, both final-parameter and learning-curve maximum errors were zero. The declared cross-environment absolute tolerance is 1e-11.

The current-only test RMSE is 1.0000261452316914. With two frames it is 0.017811815614758357. The exact constant-velocity rule has zero error on these saved noiseless trajectories. The 1,024 test trajectories are pairs built from 512 independent current positions.

The generator resets `default_rng(1701)` for each input dimension `d`. It first draws `W1` from a zero-mean normal distribution with standard deviation `sqrt(2/d)` and shape `(d,8)`. The hidden bias is eight copies of 0.3. It next draws `W2` from the same generator with standard deviation `sqrt(2/8)` and shape `(8,1)`. Output bias is zero. Inputs are divided by 3, the full-batch half-squared-error objective uses learning rate 0.03, and training lasts 10,000 updates.

The original `examples/ch03_04/experiment_v7.py` and `results.json` are included for provenance. Do not execute that original producer in this bundle. The portable checker extracts only its literal configuration and initializer, never imports its plotting/output routines, and does its own array calculation.

## Retained exploratory exercise: what did averaging discard?

This exercise was introduced in v39 following a suggestion in the external v38 review. Its original program and record are retained here, and the portable suite reruns its enumeration. It is not part of the book's frozen E3 experiment and has not been adopted as a new result in the manuscript.

```text
python -B examples/exploratory/local_rule_profiles_v39.py
```

For elementary one-dimensional binary rules use `f_r(L,C,R) = (r >> (4L+2C+R)) & 1`. Ask how often flipping a specified input changes the central output. Average over all eight equally weighted three-cell contexts for one update. For two synchronous updates, use all 32 equally weighted five-cell contexts, with no full-ring preparation or fitted model.

Before running the program, consider two proposed repairs: keep the left/right contributions separate, or observe two updates and average again. Which retains a distinction that the original mean discards?

| Quantity | Rule 24 | Rule 44 |
|---|---|---|
| One-step left / centre / right influence | 1/2, 1/2, 1/2 | 3/4, 3/4, 1/4 |
| Mean of left and right only | 1/2 | 1/2 |
| Absolute left/right imbalance | 0 | 1/2 |
| Two-step mean over the four noncentral initial positions | 1/4 | 1/4 |

Keeping the one-step profile distinguishes this pair. Looking longer and averaging again still loses that distinction in the stated two-step diagnostic. Bit-table and Boolean implementations agree on every enumerated context. Reflection and joint state/rule complement preserve the tested imbalance.

This was chosen after seeing the exposed pair. It is exploratory, not preregistered or held out, and makes no novelty claim. These are not Conway's two-dimensional rules or E3's eight full-ring probes. The calculation does not explain or predict the terminal E3 response, refit its coefficients, improve its four recorded errors, or demonstrate transfer. The frozen negative result remains negative.
