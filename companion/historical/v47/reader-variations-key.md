# Key for the four v47 variations

These are analytic answers, not reader results. Keep this file hidden until initial answers to [U1-U4](reader-variations.md) have been recorded.

## U1. The finite test does not reach beyond the cap

Even twelve successes raise three only to fifteen. Thus every twelve-update amount history, including absorption, agrees in both models. Thirteen consecutive successes first distinguish amounts: unlimited storage reaches sixteen while capped storage stays at fifteen. That statement concerns amounts; it does not assert that the two survival probabilities already differ at update thirteen.

For a fixed success probability below one, each independent block of fifteen trials has a fixed positive chance of being all failures. Such a block empties any allowed positive capped amount, so the probability of avoiding all such blocks forever is zero. This sufficient bound may be very loose. At success probability one, the failure-block argument does not apply and a positive store survives. In the unlimited process with success probability above one half, survival forever has positive probability. These are two specified models, not a theorem that every finite physical system fails or that real failures must be independent.

## U2. The readout discarded the remaining difference

The sixteen-update identity also makes the 32-update map the identity. The two preparations therefore still differ at their original perturbed site at update 32. They are not identical. Excluding that site makes the external response zero; including all sites would give one differing site out of 32. At update 33, rule 150 spreads the one-site difference to that site and its two neighbors. Excluding the central site gives two differing external sites out of 31. Earlier spreading and exact later return are compatible. This is a retrospective diagnostic of the exposed rule, not a repair of the historical failed transfer test.

## U3. Inference accuracy and resource continuation ask different questions

Three labels leave three deployment units and identify orientation correctly in 97.2% of episodes; five leave one unit and identify it correctly in 99.144%. Under the stated twelve-update accounting, continuation is approximately 97.072067% and 88.129064%, respectively. The no-purchase report-ignoring strategy keeps all six units and continues with probability approximately 90.771484%; always-follow also keeps six and reaches approximately 51.136831%. Three is best only among the displayed fixed-count procedures, not among all adaptive sampling or control rules.

Optional free information cannot decrease the best achievable expected payoff when every previous policy remains feasible: the chooser can ignore it. That is a feasible-policy inclusion argument. It does not say that every particular rule uses information well, and it does not make purchased observations free. The original equal-deployment-start result remains a separate construction.

## U4. The objective and flow complete the construction

The Fisher metric by itself specifies local comparison of probability changes, not which dataset or objective to improve or which dynamics to use. With a the fixed fraction of positive labels, the chosen mean log loss is L(p) = -a log(p) - (1-a) log(1-p). Its derivative is (p-a)/[p(1-p)]. Multiplying the negative derivative by the inverse Fisher coefficient gives the chosen optimization-time rate a-p. The exact map is p_new = a + (p_old-a) exp(-eta), where eta is nonnegative optimization time.

Here a = 3/4, p_old = 1/4 and exp(-log(2)) = 1/2, so p_new = 1/2. Along this flow, the loss rate is -(p-a)^2/[p(1-p)], which is nonpositive because the denominator is positive in the interior and the numerator is a square. For finite time, the exact map stays between the initial probability and the target fraction; it is not a general neural-network update guarantee.

Continuous flow correspondence under log odds does not imply equality of finite straight Euler steps: straight coordinate steps transform nonlinearly. The delivered checks include an explicit counterexample. The exact integrated map and a chosen finite Euler step must not be conflated.

## Executable support and scoring boundary

Run `python -B continuation-checks-v47.py` for the original-result, infinite/capped, paid-label, rule-150 and Fisher constructions. Run `python -B reconstruct-w7-v47.py` for deterministic probe and historical response reconstruction plus the rule-150 readout diagnostic. These commands print reports; they do not fit predictors or record reader performance.

Score each distinction as unassisted correct, correct after the logged clarification, incorrect, or not answered. Preserve verbatim reasoning, source location, elapsed time, reader background and any task skipped. A small pilot diagnoses explanations; it does not establish a population effect or a universally better book.
