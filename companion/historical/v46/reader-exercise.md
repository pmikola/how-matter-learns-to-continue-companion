# Exercise: enough for which question?

This companion-only exercise uses the analytic model in Chapter 30. Write an answer before opening the separate key. No empirical observation is added.

## Conditions

The sensor is correct with probability q = 0.9 or q = 0.1, with equal prior weights. That orientation stays fixed through calibration and deployment. Five correctness labels are independent given the orientation; deployment draws are independent of calibration given that same orientation. Retain FOLLOW when at least three labels are correct and REVERSE otherwise. The retained policy stays frozen for twelve deployment updates.

Deployment starts at three units. A successful choice adds one net unit; an unsuccessful choice subtracts one. Zero ends the episode. Continuation requires a positive amount at every update boundary. The hidden resource side scores the choice but is not supplied to the chooser.

## Before looking at the answer

- One calibration record has three correct reports out of five; another has five. What policy does each retain?
- Given the same next RIGHT report, what action does each choose? Would knowing only the retained bit distinguish the two original counts?
- Can the bit reproduce the original probability that q = 0.9 conditional on each count? Explain without calculating first.
- Optional calculation: use the likelihood ratio 9^(2K−5) and equal prior weights to compute the posterior for K = 3 and K = 5. Are the twelve-update probabilities conditional on these counts necessarily equal?
- If the count is discarded, does a posterior conditional only on FOLLOW still exist? Is that the same as recovering the posterior conditional on the original count?
- Explain why a difference between these conditional values does not contradict the book's aggregate 99.01% result or make the bit inadequate for its declared two-policy decision.

Record your explanation and confidence in it. Then open the [separate answer key](reader-exercise-key.md). For orientation, use [Where to return](where-to-return.md), questions 2 and 4.
