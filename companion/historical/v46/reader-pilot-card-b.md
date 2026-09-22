# Reading card B

In each episode, the sensor is correct with probability 0.9 or 0.1, with equal weights; the orientation is fixed through calibration and deployment. Five labeled calibration trials are independent given that orientation. The learner retains FOLLOW if at least three reports are correct, otherwise REVERSE. The policy then stays fixed. Deployment starts at three units: a correct choice adds one net unit and an incorrect choice subtracts one. Reaching zero ends the episode. The hidden resource side scores the choice; the learner receives only the report.

The exact twelve-update continuation calculation averages episodes over the two environments and calibration records. It gives approximately 99.01% for the learned policy and 61.23% for ignoring the report. It is an analytic resource model, not a biological lifetime or apparatus-energy measurement.

## Passage

Consider two constructed calibration records. Labels show three correct reports and two errors in one, two correct and three errors in the other. They retain "follow" and "reverse." Start both at three units and give both a left report. One chooses left, the other right. Put the resource on the left for scoring, without revealing it to either chooser. Their amounts become four and two. This illustrates one choice, not the twelve-update probability above.

## Your turn

Record one has three correct calibration labels out of five; record two has two. Both receive a RIGHT report next and both start at three. Write the two retained policies and chosen sides. Then ask the facilitator for the resource side used for scoring, calculate the next amounts, and state whether this one action proves the twelve-update probability. Please do not consult another card or answer key yet.
