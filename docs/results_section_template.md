# Template for `project.md` §3.3 Modeling and Evaluation

This is a drop-in template. Replace [PLACEHOLDERS] with the actual numbers
from `data/processed/loso_report.txt` and adapt the language to your style.
Numbers below reflect the run on 255 recordings, 3 subjects, 12 classes
(5 fall + 7 ADL).

---

## 3.3 Modeling and Evaluation

### Two-stage architecture

The system uses a two-stage classifier architecture:

- **Stage 1** is a binary classifier that decides whether a given window of
  sensor data represents a fall event or an activity of daily living (ADL).
- **Stage 2** is a 5-class classifier that, given a window which Stage 1
  identified as a fall, predicts the fall direction (backwards, forward,
  left, right, or stumble).

Both stages use Random Forest classifiers (300 trees,
`class_weight="balanced"`, scikit-learn 1.5). We use Leave-One-Subject-Out
(LOSO) cross-validation throughout: the model is trained on data from two
of our three subjects and tested on the held-out third. This is the most
honest evaluation protocol for our use case, because it directly answers
the question that matters for a deployed fall-detection app:
"Does the model work on a person it has never seen during training?"

### Feature extraction

For every window we compute 70 hand-crafted features grouped into:

- Per-axis time-domain statistics on accelerometer and gyroscope
  (mean, std, min, max, range, RMS, skewness, kurtosis).
- Signal-vector magnitude statistics on accel and gyro.
- Jerk (the time derivative of the acceleration magnitude): peak, RMS,
  and integrated absolute value.
- Frequency-domain energy of the accelerometer magnitude in three bands
  (0-3 Hz, 3-6 Hz, 6-10 Hz), computed via FFT.

### Stage 1 results: binary fall vs ADL

Stage 1 achieves a mean LOSO macro-F1 of **0.988** across the three
subjects, with per-subject scores of 1.000, 0.963, and 1.000. The random
baseline for a binary task is 0.500.

[INSERT confusion_stage1.png HERE]

This is the primary result of the project: a smartphone IMU-based fall
detector that generalises well to unseen individuals and is robust across
the three phone placements used in our data collection (pocket, hand,
chest).

### Stage 2 results and honest limitations

Stage 2 attempts to classify the direction of an already-detected fall.
On our dataset it achieves a mean LOSO macro-F1 of **0.213**, only
marginally above the random baseline of 0.200 for a 5-class problem.

[INSERT confusion_stage2.png HERE]

To investigate whether this was due to our choice of classifier, we ran
the same evaluation with four additional classifier families (shallow
Random Forest, Gradient Boosting, SVM with RBF kernel, and L2-regularised
Logistic Regression with feature selection). No classifier exceeded a
macro-F1 of 0.22, and one (Gradient Boosting) performed below the random
baseline. We also evaluated a simplified 3-class setup (frontal / sideways
/ stumble), in which the best model achieved a macro-F1 of 0.295 against
a random baseline of 0.333 - so although the raw score was higher, the
relative performance was actually slightly worse than chance.

We attribute the limited generalisation of Stage 2 to three compounding
factors:

1. **Subject-placement confounding.** In our data collection, each
   subject contributed predominantly one phone placement (Felix: pocket,
   Lea: hand, Leopold: chest). This means LOSO splits cannot distinguish
   subject-specific from placement-specific variance, and a model trained
   on two placements has no information about the third when generalising.
2. **Sample size.** With approximately 25 samples per fall class spread
   across three subjects, the per-fold training set contains only ~17
   examples per class, which is below the threshold at which subject-
   invariant geometric features of a fall direction become learnable.
3. **Inter-subject variability in fall execution.** One of our three
   subjects produced systematically weaker impact signatures, with 27%
   of their fall recordings (12 / 45) falling below the 1.8 g impact
   detection threshold. Even with a robust classifier, the underlying
   biomechanics differ enough between volunteers to limit cross-subject
   transfer at this sample size.

These findings are consistent with prior work in smartphone IMU fall
classification, in which inter-subject generalisation has been
identified as a primary failure mode (Casilari et al. 2015). Stage 2
should be regarded as a proof of concept; reaching reliable fall-type
classification would require either (a) substantially more subjects to
average out individual biomechanics, or (b) deliberate multi-placement
data collection from every subject to decouple the two confounded
factors.

### Summary

- Stage 1 (fall vs ADL) is the primary contribution and reaches very high
  accuracy (macro-F1 = 0.988).
- Stage 2 (fall type) is reported for completeness but does not exceed
  random performance at our current sample size and data-collection design.
- The two-stage architecture and the parser/preprocessing/feature
  pipeline are designed to scale: adding additional subjects or
  multi-placement recordings only requires re-running
  `python -m src.build_dataset` and `python -m src.train`.
