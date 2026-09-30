# Bland–Altman, Digital Twin vs TabPFN, RBMO prospective cohort (n = 1,026) — text for the manuscript

Figure: `Figure_bland_altman_pregnancy_pooled.png/.pdf` · statistics: `bland_altman_pregnancy_pooled.csv`
· script: `rbmo_bland_altman_pooled.py` · data: frozen cohort
`data/processed/reference/rbmo_prospective_2026/` (sha256 checked against its manifest).
All numbers below are taken from those outputs.

---

## Methods (statistical analysis, insert)

Agreement between the Digital Twin combined estimate and TabPFN predictions of clinical
pregnancy was assessed with Bland–Altman analysis in the full prospective cohort
(n = 1,026). Differences (Digital Twin minus TabPFN) were plotted against the mean of the two
predictions on the probability scale (percentage points) and on the logit scale, where the
difference equals the log odds ratio between the models. We report the mean bias and 95%
limits of agreement (bias ± 1.96 SD), with 95% confidence intervals for both (Bland and
Altman, 1999). Proportional bias was tested by regressing the difference on the mean, and
heteroscedasticity by the Spearman correlation between the absolute deviation from the bias
and the mean. Because pregnancy rates differed between centres, the SD of the differences, the
proportional-bias slope and the association with the observed outcome were also estimated
within centres (linear models with centre indicators). Because the differences were not
normally distributed, percentile-based limits (2.5th–97.5th percentiles) are reported as a
sensitivity analysis.

## Results

The two models agreed closely on average but not for individual patients (Figure X). The
Digital Twin predicted slightly lower probabilities than TabPFN, with a mean bias of −2.4
percentage points (95% CI −3.2 to −1.7), equivalent to an odds ratio of 0.90 (95% CI 0.87 to
0.94). The 95% limits of agreement ranged from −26.1 (95% CI −27.4 to −24.8) to +21.3 (95% CI
20.0 to 22.5) percentage points, and from 0.26 to 3.09 on the odds-ratio scale. For one patient
in three the two predictions differed by more than 10 percentage points, and for one in ten by
more than 20 (median absolute difference 5.8 percentage points). The differences were
right-skewed (skewness 1.01), with a tail of cycles in which the Digital Twin exceeded TabPFN
by 40–60 percentage points. The percentile-based limits (−23.6 to +25.5 percentage points)
were therefore shifted upwards relative to the parametric limits, although 94.8% of cycles
fell within the parametric limits.

Disagreement depended strongly on the predicted level (Spearman ρ = −0.40 on the probability
scale and −0.45 on the logit scale, both p < 0.001). Where both models predicted 60% or more
(n = 462), the median absolute difference was 3.4 percentage points and only 0.2% of cycles
differed by more than 20. Below 40% (n = 217), the median difference was 9.3 percentage points
and 17.5% of cycles differed by more than 20. The narrowing at higher predicted
probabilities was present within each of the four centres. Transforming to the logit scale
did not remove it, so the models genuinely disagree most for patients with a less favourable
prognosis. This is not an artefact of the probability scale.

The regression of the difference on the mean had a small positive slope (+0.12, 95% CI 0.07
to 0.17; logit scale +0.07, 95% CI 0.01 to 0.12). This slope is fully accounted for by the
wider spread of Digital Twin predictions (SD 17.1 versus 15.4 percentage points; the slope
expected from the two variances alone is 0.121). It disappeared within centres (+0.03, 95% CI
−0.03 to 0.08; logit −0.05, 95% CI −0.11 to 0.01), so it does not indicate a proportional
bias between the models. Differences between centres accounted for about 6% of the variance
of the differences (SD 12.1 overall versus 11.7 within centres), so the pooled limits of
agreement mainly reflect disagreement between individual patients rather than between
clinics. Disagreement was unrelated to the observed outcome once centre was taken into
account (pregnant minus non-pregnant +0.2 percentage points, p = 0.79). The crude difference
(+1.8 percentage points, p = 0.02) reflected the higher pregnancy rate in the centre with a
positive bias.

## Figure legend

**Figure X. Agreement between Digital Twin and TabPFN predictions of clinical pregnancy in
the prospective cohort (n = 1,026 cycles, four centres).** Difference between the two
predicted probabilities (Digital Twin minus TabPFN) plotted against their mean, (**A**) on
the probability scale and (**B**) on the logit scale, where the difference equals the log
odds ratio between the models. Solid line, mean bias with its 95% CI (dark band); dotted
lines, 95% limits of agreement (bias ± 1.96 SD) with their 95% CIs (grey bands); dash-dot
line, regression of the difference on the mean; dashed grey line, zero difference. Triangles,
cycles with clinical pregnancy (n = 526); circles, cycles without (n = 500). Mean bias −2.4
percentage points (limits −26.1 to +21.3); on the logit scale −0.10 (limits −1.33 to +1.13;
odds ratio 0.90, limits 0.26 to 3.09).

---

# Figure Y — same predicted prognosis, different outcome by centre

Figure: `Figure_prognosis_by_centre.png/.pdf` · statistics: `prognosis_by_centre.csv`,
`prognosis_by_centre_tests.csv` · script: `rbmo_prognosis_by_centre.py` (same frozen cohort).

## Methods (statistical analysis, insert)

To examine whether cycles with the same predicted prognosis had the same outcome in different
centres, cycles were grouped by the mean of the two models' predicted probabilities (<40%,
40–60%, ≥60%). Within each band we compared the observed clinical pregnancy rate (Wilson 95%
CI) with the mean predicted probability of the same cycles, and tested differences between
centres with the chi-square test. Centre–band cells with fewer than 10 cycles were not
analysed. Whether centre explained outcome beyond the models' forecast was tested with a
likelihood-ratio test, adding centre to a logistic regression of pregnancy on the logit of the
predicted probability.

## Results

Cycles with the same predicted prognosis had very different outcomes depending on the centre
(Figure Y). Among cycles with a predicted probability of 60% or more (n = 462; mean predicted
probability 63–69% in every centre), the observed pregnancy rate was 73.2% (95% CI 68.0–77.9)
at Centre B, 65.0% (43.3–81.9) at the fourth centre, 48.5% (38.8–58.3) at Centre A and 25.6%
(14.6–41.1) at Centre C (chi-square p < 0.001). Centres also differed at 40–60% (28.9% to
59.8%, p < 0.001) and below 40% (22.9% to 68.4%, p = 0.002). At Centre C the observed rate did
not rise with the predicted probability (22.9%, 28.9% and 25.6% across the three bands).
Centre remained strongly associated with pregnancy after adjustment for the predicted
probability (likelihood-ratio χ² = 75.6, 3 degrees of freedom, p < 0.001).

## Figure legend

**Figure Y. Observed clinical pregnancy rate by centre among cycles with the same predicted
prognosis.** Cycles were grouped by the mean of the Digital Twin and TabPFN predicted
probabilities. Symbols show the observed pregnancy rate in each centre with its Wilson 95%
confidence interval. Grey ticks show the mean predicted probability of the same cycles, so the
vertical distance between a symbol and its tick is the gap between observed and predicted.
Numbers along the bottom are cycles per centre. p values compare centres within each band
(chi-square). Centre–band cells with fewer than 10 cycles (fourth centre, <40% and 40–60%;
n = 2 and 6) are not shown.

---

## Discussion (both figures)

The two independently developed models agreed closely on average. For an individual patient,
however, they could differ by more than 20 percentage points, and this happened almost
exclusively when the prognosis was less favourable. Agreement between two models is not
evidence that either is accurate, and the comparison with observed outcomes showed why.
Cycles with the same predicted prognosis had very different pregnancy rates depending on the
centre: from 26% to 73% among cycles predicted at 60% or more. Centre remained a strong
predictor after the models' forecast was taken into account. In one centre the observed rate
did not change across the full range of predictions. By contrast, in a separate cohort the
framework discriminated embryological failure well (no blastocyst after fertilisation, AUC
0.82) but not pregnancy after transfer (within-centre AUC 0.56). Together, these findings
suggest that much of the variation the models do not explain arises after embryo development,
from factors that neither model observes and that differ between centres: endometrial
receptivity and its preparation, embryo transfer technique, clinical practice and the case mix
of each centre. We did not measure these factors, so this interpretation remains a hypothesis.
Because neither model included the morphology or ploidy of the transferred embryo,
embryological contributions to implantation failure cannot be excluded either. Testing the
hypothesis will require prospective data on these factors, and meanwhile the results support
recalibrating any such model to the centre where it is used.

---

### Notes for the authors (not for the manuscript)

- **Two different centre-level facts, both true.** Centres explain little of the *spread*
  (≈6% of variance), which is why pooling is defensible. They do explain the pooled *slope* and
  the crude outcome association: both disappear within centres. The text above states this, so
  a reviewer who asks "is this confounded by centre?" is already answered.
- Per-centre agreement stays available as a supplementary figure if a reviewer asks for it:
  `Figure_bland_altman_pregnancy_prospective.png` (bias from −5.7 to +1.1 percentage points
  by centre).
- The pooled observed rate (51.3%) is close to both mean predictions (52.1% and 54.5%), but
  this hides miscalibration within centres (for example, at Centre C both models predicted
  about 47–50% against an observed 26%). Do not use the pooled means as a calibration
  statement.
- **The manuscript reports two values for the no-blastocyst AUC**: 0.824 (95% CI 0.774–0.875)
  in the Abstract/Results and 0.830 in Table 7 (both n = 1,050). The Discussion above uses
  "0.82", which matches both roundings, but the two places should be reconciled before
  resubmission.
- The no-blastocyst result comes from a different cohort (1,050 cycles, one centre) than the
  pregnancy comparison (1,026 cycles, four centres). The Discussion says "in a separate
  cohort"; keep that wording, otherwise a reviewer will read it as a within-cohort contrast.
- Do not write that the model "achieves" pregnancy prediction or that causes of failure lie
  "outside the embryology stage". Figure Y shows the opposite of the first (26–73% at the same
  prediction), and the second is not testable without transferred-embryo quality/ploidy
  (`known_euploid` is empty in all 1,026 cycles, and the cohort has no morphology grade of the
  transferred embryo). The hypothesis as worded above is defensible.
- Figure Y's centre effect is the strongest single result for the argument: a
  likelihood-ratio χ² of 75.6 on 3 df for centre beyond the forecast.
