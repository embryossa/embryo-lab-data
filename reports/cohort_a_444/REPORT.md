# Cohort A (444 cycles): full Digital Twin re-run against the manuscripts

Date: 2026-09-28. Scripts: `run_dt.py` (DT .venv), `evaluate.py`. Outputs: `dt_full_444.csv`
(one row per cycle, all layers, no identifiers), `stage_spearman.csv`, `pregnancy_metrics.csv`,
`paired_auc.csv`, `cohort_characteristics.csv`.

## Source and configuration

- Cohort: `PycharmProjects/pythonProject/DNN for Fertimed/pythonProject/validation.xlsx`
  (Fertimed 1C export, 889 rows); the 444 rows with a recorded transfer outcome, 185 pregnancies
  (181 singleton + 3 multiple + 1 mixed record). One row has no age → 443 run; 5 more lack a
  CSDI/graph output → 438 in the pregnancy table.
- One row = one **transfer**: 318/444 are frozen-embryo transfers («Криоперенос»). In those rows
  `Число Bl` equals the number of embryos transferred in 82% — it is not the cycle's blastocyst yield.
- DT configuration = the 7.0 batch (`batch_analysis.py`), as used for the Two-Line / RBMO runs:
  age, attempt, follicles, OCC and inseminated (as known MII); AMH/AFC defaults 2.0/12, BMI 23;
  2PN/Bl/good Bl masked and taken from the Layer 1 simulation. N_SIM 2000, seed 42+row.
  Local DT code at commit `33101ce`; FORTUNE intercept in the code = 0.40.
- "Recalibrated" = the documented intercept −0.25 (`docs/coefficients.md`), recomputed on the
  same simulated draws and carried through BEFE.

## Pregnancy (n = 438, 184 events, observed 42.0%)

AUC 95% CI: 2,000 outcome-stratified bootstrap resamples.

| Layer | AUC (95% CI) | Brier | Mean pred. | Gap, pp |
|---|---|---|---|---|
| L7 BEFE, intercept 0.40 | 0.550 (0.496–0.603) | 0.260 | 41.8% | −0.2 |
| L7 BEFE, intercept −0.25 | 0.550 (0.496–0.603) | 0.260 | 41.5% | −0.5 |
| L3 KAT | 0.544 (0.490–0.598) | 0.262 | 38.1% | −3.9 |
| L2 per-transfer, 0.40 | 0.577 (0.523–0.630) | 0.257 | 53.7% | +11.7 |
| L2 per-transfer, −0.25 | 0.577 (0.523–0.630) | 0.245 | 46.2% | +4.2 |
| L1/2 whole-cycle, 0.40 | 0.576 (0.522–0.630) | 0.298 | 46.3% | +4.3 |
| L1/2 whole-cycle, −0.25 | 0.576 (0.522–0.630) | 0.293 | 43.1% | +1.0 |
| Prevalence only | 0.500 | 0.243 | 42.0% | 0 |

Paired ΔAUC: BEFE − KAT +0.006 (−0.011 to 0.021), p = 0.49; BEFE − L2 per-transfer −0.027
(−0.055 to −0.001), p = 0.045. Changing the intercept does not change BEFE (ΔAUC 0.000).
Excluding the mixed record or restricting to 124 fresh cycles does not change the picture
(fresh: BEFE 0.539, KAT 0.527, L2 0.583; CIs 0.42–0.69).

Manuscript (Table 3, 444 column): AUC 0.820, Brier 0.16, gap +6.2 → < 2 pp. **Not reproduced.**
No layer reaches 0.82; every upper CI bound is ≤ 0.63. Every Brier score is at or above the
prevalence-only 0.243. The −0.25 intercept moves the L2 gap +11.7 → +4.2 pp (per-transfer) or
+4.3 → +1.0 pp (whole-cycle); neither pair is +6.2 → < 2.

## Stage agreement (Spearman ρ, 95% bootstrap CI)

Forecast made by Layer 1 before the stage was observed (review item 21).

| Stage | Known to the model | n | ρ | Fresh only | Manuscript |
|---|---|---|---|---|---|
| OCC | nothing (pre-retrieval) | 355 | 0.17 (0.07–0.26) | 0.19 (n=112) | 0.718 |
| MII | OCC | 156 | 0.86 (0.80–0.92) | 0.74 (n=30) | 0.865 |
| 2PN | OCC + inseminated | 441 | 0.90 (0.87–0.92) | 0.87 (n=123) | 0.852 |
| Bl | + 2PN | 407 | −0.00 (−0.09–0.08) | **0.61 (n=95)** | 0.761 |
| Good Bl | + Bl | 333 | 0.83 (0.77–0.89) | 0.81 (n=72) | 0.876 |

Only the OCC forecast is unconditioned, and it is weak (AMH/AFC are batch defaults; follicles
known in 180/443). MII, 2PN and good-Bl ρ are high largely because the preceding count is an
input and bounds the next one. Blastocyst ρ is meaningful only in fresh rows (0.61).

## Reading

- AUC 0.820 matches neither BEFE nor KAT on this cohort. It equals the KAT / FT-Transformer
  held-out value (n = 1,606), i.e. scenario 3: carried over from the held-out result.
- Brier 0.16 likewise matches the held-out KAT Brier (0.163), not this cohort (0.26).
- The cohort is per-transfer and mostly frozen transfers; blastocyst fields in FET rows are
  transfer counts. It is a single clinic, not "three centres", and AMH is known in 225/444.
