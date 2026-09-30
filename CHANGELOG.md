# Changelog

Detailed history is in the git log. Notable milestones:

- **2026-10** — prepared for public release: MIT license, English README, citation
  metadata, CI; machine paths and internal resource IDs moved to git-ignored local
  configs; manuscript analyses in `reports/` published as code and aggregates only.
- **2026-09** — IVF Digital Twin reached only through the 7.1 REST API (7.0 batch
  removed); laboratory KPIs computed over all cycles, not only transfers; model layer for
  cycles outside the report cohort; Teable sync.
- **2026-07** — first release (ready-1): optional MII, separated OCCR/MIIR/FR and
  Model_KPIScore; sum/sum cohort KPIs with Wilson CI, direction and competency;
  strict DT + TabPFN + OPU join with versioned contracts; HMAC `cycle_uid` agent packets;
  MIRI episode analysis; DataLore notebook ported to Python; `/run-analysis` skill with
  agent roles; synthetic fixtures and contract tests.
