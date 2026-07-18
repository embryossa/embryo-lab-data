---
name: dt-ensemble
description: IVF Digital Twin ensemble layer glossary (L1-L7), verified model definitions (MC pipeline/KAT/GAT/CSDI/Digital Twin/TabPFN, plus our independent DNN cross-checker), cluster labels, calibration metrics, and probability/uncertainty language bands. Preload into narration agents that explain DT/TabPFN outputs.
---
# IVF Digital Twin — ансамбль и язык интерпретации

**Наши 7 основных моделей** (имена сверены с первоисточником —
`IVF Digital Twin Pro/IVF Digital Twin`'s собственные `befe.py`/`architecture.md`/
`GAT_method_description.md`, а не с названиями старого ноутбука, которые местами
были неточны — см. ниже):

| Модель | Колонка | Слой | Что это |
|---|---|---|---|
| **MC pipeline** | `p_per_transfer` (+`p_overall_cycle`, `p_cancel_risk`) | L1+L2 | механистический приор: Monte Carlo симуляция воронки ОКК→MII→2PN→Бласт→Хор.→Эуплоид, N=5000 |
| **KAT** | `p_kat_raw` | L3 | нейросетевой ансамбль KAN + FT-Transformer |
| **GAT** | `p_gnn_raw` (смешанный с KAT вариант — `p_gnn_ens`) | L6 | graph attention transformer (граф пациентов); `p_gnn_raw` — «сырой» GAT-скор, `p_gnn_ens` — уже смешанный с KAT |
| **CSDI** | `p_csdi` | L5 | диффузионная верификационная модель (TabDDPM-стиль) — биологический якорь, сверяется с L1/MC |
| **Digital Twin** | `DIGITAL TWIN` | L7 BEFE | Bayesian Evidence Fusion Engine — итоговый ансамбль L1+L3+L5+L6+L4(кластер) |
| **TabPFN** | `PRAI` | внешний | `p_tabpfn` переименован в `PRAI` на границе join (`reconcile.py`) — это одна и та же модель, PRAI не отдельный оценщик |
| **DNN** | `DNN` | независимая | НАША собственная независимая кросс-проверка (`external/dnn/Evo_calculations.py`, TensorFlow) — не часть ансамбля Digital Twin, отдельный проект/scaler |

Не путать с «главными» моделями (существуют, но НЕ отдельные оценщики, не показывать
как самостоятельную модель):
- `bayes_mean`/`bayes_ci_low`/`bayes_ci_high`/`bayes_prior_*` — Bayesian-обновлённый
  posterior НАД тем же L1 Monte Carlo прогоном (`res["posterior"]` в `ivf_core.py`),
  не самостоятельная архитектура и не часть KAT.
- `p_nvsa` — внутренняя KPI-якорная коррекция KAT (`clip(p_KPI/p_NN, 1/1.5, 1.5)`),
  подкомпонент L3, а не отдельная модель (и не «KAN» саму по себе — старое название
  этого поля в проекте ошибочно).
- `dominant_cluster`/`cluster_c*_prob` — L4, контекст кластера, не вероятность исхода.

Кластеры: `dominant_cluster` C0 Standard/C1 Poor/C2 High + `cluster_c*_prob`.

Метрики калибровки: Brier (↓), AUC (↑), Acc@0.5 (↑); ±30% коридор для воронки эмбриологии.

Язык вероятностей (нарратив): P<25% ниже среднего · 25–40% умеренный · 40–55% хороший ·
>55% сильный. Ширина CI: <15пп уверенно · 15–30пп умеренно · >30пп ориентировочно.
Разброс ансамбля: <15пп единодушен · 15–30пп умеренно · >30пп значимое расхождение.
Confidence grade A/B/C = комбинация CI + согласованности ансамбля.

Все категории вычисляет Python (consult.build_flags) — агент их ОБЪЯСНЯЕТ, не пересчитывает.
Числа — только из пакетов. Не ставить диагнозов, не назначать протоколов.
