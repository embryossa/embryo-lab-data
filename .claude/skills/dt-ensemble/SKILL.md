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

## Знаменатели стадии «хорошие бластоцисты» — две разные базы

В проекте сосуществуют ДВЕ «частоты формирования хороших бластоцист» с разными
знаменателями. Это не ошибка, но прямое сравнение между ними или с порогами Vienna —
ошибка интерпретации, дающая заведомо завышенное ожидание.

| Величина | Формула | Источник |
|---|---|---|
| CSDI `good_rate` | `good_Bl / Bl` | `src/embryo_csdi_v3.py` (доля от всех бластоцист) |
| Наша воронка `Bl→Good` (реал и прогноз) | `Good Bl / Bl` | `analytics.py: conversion_rates` |
| KAT-фичи / `calibrate_for_clinic.good_blast_rate` | `blasts_good / 2PN` | совпадает с TGBDR |
| Vienna **TGBDR** | `Good Bl / 2pN` | `kpi.py` — пороги 30%/40% |

CSDI `blast_rate = Bl / 2pN` совпадает с Vienna BlR и с шагом воронки `2PN→Bl` —
там расхождения нет, оно возникает ровно на одной стадии.

Мост между базами: **`TGBDR = (Bl→Good) × BlR`**. Лаборатория с `Bl→Good` 70% и
BlR 63% имеет TGBDR ≈ 44% — это одно и то же качество в двух базах, а не два разных
результата. Никогда не читать `Bl→Good` или CSDI `good_rate` против порогов TGBDR
(30%/40%) и не называть их «частотой хороших бластоцист» без указания знаменателя.

Кластеры: `dominant_cluster` C0 Standard/C1 Poor/C2 High + `cluster_c*_prob`.

Метрики калибровки: Brier (↓), AUC (↑), Acc@0.5 (↑); ±30% коридор для воронки эмбриологии.

Язык вероятностей (нарратив): P<25% ниже среднего · 25–40% умеренный · 40–55% хороший ·
>55% сильный. Неопределённость TabPFN (`CI_width_pp`, `CI_source=tabpfn_ensemble` — разброс
членов ансамбля TabPFN): <15пп уверенно · 15–30пп умеренно · >30пп ориентировочно.
`ci_width_pp` / `ci_low`–`ci_high` / `PRAI CI (TabPFN), %` / `funnel_unknown_range_pp` — НЕ
доверительный интервал: это разброс прогноза по Монте-Карло-вариантам НЕИЗВЕСТНЫХ стадий
воронки, ровно 0 при полной воронке. Называть его «ДИ» или «статистической уверенностью»
нельзя; широкий — значит «часть этапов цикла не внесена».
Разброс ансамбля моделей (DT vs PRAI): <15пп единодушен · 15–30пп умеренно · >30пп значимое расхождение.
Confidence grade A/B/C = разброс ансамбля TabPFN + согласованность DT и PRAI.

Все категории вычисляет Python (consult.build_flags) — агент их ОБЪЯСНЯЕТ, не пересчитывает.
Числа — только из пакетов. Не ставить диагнозов, не назначать протоколов.
