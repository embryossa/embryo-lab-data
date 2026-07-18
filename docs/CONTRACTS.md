# Версионированные контракты результатов

Версия контрактов: `1.0`. Полный запуск прекращается при отсутствии обязательной
колонки, дубликате ключа или несвежем результате внешнего инструмента.

| Артефакт | Ключ | Обязательные поля |
|---|---|---|
| Digital Twin | `_key` | `_key`, `DIGITAL TWIN`; `Preg` для оценки исходов |
| TabPFN | `patient_id` | `patient_id`, `p_tabpfn`, `ci_low`, `ci_high` |
| `dt_predictions.csv` | `patient_id` | `patient_id`, `cycle_uid`, `Preg`, `PRAI`, `DIGITAL TWIN`, `PRAI CI (TabPFN), %`, `ci_width_pp`, `ci_wide`, `join_status` |
| Agent packet | `cycle_uid` | `schema_version`, `cohort`, `patients[].cycle_uid`; PHI запрещена |
| MIRI event | `cycle_uid,event_id` | `cycle_uid`, `event_type`, `sensor`, `start`, `end`, `duration_minutes`, `max_deviation`, `severity` |
| Lot event (расходники, Airtable) | `cycle_uid,item_name` | `cycle_uid`, `item_name`, `attribute`, `event_type`, `lot_before`, `lot_after`, `change_date` |
| TabPFN-3 regression (количественные стадии) | `cycle_uid` | `cycle_uid`, ≥1 присутствующая стадия `TPFN3_<stage>_p10..p90/interval80_width` |
| OZEMA cycle (`ozema_cycles.csv`) | `cycle_uid` | `cycle_uid`, `ozema_flag`, `flagged`, `flagged_cycles_n`; `patient_id`/`patient_name` только локально (не в agent packet) |
| PGT-A summary (`pgt_summary.csv`) | `cycle_uid` | `cycle_uid`, `n_biopsied`, `euploid_rate`, `qc_pass_rate`; `report_date`/`pgt_run_id`/`match_reason`/`note` только локально |
| PGT-A embryo event (`pgt_embryo_events.csv`) | `cycle_uid,sample_id` | `cycle_uid`, `sample_id`, `category`, `karyotype`, `cnv_qc_status`, `overall_noise`, `sdi`, `sd_robust` |

Правила:

- `0` — известное наблюдение. Только пустое значение/`NaN` означает пропуск.
- Вероятности хранятся в диапазоне 0–1, ширина CI — в процентных пунктах.
- Join выполняется локально по нормализованному паспортному ID; Claude получает только
  `cycle_uid`. Crosswalk хранится в `data/processed/local_private/`.
- Каждый итоговый CSV/XLSX/HTML/JSON пишется атомарно: временный файл в той же папке,
  затем `os.replace`.
- `Model_FR=2PN/Inseminated` и `Model_KPIScore` повторяют формулы обучения. Их модельные
  русские алиасы используются только
  на входе старой модели и не входит в таблицу Vienna KPI.
- `indicator_class` отделяет Vienna KPI/PI, reference indicators и принятый локальный
  FR benchmark; `source_note` сохраняет происхождение порога.
- DNN действительна только при успешной загрузке scaler обучения и совпадении числа
  и порядка признаков.
- Lot event `event_type=lot_change` возникает только когда более одного периода лота
  реально пересекает окно культивирования; `event_type=stable` не подразумевает риск.
  Смена лота внутри окна — кандидат-confounder, не причина и не оценка качества лота.
- TabPFN-3 regression: стадия может отсутствовать в конкретном прогоне (внешний скрипт
  пропускает задачу без сигнала в DEV/OPU) — контракт проверяет только реально
  присутствующие стадии, не фиксированный набор из 6. Квантили каждой присутствующей
  стадии упорядочены `p10<=p25<=p50<=p75<=p90`. Сбой этого шага мягкий: причина остаётся
  в `regression_status` (`run_manifest.json`, cohort агентского пакета), остальной
  пайплайн выполняется как обычно. Это независимый количественный QA-сигнал (факт
  против P10–P90), он не переопределяет и не заменяет DT/TabPFN реконсилиацию по исходу
  переноса.
- MIRI event `gap_reason` (опциональное дополнительное поле, только для
  `event_type=gap`): `power_or_logging_outage` при длительности <8ч (потеря
  возможности ПИСАТЬ логи, например отключение питания — реальное неблагоприятное
  событие, не "нет данных"), `incubator_or_logging_stopped` при ≥8ч (явная остановка
  инкубатора или логирования). Порог — `incubator.DEFAULT_SEVERITY
  ["gap_reason_threshold_minutes"] = 480`. Не влияет на существующую severity
  (warning/critical, порог 10 минут) — это отдельное описательное измерение.
- OZEMA (`ozema.py`): один цикл — один ряд, флаги `OMA`/`FF`/`ZA`/`EEA` — односторонний
  точный биномиальный тест (α=0.05) наблюдаемой доли против порога (не просто "ниже
  черты"); стадия с n<4 — `"н/д"` (недостаточно данных), не пропуск ряда. Намеренно не
  литерал `"n/a"`: это значение входит в список `na_values` по умолчанию у
  `pandas.read_csv` и при повторном чтении CSV молча превращается в настоящий `NaN`.
  Методология:
  Van Der Kelen et al., Hum Reprod 2026 (сверено по листу OZEMA мастер-таблицы OPU на
  Google Drive). Работает на ШИРОКОЙ lab-стадийной популяции (без гейта
  transferred>0/Preg известен, без исключений TESE/open-cycle/NO-retrieval/freeze-all —
  эти исключения относятся только к таблицам компетентности эмбриолог/врач, см. ниже);
  единственное собственное исключение OZEMA — TESE/SMF из `sperm_source`, и только для
  стадии FF. Критерий "кандидат на генетическое консультирование" — ≥2 циклов с флагом
  у одного пациента (raw `patient_id`, т.к. повторные попытки делят один ID) —
  считается и отображается только локально, никогда в agent packet на этом уровне
  детализации (только агрегированный счётчик).
- Отчёт: вкладка «Прогноз беременности» (PRAI/TabPFN + Digital Twin по
  каждому циклу) берёт воронку реал/прогноз (MII/2PN/Bl/Good Bl) ИЗ P50 TabPFN-3
  regression этого же прогона (`regression_qc_frame`), а НЕ из старых полей
  Digital Twin `med_okk/med_mii/med_pn2/med_blasts/med_good` — это два разных
  источника прогноза воронки, не смешивать. Вкладка «TabPFN-3: прогноз протокола»
  — отдельная, per-stage P10-P90 QA (`_regression_tab`), не журнал по пациентам.
- Стиль/палитра отчёта: тёплая Claude.ai (`--clay:#d97757`, тёплый белый фон,
  status-цвета из шкал green/red/yellow/orange-100/800) — источник токенов:
  `<style>` сохранённой страницы Claude.ai, на которую указал владелец (сам
  артефакт-отчёт в этом файле не сохранился, восстановлена только палитра/шрифт
  chrome страницы). Меняется только по прямому запросу владельца.
- Вкладки «Кластерный анализ» и «Похожие случаи» в отчёте — прямой перенос секций
  §5/§8/§12 `notebooks/reference/Data_Twin_analytics.ipynb` (модельный боксплот,
  KAT vs PRAI, donut/профиль/исходы по кластерам, P(беременность) по кластерам,
  граф сходства пациентов + Similar Cases + Mann-Whitney intra/inter-cluster
  тест). Методология (веса признаков графа, k=5 соседей, порог сходства 70%,
  Clopper-Pearson CI, bootstrap 10000 для медиан) сохранена дословно из ноутбука;
  цвета кластеров переведены на палитру отчёта (C0=blue, C1=red, C2=green), но
  сохраняют исходную семантику ноутбука. Требует `networkx` (см.
  requirements.txt) — граф сходства (`_build_similarity_graph`) мягко деградирует
  до сообщения «недостаточно данных», если пакет недоступен или строк <4.
- **Канонические имена 7 основных моделей** — сверены с первоисточником
  (`IVF Digital Twin Pro/IVF Digital Twin`'s `befe.py`/`architecture.md`/
  `GAT_method_description.md`), не со старым ноутбуком (там `p_nvsa` ошибочно
  назван «KAN», а L6 назван «GNN» вместо «GAT»): MC pipeline=`p_per_transfer`
  (L1+L2 Monte Carlo), DNN=`DNN` (наша отдельная независимая проверка,
  `external/dnn/Evo_calculations.py`, НЕ часть ансамбля DT), KAT=`p_kat_raw`
  (L3, KAN+FT-Transformer), GAT=`p_gnn_raw`/смешанный `p_gnn_ens` (L6, graph
  attention transformer), CSDI=`p_csdi` (L5, диффузия), TabPFN=`PRAI`
  (`p_tabpfn` переименован на границе `reconcile.py` — одна и та же модель, не
  два оценщика), Digital Twin=`DIGITAL TWIN` (L7 BEFE, итоговый ансамбль).
  `bayes_mean`/`bayes_ci_*` — Bayesian-обновлённый posterior над тем же L1 MC
  прогоном (`ivf_core.py: post = res["posterior"]`), не отдельная архитектура.
  `p_nvsa` — внутренняя KPI-якорная коррекция KAT, подкомпонент L3. Ни то ни
  другое не входит в 7 основных моделей и не показывается как отдельный
  оценщик — см. `.claude/skills/dt-ensemble/SKILL.md`.
- PGT-A (`pgt.py`): eMap Cycle Report PDF, скачанные вручную из личного кабинета
  pa.ivf.uz (нет API). Сопоставление с `cycle_uid` — по имени пациента (`Subject ID`
  портала = склеенные Имя+Фамилия без пробела), разрешение повторных попыток одной
  пациентки — по попаданию даты отчёта в окно `Date OPU..Date ET` конкретного цикла;
  если это не разрешает неоднозначность — матч не делается (`match_reason` фиксирует
  причину), а не угадывается. Классификация euploid/aneuploid/mosaic — по английскому
  полю `Summary result`/`CNV Status`, НЕ по русскоязычному `Sign-off comment`: у этого
  отчёта кириллица встроена нечитаемым шрифтом без ToUnicode CMap — любая библиотека
  извлечения текста (проверено на pdfplumber и PyMuPDF) возвращает для него нечитаемые
  байты, это ограничение самого PDF, а не парсера. Признаки noise/SDI/SD Robust
  считаются только по подвыборке эуплоидных эмбрионов (прокси качества биопсии/
  секвенирования именно переносимой когорты), отдельно от общей когорты.
  `analysis_settings_variable=true` означает, что версия софта/пороги отличались
  между эмбрионами одного отчёта — сигнал для сравнения между разными прогонами/
  датами, не ожидается внутри одного цикла биопсии. Идёт по когорте review-list ∪
  известный исход переноса (см. `.claude/skills/run-analysis/SKILL.md` шаг 2b),
  raw PDF и ФИО остаются в `local_private/pgt_reports/`, никогда не коммитятся и
  никогда не попадают в agent packet.
- Таблицы компетентности эмбриолог/врач (`kpi.aggregate_kpis(..., metrics=...)` с
  `EMBRYOLOGIST_METRICS`/`DOCTOR_METRICS`): эмбриолог — FR/ClR/BlR/TGBDR/DegR/3pNR;
  врач — только OCCR/MIIR. Никогда не смешиваются в одной таблице/сравнении. Строятся
  на ШИРОКОЙ lab-популяции (без гейта transferred>0), из которой
  `kpi.competency_exclusion_mask` убирает открытые циклы без известного Bl,
  NO-retrieval и freeze-all(Cryo) — токены в `config/columns.yaml:
  competency_exclusion`. Это исключение действует ТОЛЬКО для этих двух таблиц —
  когортный KPI (срезы время/метод), review-list и OZEMA его не используют.
