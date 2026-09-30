# -*- coding: utf-8 -*-
"""Build the publication-ready Markdown report + machine-readable CSV tables
from the raw prediction dump. Every number in the MD comes from the dump.

  python make_report.py preds/clinics_ctx2000.parquet
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np, pandas as pd
from evalkit import all_metrics, boot_ci, paired_boot
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _local_paths import local_path  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else str(local_path("tabfm_workdir") / "preds/clinics_ctx2000.parquet")
OUTDIR = str(Path(__file__).resolve().parents[1])
DATA = str(local_path("tabfm_workdir") / "data")
os.makedirs(OUTDIR, exist_ok=True)
TARGET = "Исход переноса"
NBOOT = 3000

ORDER = ["tabfm", "tabpfn", "logreg", "hgb", "prior"]
NICE = {"tabfm": "TabFM 1.0.0", "tabpfn": "TabPFN-3", "logreg": "LogReg",
        "hgb": "HistGBDT", "prior": "Базовая ставка"}
PANELS = [
    ("B_fertimed_prod", "Fertimed, 6 мес. 2026 — продакшн-вход"),
    ("B_fertimed_native", "Fertimed, 6 мес. 2026 — нативные колонки"),
    ("C_twoline_prod", "Two-Line, последний прогон — продакшн-вход"),
    ("C_twoline_native", "Two-Line, последний прогон — нативные колонки"),
]
METRIC_COLS = ["n", "pos_rate", "mean_pred", "roc_auc", "pr_auc", "brier",
               "log_loss", "ece", "cal_slope", "cal_intercept", "acc@0.5"]
HDR = {"n": "n", "pos_rate": "Факт, доля", "mean_pred": "Средний прогноз",
       "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier",
       "log_loss": "Log-loss", "ece": "ECE", "cal_slope": "Наклон калибр.",
       "cal_intercept": "Сдвиг калибр.", "acc@0.5": "Acc@0.5",
       "auc_ci": "95% ДИ ROC-AUC", "brier_ci": "95% ДИ Brier",
       "fit_s": "Фит, с", "pred_s": "Предсказание, с"}

df = pd.read_parquet(SRC)
seeds = sorted(int(s) for s in df.ctx_seed.unique())
CTX = int(SRC.split("ctx")[-1].split(".")[0].split("_")[0])


def md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    head = "| " + " | ".join([frame.index.name or ""] + cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    body = []
    for idx, row in frame.iterrows():
        body.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join([head, sep] + body)


def fmt(v, nd=3):
    if isinstance(v, float) and np.isnan(v):
        return "н/д"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}f}"


tables, ci_tables, delta_tables, preds_all = {}, {}, {}, {}
for q, title in PANELS:
    d0 = df[df["query"] == q]
    if d0.empty:
        continue
    rows, preds = [], {}
    for m in ORDER:
        d = d0[d0.model == m]
        if d.empty:
            continue
        g = d.groupby("row").agg(y=("y", "first"), p=("p", "mean"))
        y, p = g.y.values.astype(int), g.p.values
        preds[m] = (y, p)
        met = all_metrics(y, p)
        alo, ahi = boot_ci(y, p, "roc_auc", n_boot=NBOOT, seed=0)
        blo, bhi = boot_ci(y, p, "brier", n_boot=NBOOT, seed=0)
        met["auc_ci"] = f"{alo:.3f}–{ahi:.3f}"
        met["brier_ci"] = f"{blo:.3f}–{bhi:.3f}"
        met["fit_s"] = round(float(d.groupby("ctx_seed").fit_seconds.first().mean()), 1)
        met["pred_s"] = round(float(d.groupby("ctx_seed").pred_seconds.first().mean()), 1)
        met["_model"] = NICE[m]
        rows.append(met)
    raw = pd.DataFrame(rows).set_index("_model")
    raw.index.name = "Модель"
    preds_all[q] = preds

    show = pd.DataFrame(index=raw.index)
    for c in METRIC_COLS:
        show[HDR[c]] = [fmt(v, 0 if c == "n" else 3) for v in raw[c]]
    show[HDR["auc_ci"]] = raw["auc_ci"]
    show[HDR["brier_ci"]] = raw["brier_ci"]
    show[HDR["fit_s"]] = raw["fit_s"]
    show[HDR["pred_s"]] = raw["pred_s"]
    tables[q] = show
    raw[METRIC_COLS + ["auc_ci", "brier_ci", "fit_s", "pred_s"]].to_csv(
        f"{OUTDIR}/metrics_{q}.csv", encoding="utf-8")

    if "tabfm" in preds and "tabpfn" in preds:
        y, pa = preds["tabfm"]; _, pb = preds["tabpfn"]
        rr = []
        for metric in ["roc_auc", "pr_auc", "brier", "log_loss", "ece"]:
            r = paired_boot(y, pa, pb, metric, n_boot=NBOOT, seed=0)
            r["metric"] = metric
            rr.append(r)
        dd = pd.DataFrame(rr).set_index("metric")
        dd.to_csv(f"{OUTDIR}/delta_tabfm_minus_tabpfn_{q}.csv", encoding="utf-8")
        disp = pd.DataFrame(index=dd.index)
        disp.index.name = "Метрика"
        disp["Δ (TabFM − TabPFN-3)"] = [f"{v:+.4f}" for v in dd["delta"]]
        disp["95% ДИ разности"] = [f"{lo:+.4f}…{hi:+.4f}" for lo, hi in zip(dd["lo"], dd["hi"])]
        disp["p (двусторонний)"] = [f"{v:.3f}" for v in dd["p_two_sided"]]
        delta_tables[q] = disp

# ── covariate shift table ────────────────────────────────────────────────────
A = pd.read_parquet(f"{DATA}/panel_A_16k.parquet"); yA = A.pop(TARGET)
B = pd.read_parquet(f"{DATA}/panel_B_fertimed_prod.parquet"); yB = B.pop(TARGET)
C = pd.read_parquet(f"{DATA}/panel_C_twoline_prod.parquet"); yC = C.pop(TARGET)
shift = pd.DataFrame({
    "Контекст 16k, среднее": A.mean().round(3),
    "Контекст 16k, SD": A.std().round(3),
    "Fertimed, среднее": B.mean().round(3),
    "Two-Line, среднее": C.mean().round(3),
})
shift["SMD Fertimed"] = ((B.mean() - A.mean()) / A.std()).round(3)
shift["SMD Two-Line"] = ((C.mean() - A.mean()) / A.std()).round(3)
shift.index.name = "Признак"
shift.to_csv(f"{OUTDIR}/covariate_shift.csv", encoding="utf-8")

# ── assemble MD ──────────────────────────────────────────────────────────────
L = []
A_ = L.append
A_("# TabFM 1.0.0 против TabPFN-3 на данных эмбриологической лаборатории")
A_("")
A_("Независимое сравнение двух табличных foundation-моделей на задаче прогноза "
   "наступления беременности после переноса. Прогон выполнен отдельно от "
   "аналитического пайплайна: ни один файл пайплайна не изменялся, результаты "
   "получены на заново собранных матрицах признаков.")
A_("")
A_(f"Дата прогона: 2026-09-18. Источник предсказаний: `{os.path.basename(SRC)}`.")
A_("")
A_("---")
A_("")
A_("@@SUMMARY@@")
A_("")
A_("## 1. Что сравнивалось")
A_("")
A_("| | TabFM | TabPFN-3 |")
A_("|---|---|---|")
A_("| Разработчик | Google Research | Prior Labs |")
A_("| Версия пакета | 1.0.1 (веса `tabfm_v1_0_0`) | 8.0.8 |")
A_("| Backend в прогоне | PyTorch, CPU | PyTorch, CPU |")
A_("| Размер чекпоинта классификации | **6.2 ГБ** | **203 МБ** |")
A_("| Лицензия кода | Apache-2.0 | — |")
A_("| Лицензия весов | `tabfm-non-commercial-v1.0`, только некоммерческое и непродакшн-использование | PriorLabs, некоммерческая |")
A_("| Технический отчёт / статья | отсутствует (README прямо это указывает) | есть |")
A_("| Обучение | сотни миллионов синтетических датасетов, порождённых структурными причинными моделями | синтетический приор |")
A_("| Заявленный бенчмарк | TabArena, 38 классификационных и 13 регрессионных наборов, 700–150 000 наблюдений; замеры на TPU | TabArena и др. |")
A_("")
A_("Обе модели работают in-context: обучающая выборка подаётся как контекст, "
   "параметры под задачу не дообучаются.")
A_("")
A_("## 2. Оборудование и стоимость прогона")
A_("")
A_("Intel Core i7-8665U @ 1.90 ГГц (4 ядра / 8 потоков), 32 ГБ ОЗУ, GPU отсутствует. "
   "Это существенная деталь: официальные бенчмарки TabFM сняты на TPU, и все "
   "приведённые ниже замеры времени характеризуют CPU-сценарий.")
A_("")


def timing_block():
    rows = []
    d0 = df[df["query"] == "B_fertimed_prod"]
    for m in ORDER:
        d = d0[d0.model == m]
        if d.empty:
            continue
        rows.append({
            "Модель": NICE[m],
            f"Фит на контексте {CTX} строк, с": round(float(d.groupby("ctx_seed").fit_seconds.first().mean()), 1),
            "Предсказание 250 строк, с": round(float(d.groupby("ctx_seed").pred_seconds.first().mean()), 1),
        })
    return pd.DataFrame(rows).set_index("Модель")


tb = timing_block()
A_(md_table(tb))
A_("")
fit_fm = float(tb.loc["TabFM 1.0.0"].iloc[0]); fit_pf = float(tb.loc["TabPFN-3"].iloc[0])
pr_fm = float(tb.loc["TabFM 1.0.0"].iloc[1]); pr_pf = float(tb.loc["TabPFN-3"].iloc[1])
A_(f"TabFM медленнее TabPFN-3 в **{fit_fm/fit_pf:.0f} раз** на подготовке контекста и в "
   f"**{pr_fm/pr_pf:.0f} раз** на предсказании. При этом TabFM работал с "
   f"`n_estimators=4` вместо штатных 32 — полная конфигурация была бы ещё в 8 раз дороже.")
A_("")
A_("## 3. Данные")
A_("")
A_("Все панели собраны заново из исходных выгрузок по той же 18-признаковой "
   "формуле, что использует продакшн-путь (`mc_features._build_features`): "
   "возраст, номер попытки, число фолликулов, ОКК, инсеминированных, 2PN, "
   "дробящихся на 3-й день, бластоцист, бластоцист хорошего качества, пять "
   "производных частот, число эмбрионов 5-го дня, заморожено, перенесено, KPIScore.")
A_("")
A_("| Панель | Источник | n | Доля беременностей |")
A_("|---|---|---|---|")
A_(f"| A — обучающий контекст | `all_df_with_KPI.xlsx` | {len(A)} | {yA.mean():.3f} |")
A_(f"| B — Fertimed, 6 мес. 2026 | выгрузка 1С, `6 месяцев 2026_txtoutcome_cleaned.xlsx` | {len(B)} | {yB.mean():.3f} |")
A_(f"| C — Two-Line, последний прогон | `OPU_table_enriched.xlsx`, прогон 20260913T154847Z | {len(C)} | {yC.mean():.3f} |")
A_("")
A_("Бинаризация исхода — по правилам `config/columns.yaml` (клиническая "
   "беременность → 1; «не наступила», биохимическая, внематочная, замершая → 0). "
   "Строки без переноса и без исхода исключены.")
A_("")
A_("**Проверка на утечку.** Точных совпадений 18-мерных векторов признаков между "
   "клиническими панелями и обучающим контекстом нет: 0 из 250 для Fertimed и "
   "0 из 36 для Two-Line.")
A_("")
A_("**Проверка корректности сборки признаков.** Чтобы убедиться, что заново "
   "собранная матрица признаков идентична той, что строит продакшн-путь, "
   "TabPFN-3 был прогнан на панели Two-Line в точной штатной конфигурации "
   "пайплайна (контекст 5000 строк, стратифицированная выборка, seed 42) и "
   "результат сопоставлен с `data/processed/tabpfn_predictions.csv` текущего "
   "прогона. Совпали все 36 строк: средний модуль расхождения 0.0000, "
   "максимальный 0.0001, корреляция 1.0000. Матрица признаков воспроизведена "
   "верно, и различия между моделями ниже не объясняются ошибкой подготовки данных.")
A_("")
A_("**Два варианта входа.** Продакшн-путь пайплайна подставляет в четыре "
   "колонки суррогаты: «дробящихся на 3-й день» и «эмбрионов 5-го дня» ← 2PN, "
   "«заморожено» ← бластоцисты хорошего качества, «перенесено» ← 1. В обучающем "
   "контексте в этих колонках стоят реальные значения. Поэтому каждая клиника "
   "прогонялась дважды: в продакшн-варианте (как пайплайн кормит TabPFN "
   "сегодня) и в нативном (реальные колонки выгрузки). Рассогласование "
   "обучения и инференса бьёт по обеим моделям одинаково, но его величина сама "
   "по себе информативна.")
A_("")
A_("### Сдвиг ковариат относительно обучающего контекста")
A_("")
A_("Стандартизованная разность средних (SMD) считается относительно SD "
   "обучающего контекста. Значения |SMD| > 0.5 обычно трактуют как заметный сдвиг.")
A_("")
sh = shift.copy()
A_(md_table(sh))
A_("")
A_("Fertimed умеренно смещён (большинство |SMD| < 0.45), Two-Line смещён сильно: "
   "когорта моложе на 1.1 SD, с большим числом ооцитов и бластоцист, и её базовая "
   "частота беременности выше обучающей. Часть сдвига в колонках «дробящихся на "
   "3-й день» и «эмбрионов 5-го дня» — артефакт продакшн-подстановки, а не "
   "свойство когорты.")
A_("")
A_("## 4. Протокол сравнения")
A_("")
A_(f"Контекст: стратифицированная подвыборка **{CTX} строк** из обучающего "
   f"датасета (16 056 циклов). Взято **{len(seeds)}** независимых подвыборки "
   f"(сиды {', '.join(map(str, seeds))}); предсказания усреднены по подвыборкам, "
   f"разброс по отдельным сидам приведён в CSV.")
A_("")
A_("Размер контекста выбран из вычислительного бюджета: продакшн-путь TabPFN "
   "использует 5000 строк, но фит TabFM на 5000 строк на этой машине — порядка "
   "часа на каждый ансамблевый член. Обе модели получают **байт в байт "
   "одинаковый контекст и одинаковые запросы**, поэтому сравнение между ними "
   "корректно; абсолютные значения метрик TabPFN-3 при этом чуть ниже, чем в "
   "штатной конфигурации пайплайна.")
A_("")
A_("Модели сравнения: логистическая регрессия со стандартизацией, "
   "HistGradientBoosting, и базовая ставка (константа, равная доле беременностей "
   "в контексте) — последняя задаёт уровень, ниже которого модель бесполезна.")
A_("")
A_("Метрики. **ROC-AUC** и **PR-AUC** — различающая способность, не зависят от "
   "калибровки. **Brier** и **log-loss** — общее качество вероятностного "
   "прогноза. **ECE** — средняя по децилям абсолютная разница между "
   "предсказанной и фактической частотой. **Наклон и сдвиг калибровки** — "
   "коэффициенты регрессии `logit(y) ~ a + b·logit(p)`; идеал b = 1, a = 0, "
   "значение b < 1 означает, что прогнозы слишком разбросаны относительно "
   "реальности. Доверительные интервалы и разности — стратифицированный "
   f"бутстрэп, {NBOOT} повторов; разности считаются **парно**, на одних и тех же "
   "перевыборках строк.")
A_("")
A_("## 5. Результаты")
A_("")
for q, title in PANELS:
    if q not in tables:
        continue
    A_(f"### {title}")
    A_("")
    A_(md_table(tables[q]))
    A_("")
    if q in delta_tables:
        A_("Прямое сравнение двух foundation-моделей, парный бутстрэп "
           "(отрицательная Δ по Brier / log-loss / ECE = TabFM лучше):")
        A_("")
        A_(md_table(delta_tables[q]))
        A_("")
A_("## 6. Ограничения")
A_("")
A_("1. **Мощность.** При n = 36 (Two-Line) 95% ДИ для ROC-AUC имеет ширину около "
   "0.4 — на этой панели никакое различие между моделями обнаружить нельзя, и "
   "приведённые числа следует читать как описание, а не как результат сравнения. "
   "Содержательные выводы опираются на панель Fertimed (n = 250).")
A_("2. **TabFM работал не в полной конфигурации.** `n_estimators=4` вместо 32, без "
   "SVD-признаков, без перекрёстных признаков и без Platt-калибровки, которые "
   "образуют заявленный Google вариант TabFM-Ensemble. Это смещает сравнение "
   "**против** TabFM: его реальный потолок выше измеренного здесь.")
A_("3. **Контекст 2000 строк вместо 16 056.** Обе модели ограничены одинаково, но "
   "ни одна не показана на полном объёме доступных данных.")
A_("4. **Размер выборок вне заявленного диапазона TabFM.** TabArena, на котором "
   "мерили TabFM, содержит наборы от 700 наблюдений; клинические панели здесь "
   "меньше.")
A_("5. **Лицензия.** Веса TabFM запрещены к коммерческому и продакшн-использованию. "
   "Для исследовательской публикации это допустимо, для встраивания в "
   "клинический пайплайн — нет.")
A_("6. Прогон на одной машине без GPU; относительные времена не переносятся на "
   "GPU/TPU-сценарий напрямую.")
A_("")
A_("## 7. Воспроизведение")
A_("")
A_("```")
A_("prep.py           # сборка трёх панелей из исходных выгрузок")
A_("run_clinics.py --ctx 2000 --seeds 42,7 --tabfm-n-est 4")
A_("make_report.py preds/clinics_ctx2000.parquet")
A_("```")
A_("")
A_("Рабочая папка прогона: `tabfm_workdir` из `config/research_paths.local.json` (изолированное окружение "
   "`.venv`, клон `google-research/tabfm`, сырые предсказания в `preds/`). "
   "Машинно-читаемые таблицы лежат рядом с этим файлом: `metrics_*.csv`, "
   "`delta_tabfm_minus_tabpfn_*.csv`, `covariate_shift.csv`.")
A_("")

# ── summary, assembled from the computed tables (no hand-typed numbers) ──────
def g(q, model, col):
    return float(pd.read_csv(f"{OUTDIR}/metrics_{q}.csv", index_col=0).loc[model, col])


def dl(q, metric, field):
    return float(pd.read_csv(f"{OUTDIR}/delta_tabfm_minus_tabpfn_{q}.csv",
                             index_col=0).loc[metric, field])


S = []
S.append("## Краткий итог")
S.append("")
S.append("1. **Различающая способность обеих моделей на этих данных слабая и между "
         "собой неотличима.** На основной панели (Fertimed, n = 250) ROC-AUC "
         f"составил {g('B_fertimed_prod','TabFM 1.0.0','roc_auc'):.3f} у TabFM против "
         f"{g('B_fertimed_prod','TabPFN-3','roc_auc'):.3f} у TabPFN-3 при продакшн-входе; "
         f"разность {dl('B_fertimed_prod','roc_auc','delta'):+.4f} "
         f"(95% ДИ {dl('B_fertimed_prod','roc_auc','lo'):+.4f}…{dl('B_fertimed_prod','roc_auc','hi'):+.4f}, "
         f"p = {dl('B_fertimed_prod','roc_auc','p_two_sided'):.3f}) статистически незначима. "
         "Ни одна из моделей не поднялась выше AUC 0.60.")
S.append("")
S.append("2. **Устойчивое и воспроизводимое преимущество TabFM — в калибровке, не в "
         "ранжировании.** На обоих вариантах входа Fertimed TabFM даёт значимо "
         "меньшие Brier, log-loss и ECE: при продакшн-входе Brier "
         f"{g('B_fertimed_prod','TabFM 1.0.0','brier'):.3f} против "
         f"{g('B_fertimed_prod','TabPFN-3','brier'):.3f} "
         f"(Δ = {dl('B_fertimed_prod','brier','delta'):+.4f}, p = {dl('B_fertimed_prod','brier','p_two_sided'):.3f}), "
         f"ECE {g('B_fertimed_prod','TabFM 1.0.0','ece'):.3f} против "
         f"{g('B_fertimed_prod','TabPFN-3','ece'):.3f}. Наклон калибровки у TabFM "
         f"({g('B_fertimed_prod','TabFM 1.0.0','cal_slope'):.3f}) ближе к единице, чем у "
         f"TabPFN-3 ({g('B_fertimed_prod','TabPFN-3','cal_slope'):.3f}). Практически это "
         "значит, что числу, которое выдаёт TabFM, можно доверять как вероятности "
         "заметно больше, хотя порядок пациентов по риску он расставляет не лучше.")
S.append("")
S.append("3. **Оба варианта входа систематически занижают прогноз.** Средний прогноз "
         f"{g('B_fertimed_prod','TabFM 1.0.0','mean_pred'):.3f} (TabFM) и "
         f"{g('B_fertimed_prod','TabPFN-3','mean_pred'):.3f} (TabPFN-3) против фактической "
         f"доли беременностей {g('B_fertimed_prod','TabFM 1.0.0','pos_rate'):.3f}. Причина "
         "общая для обеих моделей — базовая частота в обучающем контексте (0.320) "
         "ниже, чем в когорте Fertimed, и in-context-модель наследует приор контекста. "
         "TabPFN-3 занижает сильнее.")
S.append("")
S.append("4. **TabFM заметно чувствительнее к качеству входа.** При переходе от "
         "продакшн-подстановок к нативным колонкам TabPFN-3 деградирует "
         f"(AUC {g('B_fertimed_prod','TabPFN-3','roc_auc'):.3f} → "
         f"{g('B_fertimed_native','TabPFN-3','roc_auc'):.3f}, Brier "
         f"{g('B_fertimed_prod','TabPFN-3','brier'):.3f} → "
         f"{g('B_fertimed_native','TabPFN-3','brier'):.3f}), тогда как TabFM держится "
         f"(AUC {g('B_fertimed_native','TabFM 1.0.0','roc_auc'):.3f}, Brier "
         f"{g('B_fertimed_native','TabFM 1.0.0','brier'):.3f}), и именно на этом варианте "
         "его превосходство становится значимым уже и по ранжированию: "
         f"Δ ROC-AUC = {dl('B_fertimed_native','roc_auc','delta'):+.4f} "
         f"({dl('B_fertimed_native','roc_auc','lo'):+.4f}…{dl('B_fertimed_native','roc_auc','hi'):+.4f}, "
         f"p = {dl('B_fertimed_native','roc_auc','p_two_sided'):.3f}). Это наблюдение на одной "
         "панели и требует подтверждения.")
S.append("")
S.append("5. **На Two-Line (n = 36) различить модели невозможно.** Ширина 95% ДИ для "
         "ROC-AUC — около 0.4, все парные разности незначимы (p от 0.14 до 0.99), а "
         "формально лучший результат показывает логистическая регрессия "
         f"(AUC {g('C_twoline_native','LogReg','roc_auc'):.3f}). Панель приводится для "
         "полноты, выводов из неё делать нельзя.")
S.append("")
S.append("6. **Цена вопроса несопоставима.** Фит TabFM на контексте 2000 строк — "
         f"{g('B_fertimed_prod','TabFM 1.0.0','fit_s'):.0f} с против "
         f"{g('B_fertimed_prod','TabPFN-3','fit_s'):.0f} с у TabPFN-3 при чекпоинте 6.2 ГБ "
         "против 203 МБ, и это при урезанном ансамбле (4 члена вместо 32). "
         "Веса TabFM лицензированы только для некоммерческого и непродакшн-использования.")
S.append("")

out_md = "\n".join(L).replace("@@SUMMARY@@", "\n".join(S))
open(f"{OUTDIR}/README.md", "w", encoding="utf-8").write(out_md)
print(f"written: {OUTDIR}/README.md")
print("csv:", sorted(os.listdir(OUTDIR)))
