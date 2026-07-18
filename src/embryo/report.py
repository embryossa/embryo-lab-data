"""Локальный HTML-отчёт: KPI (раздельно эмбриолог/врач), модели и калибровка,
TabPFN-3 regression QA, review-list с расширенными флагами, MIRI (только эпизоды),
партии расходников, OZEMA и итоговая интерпретация lead-analyst. Разбит на вкладки
(чистый CSS, без JS-зависимостей). Числа приходят уже посчитанными из Python; здесь
только рендеринг.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, norm, pearsonr, spearmanr, wilcoxon

from ._io import atomic_write_text
from . import kpi
from . import ozema

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False


# Palette reverse-engineered from the owner's referenced Claude.ai page ("TabPFN --
# журнал прогнозов.html"): that file turned out to be a saved chat page rather than
# the standalone artifact, so the actual report DOM isn't recoverable -- but the
# page's own CDS (Claude Design System) tokens ARE present in its <style> block, and
# are what the owner asked to match. --clay:#d97757 is the literal `--cds-clay`
# token; the warm off-white/near-black and status-scale endpoints (green/red/yellow/
# orange -100 and -800 steps) are pulled from the same token set.
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"
COLORS = {"blue": "#5598E7", "red": "#B3261E", "green": "#1E7A34",
          "orange": "#D97757", "gray": "#83827B", "ink": "#262624"}
_LAYOUT = dict(font=dict(family=FONT, size=12, color=COLORS["ink"]),
               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
               margin=dict(l=50, r=30, t=55, b=45))
_plotly_embedded = False

_MODEL_BAR_COLORS = {"MC": "#5598E7", "KAT": "#8B5CF6", "GAT": "#C9A227", "PRAI": "#1E7A34",
                     "DT": "#D97757", "DNN": "#B34A24"}
_STATUS_KIND = {"benchmark": "ok", "competency": "warn", "outside_control": "crit",
                "insufficient_data": "info"}
_STATUS_LABEL = {"benchmark": "бенчмарк", "competency": "компетентность",
                 "outside_control": "вне контроля", "insufficient_data": "нет данных"}
_GAP_REASON_LABEL = {
    "power_or_logging_outage": ("⚡ Возможный перебой питания/логирования (<8ч)", "warn"),
    "incubator_or_logging_stopped": ("⏻ Инкубатор/логирование остановлены (≥8ч)", "crit"),
}
_EXCLUSION_NOTE = (
    "Из таблиц компетентности исключены: открытые циклы без известного Bl (ещё в "
    "культивировании на момент выгрузки), циклы NO-retrieval и freeze-all (Cryo) — "
    "как в референсном Vienna-style дашборде владельца. Когортный KPI (срезы "
    "время/метод), Review-list и OZEMA этих исключений не используют."
)
_TABS = [
    ("overview", "Обзор"),
    ("embryologist", "KPI эмбриологов"),
    ("doctor", "KPI врачей"),
    ("models", "Модели и калибровка"),
    ("cluster", "Кластерный анализ"),
    ("similar", "Похожие случаи"),
    ("pregnancy", "Прогноз беременности"),
    ("regression", "TabPFN-3: прогноз протокола"),
    ("review", "Review-list"),
    ("miri", "MIRI 6391"),
    ("lots", "Партии расходников"),
    ("ozema", "OZEMA"),
    ("lead", "Итоговая интерпретация"),
]

# Constants below port notebooks/reference/Data_Twin_analytics.ipynb's cluster/
# model/similarity-graph analysis onto this report's data and color language
# (the notebook's own C/CLUSTER_HEX are the old blue/red/green/orange/purple/
# teal/gray palette -- kept here only where it carries semantic meaning:
# clusters map onto this report's existing blue/red/green accents).
_CLUSTER_NAMES = {0: "C0 Standard", 1: "C1 Poor", 2: "C2 High"}
_CLUSTER_HEX = {0: "#5598E7", 1: "#B3261E", 2: "#1E7A34"}
_CLUSTER_PATTERNS = {0: "/", 1: "x", 2: ""}
# Names verified against the IVF Digital Twin project's own source (befe.py,
# architecture.md, GAT_method_description.md) -- not the older notebook's labels,
# which mislabeled p_nvsa as "KAN" and called L6 "GNN" instead of "GAT". PRAI is
# TabPFN's own p_tabpfn, renamed at the reconcile.py join boundary -- not a
# separate model. bayes_mean/p_nvsa are real sub-scores but not among these 7
# primary models (see dt-ensemble skill for what they actually are).
_NB_MODEL_COLS = {
    "MC pipeline": "p_per_transfer", "DNN": "DNN", "KAT": "p_kat_raw", "GAT": "p_gnn_raw",
    "CSDI": "p_csdi", "TabPFN (PRAI)": "PRAI", "Digital Twin": "DIGITAL TWIN",
}
_NB_PALETTE = ["#D97757", "#5598E7", "#8B5CF6", "#1E7A34", "#B34A24", "#C9A227", "#83827B"]
_GRAPH_FEATURES = {
    "Age": 1.5, "amh": 2.0, "afc": 1.5, "OCC": 1.0, "MII": 1.0,
    "Bl": 1.2, "Good Bl": 1.2, "p_per_transfer": 1.0, "p_kat_raw": 1.0, "bayes_mean": 0.8,
}
_GRAPH_K_NEIGHBORS = 5
_GRAPH_SIM_THRESHOLD = 0.70


def _hex_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color


def _interpret_r(r: float) -> str:
    ar = abs(r)
    return "Large" if ar >= 0.50 else "Medium" if ar >= 0.30 else "Small" if ar >= 0.10 else "Negligible"


def _interpret_d(d: float) -> str:
    ad = abs(d)
    return "Large" if ad >= 0.80 else "Medium" if ad >= 0.50 else "Small" if ad >= 0.20 else "Negligible"


def _fig_html(fig) -> str:
    global _plotly_embedded
    if not _HAVE_PLOTLY:
        return ""
    include = "inline" if not _plotly_embedded else False
    _plotly_embedded = True
    return pio.to_html(fig, include_plotlyjs=include, full_html=False,
                       config={"displayModeBar": False})


def _table(df: pd.DataFrame | None, max_rows=100) -> str:
    if df is None or len(df) == 0:
        return "<p class='muted'>нет данных</p>"
    return (f"<div class='table-wrap'>"
           f"{df.head(max_rows).to_html(index=False, border=0, classes='tbl', escape=True)}</div>")


def _manual_table(df: pd.DataFrame | None, raw_cols: frozenset = frozenset(), max_rows=500) -> str:
    """Like ``_table`` but lets specific columns carry pre-built HTML (badges)."""
    if df is None or df.empty:
        return "<p class='muted'>нет данных</p>"
    d = df.head(max_rows)
    header = "".join(f"<th>{escape(str(c))}</th>" for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            if c in raw_cols:
                cells.append(f"<td>{'' if pd.isna(v) else v}</td>")
            else:
                cells.append(f"<td>{'' if pd.isna(v) else escape(str(v))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (f"<div class='table-wrap'><table class='tbl'><thead><tr>{header}</tr></thead>"
           f"<tbody>{''.join(rows)}</tbody></table></div>")


def _aggregate_table(kpi_df: pd.DataFrame, supplied=None) -> pd.DataFrame:
    if isinstance(supplied, pd.DataFrame) and not supplied.empty:
        return supplied.copy()
    if hasattr(kpi, "aggregate_kpis"):
        return kpi.aggregate_kpis(kpi_df)
    return pd.DataFrame()


def _status_column(frame: pd.DataFrame) -> str | None:
    for c in ("status", "control_status", "assessment"):
        if c in frame:
            return c
    return None


def _kpi_panel(agg: pd.DataFrame) -> tuple[str, list[str]]:
    if agg.empty:
        return "", []
    name_col = next((c for c in ("kpi", "KPI", "metric") if c in agg), None)
    rate_col = next((c for c in ("rate", "value", "aggregate_rate") if c in agg), None)
    status_col = _status_column(agg)
    alerts = []
    if status_col:
        bad = agg[agg[status_col].astype(str).str.lower().isin(
            {"alert", "outside", "outside_control", "fail", "отклонение"})]
        if name_col:
            alerts = bad[name_col].astype(str).tolist()
    if not (_HAVE_PLOTLY and name_col and rate_col):
        return "", alerts
    x = agg[name_col].astype(str)
    y = pd.to_numeric(agg[rate_col], errors="coerce")
    lo_col = "ci_low_95" if "ci_low_95" in agg else ("ci_low" if "ci_low" in agg else None)
    hi_col = "ci_high_95" if "ci_high_95" in agg else ("ci_high" if "ci_high" in agg else None)
    # Same tri-tone status language as the badges/competency tables elsewhere in
    # this report: green = ok, red = alert (never the old ad-hoc blue/red pair).
    colors = [COLORS["red"] if str(s).lower() in
              {"alert", "outside", "outside_control", "fail", "отклонение"}
              else COLORS["green"] for s in agg.get(status_col, pd.Series("ok", index=agg.index))]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors, opacity=.55,
                           error_y=dict(type="data", color=COLORS["gray"],
                                        array=(pd.to_numeric(agg[hi_col], errors="coerce") - y)
                                        if hi_col else None,
                                        arrayminus=(y - pd.to_numeric(agg[lo_col], errors="coerce"))
                                        if lo_col else None)))
    fig.update_layout(title="Когортный KPI: агрегат Σ numerator / Σ denominator (вся лаборатория)",
                      yaxis_title="частота", **_LAYOUT)
    return f"<div class='chart-card'>{_fig_html(fig)}</div>", alerts


def _model_distribution_chart(df_u: pd.DataFrame) -> str:
    """Box distribution of every pipeline model's P(pregnancy), notebook style:
    boxmean='sd', translucent fill, hline at P=50%, plus a describe() table.
    """
    if not _HAVE_PLOTLY or df_u is None or df_u.empty:
        return ""
    cols = {name: col for name, col in _NB_MODEL_COLS.items() if col in df_u.columns}
    if not cols:
        return ""
    fig = go.Figure()
    stats_rows = []
    for i, (name, col) in enumerate(cols.items()):
        vals = pd.to_numeric(df_u[col], errors="coerce").dropna()
        if vals.empty:
            continue
        color = _NB_PALETTE[i % len(_NB_PALETTE)]
        fig.add_trace(go.Box(
            y=vals, name=name, boxmean="sd", marker_color=color,
            marker=dict(size=5, opacity=0.6), line_color=color, fillcolor=_hex_rgba(color, 0.20),
            hovertemplate=f"<b>{name}</b><br>%{{y:.3f}}<extra></extra>",
        ))
        stats_rows.append({
            "Модель": name, "N": len(vals), "Mean": round(float(vals.mean()), 3),
            "SD": round(float(vals.std()), 3), "Min": round(float(vals.min()), 3),
            "25%": round(float(vals.quantile(.25)), 3), "50%": round(float(vals.median()), 3),
            "75%": round(float(vals.quantile(.75)), 3), "Max": round(float(vals.max()), 3),
        })
    if not fig.data:
        return ""
    fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS["gray"],
                  annotation_text="P=50%", annotation_position="right")
    fig.update_layout(**_LAYOUT, height=420,
                      title="Распределение прогнозов беременности по всем моделям",
                      yaxis_title="P(беременность)", yaxis_range=[0, 1])
    return f"<div class='chart-card'>{_fig_html(fig)}</div>" + _table(pd.DataFrame(stats_rows))


def _kat_vs_prai_stats(df_u: pd.DataFrame) -> dict | None:
    """Pure numeric KAT-vs-PRAI comparison: correlation, bias, MAE, Wilcoxon.

    Shared by the chart (for its subplot titles/metrics line) and the
    lead-analyst packet summary (``build_analysis_summary``) — single source
    of truth so the two never drift apart.
    """
    if df_u is None or df_u.empty or not {"p_kat_raw", "PRAI"}.issubset(df_u.columns):
        return None
    sub = df_u[["p_kat_raw", "PRAI"]].dropna()
    if len(sub) < 2:
        return None
    diff = sub["p_kat_raw"] - sub["PRAI"]
    r, _ = pearsonr(sub["PRAI"], sub["p_kat_raw"])
    rho, _ = spearmanr(sub["PRAI"], sub["p_kat_raw"])
    result = {
        "n": int(len(sub)), "pearson_r": float(r), "spearman_rho": float(rho),
        "mae": float(diff.abs().mean()), "bias": float(diff.mean()),
        "sd_diff": float(diff.std()) if len(sub) > 1 else 0.0,
    }
    if len(sub) >= 10:
        w_stat, w_p = wilcoxon(sub["p_kat_raw"], sub["PRAI"])
        result["wilcoxon_stat"] = float(w_stat)
        result["wilcoxon_p"] = float(w_p)
    return result


def _kat_vs_prai_chart(df_u: pd.DataFrame) -> str:
    """KAT (simulated) vs PRAI (real full cycle): scatter + CI band + trend on the
    left, diff histogram + KDE + bias/SD annotations on the right (notebook §5).
    """
    if not _HAVE_PLOTLY or df_u is None or df_u.empty:
        return ""
    if not {"p_kat_raw", "PRAI"}.issubset(df_u.columns):
        return "<p class='muted'>нет колонок p_kat_raw/PRAI для сравнения KAT vs PRAI</p>"
    stats = _kat_vs_prai_stats(df_u)
    if stats is None:
        return "<p class='muted'>недостаточно данных для сравнения KAT vs PRAI (n&lt;2)</p>"
    keep = ["p_kat_raw", "PRAI"] + [c for c in ("ci_kat_low", "ci_kat_high", "cycle_uid",
                                                "Age", "dominant_cluster", "patient_name")
                                    if c in df_u.columns]
    sub = df_u[keep].dropna(subset=["p_kat_raw", "PRAI"]).copy()

    name_col = "patient_name" if "patient_name" in sub.columns else (
        "cycle_uid" if "cycle_uid" in sub.columns else None)
    diff = sub["p_kat_raw"] - sub["PRAI"]
    r, rho, mae, bias, sd_diff = (stats["pearson_r"], stats["spearman_rho"], stats["mae"],
                                  stats["bias"], stats["sd_diff"])

    has_ci = "ci_kat_low" in sub.columns and sub["ci_kat_low"].notna().sum() > 0
    if has_ci:
        sub["sd_kat"] = ((sub["ci_kat_high"] - sub["ci_kat_low"]) / (2 * 1.96)).clip(lower=0.005)
    else:
        sub["sd_kat"] = float(sub["p_kat_raw"].std()) * 0.5

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"KAT vs PRAI (r={r:.3f}, ρ={rho:.3f}, MAE={mae:.3f})",
        f"Распределение разниц KAT−PRAI (bias={bias:+.3f})",
    ], horizontal_spacing=0.13)

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="rgba(150,150,150,0.55)", dash="dash", width=1.5),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    x_arr = np.linspace(0, 1, 120)
    mean_sd = float(sub["sd_kat"].mean())
    fig.add_trace(go.Scatter(x=x_arr, y=np.clip(x_arr + mean_sd, 0, 1), mode="lines",
                             line=dict(color="rgba(100,100,100,0)", width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_arr, y=np.clip(x_arr - mean_sd, 0, 1), mode="lines",
                             fill="tonexty", fillcolor="rgba(150,150,150,0.10)",
                             line=dict(color="rgba(100,100,100,0.18)", dash="dot", width=0.8),
                             name=f"±SD KAT (μ={mean_sd:.3f})", showlegend=True,
                             hoverinfo="skip"), row=1, col=1)
    if len(sub) >= 4:
        z = np.polyfit(sub["PRAI"], sub["p_kat_raw"], 1)
        y_fit = np.poly1d(z)(x_arr)
        fig.add_trace(go.Scatter(x=x_arr, y=np.clip(y_fit, 0, 1), mode="lines",
                                 line=dict(color="rgba(80,80,80,0.45)", dash="dot", width=1.8),
                                 name="Линия тренда", showlegend=True, hoverinfo="skip"), row=1, col=1)

    for c in (0, 1, 2):
        s = sub[sub["dominant_cluster"] == c] if "dominant_cluster" in sub.columns else sub
        if s.empty:
            continue
        cname, chex = _CLUSTER_NAMES.get(c, str(c)), _CLUSTER_HEX.get(c, COLORS["gray"])
        text_vals = s[name_col].astype(str) if name_col else s.index.astype(str)
        fig.add_trace(go.Scatter(
            x=s["PRAI"], y=s["p_kat_raw"], mode="markers", name=cname, text=text_vals,
            error_y=dict(type="data", array=s["sd_kat"].values, color=_hex_rgba(chex, 0.35),
                        thickness=1.5, width=5),
            hovertemplate=("<b>%{text}</b><br>PRAI: %{x:.3f}<br>KAT: %{y:.3f}<br>"
                          "KAT SD: ±%{error_y.array:.3f}<extra></extra>"),
            marker=dict(size=12, color=_hex_rgba(chex, 0.68),
                       line=dict(width=1.8, color=_hex_rgba(chex, 0.95)), symbol="circle"),
        ), row=1, col=1)

    fig.add_trace(go.Histogram(x=diff, nbinsx=14,
                               marker=dict(color=_hex_rgba(COLORS["blue"], 0.55),
                                          line=dict(color="white", width=0.8)),
                               name="KAT−PRAI", showlegend=False,
                               hovertemplate="%{x:.3f}: %{y} циклов<extra></extra>"), row=1, col=2)
    if sd_diff > 0:
        x_kde = np.linspace(float(diff.min()) - 0.05, float(diff.max()) + 0.05, 200)
        n_bins = 14
        bin_w = (float(diff.max()) - float(diff.min())) / n_bins or 1.0
        scale = len(diff) * bin_w
        y_kde = norm.pdf(x_kde, loc=bias, scale=sd_diff) * scale
        fig.add_trace(go.Scatter(x=x_kde, y=y_kde, mode="lines",
                                 line=dict(color=_hex_rgba(COLORS["blue"], 0.90), width=2.5),
                                 name="KDE (норм. прибл.)",
                                 hovertemplate="%{x:.3f}: плотность=%{y:.2f}<extra></extra>"), row=1, col=2)
    for x_v, color_v, dash_v, ann, pos in [
        (0, COLORS["red"], "dash", "Нет разницы", "top right"),
        (bias, COLORS["orange"], "dot", f"bias={bias:+.3f}", "top left"),
        (bias + sd_diff, "#888", "dot", f"+SD={bias + sd_diff:+.3f}", "bottom right"),
        (bias - sd_diff, "#888", "dot", f"−SD={bias - sd_diff:+.3f}", "bottom left"),
    ]:
        fig.add_vline(x=x_v, line_dash=dash_v, line_color=color_v, line_width=1.8,
                      annotation_text=ann, annotation_position=pos, annotation_font_size=10, row=1, col=2)
    fig.add_vrect(x0=bias - sd_diff, x1=bias + sd_diff, fillcolor=_hex_rgba(COLORS["orange"], 0.08),
                  line_width=0, annotation_text=f"±1SD ({sd_diff:.3f})", annotation_position="top left",
                  annotation_font_size=9, row=1, col=2)

    fig.update_xaxes(title_text="PRAI (реальный цикл)", range=[0, 1], row=1, col=1, zeroline=False)
    fig.update_yaxes(title_text="KAT (симулированный)", range=[0, 1], row=1, col=1, zeroline=False)
    fig.update_xaxes(title_text="KAT − PRAI", row=1, col=2, zeroline=False)
    fig.update_yaxes(title_text="N циклов", row=1, col=2, zeroline=False)
    fig.update_layout(**_LAYOUT, height=490,
                      title=dict(text="KAT (симулированный) vs TabPFN/PRAI (реальный полный цикл)", font=dict(size=15)),
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.80)",
                                  bordercolor="#ddd", borderwidth=1, font=dict(size=11)))

    metrics = (f"<div class='card' style=\"border-left:3px solid {COLORS['blue']}\">"
              f"<b>Pearson r</b> = {r:.3f} &nbsp;|&nbsp; <b>Spearman ρ</b> = {rho:.3f} &nbsp;|&nbsp; "
              f"<b>MAE</b> = {mae:.3f} &nbsp;|&nbsp; <b>Bias</b> = {bias:+.3f} &nbsp;|&nbsp; "
              f"<b>SD разниц</b> = {sd_diff:.3f} &nbsp;|&nbsp; <b>N</b> = {len(sub)}</div>")
    wilcoxon_html = ""
    if "wilcoxon_p" in stats:
        w_stat, w_p = stats["wilcoxon_stat"], stats["wilcoxon_p"]
        verdict = ("Статистически значимое систематическое расхождение ⚠️" if w_p < 0.05
                  else "Значимых расхождений нет ✅")
        wilcoxon_html = (f"<div class='card' style=\"border-left:3px solid {COLORS['orange']}\">"
                         f"<b>Wilcoxon signed-rank:</b> stat={w_stat:.1f}, p={w_p:.4f} → {verdict}</div>")
    return f"<div class='chart-card'>{_fig_html(fig)}</div>" + metrics + wilcoxon_html


def _miri_warning_rows(summary: pd.DataFrame | None) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    if "warning" in summary:
        return summary[summary["warning"].fillna(False).astype(bool)]
    if "max_severity" in summary:
        return summary[summary["max_severity"].astype(str).str.lower().isin(["warning", "critical"])]
    return pd.DataFrame()


def _lot_change_rows(summary: pd.DataFrame | None) -> pd.DataFrame:
    if summary is None or summary.empty or "lot_change_flag" not in summary:
        return pd.DataFrame()
    return summary[summary["lot_change_flag"].fillna(False).astype(bool)]


# ---------------------------------------------------------------------------
# Tab shell (pure CSS, no JS): a hidden radio per tab + a sibling-selector rule
# per tab pairs the checked input to its panel and highlights its nav label.
# ---------------------------------------------------------------------------

def _tab_pair_css() -> str:
    rules = []
    for tab_id, _ in _TABS:
        rules.append(
            f"#tab-{tab_id}:checked ~ .tab-panels #panel-{tab_id}{{display:block}}"
            f"#tab-{tab_id}:checked ~ .tab-nav label[for=tab-{tab_id}]"
            "{background:var(--card);color:var(--clay-dark);border:1px solid var(--line);"
            "border-top:2px solid var(--clay);border-bottom:2px solid var(--card);margin-bottom:-1px}"
        )
    return "\n".join(rules)


def _css() -> str:
    return f"""
    <style>
    :root{{
      --bg:#F9F9F7; --card:#FFFFFF; --line:#E5E4DE; --text:#262624; --muted:#83827B; --clay:#D97757; --clay-dark:#BF5E3D;
      --red:#B3261E; --red-bg:#FAD6D6; --amber:#8A5A00; --amber-bg:#F9DCA4; --green:#1E7A34; --green-bg:#CAEAC7;
      --info:#B34A24; --info-bg:#F7D8CB;
      --warn-line:#D97757; --warn-bg:#F7D8CB; --warn-text:#7A3A1D;
    }}
    @media (prefers-color-scheme: dark){{
      :root{{
        --bg:#141413; --card:#1F1E1C; --line:#33322E; --text:#F5F4EF; --muted:#9C9A93; --clay:#E08A63; --clay-dark:#D97757;
        --red:#F2A8A0; --red-bg:#3C0E0E; --amber:#F0C274; --amber-bg:#311A00; --green:#8FD9A0; --green-bg:#11260F;
        --info:#F0B08A; --info-bg:#371407;
        --warn-line:#E08A63; --warn-bg:#371407; --warn-text:#F0B08A;
      }}
    }}
    *{{box-sizing:border-box}}
    body{{font-family:{FONT};color:var(--text);background:var(--bg);margin:0;line-height:1.45;font-size:14px}}
    .wrap{{max-width:1400px;margin:0 auto;padding:0 20px 40px}}
    header.hero{{background:linear-gradient(135deg,var(--info-bg) 0%,var(--amber-bg) 100%);
                color:var(--text);padding:26px 20px 22px;border-bottom:3px solid var(--clay)}}
    header.hero .inner{{max-width:1400px;margin:0 auto}}
    header.hero .hero-eyebrow{{font-size:11px;font-weight:700;letter-spacing:.08em;color:var(--clay-dark);
                              text-transform:uppercase;margin-bottom:6px}}
    header.hero h1{{margin:0 0 6px;font-size:24px;font-weight:800;color:var(--text)}}
    header.hero .meta{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}
    .pill{{background:rgba(0,0,0,.06);color:var(--text);border-radius:999px;padding:4px 12px;font-size:12px}}
    .warn{{margin:16px 0;border-left:4px solid var(--warn-line);background:var(--warn-bg);color:var(--warn-text);
          padding:11px 14px;border-radius:6px;font-size:13px}}
    .kpi-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:18px 0}}
    .kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
    .kpi .kl{{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;min-height:28px}}
    .kpi .kv{{font-size:24px;font-weight:800;margin-top:4px}}
    .kpi .ks{{font-size:11px;color:var(--muted);margin-top:2px}}
    h2{{font-size:17px;margin:6px 0 12px;color:var(--text)}}
    h3{{font-size:14px;margin:18px 0 8px;color:var(--text)}}
    .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}}
    .chart-card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}}
    .muted{{color:var(--muted)}}
    .table-wrap{{max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:8px}}
    table.tbl{{border-collapse:collapse;width:100%;font-size:12.5px}}
    table.tbl th{{position:sticky;top:0;background:var(--bg);text-align:left;padding:7px 8px;
                 border-bottom:2px solid var(--line);white-space:nowrap}}
    table.tbl td{{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
    table.tbl tr:hover td{{background:color-mix(in srgb, var(--clay) 6%, var(--card))}}
    .badge{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;white-space:nowrap}}
    .badge-info{{background:var(--info-bg);color:var(--info)}}
    .badge-ok{{background:var(--green-bg);color:var(--green)}}
    .badge-warn{{background:var(--amber-bg);color:var(--amber)}}
    .badge-crit{{background:var(--red-bg);color:var(--red)}}
    .row-p1 td{{background:var(--red-bg)}}
    .row-p2 td{{background:var(--amber-bg)}}
    .row-ok td{{background:var(--green-bg)}}
    .pbar{{display:inline-block;width:56px;height:7px;background:var(--line);border-radius:4px;overflow:hidden;
          vertical-align:middle;margin-right:6px}}
    .pbar > span{{display:block;height:100%;border-radius:4px}}
    .flags div{{font-size:11px;color:var(--muted);margin:1px 0}}
    .flags div:before{{content:"• "}}
    .legend{{font-size:12px;color:var(--muted);margin:8px 0 14px}}
    .legend span{{display:inline-block;margin-right:14px}}
    .swatch{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}}
    .narrative{{background:var(--card);border-left:4px solid var(--clay);padding:14px 18px;
               white-space:pre-wrap;border-radius:0 8px 8px 0}}
    .excl-note{{font-size:11.5px;color:var(--muted);margin-top:6px}}
    .candidates{{border:1px solid var(--red);background:var(--red-bg);border-radius:8px;padding:12px 16px;margin:14px 0}}
    .foot{{font-size:11px;color:var(--muted);margin-top:26px}}
    .tab-input{{display:none}}
    .tab-nav{{display:flex;flex-wrap:wrap;gap:6px;margin:18px 0 0;position:sticky;top:0;background:var(--bg);
              padding:8px 0 0;z-index:5}}
    .tab-label{{cursor:pointer;padding:8px 14px;border-radius:8px 8px 0 0;font-size:12.5px;font-weight:600;
               color:var(--muted);background:color-mix(in srgb, var(--line) 55%, var(--bg));
               border:1px solid var(--line);border-bottom:none}}
    .tab-panel{{display:none;border:1px solid var(--line);border-top:none;background:var(--card);
               border-radius:0 0 10px 10px;padding:18px}}
    /* "Прогноз беременности" tab only -- modeled on the owner's real TabPFN
       journal screenshot; scoped so no other tab's look changes. */
    .journal-eyebrow{{font-size:11px;font-weight:700;letter-spacing:.08em;color:var(--muted);
                     text-transform:uppercase;margin-bottom:6px}}
    .journal-h1{{font-size:24px;font-weight:800;color:var(--text);margin:0 0 8px}}
    .journal-meta{{font-size:12.5px;color:var(--muted);margin-bottom:16px}}
    .journal-meta code{{background:var(--line);padding:1px 5px;border-radius:4px;font-size:11.5px}}
    .journal-stats{{display:flex;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);
                   border-radius:10px;margin-bottom:14px;overflow:hidden}}
    .jstat{{flex:1 1 140px;padding:12px 16px;border-right:1px solid var(--line)}}
    .jstat:last-child{{border-right:none}}
    .jsl{{font-size:10.5px;font-weight:700;letter-spacing:.06em;color:var(--muted);text-transform:uppercase}}
    .jsv{{font-size:20px;font-weight:800;color:var(--text);margin-top:4px}}
    .journal-search{{display:block;width:100%;max-width:360px;padding:9px 12px;border:1px solid var(--line);
                    border-radius:8px;background:var(--card);color:var(--text);font-size:13px;margin-bottom:12px}}
    .journal-legend{{font-size:12px;color:var(--muted);margin-bottom:12px;display:flex;gap:16px;
                    flex-wrap:wrap;align-items:center}}
    .journal-legend .dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;
                         vertical-align:middle}}
    .dot-low{{background:var(--red)}} .dot-mid{{background:var(--amber)}} .dot-high{{background:var(--green)}}
    table.journal-table th{{text-transform:uppercase;font-size:10.5px;letter-spacing:.05em;font-weight:700;
                           color:var(--muted);background:var(--bg)}}
    table.journal-table td{{vertical-align:middle}}
    .journal-name{{font-weight:700;color:var(--text);font-size:13px}}
    .journal-id{{font-size:11px;margin-top:2px}}
    .repeat-badge{{display:inline-block;background:var(--amber-bg);color:var(--amber);font-size:10px;
                  font-weight:700;padding:1px 6px;border-radius:999px;margin-left:6px;vertical-align:middle}}
    .journal-age{{font-weight:700;font-size:13px;color:var(--text)}}
    .journal-sub{{font-size:11px;margin-top:1px}}
    .journal-funnel-chain{{font-size:12px;color:var(--text);white-space:nowrap}}
    .spark{{display:flex;align-items:flex-end;gap:2px;height:16px;margin-top:4px}}
    .spark-bar{{display:block;width:6px;background:var(--line);border-radius:1px;min-height:2px}}
    .spark-bar.spark-ok{{background:var(--green)}}
    .spark-bar.spark-bad{{background:var(--red)}}
    .prob-pill{{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:700;font-size:12.5px}}
    .prob-low{{background:var(--red-bg);color:var(--red)}}
    .prob-mid{{background:var(--amber-bg);color:var(--amber)}}
    .prob-high{{background:var(--green-bg);color:var(--green)}}
    .ci-track{{display:block;position:relative;width:90px;height:5px;background:var(--line);
              border-radius:3px;margin-top:5px}}
    .ci-range{{position:absolute;top:0;height:100%;background:var(--clay);opacity:.55;border-radius:3px}}
    .ci-tick{{position:absolute;top:-2px;width:2px;height:9px;background:var(--text);border-radius:1px}}
    {_tab_pair_css()}
    @media(max-width:1000px){{.kpi-grid{{grid-template-columns:repeat(3,1fr)}}}}
    @media(max-width:600px){{.kpi-grid{{grid-template-columns:repeat(2,1fr)}}}}
    </style>"""


def _tab_inputs() -> str:
    return "".join(
        f'<input type="radio" name="tab" id="tab-{tid}" class="tab-input"{" checked" if i == 0 else ""}>'
        for i, (tid, _) in enumerate(_TABS)
    )


def _tab_nav() -> str:
    labels = "".join(f'<label for="tab-{tid}" class="tab-label">{escape(label)}</label>'
                     for tid, label in _TABS)
    return f'<div class="tab-nav">{labels}</div>'


def _panel(tab_id: str, body: str) -> str:
    return f'<section id="panel-{tab_id}" class="tab-panel">{body}</section>'


def _kpi_tile(value: str, label: str, sub: str = "") -> str:
    sub_html = f"<div class='ks'>{escape(sub)}</div>" if sub else ""
    return (f"<div class='kpi'><div class='kl'>{escape(label)}</div>"
           f"<div class='kv'>{escape(value)}</div>{sub_html}</div>")


def _badge(text: str, kind: str = "info") -> str:
    return f"<span class='badge badge-{kind}'>{escape(text)}</span>"


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------

def _overview_tab(n_cycles: int, n_review: int, miri_bad_n: int, lot_changed_n: int,
                  ozema_flagged_n: int, ozema_candidates_n: int, banner: str,
                  kpi_fig: str, agg: pd.DataFrame) -> str:
    tiles = "".join([
        _kpi_tile(str(n_cycles), "циклов в анализе"),
        _kpi_tile(str(n_review), "в review-list"),
        _kpi_tile(str(miri_bad_n), "MIRI warning/critical"),
        _kpi_tile(str(lot_changed_n), "смен лота в окне"),
        _kpi_tile(str(ozema_flagged_n), "циклов с OZEMA-флагом"),
        _kpi_tile(str(ozema_candidates_n), "кандидатов на генконсультацию"),
    ])
    return (banner + f"<div class='kpi-grid'>{tiles}</div>"
           "<div class='card'><p class='muted'>KPI эмбриологов и врачей разделены на отдельные "
           "вкладки и никогда не сравниваются в одной таблице (эмбриолог — оплодотворение/"
           "дробление/бластоцисты; врач — только получение ОКК и зрелость). TabPFN-3 — "
           "независимый количественный QA-слой (факт против P10–P90), не заменяет реконсилиацию "
           "по исходу переноса. MIRI показывает только реальные эпизоды. OZEMA выявляет "
           "статистически значимые арест-паттерны и кандидатов на обсуждение генетической "
           "консультации.</p></div>"
           f"<h3>Когортный KPI (вся лаборатория, без разбивки по персоналу)</h3>{kpi_fig}{_table(agg)}")


# ---------------------------------------------------------------------------
# 2/3. Embryologist / doctor competency tables
# ---------------------------------------------------------------------------

def _competency_table(agg: pd.DataFrame, group_col: str, metrics: tuple[str, ...],
                      person_label: str) -> str:
    if agg is None or agg.empty or group_col not in agg.columns:
        return "<p class='muted'>нет данных для этой таблицы</p>"
    persons = sorted(agg[group_col].dropna().astype(str).unique())
    header = "".join(f"<th>{escape(m)}</th>" for m in metrics)
    rows_html = []
    for person in persons:
        sub = agg[agg[group_col].astype(str) == person]
        cells = []
        for m in metrics:
            row = sub[sub["metric"] == m] if "metric" in sub.columns else pd.DataFrame()
            if row.empty:
                cells.append("<td class='muted'>—</td>")
                continue
            r = row.iloc[0]
            rate, status, n = r.get("rate"), r.get("status"), r.get("n_cycles")
            n_txt = str(int(n)) if pd.notna(n) else "0"
            if pd.isna(rate):
                cells.append(f"<td class='muted'>н/д (n={n_txt})</td>")
                continue
            kind = _STATUS_KIND.get(status, "info")
            ci_lo, ci_hi = r.get("ci_low_95"), r.get("ci_high_95")
            ci_text = (f"95% ДИ {ci_lo * 100:.1f}–{ci_hi * 100:.1f}%"
                      if pd.notna(ci_lo) and pd.notna(ci_hi) else "")
            title = escape(f"{ci_text}; n={n_txt}; статус: {_STATUS_LABEL.get(status, status)}")
            cells.append(f"<td title='{title}'>{_badge(f'{rate * 100:.1f}%', kind)} "
                        f"<span class='muted' style='font-size:10.5px'>n={n_txt}</span></td>")
        rows_html.append(f"<tr><td><b>{escape(person)}</b></td>{''.join(cells)}</tr>")
    return (f"<div class='table-wrap'><table class='tbl'><thead><tr>"
           f"<th>{escape(person_label)}</th>{header}</tr></thead>"
           f"<tbody>{''.join(rows_html)}</tbody></table></div>")


def _below_competency_paragraph(agg: pd.DataFrame, group_col: str) -> str:
    if agg is None or agg.empty or "competency_met" not in agg.columns:
        return ""
    bad = agg[agg["competency_met"] == False]  # noqa: E712 - explicit False, not NaN
    if bad.empty:
        return "<div class='card'><h3>Поводы задуматься</h3><p class='muted'>Все показатели в пределах компетентности.</p></div>"
    items = []
    for _, r in bad.iterrows():
        person = r.get(group_col, "—")
        label = r.get("label", r.get("metric"))
        rate = r.get("rate")
        competency = r.get("competency")
        pct = f"{rate * 100:.1f}%" if pd.notna(rate) else "н/д"
        comp_pct = f"{competency * 100:.0f}%" if pd.notna(competency) else "—"
        items.append(f"<li>{escape(str(person))}: {escape(str(label))} = {pct}, ниже уровня "
                    f"компетентности {comp_pct} — повод для дополнительного разбора.</li>")
    return (f"<div class='card'><h3>Поводы задуматься</h3><ul>{''.join(items)}</ul>"
           "<p class='muted' style='font-size:11.5px'>Наблюдение, не вывод об отдельном "
           "специалисте: единичное отклонение — повод собрать больше данных, не решение.</p></div>")


# ---------------------------------------------------------------------------
# 4. Models & calibration
# ---------------------------------------------------------------------------

def _models_tab(df_u: pd.DataFrame, calibration, conversion) -> str:
    return (_model_distribution_chart(df_u) +
           "<h3>KAT (симулированный) vs TabPFN/PRAI (реальный полный цикл)</h3>" + _kat_vs_prai_chart(df_u) +
           f"<h3>Калибровка</h3>{_table(calibration)}" +
           f"<h3>Конверсии по стадиям</h3>{_table(conversion)}")


# ---------------------------------------------------------------------------
# Cluster analysis: donut + profile/outcomes + summary table + pregnancy
# box+strip by cluster (notebook §8, "КЛАСТЕРНЫЙ ДАШБОРД").
# ---------------------------------------------------------------------------

def _binom_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact binomial CI, in percent."""
    from scipy.stats import beta as _beta
    if n == 0:
        return 0.0, 0.0
    lo = _beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = _beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo) * 100, float(hi) * 100


def _cluster_donut_chart(df_u: pd.DataFrame) -> str:
    if not _HAVE_PLOTLY or "dominant_cluster" not in df_u.columns:
        return ""
    counts = pd.to_numeric(df_u["dominant_cluster"], errors="coerce").dropna().astype(int)
    counts = counts.value_counts().sort_index()
    if counts.empty:
        return ""
    fig = go.Figure(go.Pie(
        labels=[_CLUSTER_NAMES.get(c, str(c)) for c in counts.index], values=counts.values,
        marker=dict(colors=[_hex_rgba(_CLUSTER_HEX.get(c, COLORS["gray"]), 0.45) for c in counts.index],
                   line=dict(color=[_CLUSTER_HEX.get(c, COLORS["gray"]) for c in counts.index], width=2)),
        textinfo="label+percent", textfont=dict(size=11), hole=0.45,
        pull=[0.04] * len(counts), sort=False,
        hovertemplate="%{label}<br>%{value} циклов · %{percent}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{len(df_u)}</b><br>циклов", x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=13), align="center")
    fig.update_layout(**{k: v for k, v in _LAYOUT.items() if k != "margin"}, height=310,
                      title=dict(text="Состав когорты по кластерам", font=dict(size=13)),
                      margin=dict(t=50, b=20, l=30, r=30),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05,
                                  bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    return f"<div class='chart-card'>{_fig_html(fig)}</div>"


def _cluster_profile_outcomes_chart(df_u: pd.DataFrame) -> str:
    if not _HAVE_PLOTLY or "dominant_cluster" not in df_u.columns:
        return ""
    cluster_num = pd.to_numeric(df_u["dominant_cluster"], errors="coerce")
    df_preg = df_u[pd.to_numeric(df_u.get("Preg"), errors="coerce").notna()].copy()
    preg_cluster = pd.to_numeric(df_preg.get("dominant_cluster"), errors="coerce") if not df_preg.empty else pd.Series(dtype=float)

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Профиль кластеров (медианы, нормированные)", "Реальные исходы + 95% CI",
    ], column_widths=[0.58, 0.42], horizontal_spacing=0.12)

    profile_map = {col: lbl for col, lbl in [
        ("N folicules OPU", "Фолл.ОПУ"), ("MII", "MII"), ("Bl", "Бластоцисты"),
        ("Good Bl", "Хор.кач."), ("Age", "Возраст"),
    ] if col in df_u.columns}
    feat_labels = list(profile_map.values())
    norm_data, raw_data = {}, {}
    for col, lbl in profile_map.items():
        by_c = pd.to_numeric(df_u[col], errors="coerce").groupby(cluster_num).median()
        col_max = by_c.max()
        norm_data[lbl] = by_c / col_max if pd.notna(col_max) and col_max > 0 else by_c
        raw_data[lbl] = by_c

    any_profile = False
    for c in (0, 1, 2):
        chex, cname = _CLUSTER_HEX[c], _CLUSTER_NAMES[c]
        y_vals = [float(norm_data[lbl].get(c, 0) or 0) for lbl in feat_labels]
        raw_vals = [raw_data[lbl].get(c) for lbl in feat_labels]
        if not any(pd.notna(v) for v in raw_vals):
            continue
        any_profile = True
        fig.add_trace(go.Bar(
            name=cname, x=feat_labels, y=y_vals, legendgroup=f"c{c}", showlegend=True,
            marker=dict(color=_hex_rgba(chex, 0.40), line=dict(color=_hex_rgba(chex, 0.70), width=1.5),
                       pattern=dict(shape=_CLUSTER_PATTERNS[c], size=4, fgcolor=_hex_rgba(chex, 0.35))),
            text=[f"{v:.0f}" if pd.notna(v) else "" for v in raw_vals], textposition="outside",
            textfont=dict(size=9),
            hovertemplate=f"<b>{cname}</b><br>%{{x}}: %{{text}} (норм. %{{y:.2f}})<extra></extra>",
        ), row=1, col=1)
    fig.update_yaxes(title_text="Нормированная медиана", range=[0, 1.30], zeroline=False, row=1, col=1)

    if len(df_preg) >= 3 and preg_cluster.notna().any():
        preg_by_c = (pd.to_numeric(df_preg["Preg"], errors="coerce").groupby(preg_cluster)
                    .agg(["sum", "count"]).rename(columns={"sum": "k", "count": "n"}))
        x_labels = [_CLUSTER_NAMES.get(int(c), str(c)) for c in preg_by_c.index]
        y_vals = (preg_by_c["k"] / preg_by_c["n"] * 100).values
        ci_lo, ci_hi = [], []
        for i, (_, r) in enumerate(preg_by_c.iterrows()):
            lo, hi = _binom_ci(int(r["k"]), int(r["n"]))
            ci_lo.append(y_vals[i] - lo)
            ci_hi.append(hi - y_vals[i])
        fig.add_trace(go.Bar(
            x=x_labels, y=y_vals,
            marker=dict(color=[_hex_rgba(_CLUSTER_HEX.get(int(c), COLORS["gray"]), 0.40) for c in preg_by_c.index],
                       line=dict(color=[_CLUSTER_HEX.get(int(c), COLORS["gray"]) for c in preg_by_c.index], width=1.8),
                       pattern=dict(shape="/", size=5,
                                   fgcolor=[_hex_rgba(_CLUSTER_HEX.get(int(c), COLORS["gray"]), 0.30)
                                           for c in preg_by_c.index])),
            error_y=dict(type="data", symmetric=False, array=ci_hi, arrayminus=ci_lo,
                        color="rgba(180,180,180,0.70)", thickness=2.0, width=9),
            text=[f"{int(r['k'])}/{int(r['n'])}" for _, r in preg_by_c.iterrows()],
            textposition="outside", textfont=dict(size=11), showlegend=False,
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<br>%{text}<extra></extra>",
        ), row=1, col=2)
        overall = float(pd.to_numeric(df_preg["Preg"], errors="coerce").mean() * 100)
        fig.add_hline(y=overall, line_dash="dot", line_color="rgba(150,150,150,0.55)", line_width=1.5,
                      annotation_text=f"Когорта {overall:.0f}%", annotation_position="top right",
                      annotation_font_size=9, row=1, col=2)
    else:
        fig.add_annotation(text="Недостаточно данных об исходах", x=0.5, y=0.5,
                           xref="x2 domain", yref="y2 domain", showarrow=False,
                           font=dict(size=11, color="gray"))
    fig.update_yaxes(title_text="% беременности", range=[0, 115], zeroline=False, row=1, col=2)

    if not any_profile:
        return ""
    fig.update_layout(**{k: v for k, v in _LAYOUT.items() if k != "margin"}, height=430, barmode="group",
                      title=dict(text="Профиль и исходы по кластерам", font=dict(size=14)),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12, yanchor="top",
                                  bgcolor="rgba(0,0,0,0)", bordercolor="rgba(150,150,150,0.35)",
                                  borderwidth=1, font=dict(size=10), tracegroupgap=4),
                      margin=dict(t=60, b=90, l=60, r=30))
    return f"<div class='chart-card'>{_fig_html(fig)}</div>"


def _cluster_summary_rows(df_u: pd.DataFrame) -> list[dict]:
    """Clean numeric per-cluster summary (cohort-level, no identifiers) for the
    lead-analyst packet -- distinct from ``_cluster_summary_table``'s
    display-formatted strings, which are for the HTML table only.
    """
    if df_u is None or df_u.empty or "dominant_cluster" not in df_u.columns:
        return []
    cluster_num = pd.to_numeric(df_u["dominant_cluster"], errors="coerce")
    df_preg = df_u[pd.to_numeric(df_u.get("Preg"), errors="coerce").notna()].copy()
    preg_cluster = (pd.to_numeric(df_preg.get("dominant_cluster"), errors="coerce")
                    if not df_preg.empty else pd.Series(dtype=float))
    rows = []
    for c in sorted(cluster_num.dropna().unique().astype(int)):
        sub = df_u[cluster_num == c]
        sub_p = df_preg[preg_cluster == c]
        k = int(pd.to_numeric(sub_p.get("Preg"), errors="coerce").sum()) if len(sub_p) else 0
        n = len(sub_p)
        lo, hi = _binom_ci(k, n)

        def _median(col):
            return (float(pd.to_numeric(sub[col], errors="coerce").median())
                    if col in sub.columns and pd.to_numeric(sub[col], errors="coerce").notna().any() else None)

        rows.append({
            "cluster": _CLUSTER_NAMES.get(c, str(c)), "n_cycles": int(len(sub)),
            "median_age": _median("Age"), "median_occ": _median("OCC"),
            "median_kat": _median("p_kat_raw"), "median_prai": _median("PRAI"),
            "pregnancy_k": k, "pregnancy_n": n,
            "pregnancy_rate": round(k / n, 3) if n else None,
            "pregnancy_ci_low_pct": round(lo, 1) if n else None,
            "pregnancy_ci_high_pct": round(hi, 1) if n else None,
        })
    return rows


def _cluster_summary_table(df_u: pd.DataFrame) -> str:
    if "dominant_cluster" not in df_u.columns:
        return ""
    cluster_num = pd.to_numeric(df_u["dominant_cluster"], errors="coerce")
    df_preg = df_u[pd.to_numeric(df_u.get("Preg"), errors="coerce").notna()].copy()
    preg_cluster = pd.to_numeric(df_preg.get("dominant_cluster"), errors="coerce") if not df_preg.empty else pd.Series(dtype=float)
    rows = []
    for c in sorted(cluster_num.dropna().unique().astype(int)):
        sub = df_u[cluster_num == c]
        sub_p = df_preg[preg_cluster == c]
        k = int(pd.to_numeric(sub_p.get("Preg"), errors="coerce").sum()) if len(sub_p) else 0
        n = len(sub_p)
        lo, hi = _binom_ci(k, n)
        rows.append({
            "Кластер": _CLUSTER_NAMES.get(c, str(c)), "N": len(sub),
            "Мед. возраст": f"{pd.to_numeric(sub.get('Age'), errors='coerce').median():.0f}" if "Age" in sub.columns else "—",
            "Мед. OCC": f"{pd.to_numeric(sub.get('OCC'), errors='coerce').median():.0f}" if "OCC" in sub.columns else "—",
            "Мед. KAT": f"{pd.to_numeric(sub.get('p_kat_raw'), errors='coerce').median():.2f}" if "p_kat_raw" in sub.columns else "—",
            "Мед. PRAI": f"{pd.to_numeric(sub.get('PRAI'), errors='coerce').median():.2f}" if "PRAI" in sub.columns else "—",
            "Беременность": f"{k}/{n} ({k / n * 100:.0f}%)" if n else "—",
            "95% CI": f"{lo:.0f}–{hi:.0f}%" if n else "—",
        })
    return _table(pd.DataFrame(rows))


def _cluster_pregnancy_boxstrip_chart(df_u: pd.DataFrame) -> str:
    """Box + jittered strip: color=cluster, size=age, marker outline=outcome."""
    if not _HAVE_PLOTLY or "dominant_cluster" not in df_u.columns or "p_per_transfer" not in df_u.columns:
        return ""
    name_col = "patient_name" if "patient_name" in df_u.columns else (
        "cycle_uid" if "cycle_uid" in df_u.columns else None)
    fig = go.Figure()
    rng = np.random.default_rng(42)
    cluster_num = pd.to_numeric(df_u["dominant_cluster"], errors="coerce")
    any_data = False
    for ci, c in enumerate((0, 1, 2)):
        sub_c = df_u[cluster_num == c].reset_index(drop=True)
        vals = pd.to_numeric(sub_c.get("p_per_transfer"), errors="coerce").dropna()
        if vals.empty:
            continue
        any_data = True
        chex, cname = _CLUSTER_HEX[c], _CLUSTER_NAMES[c]
        fig.add_trace(go.Box(
            x=[ci] * len(vals), y=vals.values, name=cname, boxmean=True,
            fillcolor=_hex_rgba(chex, 0.15), line=dict(color=_hex_rgba(chex, 0.55), width=2),
            marker=dict(size=3, opacity=0), width=0.55, legendgroup=f"c{c}", showlegend=True,
            hoverinfo="skip",
        ))
        preg = (pd.to_numeric(sub_c["Preg"], errors="coerce") if "Preg" in sub_c.columns
               else pd.Series(np.nan, index=sub_c.index))
        border_c = ["rgba(0,200,0,1.0)" if v == 1 else "rgba(220,0,0,1.0)" if v == 0
                   else "rgba(255,255,255,0.75)" for v in preg]
        border_w = [3.0 if v in (0, 1) else 1.2 for v in preg]
        age = (pd.to_numeric(sub_c["Age"], errors="coerce") if "Age" in sub_c.columns
              else pd.Series(np.nan, index=sub_c.index))
        if age.notna().any():
            age_min, age_max = float(age.min()), float(age.max())
            sizes = (8 + (age - age_min) / (age_max - age_min + 1e-9) * 12).fillna(10).to_numpy()
        else:
            sizes = np.full(len(sub_c), 10.0)
        idx_p = vals.index.to_numpy()
        jitter = rng.uniform(-0.17, 0.17, len(idx_p))
        names = sub_c.loc[idx_p, name_col].astype(str) if name_col else pd.Series(idx_p.astype(str))
        fig.add_trace(go.Scatter(
            x=ci + jitter, y=vals.values, mode="markers", name=cname, showlegend=False,
            legendgroup=f"c{c}", text=names,
            hovertemplate="<b>%{text}</b><br>MC: %{y:.1%}<extra></extra>",
            marker=dict(size=sizes[idx_p], color=_hex_rgba(chex, 0.55),
                       line=dict(color=[border_c[i] for i in idx_p], width=[border_w[i] for i in idx_p])),
        ))
    if not any_data:
        return ""
    cohort_mean = float(pd.to_numeric(df_u["p_per_transfer"], errors="coerce").mean())
    fig.add_hline(y=cohort_mean, line_dash="dot", line_color="rgba(150,150,150,0.50)", line_width=1.5,
                  annotation_text=f"Когорта {cohort_mean * 100:.0f}%", annotation_position="top right",
                  annotation_font_size=9)
    fig.update_xaxes(tickvals=[0, 1, 2], ticktext=[_CLUSTER_NAMES[c] for c in (0, 1, 2)], zeroline=False)
    fig.update_yaxes(title_text="P(беременность) — MC прогноз", range=[0, 1.08], zeroline=False,
                     gridcolor="rgba(200,200,200,0.20)")
    fig.update_layout(**{k: v for k, v in _LAYOUT.items() if k != "margin"}, height=500,
                      title=dict(text="P(беременность) по кластерам", font=dict(size=14)),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.10, yanchor="top",
                                  bgcolor="rgba(0,0,0,0)", bordercolor="rgba(150,150,150,0.35)",
                                  borderwidth=1, font=dict(size=11)),
                      margin=dict(t=60, b=80, l=70, r=30))
    note = ("<p class='muted' style='font-size:11.5px'><b>Размер</b> = возраст (больше — старше) · "
           "<b>Обводка:</b> зелёная = беременность, красная = нет, белая = нет данных · "
           "точки = отдельные циклы.</p>")
    return f"<div class='chart-card'>{_fig_html(fig)}</div>" + note


def _cluster_analysis_tab(df_u: pd.DataFrame) -> str:
    if df_u is None or df_u.empty or "dominant_cluster" not in df_u.columns:
        return "<p class='muted'>нет колонки dominant_cluster — кластерный анализ недоступен</p>"
    return (_cluster_donut_chart(df_u) + _cluster_profile_outcomes_chart(df_u) +
           "<h3>Сводная таблица по кластерам</h3>" + _cluster_summary_table(df_u) +
           "<h3>P(беременность) по кластерам</h3>" + _cluster_pregnancy_boxstrip_chart(df_u))


# ---------------------------------------------------------------------------
# Similar Cases: weighted-cosine similarity graph over clinical/lab/prediction
# features (notebook §12), per-patient nearest-neighbor breakdown, and the
# intra- vs inter-cluster Mann-Whitney significance test.
# ---------------------------------------------------------------------------

def _build_similarity_graph(df_u: pd.DataFrame):
    """Returns (G, name_col) or (None, None) if deps/data are insufficient."""
    try:
        import networkx as nx
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None, None
    if df_u is None or df_u.empty:
        return None, None
    feat_cols = [c for c in _GRAPH_FEATURES if c in df_u.columns]
    name_col = "patient_name" if "patient_name" in df_u.columns else (
        "cycle_uid" if "cycle_uid" in df_u.columns else None)
    if len(feat_cols) < 3 or len(df_u) < 4 or name_col is None:
        return None, None

    df_reset = df_u.reset_index(drop=True)
    weights = np.array([_GRAPH_FEATURES[c] for c in feat_cols])
    x_raw = df_reset[feat_cols].apply(pd.to_numeric, errors="coerce")
    x_raw = x_raw.fillna(x_raw.median())
    x_scaled = StandardScaler().fit_transform(x_raw) * weights
    sim_matrix = cosine_similarity(x_scaled)
    np.fill_diagonal(sim_matrix, 0)

    # De-duplicate display names (repeat-attempt patients can share one name).
    seen: dict[str, int] = {}
    node_names = []
    for raw_name in df_reset[name_col].astype(str):
        n = seen.get(raw_name, 0)
        seen[raw_name] = n + 1
        node_names.append(raw_name if n == 0 else f"{raw_name} #{n + 1}")

    graph = nx.Graph()
    for i, (_, row) in enumerate(df_reset.iterrows()):
        cluster_raw = pd.to_numeric(pd.Series([row.get("dominant_cluster")]), errors="coerce").iloc[0]
        outcome_raw = pd.to_numeric(pd.Series([row.get("Preg")]), errors="coerce").iloc[0]
        graph.add_node(
            node_names[i],
            cluster=int(cluster_raw) if pd.notna(cluster_raw) else -1,
            age=float(pd.to_numeric(pd.Series([row.get("Age")]), errors="coerce").iloc[0] or 0),
            amh=float(pd.to_numeric(pd.Series([row.get("amh")]), errors="coerce").iloc[0] or 0),
            p_mc=float(pd.to_numeric(pd.Series([row.get("p_per_transfer")]), errors="coerce").iloc[0] or 0),
            p_kat=float(pd.to_numeric(pd.Series([row.get("p_kat_raw")]), errors="coerce").iloc[0] or 0),
            outcome=(float(outcome_raw) if pd.notna(outcome_raw) else None),
        )
    for i in range(len(df_reset)):
        top_k = np.argsort(sim_matrix[i])[::-1][:_GRAPH_K_NEIGHBORS]
        for j in top_k:
            sim_val = sim_matrix[i][j]
            if sim_val >= _GRAPH_SIM_THRESHOLD and not graph.has_edge(node_names[i], node_names[j]):
                graph.add_edge(node_names[i], node_names[j], weight=round(float(sim_val), 4))
    for node in graph.nodes():
        graph.nodes[node]["degree"] = graph.degree(node)
    return graph, name_col


def _find_similar_cases(graph, patient_name: str, n_top: int = 5) -> list[dict]:
    if patient_name not in graph:
        return []
    nbrs = sorted(((nbr, graph[patient_name][nbr]["weight"]) for nbr in graph.neighbors(patient_name)),
                 key=lambda x: x[1], reverse=True)[:n_top]
    rows = []
    for nbr_name, sim in nbrs:
        nd = graph.nodes[nbr_name]
        out = nd.get("outcome")
        rows.append({
            "Пациент": nbr_name[:25], "Сходство": f"{sim * 100:.1f}%",
            "Возраст": int(nd.get("age", 0)), "AMH": round(nd.get("amh", 0), 1),
            "MC%": f"{nd.get('p_mc', 0) * 100:.1f}%", "KAT%": f"{nd.get('p_kat', 0) * 100:.1f}%",
            "Кластер": _CLUSTER_NAMES.get(nd.get("cluster", -1), "—"),
            "Исход": "✅" if out == 1.0 else ("❌" if out == 0.0 else "—"),
        })
    return rows


def _similar_cases_section(df_u: pd.DataFrame, graph, name_col: str) -> str:
    if graph is None or graph.number_of_nodes() == 0:
        return "<p class='muted'>недостаточно данных для графа сходства (нужно ≥4 циклов и ≥3 общих признаков)</p>"
    df_local = df_u.reset_index(drop=True).copy()
    # The 7 primary models only (p_nvsa is KAT's own internal correction, not a
    # separate estimator -- including it here would double-count KAT).
    pred_cols = [c for c in ("p_per_transfer", "DNN", "p_kat_raw", "p_gnn_raw", "p_csdi",
                             "PRAI", "DIGITAL TWIN") if c in df_local.columns]
    df_local["_pred_std"] = (df_local[pred_cols].apply(pd.to_numeric, errors="coerce").std(axis=1)
                            if pred_cols else np.nan)
    seen: dict[str, int] = {}
    node_names = []
    for raw_name in df_local[name_col].astype(str):
        n = seen.get(raw_name, 0)
        seen[raw_name] = n + 1
        node_names.append(raw_name if n == 0 else f"{raw_name} #{n + 1}")
    df_local["_node_name"] = node_names

    priority = (df_local.nlargest(5, "_pred_std")["_node_name"].tolist()
               if df_local["_pred_std"].notna().any() else [])
    fp_mask = ((pd.to_numeric(df_local.get("Preg"), errors="coerce") == 0)
              & (pd.to_numeric(df_local.get("p_per_transfer"), errors="coerce") > 0.60))
    for p in df_local.loc[fp_mask, "_node_name"].tolist():
        if p not in priority:
            priority.insert(0, p)
    priority = [p for p in priority if p in graph][:6]
    if not priority:
        return "<p class='muted'>нет приоритетных циклов для разбора похожих случаев</p>"

    blocks = ["<p class='muted'>Приоритет: ложноположительные прогнозы + высокая неопределённость "
             "моделей (разброс между MC/KAT/NVSA/CSDI/DT/PRAI).</p>"]
    for pname in priority:
        row = df_local.loc[df_local["_node_name"] == pname]
        if row.empty:
            continue
        ref = row.iloc[0]
        preg = pd.to_numeric(pd.Series([ref.get("Preg")]), errors="coerce").iloc[0]
        mc = pd.to_numeric(pd.Series([ref.get("p_per_transfer")]), errors="coerce").iloc[0]
        std = pd.to_numeric(pd.Series([ref.get("_pred_std")]), errors="coerce").iloc[0]
        preg_badge = "✅" if preg == 1 else ("❌" if preg == 0 else "— нет данных")
        header_color = (COLORS["red"] if (preg == 0 and pd.notna(mc) and mc > 0.60)
                        else COLORS["blue"] if pd.isna(preg) else COLORS["green"])
        mc_txt = f"{mc * 100:.1f}%" if pd.notna(mc) else "—"
        std_txt = f"{std * 100:.1f}пп" if pd.notna(std) else "—"
        sim_rows = _find_similar_cases(graph, pname)
        if not sim_rows:
            continue
        sim_df = pd.DataFrame(sim_rows)
        n_preg = int((sim_df["Исход"] == "✅").sum())
        n_no = int((sim_df["Исход"] == "❌").sum())
        with_outcome = n_preg + n_no
        outcome_note = (f"<p class='muted' style='font-size:11.5px'>Среди {with_outcome} похожих с "
                        f"известным исходом: <b style='color:{COLORS['green']}'>{n_preg} беременность</b> · "
                        f"<b style='color:{COLORS['red']}'>{n_no} нет</b></p>" if with_outcome else "")
        blocks.append(
            f"<div class='card' style=\"border-left:3px solid {header_color}\">"
            f"<b>{escape(pname[:30])}</b> &nbsp; MC={mc_txt} · SD={std_txt} · Исход: {preg_badge}</div>"
            + outcome_note + _table(sim_df)
        )
    return "".join(blocks)


def _intra_inter_cluster_stats(graph) -> dict | None:
    if graph is None:
        return None
    intra, inter, edge_records = [], [], []
    for u, v, data in graph.edges(data=True):
        c_u, c_v = graph.nodes[u].get("cluster", -1), graph.nodes[v].get("cluster", -1)
        sim = data.get("weight", 0)
        edge_records.append((c_u, c_v, sim))
        (intra if c_u == c_v else inter).append(sim)
    if len(intra) < 2 or len(inter) < 2:
        return None
    intra_arr, inter_arr = np.array(intra), np.array(inter)
    n1, n2 = len(intra_arr), len(inter_arr)
    u_stat, p_value = mannwhitneyu(intra_arr, inter_arr, alternative="greater")
    r_rb = 1 - (2 * u_stat) / (n1 * n2)
    pooled_sd = (np.sqrt(((n1 - 1) * intra_arr.std() ** 2 + (n2 - 1) * inter_arr.std() ** 2) / (n1 + n2 - 2))
                if n1 + n2 > 2 else 0.0)
    cohens_d = (intra_arr.mean() - inter_arr.mean()) / pooled_sd if pooled_sd > 0 else 0.0

    rng = np.random.default_rng(42)

    def _boot_ci(arr, n_boot=10_000):
        meds = np.array([np.median(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
        return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

    return dict(intra=intra_arr, inter=inter_arr, edge_records=edge_records, n1=n1, n2=n2,
               U=float(u_stat), p=float(p_value), r_rb=float(r_rb), cohens_d=float(cohens_d),
               intra_ci=_boot_ci(intra_arr), inter_ci=_boot_ci(inter_arr))


def _intra_inter_stats_chart(stats: dict | None) -> str:
    if not _HAVE_PLOTLY or stats is None:
        return ""
    intra_arr, inter_arr = stats["intra"], stats["inter"]
    n1, n2, u_stat, p_value = stats["n1"], stats["n2"], stats["U"], stats["p"]
    r_rb = stats["r_rb"]
    intra_ci_lo, intra_ci_hi = stats["intra_ci"]
    inter_ci_lo, inter_ci_hi = stats["inter_ci"]
    sig_label = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        "Распределения сходства", "Медианы с 95% Bootstrap CI", "Матрица сходства по кластерам",
    ], horizontal_spacing=0.11)

    for arr, label, color in [(intra_arr, f"Intra-cluster (n={n1})", COLORS["blue"]),
                              (inter_arr, f"Inter-cluster (n={n2})", COLORS["red"])]:
        fig.add_trace(go.Histogram(x=arr, nbinsx=14, name=label,
                                   marker=dict(color=_hex_rgba(color, 0.52),
                                              line=dict(color=_hex_rgba(color, 0.80), width=0.8)),
                                   opacity=0.80, hovertemplate=f"{label}: %{{x:.3f}}<extra></extra>"), row=1, col=1)
    for val, color, lbl, pos in [
        (float(np.median(intra_arr)), COLORS["blue"], f"Intra med={np.median(intra_arr):.3f}", "top left"),
        (float(np.median(inter_arr)), COLORS["red"], f"Inter med={np.median(inter_arr):.3f}", "bottom right"),
    ]:
        fig.add_vline(x=val, line_dash="dash", line_color=color, line_width=2,
                      annotation_text=lbl, annotation_position=pos, annotation_font_size=9, col=1)

    for arr, label, color, ci_lo, ci_hi in [
        (intra_arr, "Intra-cluster", COLORS["blue"], intra_ci_lo, intra_ci_hi),
        (inter_arr, "Inter-cluster", COLORS["red"], inter_ci_lo, inter_ci_hi),
    ]:
        med = float(np.median(arr))
        fig.add_trace(go.Violin(x=[label] * len(arr), y=arr, name=label, showlegend=False,
                                fillcolor=_hex_rgba(color, 0.18), line_color=_hex_rgba(color, 0.55),
                                box_visible=True, meanline_visible=True, width=0.5, hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(x=[label], y=[med], mode="markers", showlegend=False,
                                 error_y=dict(type="data", array=[ci_hi - med], arrayminus=[med - ci_lo],
                                             color=_hex_rgba(color, 0.80), thickness=2.5, width=10),
                                 marker=dict(size=12, color=_hex_rgba(color, 0.85), line=dict(color="white", width=1.5)),
                                 hovertemplate=(f"<b>{label}</b><br>Median: {med:.4f}<br>"
                                              f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]<extra></extra>")), row=1, col=2)
    # U/p/r already appear in the figure title below; a second annotation here
    # would sit directly on top of this panel's own subplot title.
    fig.update_yaxes(title_text="Cosine similarity", range=[0.65, 1.02], zeroline=False, row=1, col=2)

    heat_z = np.zeros((3, 3))
    heat_n = np.zeros((3, 3), dtype=int)
    for c_u, c_v, sim in stats["edge_records"]:
        if c_u in (0, 1, 2) and c_v in (0, 1, 2):
            heat_z[c_u][c_v] += sim
            heat_n[c_u][c_v] += 1
            if c_u != c_v:
                heat_z[c_v][c_u] += sim
                heat_n[c_v][c_u] += 1
    heat_mean = np.where(heat_n > 0, heat_z / np.maximum(heat_n, 1), 0)
    for ci in range(3):
        intra_c = [sim for c_u, c_v, sim in stats["edge_records"] if c_u == ci and c_v == ci]
        heat_mean[ci][ci] = np.mean(intra_c) if intra_c else 0
    cluster_labels = [_CLUSTER_NAMES[c] for c in (0, 1, 2)]
    text_annot = [[f"{heat_mean[i][j]:.3f}<br>n={int(heat_n[i][j])}" for j in range(3)] for i in range(3)]
    fig.add_trace(go.Heatmap(z=heat_mean, x=cluster_labels, y=cluster_labels, colorscale="Blues",
                             zmin=0.75, zmax=0.95, text=text_annot, texttemplate="%{text}",
                             textfont=dict(size=10),
                             colorbar=dict(title="Mean sim", len=0.55, x=1.02, thickness=12, tickfont=dict(size=9)),
                             hovertemplate="%{y} ↔ %{x}<br>Mean sim: %{z:.3f}<extra></extra>"), row=1, col=3)

    fig.update_layout(**{k: v for k, v in _LAYOUT.items() if k != "margin"}, height=500, barmode="overlay",
                      title=dict(text=(f"Mann-Whitney U test: Intra- vs Inter-cluster Cosine Similarity | "
                                       f"p={p_value:.4f} {sig_label} | r={r_rb:.3f} ({_interpret_r(r_rb)})"),
                                font=dict(size=13), y=0.98, yanchor="top"),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12, yanchor="top",
                                  bgcolor="rgba(0,0,0,0)", bordercolor="rgba(150,150,150,0.35)",
                                  borderwidth=1, font=dict(size=10)),
                      margin=dict(t=110, b=80, l=60, r=80))
    fig.update_xaxes(zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="N рёбер", zeroline=False, row=1, col=1)
    return f"<div class='chart-card'>{_fig_html(fig)}</div>"


def _intra_inter_stats_summary(stats: dict | None) -> str:
    if stats is None:
        return ("<p class='muted'>недостаточно рёбер графа для intra/inter статистического теста "
               "(нужно ≥2 в каждой группе)</p>")
    intra_arr, inter_arr = stats["intra"], stats["inter"]
    n1, n2, u_stat, p_value = stats["n1"], stats["n2"], stats["U"], stats["p"]
    r_rb, cohens_d = stats["r_rb"], stats["cohens_d"]
    intra_ci_lo, intra_ci_hi = stats["intra_ci"]
    inter_ci_lo, inter_ci_hi = stats["inter_ci"]
    verdict = "значимо выше" if p_value < 0.05 else "статистически не отличается от"
    text = (
        f"Сходство внутри кластера (медиана {np.median(intra_arr):.3f}, 95% bootstrap ДИ "
        f"[{intra_ci_lo:.3f}, {intra_ci_hi:.3f}], n={n1}) {verdict} сходства между кластерами "
        f"(медиана {np.median(inter_arr):.3f}, 95% bootstrap ДИ [{inter_ci_lo:.3f}, {inter_ci_hi:.3f}], n={n2}). "
        f"Mann-Whitney U={u_stat:.1f}, односторонний p={p_value:.4f}. "
        f"Rank-biserial r={r_rb:.3f} ({_interpret_r(r_rb)}), Cohen's d={cohens_d:.3f} ({_interpret_d(cohens_d)})."
    )
    return f"<div class='card'>{escape(text)}</div>"


def _similar_cases_tab(df_u: pd.DataFrame) -> str:
    graph, name_col = _build_similarity_graph(df_u)
    if graph is None:
        return ("<p class='muted'>Недостаточно данных, либо пакеты networkx/scikit-learn недоступны "
               "— граф сходства пациентов не построен (нужно ≥4 циклов и ≥3 общих признаков).</p>")
    intro = ("<p class='muted'>Граф сходства строится по взвешенному косинусному сходству клинических "
            "входных, лабораторных и модельных признаков (возраст, AMH, AFC, OCC, MII, Bl, Good Bl, "
            "MC pipeline, KAT, и Bayesian-posterior над MC); "
            f"k={_GRAPH_K_NEIGHBORS} ближайших соседей, "
            f"порог сходства {_GRAPH_SIM_THRESHOLD:.0%}.</p>")
    stats = _intra_inter_cluster_stats(graph)
    return (intro +
           "<h3>Similar Cases — клинически похожие циклы</h3>" + _similar_cases_section(df_u, graph, name_col) +
           "<h3>Статистический тест: Intra- vs Inter-cluster Similarity</h3>" +
           _intra_inter_stats_chart(stats) + _intra_inter_stats_summary(stats))


def _intra_inter_stats_json(stats: dict | None) -> dict | None:
    """JSON-safe (no numpy types) view of ``_intra_inter_cluster_stats``'s output."""
    if stats is None:
        return None
    return {
        "n_intra_edges": stats["n1"], "n_inter_edges": stats["n2"],
        "mann_whitney_U": round(stats["U"], 2), "p_value_one_tailed": round(stats["p"], 6),
        "rank_biserial_r": round(stats["r_rb"], 4), "cohens_d": round(stats["cohens_d"], 4),
        "intra_median": round(float(np.median(stats["intra"])), 4),
        "inter_median": round(float(np.median(stats["inter"])), 4),
        "intra_95ci": [round(v, 4) for v in stats["intra_ci"]],
        "inter_95ci": [round(v, 4) for v in stats["inter_ci"]],
    }


def build_analysis_summary(df_u: pd.DataFrame) -> dict:
    """Cohort-level summary (no patient identifiers) of the model-agreement,
    cluster, and similarity analyses added to this report -- consumed by
    ``scripts/run_analysis.py`` to feed the lead-analyst agent packet, so the
    final synthesis can reference and explain this data, not just the report.
    """
    graph, _ = _build_similarity_graph(df_u) if df_u is not None else (None, None)
    return {
        "cluster_summary": _cluster_summary_rows(df_u),
        "kat_vs_prai": _kat_vs_prai_stats(df_u),
        "intra_inter_cluster_similarity": _intra_inter_stats_json(_intra_inter_cluster_stats(graph)),
    }


# ---------------------------------------------------------------------------
# 5. TabPFN pregnancy-prediction journal (PRAI + Digital Twin), funnel sourced
# from THIS report's own TabPFN-3 regression P50 -- never Digital Twin's own
# med_* funnel numbers, to avoid mixing two different funnel-prediction models.
# ---------------------------------------------------------------------------

# Layout modeled directly on the owner's real reference screenshot (TabPFN ·
# журнал прогнозов): eyebrow label + big title + source meta, a bordered strip of
# stat tiles, a search box, a relative-probability-tier legend, then a table with
# a patient block (name + repeat-attempt badge + ID), a stacked age/AMH/BMI block,
# a funnel arrow-chain with a mini sparkline, a colored probability pill with a
# 95% CI range bar, and a plain Digital Twin comparison number. Data is exactly
# what this pipeline already computes (PRAI/CI, Digital Twin, real funnel counts,
# TabPFN-3 regression in-interval verdicts) -- only the presentation is new.
_JOURNAL_FUNNEL_STAGES = ("Фолликулы", "ООК", "MII", "2PN", "Бласт", "Хор.бласт")
_JOURNAL_REAL_COLUMN = {
    "Фолликулы": "N folicules OPU", "ООК": "OCC", "MII": "MII",
    "2PN": "2pN", "Бласт": "Bl", "Хор.бласт": "Good Bl",
}
_JOURNAL_REGRESSION_STAGE = {
    "MII": ("mii",), "2PN": ("pn2",),
    "Бласт": ("blasts_after_fert", "blasts_early"), "Хор.бласт": ("good",),
}
_RU_MONTH = {1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "май", 6: "июн",
            7: "июл", 8: "авг", 9: "сен", 10: "окт", 11: "ноя", 12: "дек"}
_JOURNAL_SEARCH_JS = """<script>
function journalFilter(input){
  var q = input.value.trim().toLowerCase();
  document.querySelectorAll('#journal-table tbody tr').forEach(function(r){
    r.style.display = r.getAttribute('data-search').indexOf(q) >= 0 ? '' : 'none';
  });
}
</script>"""


def _ru_date(ts) -> str:
    # format="mixed" lets pandas infer per-value: raw OPU dates are DD.MM.YYYY,
    # but values already parsed upstream (e.g. by Digital Twin) round-trip as
    # ISO YYYY-MM-DD -- dayfirst=True alone warns on that mix.
    t = pd.to_datetime(pd.Series([ts]), errors="coerce", dayfirst=True, format="mixed").iloc[0]
    return "—" if pd.isna(t) else f"{t.day} {_RU_MONTH.get(t.month, '')}"


def _regression_funnel_map(regression_qc_frame: pd.DataFrame, stage_map: dict) -> dict:
    """cycle_uid -> {display_stage: {"ok": in_interval80 bool|None}}."""
    if (regression_qc_frame is None or regression_qc_frame.empty
            or "cycle_uid" not in regression_qc_frame.columns):
        return {}
    out: dict = {}
    for _, r in regression_qc_frame.iterrows():
        stage_verdicts = {}
        for display, candidates in stage_map.items():
            for cand in candidates:
                p50_col, ok_col = f"{cand}_p50", f"{cand}_in_interval80"
                if p50_col in regression_qc_frame.columns and pd.notna(r.get(p50_col)):
                    stage_verdicts[display] = r.get(ok_col)
                    break
        out[r.get("cycle_uid")] = stage_verdicts
    return out


def _funnel_chain_text(values: list) -> str:
    parts = []
    for v in values:
        vn = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        parts.append("—" if pd.isna(vn) else str(int(vn)))
    return " → ".join(parts)


def _sparkline(values: list, tones: list) -> str:
    numeric = [float(v) for v in values if v is not None and pd.notna(v)]
    peak = max(numeric) if numeric else 0
    bars = []
    for v, tone in zip(values, tones):
        cls = "spark-bar" + (f" spark-{tone}" if tone else "")
        if v is None or pd.isna(v) or peak <= 0:
            bars.append(f"<span class='{cls}' style='height:2px'></span>")
        else:
            pct = max(6, round(float(v) / peak * 100))
            bars.append(f"<span class='{cls}' style='height:{pct}%'></span>")
    return f"<span class='spark'>{''.join(bars)}</span>"


def _prob_tier(v: float) -> str:
    return "low" if v < 0.40 else ("mid" if v < 0.60 else "high")


def _journal_prob_pill(prai, ci_low, ci_high) -> str:
    v = pd.to_numeric(pd.Series([prai]), errors="coerce").iloc[0]
    if pd.isna(v):
        return "<span class='muted'>—</span>"
    tier = _prob_tier(float(v))
    pct = f"{float(v) * 100:.1f}%"
    lo = pd.to_numeric(pd.Series([ci_low]), errors="coerce").iloc[0]
    hi = pd.to_numeric(pd.Series([ci_high]), errors="coerce").iloc[0]
    bar = ""
    if pd.notna(lo) and pd.notna(hi):
        lo_pct = max(0.0, min(100.0, float(lo) * 100))
        hi_pct = max(0.0, min(100.0, float(hi) * 100))
        v_pct = max(0.0, min(100.0, float(v) * 100))
        bar = (f"<span class='ci-track'><span class='ci-range' "
              f"style='left:{lo_pct:.0f}%;width:{max(1.0, hi_pct - lo_pct):.0f}%'></span>"
              f"<span class='ci-tick' style='left:{v_pct:.0f}%'></span></span>")
    return f"<span class='prob-pill prob-{tier}'>{pct}</span>{bar}"


def _dt_compare_text(value) -> str:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "—" if pd.isna(v) else f"{float(v):.3f}"


def _journal_patient_block(name, pid, repeat_n: int) -> str:
    badge = f"<span class='repeat-badge'>×{repeat_n}</span>" if repeat_n > 1 else ""
    return (f"<div class='journal-name'>{escape(str(name))[:28]}{badge}</div>"
           f"<div class='journal-id muted'>{escape(str(pid))}</div>")


def _journal_demo_block(age, amh, bmi) -> str:
    age_num = pd.to_numeric(pd.Series([age]), errors="coerce").iloc[0]
    age_txt = "—" if pd.isna(age_num) else f"{int(age_num)} лет"
    amh_num = pd.to_numeric(pd.Series([amh]), errors="coerce").iloc[0]
    bmi_num = pd.to_numeric(pd.Series([bmi]), errors="coerce").iloc[0]
    amh_txt = "—" if pd.isna(amh_num) else f"{amh_num:.2f}"
    bmi_txt = "—" if pd.isna(bmi_num) else f"{bmi_num:.1f}"
    return (f"<div class='journal-age'>{age_txt}</div>"
           f"<div class='journal-sub muted'>АМН <b>{amh_txt}</b></div>"
           f"<div class='journal-sub muted'>BMI <b>{bmi_txt}</b></div>")


def _journal_stat(label: str, value: str) -> str:
    return (f"<div class='jstat'><div class='jsl'>{escape(label)}</div>"
           f"<div class='jsv'>{escape(value)}</div></div>")


def _pregnancy_journal_tab(analytics_result: dict, regression_qc_frame: pd.DataFrame) -> str:
    df_u = analytics_result.get("df_u", pd.DataFrame())
    if df_u is None or df_u.empty:
        return "<p class='muted'>нет данных</p>"
    funnel_map = _regression_funnel_map(regression_qc_frame, _JOURNAL_REGRESSION_STAGE)
    has_regression = bool(funnel_map)

    prai = pd.to_numeric(df_u.get("PRAI"), errors="coerce")
    id_col = "patient_id" if "patient_id" in df_u.columns else ("ID" if "ID" in df_u.columns else None)
    repeat_counts = df_u[id_col].value_counts() if id_col else pd.Series(dtype=int)
    n_repeat_patients = int((repeat_counts > 1).sum())
    n_incomplete = int(prai.isna().sum())
    dates = (pd.to_datetime(df_u["Date OPU"], errors="coerce", dayfirst=True, format="mixed")
            if "Date OPU" in df_u.columns else pd.Series([], dtype="datetime64[ns]")).dropna()
    period_txt = (f"{_ru_date(dates.min())} – {_ru_date(dates.max())} {int(dates.max().year)}"
                 if not dates.empty else "—")

    eyebrow = ("<div class='journal-eyebrow'>TABPFN · ПРОГНОЗ БЕРЕМЕННОСТИ</div>"
              "<h2 class='journal-h1'>Прогнозы вероятности исхода по эмбриологическим протоколам</h2>"
              "<div class='journal-meta'>Источник: <code>dt_predictions.csv</code> · PRAI (TabPFN) "
              "+ Digital Twin по каждому циклу; воронка — реальные значения" +
              (", подсвечена по совпадению с P10–P90 TabPFN-3 regression этого отчёта.</div>"
               if has_regression else
               "; TabPFN-3 regression в этом прогоне не запускалась — без сверки с интервалом.</div>"))
    stats = "".join([
        _journal_stat("Всего записей", str(len(df_u))),
        _journal_stat("Период", period_txt),
        _journal_stat("Средняя P", "—" if prai.dropna().empty else f"{prai.mean() * 100:.1f}%"),
        _journal_stat("Медиана P", "—" if prai.dropna().empty else f"{prai.median() * 100:.1f}%"),
        _journal_stat("Повторные ID", f"{n_repeat_patients} пациенток" if n_repeat_patients else "0"),
        _journal_stat("Неполные / тест.", f"{n_incomplete} записей"),
    ])
    search = ("<input type='text' class='journal-search' "
             "placeholder='Поиск по имени или ID пациентки...' oninput='journalFilter(this)'>")
    legend = ("<div class='journal-legend'>"
             "<span><i class='dot dot-low'></i>ниже 0.40</span>"
             "<span><i class='dot dot-mid'></i>0.40–0.60</span>"
             "<span><i class='dot dot-high'></i>выше 0.60</span>"
             "<span class='muted'>— шкала относительная, в пределах этой выборки</span></div>")

    sorted_df = df_u.copy()
    if "Date OPU" in sorted_df.columns:
        sorted_df["_journal_sort"] = pd.to_datetime(sorted_df["Date OPU"], errors="coerce",
                                                    dayfirst=True, format="mixed")
        sorted_df = sorted_df.sort_values("_journal_sort", ascending=False)
    header = ("<tr><th>№</th><th>Дата</th><th>Пациентка</th><th>Возраст · АМН · BMI</th>"
             "<th>Воронка: фолликулы → ООК → MII → 2PN → бласт → хор.бласт</th>"
             "<th>Прогноз P (95% ДИ)</th><th>DT_COMPARE</th></tr>")
    rows_html = []
    for i, (_, row) in enumerate(sorted_df.iterrows(), start=1):
        name = row.get("patient_name") or row.get("Patient Full Name") or "—"
        pid = row.get("patient_id") or row.get("ID") or "—"
        repeat_n = int(repeat_counts.get(row.get(id_col), 1)) if id_col else 1
        real_values = [row.get(_JOURNAL_REAL_COLUMN[s]) for s in _JOURNAL_FUNNEL_STAGES]
        stages = funnel_map.get(row.get("cycle_uid"), {})
        tones = ["bad" if stages.get(s) is False else ("ok" if stages.get(s) is True else None)
                for s in _JOURNAL_FUNNEL_STAGES]
        funnel_html = (f"<div class='journal-funnel-chain'>{escape(_funnel_chain_text(real_values))}</div>"
                      f"{_sparkline(real_values, tones)}")
        search_blob = escape(f"{name} {pid}".lower())
        rows_html.append(
            f"<tr data-search='{search_blob}'>"
            f"<td class='muted'>{i}</td><td>{escape(_ru_date(row.get('Date OPU')))}</td>"
            f"<td>{_journal_patient_block(name, pid, repeat_n)}</td>"
            f"<td>{_journal_demo_block(row.get('age', row.get('Age')), row.get('AMH'), row.get('BMI'))}</td>"
            f"<td>{funnel_html}</td>"
            f"<td>{_journal_prob_pill(row.get('PRAI'), row.get('ci_low'), row.get('ci_high'))}</td>"
            f"<td class='muted'>{_dt_compare_text(row.get('DIGITAL TWIN'))}</td></tr>")
    table = (f"<div class='table-wrap'><table class='tbl journal-table' id='journal-table'>"
            f"<thead>{header}</thead><tbody>{''.join(rows_html)}</tbody></table></div>")
    return eyebrow + f"<div class='journal-stats'>{stats}</div>" + search + legend + table + _JOURNAL_SEARCH_JS


# ---------------------------------------------------------------------------
# 6. TabPFN-3 quantitative regression (protocol/funnel forecast QA)
# ---------------------------------------------------------------------------

def _regression_long(regression_qc_frame: pd.DataFrame) -> pd.DataFrame:
    if regression_qc_frame is None or regression_qc_frame.empty:
        return pd.DataFrame()
    from .contracts import detect_regression_stages
    stages = detect_regression_stages(regression_qc_frame)
    rows = []
    for stage in stages:
        cols = {s: f"TPFN3_{stage}_{s}" for s in ("p10", "p25", "p50", "p75", "p90")}
        if not all(c in regression_qc_frame.columns for c in cols.values()):
            continue
        actual_col = f"{stage}_actual"
        interval_col = f"{stage}_in_interval80"
        for _, r in regression_qc_frame.iterrows():
            p50 = pd.to_numeric(pd.Series([r.get(cols["p50"])]), errors="coerce").iloc[0]
            actual = (pd.to_numeric(pd.Series([r.get(actual_col)]), errors="coerce").iloc[0]
                     if actual_col in regression_qc_frame.columns else np.nan)
            rows.append({
                "cycle_uid": r.get("cycle_uid"), "Стадия": stage,
                "P10": r.get(cols["p10"]), "P25": r.get(cols["p25"]), "P50": r.get(cols["p50"]),
                "P75": r.get(cols["p75"]), "P90": r.get(cols["p90"]), "Факт": actual,
                "Факт-P50": (actual - p50) if pd.notna(actual) and pd.notna(p50) else np.nan,
                "в интервале 80%": r.get(interval_col) if interval_col in regression_qc_frame.columns else None,
            })
    return pd.DataFrame(rows)


def _regression_chart(long_df: pd.DataFrame) -> str:
    if not _HAVE_PLOTLY or long_df is None or long_df.empty:
        return ""
    palette = ["#D97757", "#1E7A34", "#8B5CF6", "#B34A24", "#5598E7", "#C9A227"]
    fig = go.Figure()
    stages = list(dict.fromkeys(long_df["Стадия"]))
    for i, stage in enumerate(stages):
        sub = long_df[long_df["Стадия"] == stage].copy()
        sub["Факт"] = pd.to_numeric(sub["Факт"], errors="coerce")
        sub["P50"] = pd.to_numeric(sub["P50"], errors="coerce")
        sub = sub.dropna(subset=["Факт", "P50"])
        if sub.empty:
            continue
        p10 = pd.to_numeric(sub["P10"], errors="coerce")
        p90 = pd.to_numeric(sub["P90"], errors="coerce")
        fig.add_trace(go.Scatter(
            x=sub["Факт"], y=sub["P50"], mode="markers", name=stage,
            marker=dict(color=palette[i % len(palette)], size=7, opacity=.55),
            error_y=dict(type="data", symmetric=False, array=p90 - sub["P50"], arrayminus=sub["P50"] - p10)))
    if fig.data:
        all_vals = pd.concat([pd.to_numeric(long_df["Факт"], errors="coerce"),
                              pd.to_numeric(long_df["P50"], errors="coerce")]).dropna()
        if not all_vals.empty:
            lo, hi = float(all_vals.min()), float(all_vals.max())
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                     line=dict(color="#9aa5b1", dash="dash"), name="идеал (факт=P50)"))
        fig.update_layout(title="TabPFN-3: факт против P10–P50–P90 по стадиям",
                          xaxis_title="факт", yaxis_title="прогноз (P50, интервал P10–P90)", **_LAYOUT)
        return f"<div class='chart-card'>{_fig_html(fig)}</div>"
    return ""


def _regression_tab(regression_qc_frame: pd.DataFrame, regression_calibration: pd.DataFrame) -> str:
    if regression_qc_frame is None or regression_qc_frame.empty:
        return ("<p class='muted'>TabPFN-3 regression не запускалась или не дала результата в "
               "этом прогоне (мягкий шаг — см. run_manifest.json: regression_status). Это "
               "независимый QA-слой; его отсутствие не блокирует остальной отчёт.</p>")
    long_df = _regression_long(regression_qc_frame)
    chart = _regression_chart(long_df)
    calib = regression_calibration.copy() if isinstance(regression_calibration, pd.DataFrame) else pd.DataFrame()
    if not calib.empty:
        calib = calib.rename(columns={"stage": "Стадия", "n_evaluated": "N",
                                      "share_in_interval80": "Покрытие 80%", "mae_p50": "MAE (P50)"})
        if "Покрытие 80%" in calib:
            calib["Покрытие 80%"] = (
                pd.to_numeric(calib["Покрытие 80%"], errors="coerce") * 100
            ).round(1).astype(str) + "%"
    disclaimer = ("<div class='warn'>Независимый количественный QA-слой: факт сверяется против "
                 "P10–P90 по стадиям воронки (MII/2PN/Bl/Good Bl/Frozen). Не заменяет и не "
                 "переопределяет DT/TabPFN реконсилиацию по исходу переноса; сбой этого шага "
                 "мягкий и не блокирует остальной пайплайн.</div>")
    display_long = long_df.copy()
    for col in ("P10", "P25", "P50", "P75", "P90", "Факт", "Факт-P50"):
        if col in display_long.columns:
            display_long[col] = pd.to_numeric(display_long[col], errors="coerce").round(2)
    return (disclaimer + chart +
           f"<h3>Модельная точность по стадиям</h3>{_table(calib)}"
           f"<h3>По циклам</h3>{_table(display_long, max_rows=500)}")


# ---------------------------------------------------------------------------
# 7. Review-list
# ---------------------------------------------------------------------------

def _prob_cell(value, color: str) -> str:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(v):
        return "<span class='muted'>—</span>"
    pct = max(0.0, min(1.0, float(v))) * 100
    return f"<span class='pbar'><span style='width:{pct:.0f}%;background:{color}'></span></span>{pct:.0f}%"


def _cycle_flags(reason, out_of_interval, miri_row: dict | None, lot_row: dict | None) -> str:
    items = []
    if isinstance(reason, str) and reason and reason != "в пределах ожиданий":
        items.append(escape(reason))
    if isinstance(out_of_interval, list) and out_of_interval:
        items.append("TabPFN-3 вне P10–P90: " + escape(", ".join(out_of_interval)))
    if miri_row:
        sev = miri_row.get("max_severity")
        if sev in ("warning", "critical"):
            items.append(f"MIRI {escape(str(sev))} в окне культивирования (см. вкладку MIRI: "
                        "&lt;8ч = перебой питания/логирования, ≥8ч = остановка)")
    if lot_row and lot_row.get("lot_change_flag"):
        items.append("смена лота расходника в окне культивирования (кандидат-конфаундер)")
    if not items:
        return "<span class='muted'>без флагов</span>"
    return "".join(f"<div>{i}</div>" for i in items)


def _review_section(analytics_result: dict, regression_qc_frame: pd.DataFrame,
                    miri_summary: pd.DataFrame | None, lot_summary: pd.DataFrame | None) -> str:
    review = analytics_result.get("review_list", pd.DataFrame())
    df_u = analytics_result.get("df_u", pd.DataFrame())
    if review is None or review.empty:
        return "<p class='muted'>Review-list пуст: явных отклонений/широких CI TabPFN не обнаружено.</p>"
    uids = set(review["cycle_uid"].dropna()) if "cycle_uid" in review.columns else set()
    if isinstance(df_u, pd.DataFrame) and "cycle_uid" in df_u.columns and uids:
        detail = df_u[df_u["cycle_uid"].isin(uids)].copy()
    else:
        detail = review.copy()

    ooi_map = {}
    if isinstance(regression_qc_frame, pd.DataFrame) and not regression_qc_frame.empty \
            and "out_of_interval" in regression_qc_frame.columns and "cycle_uid" in regression_qc_frame.columns:
        ooi_map = dict(zip(regression_qc_frame["cycle_uid"], regression_qc_frame["out_of_interval"]))
    miri_map = {}
    if isinstance(miri_summary, pd.DataFrame) and not miri_summary.empty and "cycle_uid" in miri_summary.columns:
        miri_map = miri_summary.set_index("cycle_uid").to_dict("index")
    lot_map = {}
    if isinstance(lot_summary, pd.DataFrame) and not lot_summary.empty and "cycle_uid" in lot_summary.columns:
        lot_map = lot_summary.set_index("cycle_uid").to_dict("index")

    legend = ("<div class='legend'>"
             "<span><span class='swatch' style='background:var(--red-bg);border:1px solid var(--red)'></span>"
             "P1 — обязательный разбор</span>"
             "<span><span class='swatch' style='background:var(--amber-bg);border:1px solid var(--amber)'></span>"
             "P2 — рекомендован</span>"
             "<span><span class='swatch' style='background:var(--green-bg);border:1px solid var(--green)'></span>"
             "без приоритета / широкий CI TabPFN</span></div>")

    rows_html = []
    for _, row in detail.iterrows():
        priority = pd.to_numeric(pd.Series([row.get("priority")]), errors="coerce").iloc[0]
        row_cls = "row-p1" if priority == 1 else ("row-p2" if priority == 2 else "row-ok")
        name = row.get("patient_name") or row.get("Patient Full Name") or "—"
        pid = row.get("patient_id") or row.get("ID") or "—"
        age = row.get("age", row.get("Age", "—"))
        preg = pd.to_numeric(pd.Series([row.get("Preg")]), errors="coerce").iloc[0]
        outcome_txt = "✅" if preg == 1 else ("❌" if preg == 0 else "—")
        cluster_code = pd.to_numeric(pd.Series([row.get("dominant_cluster")]), errors="coerce").iloc[0]
        cluster = {0: "C0 Standard", 1: "C1 Poor", 2: "C2 High"}.get(cluster_code, "—")
        uid = row.get("cycle_uid")
        flags = _cycle_flags(row.get("reason"), ooi_map.get(uid), miri_map.get(uid), lot_map.get(uid))
        rows_html.append(
            f"<tr class='{row_cls}'>"
            f"<td>{escape(str(name))[:25]}</td><td>{escape(str(pid))}</td><td>{escape(str(age))}</td>"
            f"<td>{_prob_cell(row.get('p_per_transfer'), _MODEL_BAR_COLORS['MC'])}</td>"
            f"<td>{_prob_cell(row.get('p_kat_raw'), _MODEL_BAR_COLORS['KAT'])}</td>"
            f"<td>{_prob_cell(row.get('p_gnn_raw'), _MODEL_BAR_COLORS['GAT'])}</td>"
            f"<td>{_prob_cell(row.get('PRAI'), _MODEL_BAR_COLORS['PRAI'])}</td>"
            f"<td>{_prob_cell(row.get('DIGITAL TWIN'), _MODEL_BAR_COLORS['DT'])}</td>"
            f"<td>{_prob_cell(row.get('DNN'), _MODEL_BAR_COLORS['DNN'])}</td>"
            f"<td>{outcome_txt}</td><td>{escape(str(cluster))}</td>"
            f"<td class='flags'>{flags}</td></tr>")
    header = ("<tr><th>Пациент</th><th>ID</th><th>Возраст</th><th>MC%</th><th>KAT%</th>"
             "<th>GAT%</th><th>TabPFN%</th><th>Digital Twin%</th><th>DNN%</th>"
             "<th>Исход</th><th>Кластер</th><th>Флаги и суждения</th></tr>")
    table = (f"<div class='table-wrap'><table class='tbl'><thead>{header}</thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table></div>")
    return legend + table


# ---------------------------------------------------------------------------
# 8. MIRI (episodes only — no cycle-summary table)
# ---------------------------------------------------------------------------

def _gap_badge(value) -> str:
    if pd.isna(value) or value not in _GAP_REASON_LABEL:
        return ""
    label, kind = _GAP_REASON_LABEL[value]
    return _badge(label, kind)


def _miri_tab(events: pd.DataFrame | None) -> str:
    note = ("<p class='muted' style='font-size:11.5px'>Пропуск логов &lt;8ч трактуется как потеря "
           "возможности ПИСАТЬ логи (например, отключение питания) — реальное неблагоприятное "
           "событие, а не «отсутствие данных». Пропуск ≥8ч означает явную остановку "
           "инкубатора или логирования.</p>")
    if events is None or events.empty:
        return note + "<p class='muted'>эпизодов не найдено (логов в окне нет либо отклонений не обнаружено)</p>"
    disp = events.copy()
    if "gap_reason" in disp.columns:
        disp["Тип пропуска"] = disp["gap_reason"].map(_gap_badge)
        disp = disp.drop(columns=["gap_reason"])
    rename = {"event_type": "Тип", "sensor": "Датчик", "start": "Начало", "end": "Конец",
             "duration_minutes": "Длительность, мин", "max_deviation": "Макс. отклонение",
             "severity": "Severity", "recovery_time_minutes": "Восстановление, мин"}
    disp = disp.rename(columns={k: v for k, v in rename.items() if k in disp.columns})
    return note + _manual_table(disp, raw_cols=frozenset({"Тип пропуска"}))


# ---------------------------------------------------------------------------
# 9. Consumable lots
# ---------------------------------------------------------------------------

def _lots_tab(lot_summary: pd.DataFrame | None, lot_events: pd.DataFrame | None) -> str:
    return f"<h3>Сводка по циклам</h3>{_table(lot_summary)}<h3>События смены лота</h3>{_table(lot_events)}"


# ---------------------------------------------------------------------------
# 10. OZEMA
# ---------------------------------------------------------------------------

def _ozema_flag_kind(flag: str) -> str:
    if flag in ("—", "н/д"):
        return "info" if flag == "н/д" else "ok"
    return "crit"


def _ozema_legend() -> str:
    rows = "".join(
        f"<tr><td><b>{escape(code)}</b></td><td>{escape(label)}</td></tr>"
        for code, label in [
            ("OMA", ozema.STAGE_LABELS["OMA"]), ("FF", ozema.STAGE_LABELS["FF"]),
            ("ZA", ozema.STAGE_LABELS["ZA"]), ("EEA", ozema.STAGE_LABELS["EEA"]),
            ("—", "В пределах порогов — значимой остановки не выявлено"),
            ("н/д", f"Недостаточно наблюдений (n&lt;{ozema.MIN_N}) — оценить нельзя"),
            ("≥2 флагов у пациента", "Критерий OZEMA выполнен → рассмотреть направление на генетическую консультацию"),
        ]
    )
    return (f"<div class='card'><h3>Легенда OZEMA</h3>"
           f"<p class='muted' style='font-size:12px'>{escape(ozema.METHODOLOGY_CITATION)}. "
           f"Односторонний точный биномиальный тест, α={ozema.ALPHA}, против Vienna-style "
           "порогов компетентности — значимо ниже порога, не просто «ниже линии».</p>"
           f"<table class='tbl'><tbody>{rows}</tbody></table></div>")


def _ozema_table(ozema_cycles: pd.DataFrame) -> str:
    if ozema_cycles is None or ozema_cycles.empty:
        return "<p class='muted'>нет данных OZEMA в этом прогоне</p>"
    d = ozema_cycles.copy()
    d["OZEMA flag"] = d.get("ozema_flag", "").map(lambda f: _badge(str(f), _ozema_flag_kind(str(f))))
    display_cols = {
        "patient_name": "Пациент", "patient_id": "ID", "date_opu": "Date OPU", "age": "Возраст",
        "COC": "COC", "MII": "MII", "2PN": "2PN", "Cleav": "Cleav", "Bl": "Bl",
        "OZEMA flag": "OZEMA flag", "flagged_cycles_n": "Флагов у пациента",
        "embryologist": "Эмбриолог", "doctor": "Врач", "monitoring_note": "Наблюдение/заключение",
    }
    present = [c for c in display_cols if c in d.columns]
    out = d[present].rename(columns={c: display_cols[c] for c in present})
    return _manual_table(out, raw_cols=frozenset({"OZEMA flag"}))


def _ozema_candidates_block(candidates: pd.DataFrame | None) -> str:
    if candidates is None or candidates.empty:
        return ("<div class='card'><h3>Кандидаты на обсуждение генетического консультирования</h3>"
               "<p class='muted'>Не выявлено в этом прогоне.</p></div>")
    n_patients = int(candidates["patient_id"].nunique()) if "patient_id" in candidates.columns else len(candidates)
    return (f"<div class='candidates'><h3>Кандидаты на обсуждение генетического консультирования "
           f"({n_patients})</h3><p style='font-size:12.5px'>Критерий OZEMA: ≥2 циклов с "
           "флагом ареста у одного пациента. Это НЕ диагноз — повод для клинического обсуждения "
           f"и, при показаниях, направления на генетическую консультацию.</p>{_ozema_table(candidates)}</div>")


def _ozema_tab(ozema_cycles: pd.DataFrame | None, ozema_candidates: pd.DataFrame | None) -> str:
    return (_ozema_legend() + _ozema_candidates_block(ozema_candidates) +
           f"<h3>Все циклы</h3>{_ozema_table(ozema_cycles)}")


# ---------------------------------------------------------------------------
# 11. Lead-analyst synthesis (markers preserved for finalize_report.py)
# ---------------------------------------------------------------------------

def _lead_tab(lead_analyst_text: str) -> str:
    if lead_analyst_text:
        body = f"<div class='narrative' id='lead-analysis'>{escape(lead_analyst_text)}</div>"
    else:
        body = ("<p class='muted'>Итоговый текст формируется агентом lead-analyst по "
               "псевдонимизированному пакету.</p>")
    return f"<!-- LEAD_START -->{body}<!-- LEAD_END -->"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_report(analytics_result: dict, kpi_df: pd.DataFrame,
                 incubator_findings: pd.DataFrame | None = None,
                 incubator_events: pd.DataFrame | None = None,
                 lead_analyst_text: str = "",
                 lot_summary: pd.DataFrame | None = None,
                 lot_events: pd.DataFrame | None = None,
                 regression_qc_frame: pd.DataFrame | None = None,
                 regression_calibration: pd.DataFrame | None = None,
                 ozema_cycles: pd.DataFrame | None = None,
                 ozema_candidates: pd.DataFrame | None = None,
                 out_html="data/processed/DT_Analytics_Report.html") -> dict:
    """Строит полностью локальный HTML. Наличие логов само по себе warning не создаёт."""
    global _plotly_embedded
    _plotly_embedded = False
    df_u = analytics_result.get("df_u", pd.DataFrame())
    review = analytics_result.get("review_list", pd.DataFrame())
    kpi_slices = analytics_result.get("kpi_slices", {}) or {}
    embryologist_agg = kpi_slices.get("embryologist", pd.DataFrame())
    doctor_agg = kpi_slices.get("doctor", pd.DataFrame())
    agg = _aggregate_table(kpi_df, kpi_slices.get("overall"))
    kpi_fig, kpi_alerts = _kpi_panel(agg)
    miri_bad = _miri_warning_rows(incubator_findings)
    lot_changed = _lot_change_rows(lot_summary)
    ozema_cycles = ozema_cycles if isinstance(ozema_cycles, pd.DataFrame) else pd.DataFrame()
    ozema_candidates = ozema_candidates if isinstance(ozema_candidates, pd.DataFrame) else pd.DataFrame()
    regression_qc_frame = regression_qc_frame if isinstance(regression_qc_frame, pd.DataFrame) else pd.DataFrame()
    regression_calibration = (regression_calibration if isinstance(regression_calibration, pd.DataFrame)
                             else pd.DataFrame())
    ozema_flagged_n = int(ozema_cycles["flagged"].sum()) if "flagged" in ozema_cycles.columns else 0
    ozema_candidates_n = (int(ozema_candidates["patient_id"].nunique())
                         if "patient_id" in ozema_candidates.columns else 0)

    warnings = []
    if kpi_alerts:
        warnings.append("KPI вне контрольного предела: " + ", ".join(kpi_alerts))
    if isinstance(review, pd.DataFrame) and "priority" in review:
        n1 = int((pd.to_numeric(review["priority"], errors="coerce") == 1).sum())
        if n1:
            warnings.append(f"критических циклов P1: {n1}")
    if not miri_bad.empty:
        warnings.append(f"циклов с MIRI warning/critical: {len(miri_bad)}")
    if not lot_changed.empty:
        warnings.append(f"циклов со сменой лота расходника в окне культивирования: {len(lot_changed)}")
    if ozema_candidates_n:
        warnings.append(f"кандидатов на обсуждение генетического консультирования (OZEMA): {ozema_candidates_n}")
    banner = ("<div class='warn'>⚠ Раннее предупреждение: " +
              escape(" · ".join(warnings)) + "</div>") if warnings else ""

    panels = "".join([
        _panel("overview", _overview_tab(len(df_u), len(review), len(miri_bad), len(lot_changed),
                                         ozema_flagged_n, ozema_candidates_n, banner, kpi_fig, agg)),
        _panel("embryologist", _competency_table(embryologist_agg, "embryologist",
                                                  kpi.EMBRYOLOGIST_METRICS, "Эмбриолог")
              + _below_competency_paragraph(embryologist_agg, "embryologist")
              + f"<p class='excl-note'>{_EXCLUSION_NOTE}</p>"),
        _panel("doctor", _competency_table(doctor_agg, "doctor", kpi.DOCTOR_METRICS, "Врач")
              + _below_competency_paragraph(doctor_agg, "doctor")
              + f"<p class='excl-note'>{_EXCLUSION_NOTE}</p>"),
        _panel("models", _models_tab(df_u, analytics_result.get("calibration"),
                                     analytics_result.get("conversion"))),
        _panel("cluster", _cluster_analysis_tab(df_u)),
        _panel("similar", _similar_cases_tab(df_u)),
        _panel("pregnancy", _pregnancy_journal_tab(analytics_result, regression_qc_frame)),
        _panel("regression", _regression_tab(regression_qc_frame, regression_calibration)),
        _panel("review", _review_section(analytics_result, regression_qc_frame, incubator_findings, lot_summary)),
        _panel("miri", _miri_tab(incubator_events)),
        _panel("lots", _lots_tab(lot_summary, lot_events)),
        _panel("ozema", _ozema_tab(ozema_cycles, ozema_candidates)),
        _panel("lead", _lead_tab(lead_analyst_text)),
    ])

    ts = datetime.now().astimezone().isoformat(timespec="minutes")
    html = f"""<!doctype html><html lang='ru'><head><meta charset='utf-8'>
    <title>IVF Digital Twin — аналитический отчёт</title>{_css()}</head><body>
    <header class="hero"><div class="inner">
    <div class="hero-eyebrow">IVF DIGITAL TWIN</div>
    <h1>Аналитический отчёт</h1>
    <div class="meta"><span class="pill">Сформировано {escape(ts)}</span>
    <span class="pill">циклов: {len(df_u)}</span>
    <span class="pill">review-list: {len(review)}</span></div>
    </div></header>
    <div class="wrap">
    {_tab_inputs()}
    {_tab_nav()}
    <div class="tab-panels">{panels}</div>
    <p class="foot">Причинные выводы не устанавливаются: причины неудач представлены как проверяемые
    гипотезы. Диагнозы и назначения этим отчётом не даются.</p>
    </div>
    </body></html>"""
    atomic_write_text(html, out_html)
    return {"report_html": out_html, "warnings": warnings,
            "kpi_alerts": kpi_alerts, "plotly": _HAVE_PLOTLY}
