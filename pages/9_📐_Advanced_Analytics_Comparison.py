"""
9_📐_Análisis_Comparativo_Avanzado.py  —  v4

Cambios v4
──────────
- Eliminados: expanders de líderes Alto-Alto y sección de metodología.
- Multiselect con colores distintos por cada entidad seleccionada (empresa/área/pozo).
- Nuevo panel de "Evolución histórica de la selección": cuando hay multiselect activo,
  gráfico de líneas Q_pico mediano y acumulada mediana por año de campaña,
  una serie por empresa/área/pozo seleccionado.
- color_by_year y multiselect siguen siendo mutuamente excluyentes.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils import (
    COMPANY_REPLACEMENTS,
    get_fluid_classification,
    load_frac_data,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

INTERVALS = {"180 días": 180, "1 año (365d)": 365, "5 años (1825d)": 1825}

# Palettes
YEAR_COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
]
SEL_COLORS = [
    "#E63946","#2196F3","#FF9800","#4CAF50","#9C27B0",
    "#00BCD4","#F44336","#3F51B5","#8BC34A","#FF5722",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — CUMULATIVE PRODUCTION  (Watchlist logic)
# ══════════════════════════════════════════════════════════════════════════════

def cumulative_at_interval(df: pd.DataFrame, interval_days: int) -> pd.DataFrame:
    grp = (
        df.groupby("sigla")
        .agg(
            oil_cum        =("prod_pet",       "sum"),
            gas_cum        =("prod_gas",       "sum"),
            water_cum      =("prod_agua",      "sum"),
            tef_cum        =("tef",            "sum"),
            empresaNEW     =("empresaNEW",     "first"),
            areayacimiento =("areayacimiento", "first"),
            tipopozoNEW    =("tipopozoNEW",    "first"),
            start_year     =("anio",           "min"),
        )
        .reset_index()
    )
    return grp[grp["tef_cum"] >= interval_days].copy()


def build_prod_summary(data_filtered: pd.DataFrame, interval_days: int) -> pd.DataFrame:
    cum = cumulative_at_interval(data_filtered, interval_days)
    peaks = (
        data_filtered.groupby("sigla")
        .agg(
            Qo_peak =("oil_rate",   "max"),
            Qg_peak =("gas_rate",   "max"),
            Qw_peak =("water_rate", "max"),
        )
        .reset_index()
    )
    return cum.merge(peaks, on="sigla", how="left")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — QUADRANT CHART
# ══════════════════════════════════════════════════════════════════════════════

def pct_q(s: pd.Series, q: float) -> float:
    c = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(c.quantile(q)) if not c.empty else np.nan


def _safe_log_range(s: pd.Series) -> list:
    pos = s[s > 0].replace([np.inf, -np.inf], np.nan).dropna()
    if pos.empty:
        return [None, None]
    return [np.log10(np.nanpercentile(pos, 1)) * 0.97,
            np.log10(np.nanpercentile(pos, 99)) * 1.03]


def _clip(s: pd.Series) -> pd.Series:
    c = s.replace([np.inf, -np.inf], np.nan).dropna()
    if c.empty:
        return s
    lo, hi = np.nanpercentile(c, [0.5, 99.5])
    return s.clip(lo, hi)


def _add_ref_lines(fig, x_p50, y_p50, x_p10, y_p10, x_p90, y_p90, xl, yl):
    kw = dict(line_color="rgba(80,80,80,0.45)", line_width=1.1)
    fig.add_vline(x=x_p50, line_dash="solid", line_color="rgba(35,35,35,0.70)",
                  line_width=1.8, annotation_text=f"P50 {xl}",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color="rgba(35,35,35,0.80)"))
    fig.add_hline(y=y_p50, line_dash="solid", line_color="rgba(35,35,35,0.70)",
                  line_width=1.8, annotation_text=f"P50 {yl}",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color="rgba(35,35,35,0.80)"))
    for xv, lbl, pos in [(x_p10, f"P10 {xl}", "bottom right"),
                          (x_p90, f"P90 {xl}", "bottom right")]:
        fig.add_vline(x=xv, line_dash="dash" if "P10" in lbl else "dot", **kw,
                      annotation_text=lbl, annotation_position=pos,
                      annotation_font=dict(size=8, color="rgba(80,80,80,0.65)"))
    for yv, lbl, pos in [(y_p10, f"P10 {yl}", "top left"),
                          (y_p90, f"P90 {yl}", "top left")]:
        fig.add_hline(y=yv, line_dash="dash" if "P10" in lbl else "dot", **kw,
                      annotation_text=lbl, annotation_position=pos,
                      annotation_font=dict(size=8, color="rgba(80,80,80,0.65)"))
    return fig


def _add_zone_labels(fig, x_p50, y_p50, x_max, y_max, x_min, y_min):
    for x, y, txt in [
        (x_p50 + (x_max - x_p50) * 0.50, y_p50 + (y_max - y_p50) * 0.55, "⭐ Alto–Alto"),
        (x_min + (x_p50 - x_min) * 0.50, y_p50 + (y_max - y_p50) * 0.55, "Alta Q · Baja Acum."),
        (x_p50 + (x_max - x_p50) * 0.50, y_min + (y_p50 - y_min) * 0.45, "Alta Acum. · Baja Q"),
        (x_min + (x_p50 - x_min) * 0.50, y_min + (y_p50 - y_min) * 0.45, "⚠️ Bajo–Bajo"),
    ]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(size=9, color="rgba(60,60,60,0.50)"))
    return fig


def build_quadrant_chart(
    df_all:          pd.DataFrame,
    # For multiselect: list of (label, df_subset, color) tuples; empty list = no highlights
    highlights:      list,
    x_col:           str,
    y_col:           str,
    x_label:         str,
    y_label:         str,
    title:           str,
    log_x:           bool = False,
    log_y:           bool = False,
    color_by_year:   bool = False,
) -> go.Figure:

    df_all = df_all.copy()
    df_all[x_col] = _clip(df_all[x_col])
    df_all[y_col] = _clip(df_all[y_col])

    x_p10 = pct_q(df_all[x_col], 0.90)
    x_p50 = pct_q(df_all[x_col], 0.50)
    x_p90 = pct_q(df_all[x_col], 0.10)
    y_p10 = pct_q(df_all[y_col], 0.90)
    y_p50 = pct_q(df_all[y_col], 0.50)
    y_p90 = pct_q(df_all[y_col], 0.10)
    x_min, x_max = df_all[x_col].min(), df_all[x_col].max()
    y_min, y_max = df_all[y_col].min(), df_all[y_col].max()

    cd_cols = [c for c in ["sigla", "empresaNEW", "areayacimiento"] if c in df_all.columns]
    htmpl   = (
        "<b>%{customdata[0]}</b><br>"
        + (f"Empresa: %{{customdata[1]}}<br>" if len(cd_cols) > 1 else "")
        + (f"Área: %{{customdata[2]}}<br>"    if len(cd_cols) > 2 else "")
        + f"{x_label}: %{{x:,.1f}}<br>{y_label}: %{{y:,.1f}}"
    )

    fig = go.Figure()

    # ── Universe layer ─────────────────────────────────────────────────────────
    if color_by_year and "start_year" in df_all.columns:
        for i, yr in enumerate(sorted(df_all["start_year"].dropna().unique())):
            sub = df_all[df_all["start_year"] == yr]
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[y_col], mode="markers",
                name=str(int(yr)), legendgroup=str(int(yr)),
                marker=dict(color=YEAR_COLORS[i % len(YEAR_COLORS)],
                            size=5, opacity=0.65, line=dict(width=0)),
                hovertemplate=htmpl + f"<extra>Campaña {int(yr)}</extra>",
                customdata=sub[cd_cols].values,
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df_all[x_col], y=df_all[y_col], mode="markers",
            name="Todos los pozos",
            marker=dict(color="rgba(175,175,175,0.28)", size=5, line=dict(width=0)),
            hovertemplate=htmpl + "<extra>Universo</extra>",
            customdata=df_all[cd_cols].values,
        ))

    # ── Highlight layers — one per selected entity, each with its own color ───
    for hl_label, df_hl, hl_color in highlights:
        if df_hl.empty:
            continue
        dh = df_hl.copy()
        dh[x_col] = _clip(dh[x_col])
        dh[y_col] = _clip(dh[y_col])
        fig.add_trace(go.Scatter(
            x=dh[x_col], y=dh[y_col], mode="markers",
            name=hl_label,
            marker=dict(color=hl_color, size=9,
                        line=dict(width=1.5, color="white")),
            hovertemplate=htmpl + f"<extra>{hl_label}</extra>",
            customdata=dh[cd_cols].values,
        ))

    _add_ref_lines(fig, x_p50, y_p50, x_p10, y_p10, x_p90, y_p90, x_label, y_label)
    _add_zone_labels(fig, x_p50, y_p50, x_max, y_max, x_min, y_min)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(
            title=x_label,
            type="log" if log_x else "linear",
            range=_safe_log_range(df_all[x_col]) if log_x else [None, None],
            showgrid=True, gridcolor="rgba(200,200,200,0.28)", zeroline=False,
        ),
        yaxis=dict(
            title=y_label,
            type="log" if log_y else "linear",
            range=_safe_log_range(df_all[y_col]) if log_y else [None, None],
            showgrid=True, gridcolor="rgba(200,200,200,0.28)", zeroline=False,
        ),
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            title_text="Campaña" if color_by_year else "",
        ),
        height=540,
        margin=dict(l=65, r=35, t=80, b=65),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — HISTORICAL EVOLUTION CHART
# Shown below quadrant when multiselect is active.
# One line per selected entity; X = start_year, Y = median of metric.
# ══════════════════════════════════════════════════════════════════════════════

def build_evolution_chart(
    highlights:  list,      # list of (label, df_subset, color)
    y_col:       str,
    y_label:     str,
    title:       str,
    group_col:   str = "start_year",   # always start_year here
) -> go.Figure | None:
    """
    Line chart: median of y_col per start_year, one line per highlight entity.
    Returns None if there is nothing to plot.
    """
    fig = go.Figure()
    has_data = False
    for hl_label, df_hl, hl_color in highlights:
        if df_hl.empty or y_col not in df_hl.columns or group_col not in df_hl.columns:
            continue
        agg = (
            df_hl.dropna(subset=[group_col, y_col])
            .groupby(group_col)[y_col]
            .agg(median="median", count="count")
            .reset_index()
        )
        if agg.empty:
            continue
        has_data = True
        fig.add_trace(go.Scatter(
            x=agg[group_col],
            y=agg["median"],
            mode="lines+markers+text",
            name=hl_label,
            line=dict(color=hl_color, width=2.5),
            marker=dict(size=8, color=hl_color,
                        line=dict(width=1.5, color="white")),
            text=agg["median"].round(0).astype(int).astype(str),
            textposition="top center",
            textfont=dict(size=8, color=hl_color),
            customdata=agg["count"].values,
            hovertemplate=(
                f"<b>{hl_label}</b><br>"
                f"Campaña: %{{x}}<br>"
                f"P50 {y_label}: %{{y:,.1f}}<br>"
                "N° pozos: %{customdata}<extra></extra>"
            ),
        ))

    if not has_data:
        return None

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Año de Campaña", dtick=1, showgrid=True,
                   gridcolor="rgba(200,200,200,0.28)"),
        yaxis=dict(title=y_label, showgrid=True,
                   gridcolor="rgba(200,200,200,0.28)"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        height=360,
        margin=dict(l=60, r=30, t=55, b=55),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — MULTI-SELECT FILTER returning per-entity highlight list
# ══════════════════════════════════════════════════════════════════════════════

def multiselect_filter(df: pd.DataFrame, key_prefix: str):
    """
    Cascaded multiselect: empresa → area → sigla.
    Returns:
      highlights: list of (label, df_subset, color) — one entry per selected entity
      mode:       "sigla" | "area" | "empresa" | None
    """
    col1, col2, col3 = st.columns(3)

    all_companies = sorted(df["empresaNEW"].dropna().unique())
    sel_cos = col1.multiselect("Empresas:", all_companies, key=f"{key_prefix}_co")

    df_area   = df if not sel_cos else df[df["empresaNEW"].isin(sel_cos)]
    all_areas = sorted(df_area["areayacimiento"].dropna().unique())
    sel_ars   = col2.multiselect("Áreas:", all_areas, key=f"{key_prefix}_ar")

    df_sig    = df_area if not sel_ars else df_area[df_area["areayacimiento"].isin(sel_ars)]
    all_sigs  = sorted(df_sig["sigla"].dropna().unique())
    sel_sigs  = col3.multiselect("Siglas:", all_sigs, key=f"{key_prefix}_sig")

    highlights = []
    mode       = None

    if sel_sigs:
        mode = "sigla"
        for i, sig in enumerate(sel_sigs):
            highlights.append((sig, df[df["sigla"] == sig], SEL_COLORS[i % len(SEL_COLORS)]))
    elif sel_ars:
        mode = "area"
        for i, ar in enumerate(sel_ars):
            highlights.append((ar, df[df["areayacimiento"] == ar], SEL_COLORS[i % len(SEL_COLORS)]))
    elif sel_cos:
        mode = "empresa"
        for i, co in enumerate(sel_cos):
            highlights.append((co, df[df["empresaNEW"] == co], SEL_COLORS[i % len(SEL_COLORS)]))

    return highlights, mode


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE & DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(layout="wide")

if "df" in st.session_state:
    data_sorted = st.session_state["df"]
    data_sorted["date"]       = pd.to_datetime(data_sorted["anio"].astype(str) + "-" + data_sorted["mes"].astype(str) + "-1")
    data_sorted["gas_rate"]   = data_sorted["prod_gas"]  / data_sorted["tef"]
    data_sorted["oil_rate"]   = data_sorted["prod_pet"]  / data_sorted["tef"]
    data_sorted["water_rate"] = data_sorted["prod_agua"] / data_sorted["tef"]
    data_sorted               = data_sorted.sort_values(by=["sigla", "date"], ascending=True)
    data_sorted["empresaNEW"] = data_sorted["empresa"].replace(COMPANY_REPLACEMENTS)
    data_sorted               = get_fluid_classification(data_sorted)
    st.info("Utilizando datos recuperados de la memoria.")
else:
    st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la Página Principal.")
    st.stop()

st.header(":blue[📐 Análisis Comparativo Avanzado]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))

data_filtered = data_sorted[data_sorted["tef"] > 0]

with st.spinner("Cargando datos de fractura…"):
    df_frac = load_frac_data()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BENCHMARK DE PRODUCCIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🛢️ Benchmark de Producción", divider="blue")

# ── Global controls ───────────────────────────────────────────────────────────
ctrl1, ctrl2 = st.columns(2)
fluid_prod    = ctrl1.radio("Tipo de Fluido:", ["Petrolífero", "Gasífero"],
                             horizontal=True, key="fluid_prod")
interval_lbl  = ctrl2.selectbox("Intervalo de acumulada:",
                                  list(INTERVALS.keys()), index=1, key="int_prod")
interval_days = INTERVALS[interval_lbl]

is_oil   = fluid_prod == "Petrolífero"
peak_hc  = "Qo_peak"  if is_oil else "Qg_peak"
cum_hc   = "oil_cum"  if is_oil else "gas_cum"
peak_lbl = "Qo Pico (m³/d)"             if is_oil else "Qg Pico (km³/d)"
cum_lbl  = f"Np @ {interval_lbl} (m³)"  if is_oil else f"Gp @ {interval_lbl} (km³)"
wat_lbl  = f"Wp @ {interval_lbl} (m³)"

# Visual mode row
vis1, vis2 = st.columns(2)
color_by_year_prod = vis1.checkbox(
    "🎨 Colorear por año de campaña", key="cby_prod",
    help="Pinta cada punto según campaña. Incompatible con resaltado múltiple.",
)
normalize_stages = vis2.checkbox(
    "➗ Normalizar por etapas (Q/etapa · Acum/etapa)", key="norm_stages",
)

log_col1, log_col2 = st.columns(2)
log_x_p = log_col1.checkbox("Log eje X", key="lx_p")
log_y_p = log_col2.checkbox("Log eje Y", key="ly_p")

# Build summary
with st.spinner(f"Calculando acumuladas @ {interval_lbl}…"):
    prod_sum = build_prod_summary(data_filtered, interval_days)

summary_fluid = prod_sum[prod_sum["tipopozoNEW"] == fluid_prod].copy()
summary_fluid = summary_fluid.dropna(subset=[peak_hc, cum_hc])

if summary_fluid.empty:
    st.warning(f"No hay pozos con TEF ≥ {interval_days} días para {fluid_prod}.")
    st.stop()

# Normalise by stages if requested
if normalize_stages:
    stages_pw = (
        df_frac.drop_duplicates("sigla")[["sigla", "cantidad_fracturas"]]
        .rename(columns={"cantidad_fracturas": "n_stages"})
    )
    sf_n = summary_fluid.merge(stages_pw, on="sigla", how="inner")
    sf_n = sf_n[sf_n["n_stages"] > 0].copy()
    for col in [peak_hc, cum_hc, "water_cum", "Qw_peak"]:
        if col in sf_n.columns:
            sf_n[col] = sf_n[col] / sf_n["n_stages"]
    sf_work    = sf_n
    peak_lbl_w = peak_lbl.replace("(m³/d)", "(m³/d/etapa)").replace("(km³/d)", "(km³/d/etapa)")
    cum_lbl_w  = cum_lbl.replace("(m³)", "(m³/etapa)").replace("(km³)", "(km³/etapa)")
    wat_lbl_w  = wat_lbl.replace("(m³)", "(m³/etapa)")
    qw_lbl_w   = "Qw Pico (m³/d/etapa)"
    st.info(f"Normalización activa — {len(sf_work)} pozos con datos de fractura.")
else:
    sf_work    = summary_fluid
    peak_lbl_w = peak_lbl
    cum_lbl_w  = cum_lbl
    wat_lbl_w  = wat_lbl
    qw_lbl_w   = "Qw Pico (m³/d)"

# Filter section
st.markdown("#### 🎯 Filtro de Resaltado")
if color_by_year_prod:
    st.info("ℹ️ Modo 'Colorear por año' activo — el resaltado múltiple está desactivado.")
    highlights_p = []
    mode_p       = None
else:
    highlights_p, mode_p = multiselect_filter(sf_work, "prod")

# ── Chart 1a ──────────────────────────────────────────────────────────────────
st.markdown(f"#### {peak_lbl_w} vs {cum_lbl_w}")
fig1a = build_quadrant_chart(
    df_all=sf_work, highlights=highlights_p,
    x_col=cum_hc, y_col=peak_hc,
    x_label=cum_lbl_w, y_label=peak_lbl_w,
    title=f"Benchmark {fluid_prod}: Caudal Pico vs Producción Acumulada ({interval_lbl})"
          + (" — por etapa" if normalize_stages else ""),
    log_x=log_x_p, log_y=log_y_p,
    color_by_year=color_by_year_prod,
)
st.plotly_chart(fig1a, use_container_width=True)

# KPIs for single-entity highlights
if len(highlights_p) == 1:
    hl_lbl_s, df_hl_s, _ = highlights_p[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(peak_lbl_w, f"{df_hl_s[peak_hc].median():,.1f}")
    c2.metric(cum_lbl_w,  f"{df_hl_s[cum_hc].median():,.1f}")
    c3.metric("N° Pozos", len(df_hl_s))
    rx = (sf_work[cum_hc].dropna()  < df_hl_s[cum_hc].median()).mean()  * 100
    ry = (sf_work[peak_hc].dropna() < df_hl_s[peak_hc].median()).mean() * 100
    st.caption(f"Percentil **{rx:.0f}** en acumulada · **{ry:.0f}** en caudal pico vs universo {fluid_prod}.")

# ── Evolution chart 1a ────────────────────────────────────────────────────────
if highlights_p and mode_p in ("empresa", "area"):
    st.markdown(f"##### 📈 Evolución histórica — P50 {peak_lbl_w} por campaña")
    fig_ev1a = build_evolution_chart(
        highlights=highlights_p,
        y_col=peak_hc,
        y_label=peak_lbl_w,
        title=f"Evolución P50 {peak_lbl_w} por Campaña",
    )
    if fig_ev1a:
        st.plotly_chart(fig_ev1a, use_container_width=True)

    fig_ev1a_cum = build_evolution_chart(
        highlights=highlights_p,
        y_col=cum_hc,
        y_label=cum_lbl_w,
        title=f"Evolución P50 {cum_lbl_w} por Campaña",
    )
    if fig_ev1a_cum:
        st.plotly_chart(fig_ev1a_cum, use_container_width=True)

st.divider()

# ── Chart 1b: Qw_pico vs Agua acumulada ──────────────────────────────────────
st.markdown(f"#### {qw_lbl_w} vs {wat_lbl_w}")
sf_w = sf_work.dropna(subset=["Qw_peak", "water_cum"])

# Re-filter highlights on water-valid subset
highlights_w = [
    (lbl, df_hl[df_hl["sigla"].isin(sf_w["sigla"])], col)
    for lbl, df_hl, col in highlights_p
    if not df_hl[df_hl["sigla"].isin(sf_w["sigla"])].empty
]

fig1b = build_quadrant_chart(
    df_all=sf_w, highlights=highlights_w,
    x_col="water_cum", y_col="Qw_peak",
    x_label=wat_lbl_w, y_label=qw_lbl_w,
    title=f"Benchmark {fluid_prod}: Qw Pico vs Agua Acumulada ({interval_lbl})"
          + (" — por etapa" if normalize_stages else ""),
    log_x=log_x_p, log_y=log_y_p,
    color_by_year=color_by_year_prod,
)
st.plotly_chart(fig1b, use_container_width=True)

if highlights_w and mode_p in ("empresa", "area"):
    st.markdown(f"##### 📈 Evolución histórica — P50 {qw_lbl_w} por campaña")
    fig_ev1b = build_evolution_chart(
        highlights=highlights_w,
        y_col="Qw_peak",
        y_label=qw_lbl_w,
        title=f"Evolución P50 {qw_lbl_w} por Campaña",
    )
    if fig_ev1b:
        st.plotly_chart(fig_ev1b, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BENCHMARK DE COMPLETACIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🔩 Benchmark de Completación", divider="blue")

# Build completion base
peaks_all = (
    data_filtered.groupby("sigla")
    .agg(
        Qo_peak        =("oil_rate",       "max"),
        Qg_peak        =("gas_rate",       "max"),
        Qw_peak        =("water_rate",     "max"),
        empresaNEW     =("empresaNEW",     "first"),
        areayacimiento =("areayacimiento", "first"),
        tipopozoNEW    =("tipopozoNEW",    "first"),
        start_year     =("anio",           "min"),
    )
    .reset_index()
)
df_comp_base = df_frac.merge(peaks_all, on="sigla", how="inner").drop_duplicates(subset="sigla")

for lbl, days in INTERVALS.items():
    c_int = cumulative_at_interval(data_filtered, days)[
        ["sigla", "oil_cum", "gas_cum", "water_cum"]
    ].rename(columns={
        "oil_cum":   f"oil_{lbl}",
        "gas_cum":   f"gas_{lbl}",
        "water_cum": f"wtr_{lbl}",
    })
    df_comp_base = df_comp_base.merge(c_int, on="sigla", how="left")

df_comp_base["fracspacing"]        = df_comp_base["longitud_rama_horizontal_m"] / df_comp_base["cantidad_fracturas"]
df_comp_base["prop_x_etapa"]       = df_comp_base["arena_total_tn"] / df_comp_base["cantidad_fracturas"]
df_comp_base["proppant_intensity"]  = df_comp_base["arena_total_tn"] / df_comp_base["longitud_rama_horizontal_m"]
df_comp_base["AS_x_vol"]           = df_comp_base["arena_total_tn"] / (df_comp_base["agua_inyectada_m3"].replace(0, np.nan) / 1000)
df_comp_base["Qo_peak_x_etapa"]    = df_comp_base["Qo_peak"] / df_comp_base["cantidad_fracturas"]
df_comp_base["Qg_peak_x_etapa"]    = df_comp_base["Qg_peak"] / df_comp_base["cantidad_fracturas"]
df_comp_base["Qw_peak_x_etapa"]    = df_comp_base["Qw_peak"] / df_comp_base["cantidad_fracturas"]
df_comp_base = df_comp_base.replace([np.inf, -np.inf], np.nan)

COMP_VARS: dict = {
    "Longitud Rama (m)":                "longitud_rama_horizontal_m",
    "Cantidad de Etapas":               "cantidad_fracturas",
    "Arena Total (tn)":                 "arena_total_tn",
    "Arena Nacional (tn)":              "arena_bombeada_nacional_tn",
    "Arena Importada (tn)":             "arena_bombeada_importada_tn",
    "Agua Inyectada (m³)":              "agua_inyectada_m3",
    "Fracspacing (m)":                  "fracspacing",
    "Propante x Etapa (tn/etapa)":      "prop_x_etapa",
    "Intensidad Propante (tn/m)":       "proppant_intensity",
    "AS x Vol. Inyectado (tn/1000m³)":  "AS_x_vol",
    "Qo Pico (m³/d)":                   "Qo_peak",
    "Qg Pico (km³/d)":                  "Qg_peak",
    "Qw Pico (m³/d)":                   "Qw_peak",
    "Qo Pico x Etapa (m³/d/etapa)":    "Qo_peak_x_etapa",
    "Qg Pico x Etapa (km³/d/etapa)":   "Qg_peak_x_etapa",
    "Qw Pico x Etapa (m³/d/etapa)":    "Qw_peak_x_etapa",
}
for lbl in INTERVALS.keys():
    COMP_VARS[f"Np @ {lbl} (m³)"]  = f"oil_{lbl}"
    COMP_VARS[f"Gp @ {lbl} (km³)"] = f"gas_{lbl}"
    COMP_VARS[f"Wp @ {lbl} (m³)"]  = f"wtr_{lbl}"

ca1, ca2 = st.columns(2)
xvl = ca1.selectbox("Variable eje X:", list(COMP_VARS.keys()), index=0, key="xvc")
yvl = ca2.selectbox("Variable eje Y:", list(COMP_VARS.keys()), index=6, key="yvc")
xv  = COMP_VARS[xvl]
yv  = COMP_VARS[yvl]

cv1, cv2 = st.columns(2)
fluid_c = cv1.radio("Tipo de Fluido:", ["Todos", "Petrolífero", "Gasífero"],
                     horizontal=True, key="fluid_c")
color_by_year_comp = cv2.checkbox(
    "🎨 Colorear por año de campaña", key="cby_comp",
    help="Incompatible con resaltado múltiple.",
)

log_c1, log_c2 = st.columns(2)
log_xc = log_c1.checkbox("Log eje X", key="lxc")
log_yc = log_c2.checkbox("Log eje Y", key="lyc")

df_c = df_comp_base.copy()
if fluid_c != "Todos":
    df_c = df_c[df_c["tipopozoNEW"] == fluid_c]
df_c = df_c.dropna(subset=[xv, yv])

if df_c.empty:
    st.warning("No hay datos con ambas variables seleccionadas. Probá otra combinación.")
    st.stop()

st.markdown("#### 🎯 Filtro de Resaltado")
if color_by_year_comp:
    st.info("ℹ️ Modo 'Colorear por año' activo — el resaltado múltiple está desactivado.")
    highlights_c = []
    mode_c       = None
else:
    highlights_c, mode_c = multiselect_filter(df_c, "comp")

fig2 = build_quadrant_chart(
    df_all=df_c, highlights=highlights_c,
    x_col=xv, y_col=yv,
    x_label=xvl, y_label=yvl,
    title=f"Benchmark Completación — {xvl} vs {yvl}",
    log_x=log_xc, log_y=log_yc,
    color_by_year=color_by_year_comp,
)
st.plotly_chart(fig2, use_container_width=True)

if len(highlights_c) == 1:
    hl_lbl_cs, df_hl_cs, _ = highlights_c[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(xvl,        f"{df_hl_cs[xv].median():,.1f}")
    c2.metric(yvl,        f"{df_hl_cs[yv].median():,.1f}")
    c3.metric("N° Pozos", len(df_hl_cs))
    rx = (df_c[xv].dropna() < df_hl_cs[xv].median()).mean() * 100
    ry = (df_c[yv].dropna() < df_hl_cs[yv].median()).mean() * 100
    st.caption(f"Percentil **{rx:.0f}** en {xvl} · **{ry:.0f}** en {yvl} vs universo de completación.")

if len(highlights_c) > 1:
    all_hl = pd.concat([df for _, df, _ in highlights_c]).drop_duplicates("sigla")
    with st.expander("Ver detalle de pozos seleccionados"):
        dcols = [c for c in ["sigla", "empresaNEW", "areayacimiento",
                              "start_year", xv, yv] if c in all_hl.columns]
        ren   = {"sigla": "Pozo", "empresaNEW": "Empresa", "areayacimiento": "Área",
                 "start_year": "Campaña", xv: xvl, yv: yvl}
        st.dataframe(
            all_hl[dcols].rename(columns=ren)
            .sort_values(yvl, ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )

# ── Evolution chart — completion ──────────────────────────────────────────────
if highlights_c and mode_c in ("empresa", "area"):
    st.markdown(f"##### 📈 Evolución histórica — P50 {yvl} por campaña")
    fig_ev2 = build_evolution_chart(
        highlights=highlights_c,
        y_col=yv,
        y_label=yvl,
        title=f"Evolución P50 {yvl} por Campaña",
    )
    if fig_ev2:
        st.plotly_chart(fig_ev2, use_container_width=True)

    fig_ev2x = build_evolution_chart(
        highlights=highlights_c,
        y_col=xv,
        y_label=xvl,
        title=f"Evolución P50 {xvl} por Campaña",
    )
    if fig_ev2x:
        st.plotly_chart(fig_ev2x, use_container_width=True)