"""
9_📐_Análisis_Comparativo_Avanzado.py

Benchmark de Producción y Completación — análisis de cuadrantes P10/P50/P90.

Secciones
─────────
1. Benchmark de Producción (Gas / Petróleo)
   • Cuadrante: Q_pico vs Acumulada (Gp o Np)
   • Cuadrante: Q_pico vs Acumulada de Agua (Wp)
   Visualiza TODOS los pozos en gris; el usuario resalta área, empresa o sigla.

2. Benchmark de Completación
   • Ejes configurables con variables del dataset de fractura.
   Misma lógica de resaltado.
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
    create_summary_dataframe,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def percentile_lines(series: pd.Series, q: float) -> float:
    """Return quantile ignoring NaN/inf."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.quantile(q)) if not clean.empty else np.nan


def add_quadrant_lines(fig, x_p50, y_p50, x_p10, y_p10, x_p90, y_p90,
                        x_label, y_label, x_range=None, y_range=None):
    """
    Adds P50 crosshairs and P10/P90 sub-quadrant dashed rectangles.
    Uses production convention: P10 = optimistic (high value).
    """
    line_kw = dict(line_color="rgba(80,80,80,0.55)", line_width=1.2)

    # P50 solid lines
    fig.add_vline(x=x_p50, line_dash="solid", line_color="rgba(50,50,50,0.70)",
                  line_width=1.8, annotation_text=f"P50 {x_label}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="rgba(50,50,50,0.85)"))
    fig.add_hline(y=y_p50, line_dash="solid", line_color="rgba(50,50,50,0.70)",
                  line_width=1.8, annotation_text=f"P50 {y_label}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="rgba(50,50,50,0.85)"))

    # P10 dashed lines (optimistic threshold)
    fig.add_vline(x=x_p10, line_dash="dash", **line_kw,
                  annotation_text=f"P10 {x_label}",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="rgba(80,80,80,0.75)"))
    fig.add_hline(y=y_p10, line_dash="dash", **line_kw,
                  annotation_text=f"P10 {y_label}",
                  annotation_position="top left",
                  annotation_font=dict(size=9, color="rgba(80,80,80,0.75)"))

    # P90 dashed lines (conservative threshold)
    fig.add_vline(x=x_p90, line_dash="dot", **line_kw,
                  annotation_text=f"P90 {x_label}",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="rgba(80,80,80,0.75)"))
    fig.add_hline(y=y_p90, line_dash="dot", **line_kw,
                  annotation_text=f"P90 {y_label}",
                  annotation_position="top left",
                  annotation_font=dict(size=9, color="rgba(80,80,80,0.75)"))

    return fig


def add_quadrant_labels(fig, x_p50, y_p50, x_max, y_max, x_min, y_min):
    """Annotate the four quadrant zones."""
    labels = [
        (x_p50 + (x_max - x_p50) * 0.5, y_p50 + (y_max - y_p50) * 0.5,
         "⭐ Alto-Alto", "rgba(34,139,34,0.18)"),
        (x_min + (x_p50 - x_min) * 0.5, y_p50 + (y_max - y_p50) * 0.5,
         "Alta Q, Baja Acum.", "rgba(255,165,0,0.12)"),
        (x_p50 + (x_max - x_p50) * 0.5, y_min + (y_p50 - y_min) * 0.5,
         "Alta Acum., Baja Q", "rgba(100,149,237,0.12)"),
        (x_min + (x_p50 - x_min) * 0.5, y_min + (y_p50 - y_min) * 0.5,
         "⚠️ Bajo-Bajo", "rgba(220,20,60,0.10)"),
    ]
    for x, y, txt, _ in labels:
        fig.add_annotation(
            x=x, y=y, text=txt, showarrow=False,
            font=dict(size=9, color="rgba(60,60,60,0.60)"),
            bgcolor="rgba(255,255,255,0.0)",
        )
    return fig


def build_quadrant_chart(
    df_all: pd.DataFrame,
    df_highlight: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    highlight_label: str,
    highlight_color: str = "#E63946",
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    """
    Scatter quadrant chart.
    df_all      → all wells in light grey.
    df_highlight → selected subset in highlight colour.
    """
    # Clip extremes for cleaner axes (1st–99th pct)
    def clip(s):
        lo, hi = np.nanpercentile(s.replace([np.inf, -np.inf], np.nan).dropna(), [0.5, 99.5])
        return s.clip(lo, hi)

    df_all = df_all.copy()
    df_all[x_col] = clip(df_all[x_col])
    df_all[y_col] = clip(df_all[y_col])

    # Percentiles (production convention: P10 = high = optimistic)
    x_p10 = percentile_lines(df_all[x_col], 0.90)
    x_p50 = percentile_lines(df_all[x_col], 0.50)
    x_p90 = percentile_lines(df_all[x_col], 0.10)
    y_p10 = percentile_lines(df_all[y_col], 0.90)
    y_p50 = percentile_lines(df_all[y_col], 0.50)
    y_p90 = percentile_lines(df_all[y_col], 0.10)

    x_min = df_all[x_col].min()
    x_max = df_all[x_col].max()
    y_min = df_all[y_col].min()
    y_max = df_all[y_col].max()

    fig = go.Figure()

    # Background — all wells
    fig.add_trace(go.Scatter(
        x=df_all[x_col],
        y=df_all[y_col],
        mode="markers",
        name="Todos los pozos",
        marker=dict(color="rgba(180,180,180,0.35)", size=5, line=dict(width=0)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            f"{x_label}: %{{x:,.0f}}<br>"
            f"{y_label}: %{{y:,.0f}}"
            "<extra>Universo</extra>"
        ),
        customdata=df_all[["sigla"]].values,
    ))

    # Highlight
    if not df_highlight.empty:
        df_hl = df_highlight.copy()
        df_hl[x_col] = clip(df_hl[x_col])
        df_hl[y_col] = clip(df_hl[y_col])
        fig.add_trace(go.Scatter(
            x=df_hl[x_col],
            y=df_hl[y_col],
            mode="markers",
            name=highlight_label,
            marker=dict(
                color=highlight_color,
                size=9,
                line=dict(width=1.2, color="white"),
                symbol="circle",
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{x_label}: %{{x:,.0f}}<br>"
                f"{y_label}: %{{y:,.0f}}"
                "<extra>" + highlight_label + "</extra>"
            ),
            customdata=df_hl[["sigla"]].values,
        ))

    # Quadrant lines
    add_quadrant_lines(fig, x_p50, y_p50, x_p10, y_p10, x_p90, y_p90,
                       x_label, y_label)
    add_quadrant_labels(fig, x_p50, y_p50, x_max, y_max, x_min, y_min)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(
            title=x_label,
            type="log" if log_x else "linear",
            showgrid=True, gridcolor="rgba(200,200,200,0.3)",
            zeroline=False,
        ),
        yaxis=dict(
            title=y_label,
            type="log" if log_y else "linear",
            showgrid=True, gridcolor="rgba(200,200,200,0.3)",
            zeroline=False,
        ),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520,
        margin=dict(l=60, r=30, t=70, b=60),
    )
    return fig


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

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.header(":blue[📐 Análisis Comparativo Avanzado]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))

data_filtered = data_sorted[data_sorted["tef"] > 0]

# Summary per well
with st.spinner("Calculando métricas por pozo…"):
    summary_df = create_summary_dataframe(data_filtered)

# Fracture data
with st.spinner("Cargando datos de fractura…"):
    df_frac = load_frac_data()

# Merge for completation benchmark
df_merged = (
    pd.merge(df_frac, summary_df, on="sigla", how="inner")
    .drop_duplicates(subset="sigla")
)
df_merged["fracspacing"]       = df_merged["longitud_rama_horizontal_m"] / df_merged["cantidad_fracturas"]
df_merged["prop_x_etapa"]      = df_merged["arena_total_tn"] / df_merged["cantidad_fracturas"]
df_merged["proppant_intensity"] = df_merged["arena_total_tn"] / df_merged["longitud_rama_horizontal_m"]
df_merged["AS_x_vol"]          = df_merged["arena_total_tn"] / (df_merged["agua_inyectada_m3"].replace(0, np.nan) / 1000)
df_merged["Qo_peak_x_etapa"]   = df_merged["Qo_peak"] / df_merged["cantidad_fracturas"]
df_merged["Qg_peak_x_etapa"]   = df_merged["Qg_peak"] / df_merged["cantidad_fracturas"]
df_merged = df_merged.replace([np.inf, -np.inf], np.nan)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BENCHMARK DE PRODUCCIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🛢️ Benchmark de Producción", divider="blue")

st.markdown("""
Todos los pozos del dataset se grafican en **gris claro** como universo de referencia.  
Los cuadrantes P10 / P50 / P90 se calculan sobre ese universo.  
Usá los filtros para resaltar un área, empresa o pozo específico.
""")

# ── Fluid selector ────────────────────────────────────────────────────────────
fluid_prod = st.radio(
    "Tipo de Fluido:",
    ["Petrolífero", "Gasífero"],
    horizontal=True,
    key="fluid_prod",
)

is_oil = fluid_prod == "Petrolífero"
peak_col   = "Qo_peak"  if is_oil else "Qg_peak"
cum_col    = "Np"       if is_oil else "Gp"
peak_label = "Qo Pico (m³/d)" if is_oil else "Qg Pico (km³/d)"
cum_label  = "Np Acumulada (m³)" if is_oil else "Gp Acumulada (km³)"
water_col  = "Wp"
water_label = "Wp Acumulada Agua (m³)"

# Universe for selected fluid type
summary_fluid = summary_df[summary_df["tipopozoNEW"] == fluid_prod].copy()
summary_fluid = summary_fluid.dropna(subset=[peak_col, cum_col, water_col])

# ── Highlight filter ──────────────────────────────────────────────────────────
st.markdown("#### 🎯 Filtro de Resaltado")
col_filt1, col_filt2, col_filt3 = st.columns(3)

with col_filt1:
    areas_prod = ["(Ninguna)"] + sorted(
        data_filtered["areayacimiento"].dropna().unique()
    )
    selected_area_prod = st.selectbox("Área de Yacimiento:", areas_prod, key="area_prod")

with col_filt2:
    companies_prod = ["(Ninguna)"] + sorted(
        summary_fluid["empresaNEW"].dropna().unique()
    )
    selected_company_prod = st.selectbox("Empresa:", companies_prod, key="company_prod")

with col_filt3:
    # Build sigla list from the fluid-filtered summary
    siglas_prod = ["(Ninguna)"] + sorted(summary_fluid["sigla"].dropna().unique())
    selected_sigla_prod = st.selectbox("Sigla (pozo):", siglas_prod, key="sigla_prod")

# Log scale toggle
col_log1, col_log2 = st.columns(2)
log_x_prod = col_log1.checkbox("Escala log eje X (acumulada)", key="logx_prod")
log_y_prod = col_log2.checkbox("Escala log eje Y (caudal pico)", key="logy_prod")

# Build highlight mask
def build_highlight_mask(df: pd.DataFrame, area, company, sigla, area_map: dict) -> pd.Series:
    """Returns boolean mask. Priority: sigla > company > area."""
    mask = pd.Series(False, index=df.index)
    if sigla != "(Ninguna)":
        mask = df["sigla"] == sigla
    elif company != "(Ninguna)":
        mask = df["empresaNEW"] == company
    elif area != "(Ninguna)":
        siglas_in_area = area_map.get(area, set())
        mask = df["sigla"].isin(siglas_in_area)
    return mask

# Map area → siglas (from production data)
area_to_siglas = (
    data_filtered.groupby("areayacimiento")["sigla"]
    .apply(set)
    .to_dict()
)

hl_mask_prod = build_highlight_mask(
    summary_fluid,
    selected_area_prod,
    selected_company_prod,
    selected_sigla_prod,
    area_to_siglas,
)

df_hl_prod = summary_fluid[hl_mask_prod]

hl_label_prod = (
    selected_sigla_prod   if selected_sigla_prod   != "(Ninguna)" else
    selected_company_prod if selected_company_prod != "(Ninguna)" else
    selected_area_prod    if selected_area_prod    != "(Ninguna)" else
    "Selección"
)

# ── Chart 1a: Q_pico vs Acumulada ─────────────────────────────────────────────
st.markdown(f"#### Cuadrante: {peak_label} vs {cum_label}")
fig_prod1 = build_quadrant_chart(
    df_all=summary_fluid,
    df_highlight=df_hl_prod,
    x_col=cum_col,
    y_col=peak_col,
    x_label=cum_label,
    y_label=peak_label,
    title=f"Benchmark {fluid_prod}: Caudal Pico vs Producción Acumulada",
    highlight_label=hl_label_prod,
    highlight_color="#E63946" if is_oil else "#2196F3",
    log_x=log_x_prod,
    log_y=log_y_prod,
)
st.plotly_chart(fig_prod1, use_container_width=True)

# ── Chart 1b: Q_pico vs Agua acumulada ────────────────────────────────────────
st.markdown(f"#### Cuadrante: {peak_label} vs {water_label}")

summary_fluid_w = summary_fluid.dropna(subset=[peak_col, water_col])
df_hl_prod_w    = summary_fluid_w[summary_fluid_w["sigla"].isin(df_hl_prod["sigla"])]

fig_prod2 = build_quadrant_chart(
    df_all=summary_fluid_w,
    df_highlight=df_hl_prod_w,
    x_col=water_col,
    y_col=peak_col,
    x_label=water_label,
    y_label=peak_label,
    title=f"Benchmark {fluid_prod}: Caudal Pico vs Agua Acumulada",
    highlight_label=hl_label_prod,
    highlight_color="#E63946" if is_oil else "#2196F3",
    log_x=log_x_prod,
    log_y=log_y_prod,
)
st.plotly_chart(fig_prod2, use_container_width=True)

# ── KPIs for selection ─────────────────────────────────────────────────────────
if not df_hl_prod.empty:
    st.markdown(f"##### 📊 Estadísticas — {hl_label_prod} ({len(df_hl_prod)} pozos)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"P50 {peak_label}",  f"{df_hl_prod[peak_col].median():,.0f}")
    c2.metric(f"P50 {cum_label}",   f"{df_hl_prod[cum_col].median():,.0f}")
    c3.metric(f"P50 {water_label}", f"{df_hl_prod[water_col].median():,.0f}")
    c4.metric("N° Pozos", len(df_hl_prod))

    # Percentile position vs universe
    def pct_rank(val, series):
        clean = series.dropna()
        if clean.empty or pd.isna(val):
            return np.nan
        return (clean < val).mean() * 100

    p50_x_hl  = df_hl_prod[cum_col].median()
    p50_y_hl  = df_hl_prod[peak_col].median()
    rank_x     = pct_rank(p50_x_hl, summary_fluid[cum_col])
    rank_y     = pct_rank(p50_y_hl, summary_fluid[peak_col])

    st.caption(
        f"La selección se ubica en el **percentil {rank_x:.0f}** en {cum_label} "
        f"y en el **percentil {rank_y:.0f}** en {peak_label} respecto al universo {fluid_prod}."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BENCHMARK DE COMPLETACIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🔩 Benchmark de Completación", divider="blue")

st.markdown("""
Compará la completación de cualquier pozo, empresa o área vs el universo de pozos con datos de fractura.  
Elegí qué variable graficar en cada eje.
""")

# ── Axis variable selector ────────────────────────────────────────────────────
COMPLETION_VARS = {
    "Longitud Rama (m)":              "longitud_rama_horizontal_m",
    "Cantidad de Etapas":             "cantidad_fracturas",
    "Arena Total (tn)":               "arena_total_tn",
    "Arena Nacional (tn)":            "arena_bombeada_nacional_tn",
    "Arena Importada (tn)":           "arena_bombeada_importada_tn",
    "Agua Inyectada (m³)":            "agua_inyectada_m3",
    "Fracspacing (m)":                "fracspacing",
    "Propante x Etapa (tn/etapa)":    "prop_x_etapa",
    "Intensidad Propante (tn/m)":     "proppant_intensity",
    "AS x Vol. Inyectado (tn/1000m³)":"AS_x_vol",
    "Qo Pico (m³/d)":                 "Qo_peak",
    "Qg Pico (km³/d)":                "Qg_peak",
    "Qo Pico x Etapa (m³/d/etapa)":  "Qo_peak_x_etapa",
    "Qg Pico x Etapa (km³/d/etapa)": "Qg_peak_x_etapa",
    "Np Acumulada (m³)":              "Np",
    "Gp Acumulada (km³)":             "Gp",
    "Wp Acumulada (m³)":              "Wp",
}

col_ax1, col_ax2 = st.columns(2)
x_var_label = col_ax1.selectbox(
    "Variable eje X:",
    list(COMPLETION_VARS.keys()),
    index=0,
    key="x_var_comp",
)
y_var_label = col_ax2.selectbox(
    "Variable eje Y:",
    list(COMPLETION_VARS.keys()),
    index=6,
    key="y_var_comp",
)

x_var = COMPLETION_VARS[x_var_label]
y_var = COMPLETION_VARS[y_var_label]

# Fluid filter for completion
fluid_comp = st.radio(
    "Filtrar por Tipo de Fluido (opcional):",
    ["Todos", "Petrolífero", "Gasífero"],
    horizontal=True,
    key="fluid_comp",
)

df_comp_base = df_merged.copy()
if fluid_comp != "Todos":
    df_comp_base = df_comp_base[df_comp_base["tipopozoNEW"] == fluid_comp]

df_comp_base = df_comp_base.dropna(subset=[x_var, y_var])

if df_comp_base.empty:
    st.warning("No hay datos con ambas variables seleccionadas. Probá otra combinación.")
    st.stop()

# ── Highlight filter ──────────────────────────────────────────────────────────
st.markdown("#### 🎯 Filtro de Resaltado")
col_fc1, col_fc2, col_fc3 = st.columns(3)

with col_fc1:
    areas_comp = ["(Ninguna)"] + sorted(
        data_filtered["areayacimiento"].dropna().unique()
    )
    selected_area_comp = st.selectbox("Área de Yacimiento:", areas_comp, key="area_comp")

with col_fc2:
    companies_comp = ["(Ninguna)"] + sorted(
        df_comp_base["empresaNEW"].dropna().unique()
    )
    selected_company_comp = st.selectbox("Empresa:", companies_comp, key="company_comp")

with col_fc3:
    siglas_comp = ["(Ninguna)"] + sorted(df_comp_base["sigla"].dropna().unique())
    selected_sigla_comp = st.selectbox("Sigla (pozo):", siglas_comp, key="sigla_comp")

col_log3, col_log4 = st.columns(2)
log_x_comp = col_log3.checkbox("Escala log eje X", key="logx_comp")
log_y_comp = col_log4.checkbox("Escala log eje Y", key="logy_comp")

hl_mask_comp = build_highlight_mask(
    df_comp_base,
    selected_area_comp,
    selected_company_comp,
    selected_sigla_comp,
    area_to_siglas,
)
df_hl_comp = df_comp_base[hl_mask_comp]

hl_label_comp = (
    selected_sigla_comp   if selected_sigla_comp   != "(Ninguna)" else
    selected_company_comp if selected_company_comp != "(Ninguna)" else
    selected_area_comp    if selected_area_comp    != "(Ninguna)" else
    "Selección"
)

# ── Chart 2: Completion quadrant ──────────────────────────────────────────────
fig_comp = build_quadrant_chart(
    df_all=df_comp_base,
    df_highlight=df_hl_comp,
    x_col=x_var,
    y_col=y_var,
    x_label=x_var_label,
    y_label=y_var_label,
    title=f"Benchmark de Completación — {x_var_label} vs {y_var_label}",
    highlight_label=hl_label_comp,
    highlight_color="#FF6B35",
    log_x=log_x_comp,
    log_y=log_y_comp,
)
st.plotly_chart(fig_comp, use_container_width=True)

# ── Stats for completion selection ─────────────────────────────────────────────
if not df_hl_comp.empty:
    st.markdown(f"##### 📊 Estadísticas — {hl_label_comp} ({len(df_hl_comp)} pozos)")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"P50 {x_var_label}", f"{df_hl_comp[x_var].median():,.1f}")
    c2.metric(f"P50 {y_var_label}", f"{df_hl_comp[y_var].median():,.1f}")
    c3.metric("N° Pozos",           len(df_hl_comp))

    rank_x = (df_comp_base[x_var].dropna() < df_hl_comp[x_var].median()).mean() * 100
    rank_y = (df_comp_base[y_var].dropna() < df_hl_comp[y_var].median()).mean() * 100

    st.caption(
        f"La selección se ubica en el **percentil {rank_x:.0f}** en {x_var_label} "
        f"y en el **percentil {rank_y:.0f}** en {y_var_label} respecto al universo de completación."
    )

    # Additional breakdown table for multi-well selections
    if len(df_hl_comp) > 1:
        with st.expander("Ver detalle de pozos de la selección"):
            display_cols = ["sigla", "empresaNEW", x_var, y_var]
            rename = {"sigla": "Pozo", "empresaNEW": "Empresa",
                      x_var: x_var_label, y_var: y_var_label}
            st.dataframe(
                df_hl_comp[display_cols].rename(columns=rename)
                .sort_values(y_var_label, ascending=False)
                .reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

# ── Legend / methodology note ─────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ Metodología y lectura de los gráficos"):
    st.markdown("""
    **Cuadrantes y percentiles**
    - Los percentiles se calculan sobre el **universo completo** de pozos del tipo de fluido seleccionado.
    - Se usa la **convención de producción de hidrocarburos**:
      - **P10** = pozo optimista (valor alto, percentil 90 estadístico).
      - **P50** = mediana del universo.
      - **P90** = pozo conservador (valor bajo, percentil 10 estadístico).
    - Las líneas **sólidas** representan P50 (cruce de cuadrantes principal).
    - Las líneas **punteadas** delimitan las zonas P10 y P90.

    **Lectura del cuadrante superior derecho (⭐ Alto-Alto)**
    - Pozos con alta producción acumulada Y alto caudal pico → mejores performers.

    **Benchmark de Completación**
    - Se cruzan datos del dataset de fractura con el resumen de producción por sigla.
    - Solo se grafican pozos con **datos válidos en ambos ejes** seleccionados.
    - Se aplican los mismos filtros de calidad de datos que en el resto del reporte
      (longitud rama > 100 m, etapas > 6, arena total > 100 tn).

    **Prioridad del filtro de resaltado**  
    Sigla > Empresa > Área (solo se aplica el primer filtro activo distinto de "(Ninguna)").
    """)