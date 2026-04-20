"""
9_📐_Análisis_Comparativo_Avanzado.py  —  v3

Cambios v3
──────────
- Checkbox: colorear nube de puntos por año de campaña (start_year).
  Mutuamente exclusivo con el resaltado de selección múltiple.
- Filtro de resaltado: multiselect de empresas, áreas Y pozos (cualquier combo).
- Normalización por etapas: checkbox para ver Q/etapa y Acum/etapa en benchmark producción.
- Auto-zoom al percentil 1–99 cuando se activa escala log.
- Gráfico Qpico vs Agua: eje Y = Qw (water_rate peak) en lugar de Q hidrocarburo.
- Líderes Alto-Alto + heatmap de consistencia al pie de cada cuadrante (expander).
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

YEAR_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — CUMULATIVE PRODUCTION  (same logic as Watchlist)
# ══════════════════════════════════════════════════════════════════════════════

def cumulative_at_interval(df: pd.DataFrame, interval_days: int) -> pd.DataFrame:
    grp = (
        df.groupby("sigla")
        .agg(
            oil_cum        =("prod_pet",        "sum"),
            gas_cum        =("prod_gas",        "sum"),
            water_cum      =("prod_agua",       "sum"),
            tef_cum        =("tef",             "sum"),
            empresaNEW     =("empresaNEW",      "first"),
            areayacimiento =("areayacimiento",  "first"),
            tipopozoNEW    =("tipopozoNEW",     "first"),
            start_year     =("anio",            "min"),
        )
        .reset_index()
    )
    return grp[grp["tef_cum"] >= interval_days].copy()


def build_prod_summary(data_filtered: pd.DataFrame, interval_days: int) -> pd.DataFrame:
    cum = cumulative_at_interval(data_filtered, interval_days)
    peaks = (
        data_filtered.groupby("sigla")
        .agg(
            Qo_peak  =("oil_rate",   "max"),
            Qg_peak  =("gas_rate",   "max"),
            Qw_peak  =("water_rate", "max"),
        )
        .reset_index()
    )
    return cum.merge(peaks, on="sigla", how="left")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — QUADRANT CHART (v3)
# ══════════════════════════════════════════════════════════════════════════════

def pct_q(s: pd.Series, q: float) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.quantile(q)) if not clean.empty else np.nan


def _safe_range(s: pd.Series, log: bool) -> list:
    """
    For log scale: zoom to 1st–99th percentile of positive values.
    For linear: return None (Plotly autoscale).
    """
    if not log:
        return [None, None]
    pos = s[s > 0].replace([np.inf, -np.inf], np.nan).dropna()
    if pos.empty:
        return [None, None]
    lo = np.log10(np.nanpercentile(pos, 1))
    hi = np.log10(np.nanpercentile(pos, 99)) * 1.05
    return [lo, hi]


def _clip(s: pd.Series) -> pd.Series:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return s
    lo, hi = np.nanpercentile(clean, [0.5, 99.5])
    return s.clip(lo, hi)


def add_quadrant_lines(fig, x_p50, y_p50, x_p10, y_p10, x_p90, y_p90, xl, yl):
    kw = dict(line_color="rgba(80,80,80,0.48)", line_width=1.1)
    fig.add_vline(x=x_p50, line_dash="solid", line_color="rgba(35,35,35,0.72)",
                  line_width=1.8, annotation_text=f"P50 {xl}",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color="rgba(35,35,35,0.82)"))
    fig.add_hline(y=y_p50, line_dash="solid", line_color="rgba(35,35,35,0.72)",
                  line_width=1.8, annotation_text=f"P50 {yl}",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color="rgba(35,35,35,0.82)"))
    for xv, lbl, pos in [(x_p10, f"P10 {xl}", "bottom right"),
                          (x_p90, f"P90 {xl}", "bottom right")]:
        fig.add_vline(x=xv, line_dash="dash" if "P10" in lbl else "dot", **kw,
                      annotation_text=lbl, annotation_position=pos,
                      annotation_font=dict(size=8, color="rgba(80,80,80,0.68)"))
    for yv, lbl, pos in [(y_p10, f"P10 {yl}", "top left"),
                          (y_p90, f"P90 {yl}", "top left")]:
        fig.add_hline(y=yv, line_dash="dash" if "P10" in lbl else "dot", **kw,
                      annotation_text=lbl, annotation_position=pos,
                      annotation_font=dict(size=8, color="rgba(80,80,80,0.68)"))
    return fig


def add_quadrant_labels(fig, x_p50, y_p50, x_max, y_max, x_min, y_min):
    for x, y, txt in [
        (x_p50 + (x_max - x_p50) * 0.50, y_p50 + (y_max - y_p50) * 0.55, "⭐ Alto–Alto"),
        (x_min + (x_p50 - x_min) * 0.50, y_p50 + (y_max - y_p50) * 0.55, "Alta Q · Baja Acum."),
        (x_p50 + (x_max - x_p50) * 0.50, y_min + (y_p50 - y_min) * 0.45, "Alta Acum. · Baja Q"),
        (x_min + (x_p50 - x_min) * 0.50, y_min + (y_p50 - y_min) * 0.45, "⚠️ Bajo–Bajo"),
    ]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(size=9, color="rgba(60,60,60,0.52)"))
    return fig


def build_quadrant_chart(
    df_all:          pd.DataFrame,
    df_highlight:    pd.DataFrame,   # empty → no highlight
    x_col:           str,
    y_col:           str,
    x_label:         str,
    y_label:         str,
    title:           str,
    highlight_label: str,
    highlight_color: str = "#E63946",
    log_x:           bool = False,
    log_y:           bool = False,
    color_by_year:   bool = False,   # NEW: color universe by start_year
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

    if color_by_year and "start_year" in df_all.columns:
        # One trace per year so the legend shows campaign colours
        years = sorted(df_all["start_year"].dropna().unique())
        for i, yr in enumerate(years):
            sub = df_all[df_all["start_year"] == yr]
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub[y_col],
                mode="markers", name=str(int(yr)),
                legendgroup=str(int(yr)),
                marker=dict(
                    color=YEAR_COLORS[i % len(YEAR_COLORS)],
                    size=5, opacity=0.65, line=dict(width=0),
                ),
                hovertemplate=htmpl + f"<extra>Campaña {int(yr)}</extra>",
                customdata=sub[cd_cols].values,
            ))
    else:
        # Uniform grey universe
        fig.add_trace(go.Scatter(
            x=df_all[x_col], y=df_all[y_col],
            mode="markers", name="Todos los pozos",
            marker=dict(color="rgba(175,175,175,0.30)", size=5, line=dict(width=0)),
            hovertemplate=htmpl + "<extra>Universo</extra>",
            customdata=df_all[cd_cols].values,
        ))

    # Highlighted selection (only shown when NOT color_by_year OR if user wants both)
    if not df_highlight.empty:
        df_hl = df_highlight.copy()
        df_hl[x_col] = _clip(df_hl[x_col])
        df_hl[y_col] = _clip(df_hl[y_col])
        # If color_by_year is active, override highlight with black outline markers
        hl_color = "#111111" if color_by_year else highlight_color
        fig.add_trace(go.Scatter(
            x=df_hl[x_col], y=df_hl[y_col],
            mode="markers", name=highlight_label,
            marker=dict(color=hl_color, size=11,
                        line=dict(width=2, color="white"),
                        symbol="circle-open-dot"),
            hovertemplate=htmpl + f"<extra>{highlight_label}</extra>",
            customdata=df_hl[cd_cols].values,
        ))

    add_quadrant_lines(fig, x_p50, y_p50, x_p10, y_p10, x_p90, y_p90, x_label, y_label)
    add_quadrant_labels(fig, x_p50, y_p50, x_max, y_max, x_min, y_min)

    x_range = _safe_range(df_all[x_col], log_x)
    y_range = _safe_range(df_all[y_col], log_y)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(
            title=x_label,
            type="log" if log_x else "linear",
            range=x_range if log_x else [None, None],
            showgrid=True, gridcolor="rgba(200,200,200,0.28)", zeroline=False,
        ),
        yaxis=dict(
            title=y_label,
            type="log" if log_y else "linear",
            range=y_range if log_y else [None, None],
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
# HELPERS — ALTO-ALTO LEADERS + CONSISTENCY HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def render_leaders(
    df_universe: pd.DataFrame,
    x_col: str, y_col: str,
    x_label: str, y_label: str,
    fluid_label: str, section_key: str,
):
    x_p50 = pct_q(df_universe[x_col], 0.50)
    y_p50 = pct_q(df_universe[y_col], 0.50)
    df_aa = df_universe[
        (df_universe[x_col] >= x_p50) & (df_universe[y_col] >= y_p50)
    ].copy()

    if df_aa.empty:
        st.caption("No hay pozos en el cuadrante Alto-Alto con los datos actuales.")
        return

    st.markdown(
        f"**{len(df_aa)} pozos** ({len(df_aa)/len(df_universe)*100:.1f}% del universo {fluid_label})"
        f" en el cuadrante ⭐ Alto–Alto."
    )

    show_cols  = [c for c in ["sigla", "empresaNEW", "areayacimiento",
                               "start_year", x_col, y_col] if c in df_aa.columns]
    rename_map = {"sigla": "Pozo", "empresaNEW": "Empresa",
                  "areayacimiento": "Área", "start_year": "Campaña",
                  x_col: x_label, y_col: y_label}
    st.markdown("**Top 15 pozos del cuadrante Alto–Alto** (ordenados por eje Y)")
    st.dataframe(
        df_aa[show_cols].rename(columns=rename_map)
        .sort_values(y_label, ascending=False).head(15).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

    if "empresaNEW" not in df_aa.columns or "areayacimiento" not in df_aa.columns:
        return

    aa_cnt  = (df_aa.groupby(["empresaNEW", "areayacimiento"])["sigla"]
               .nunique().reset_index(name="aa_n"))
    tot_cnt = (df_universe.groupby(["empresaNEW", "areayacimiento"])["sigla"]
               .nunique().reset_index(name="tot_n"))
    heat    = tot_cnt.merge(aa_cnt, on=["empresaNEW", "areayacimiento"], how="left")
    heat["aa_n"] = heat["aa_n"].fillna(0)
    heat["pct"]  = (heat["aa_n"] / heat["tot_n"] * 100).round(1)
    heat         = heat[heat["tot_n"] >= 2]

    if heat.empty:
        st.caption("No hay suficientes datos para el heatmap (mínimo 2 pozos por celda).")
        return

    pivot   = heat.pivot_table(index="empresaNEW", columns="areayacimiento",
                                values="pct", fill_value=np.nan)
    row_ord = pivot.mean(axis=1).sort_values(ascending=False).index
    col_ord = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot   = pivot.loc[row_ord, col_ord]
    txt_v   = pivot.applymap(lambda v: f"{v:.0f}%" if pd.notna(v) else "—").values

    fig_h = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="YlGn", zmin=0, zmax=100,
        text=txt_v, texttemplate="%{text}", textfont=dict(size=9),
        hoverongaps=False,
        colorbar=dict(title="% Alto-Alto", ticksuffix="%"),
        hovertemplate="Empresa: %{y}<br>Área: %{x}<br>% Alto-Alto: %{z:.1f}%<extra></extra>",
    ))
    fig_h.update_layout(
        template="plotly_white",
        title=f"Heatmap de consistencia — % pozos Alto-Alto · {fluid_label}",
        xaxis=dict(title="Área de Yacimiento", tickangle=-35),
        yaxis_title="Empresa",
        height=max(300, 32 * len(pivot) + 130),
        margin=dict(l=165, r=30, t=60, b=130),
    )
    st.markdown("**Heatmap de consistencia — % de pozos Alto-Alto por Empresa × Área:**")
    st.caption("Verde intenso = dominancia estructural. Solo combinaciones con ≥ 2 pozos.")
    st.plotly_chart(fig_h, use_container_width=True, key=f"heat_{section_key}")

    emp_rank = (
        heat.groupby("empresaNEW")
        .apply(lambda g: pd.Series({
            "Pozos Alto-Alto": int(g["aa_n"].sum()),
            "Pozos Universo":  int(g["tot_n"].sum()),
            "% Alto-Alto":     round(g["aa_n"].sum() / g["tot_n"].sum() * 100, 1),
        }))
        .reset_index()
        .rename(columns={"empresaNEW": "Empresa"})
        .sort_values("% Alto-Alto", ascending=False)
        .reset_index(drop=True)
    )
    st.markdown("**Ranking de empresas por % de pozos en Alto-Alto:**")
    st.dataframe(emp_rank, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — MULTI-SELECT FILTER  (empresa → area → sigla, any combination)
# ══════════════════════════════════════════════════════════════════════════════

def multiselect_filter(df: pd.DataFrame, key_prefix: str):
    """
    Cascaded multiselect: empresa → area (filtered by selected companies) → sigla.
    Returns (mask: pd.Series, label: str).
    Priority for label: siglas > areas > companies.
    When nothing is selected the mask is all-False.
    """
    col1, col2, col3 = st.columns(3)

    all_companies = sorted(df["empresaNEW"].dropna().unique())
    sel_cos = col1.multiselect("Empresas:", all_companies, key=f"{key_prefix}_co")

    df_area = df if not sel_cos else df[df["empresaNEW"].isin(sel_cos)]
    all_areas = sorted(df_area["areayacimiento"].dropna().unique())
    sel_ars = col2.multiselect("Áreas de Yacimiento:", all_areas, key=f"{key_prefix}_ar")

    df_sig = df_area if not sel_ars else df_area[df_area["areayacimiento"].isin(sel_ars)]
    all_siglas = sorted(df_sig["sigla"].dropna().unique())
    sel_sigs = col3.multiselect("Siglas (pozos):", all_siglas, key=f"{key_prefix}_sig")

    # Build mask
    if sel_sigs:
        mask  = df["sigla"].isin(sel_sigs)
        label = ", ".join(sel_sigs[:3]) + ("…" if len(sel_sigs) > 3 else "")
    elif sel_ars:
        mask  = df["areayacimiento"].isin(sel_ars)
        label = ", ".join(sel_ars[:3]) + ("…" if len(sel_ars) > 3 else "")
    elif sel_cos:
        mask  = df["empresaNEW"].isin(sel_cos)
        label = ", ".join(sel_cos[:3]) + ("…" if len(sel_cos) > 3 else "")
    else:
        mask  = pd.Series(False, index=df.index)
        label = "Selección"

    return mask, label


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

st.markdown("""
Todos los pozos con **TEF acumulado ≥ intervalo seleccionado** integran el universo  
(misma lógica que hoja **Watchlist**). Los cuadrantes P10/P50/P90 se calculan sobre ese universo.
""")

# ── Global controls ───────────────────────────────────────────────────────────
ctrl1, ctrl2 = st.columns(2)
fluid_prod    = ctrl1.radio("Tipo de Fluido:", ["Petrolífero", "Gasífero"],
                             horizontal=True, key="fluid_prod")
interval_lbl  = ctrl2.selectbox("Intervalo de acumulada:",
                                  list(INTERVALS.keys()), index=1, key="int_prod")
interval_days = INTERVALS[interval_lbl]

is_oil     = fluid_prod == "Petrolífero"
peak_hc    = "Qo_peak"   if is_oil else "Qg_peak"
cum_hc     = "oil_cum"   if is_oil else "gas_cum"
peak_lbl   = "Qo Pico (m³/d)"              if is_oil else "Qg Pico (km³/d)"
cum_lbl    = f"Np @ {interval_lbl} (m³)"   if is_oil else f"Gp @ {interval_lbl} (km³)"
wat_lbl    = f"Wp @ {interval_lbl} (m³)"

# ── Visual mode ───────────────────────────────────────────────────────────────
vis_col1, vis_col2, vis_col3 = st.columns(3)
color_by_year_prod = vis_col1.checkbox(
    "🎨 Colorear por año de campaña", key="cby_prod",
    help="Pinta cada punto según el año de inicio del pozo. Incompatible con resaltado múltiple."
)
normalize_stages = vis_col2.checkbox(
    "➗ Normalizar por etapas (÷ cantidad de fracturas)",
    key="norm_stages",
    help="Muestra Q/etapa y Acum/etapa. Requiere datos de fractura para los pozos.",
)
log1, log2 = vis_col3.columns(2)
log_x_p = vis_col3.checkbox("Log eje X", key="lx_p")
log_y_p = vis_col3.checkbox("Log eje Y", key="ly_p")

# ── Build per-well summary ─────────────────────────────────────────────────────
with st.spinner(f"Calculando acumuladas @ {interval_lbl}…"):
    prod_sum = build_prod_summary(data_filtered, interval_days)

summary_fluid = prod_sum[prod_sum["tipopozoNEW"] == fluid_prod].copy()
summary_fluid = summary_fluid.dropna(subset=[peak_hc, cum_hc])

if summary_fluid.empty:
    st.warning(f"No hay pozos con TEF ≥ {interval_days} días para {fluid_prod}.")
    st.stop()

# ── Normalisation by stages (join frac data) ──────────────────────────────────
if normalize_stages:
    stages_per_well = (
        df_frac.drop_duplicates(subset="sigla")[["sigla", "cantidad_fracturas"]]
        .rename(columns={"cantidad_fracturas": "n_stages"})
    )
    sf_norm = summary_fluid.merge(stages_per_well, on="sigla", how="inner")
    sf_norm = sf_norm[sf_norm["n_stages"] > 0].copy()
    sf_norm[peak_hc]  = sf_norm[peak_hc]  / sf_norm["n_stages"]
    sf_norm[cum_hc]   = sf_norm[cum_hc]   / sf_norm["n_stages"]
    sf_norm["water_cum"] = sf_norm["water_cum"] / sf_norm["n_stages"]
    sf_norm["Qw_peak"]   = sf_norm["Qw_peak"]   / sf_norm["n_stages"]
    sf_work = sf_norm
    peak_lbl_w  = peak_lbl.replace("(m³/d)", "(m³/d/etapa)").replace("(km³/d)", "(km³/d/etapa)")
    cum_lbl_w   = cum_lbl.replace("(m³)", "(m³/etapa)").replace("(km³)", "(km³/etapa)")
    wat_lbl_w   = wat_lbl.replace("(m³)", "(m³/etapa)")
    qw_lbl_w    = "Qw Pico (m³/d/etapa)"
    st.info(f"Normalización activa: mostrando {len(sf_work)} pozos con datos de fractura y etapas > 0.")
else:
    sf_work    = summary_fluid
    peak_lbl_w = peak_lbl
    cum_lbl_w  = cum_lbl
    wat_lbl_w  = wat_lbl
    qw_lbl_w   = "Qw Pico (m³/d)"

# ── Filter: disable multiselect when color_by_year is active ─────────────────
st.markdown("#### 🎯 Filtro de Resaltado")
if color_by_year_prod:
    st.info("ℹ️ Modo 'Colorear por año' activo. El resaltado de selección múltiple está desactivado.")
    df_hl_p   = pd.DataFrame(columns=sf_work.columns)
    hl_lbl_p  = "Selección"
else:
    hl_mask_p, hl_lbl_p = multiselect_filter(sf_work, "prod")
    df_hl_p = sf_work[hl_mask_p]

# ── Chart 1a: Q_pico vs Acumulada hidrocarburo ────────────────────────────────
st.markdown(f"#### Cuadrante: {peak_lbl_w} vs {cum_lbl_w}")
fig1a = build_quadrant_chart(
    df_all=sf_work, df_highlight=df_hl_p,
    x_col=cum_hc, y_col=peak_hc,
    x_label=cum_lbl_w, y_label=peak_lbl_w,
    title=f"Benchmark {fluid_prod}: Caudal Pico vs Producción Acumulada ({interval_lbl})"
          + (" — por etapa" if normalize_stages else ""),
    highlight_label=hl_lbl_p,
    highlight_color="#E63946" if is_oil else "#1565C0",
    log_x=log_x_p, log_y=log_y_p,
    color_by_year=color_by_year_prod,
)
st.plotly_chart(fig1a, use_container_width=True)

if not df_hl_p.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric(peak_lbl_w, f"{df_hl_p[peak_hc].median():,.1f}")
    c2.metric(cum_lbl_w,  f"{df_hl_p[cum_hc].median():,.1f}")
    c3.metric("N° Pozos", len(df_hl_p))
    rx = (sf_work[cum_hc].dropna()  < df_hl_p[cum_hc].median()).mean()  * 100
    ry = (sf_work[peak_hc].dropna() < df_hl_p[peak_hc].median()).mean() * 100
    st.caption(f"Percentil **{rx:.0f}** en acumulada · **{ry:.0f}** en caudal pico vs universo {fluid_prod}.")

with st.expander("⭐ Líderes del cuadrante Alto-Alto", expanded=False):
    render_leaders(sf_work, cum_hc, peak_hc, cum_lbl_w, peak_lbl_w,
                   fluid_prod, "p1a")

st.divider()

# ── Chart 1b: Qw_pico vs Agua acumulada ──────────────────────────────────────
st.markdown(f"#### Cuadrante: {qw_lbl_w} vs {wat_lbl_w}")
sf_w   = sf_work.dropna(subset=["Qw_peak", "water_cum"])
df_hl_w = sf_w[sf_w["sigla"].isin(df_hl_p["sigla"])] if not df_hl_p.empty else pd.DataFrame(columns=sf_w.columns)

fig1b = build_quadrant_chart(
    df_all=sf_w, df_highlight=df_hl_w,
    x_col="water_cum", y_col="Qw_peak",
    x_label=wat_lbl_w, y_label=qw_lbl_w,
    title=f"Benchmark {fluid_prod}: Qw Pico vs Agua Acumulada ({interval_lbl})"
          + (" — por etapa" if normalize_stages else ""),
    highlight_label=hl_lbl_p,
    highlight_color="#E63946" if is_oil else "#1565C0",
    log_x=log_x_p, log_y=log_y_p,
    color_by_year=color_by_year_prod,
)
st.plotly_chart(fig1b, use_container_width=True)

with st.expander("⭐ Líderes del cuadrante Alto-Alto (Qw vs Agua)", expanded=False):
    render_leaders(sf_w, "water_cum", "Qw_peak", wat_lbl_w, qw_lbl_w,
                   fluid_prod, "p1b")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BENCHMARK DE COMPLETACIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🔩 Benchmark de Completación", divider="blue")

st.markdown("""
Compará la completación de cualquier pozo, empresa o área vs el universo con datos de fractura.  
Las **acumuladas de producción** siguen la misma lógica de TEF acumulado (Watchlist).
""")

# Build frac + peaks dataset
peaks_all = (
    data_filtered.groupby("sigla")
    .agg(
        Qo_peak        =("oil_rate",        "max"),
        Qg_peak        =("gas_rate",        "max"),
        Qw_peak        =("water_rate",      "max"),
        empresaNEW     =("empresaNEW",      "first"),
        areayacimiento =("areayacimiento",  "first"),
        tipopozoNEW    =("tipopozoNEW",     "first"),
        start_year     =("anio",            "min"),
    )
    .reset_index()
)

df_comp_base = df_frac.merge(peaks_all, on="sigla", how="inner").drop_duplicates(subset="sigla")

# Attach all interval cumulative volumes
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

# Variable catalog
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

# Axis selectors
ca1, ca2 = st.columns(2)
xvl = ca1.selectbox("Variable eje X:", list(COMP_VARS.keys()), index=0, key="xvc")
yvl = ca2.selectbox("Variable eje Y:", list(COMP_VARS.keys()), index=6, key="yvc")
xv  = COMP_VARS[xvl]
yv  = COMP_VARS[yvl]

cv1, cv2, cv3 = st.columns(3)
fluid_c = cv1.radio("Tipo de Fluido:", ["Todos", "Petrolífero", "Gasífero"],
                     horizontal=True, key="fluid_c")
color_by_year_comp = cv2.checkbox(
    "🎨 Colorear por año de campaña", key="cby_comp",
    help="Incompatible con resaltado múltiple.",
)
log_xc = cv3.checkbox("Log eje X", key="lxc")
log_yc = cv3.checkbox("Log eje Y", key="lyc")

df_c = df_comp_base.copy()
if fluid_c != "Todos":
    df_c = df_c[df_c["tipopozoNEW"] == fluid_c]
df_c = df_c.dropna(subset=[xv, yv])

if df_c.empty:
    st.warning("No hay datos con ambas variables seleccionadas. Probá otra combinación.")
    st.stop()

st.markdown("#### 🎯 Filtro de Resaltado")
if color_by_year_comp:
    st.info("ℹ️ Modo 'Colorear por año' activo. El resaltado de selección múltiple está desactivado.")
    df_hl_c  = pd.DataFrame(columns=df_c.columns)
    hl_lbl_c = "Selección"
else:
    hl_mask_c, hl_lbl_c = multiselect_filter(df_c, "comp")
    df_hl_c = df_c[hl_mask_c]

fig2 = build_quadrant_chart(
    df_all=df_c, df_highlight=df_hl_c,
    x_col=xv, y_col=yv,
    x_label=xvl, y_label=yvl,
    title=f"Benchmark Completación — {xvl} vs {yvl}",
    highlight_label=hl_lbl_c,
    highlight_color="#FF6B35",
    log_x=log_xc, log_y=log_yc,
    color_by_year=color_by_year_comp,
)
st.plotly_chart(fig2, use_container_width=True)

if not df_hl_c.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric(xvl,        f"{df_hl_c[xv].median():,.1f}")
    c2.metric(yvl,        f"{df_hl_c[yv].median():,.1f}")
    c3.metric("N° Pozos", len(df_hl_c))
    rx = (df_c[xv].dropna() < df_hl_c[xv].median()).mean() * 100
    ry = (df_c[yv].dropna() < df_hl_c[yv].median()).mean() * 100
    st.caption(f"Percentil **{rx:.0f}** en {xvl} · **{ry:.0f}** en {yvl} vs universo de completación.")
    if len(df_hl_c) > 1:
        with st.expander("Ver detalle de pozos seleccionados"):
            dcols = [c for c in ["sigla", "empresaNEW", "areayacimiento",
                                  "start_year", xv, yv] if c in df_hl_c.columns]
            ren   = {"sigla": "Pozo", "empresaNEW": "Empresa", "areayacimiento": "Área",
                     "start_year": "Campaña", xv: xvl, yv: yvl}
            st.dataframe(
                df_hl_c[dcols].rename(columns=ren)
                .sort_values(yvl, ascending=False).reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

with st.expander("⭐ Líderes del cuadrante Alto-Alto (Completación)", expanded=False):
    render_leaders(df_c, xv, yv, xvl, yvl, fluid_c, "comp1")


# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
with st.expander("ℹ️ Metodología y lectura de los gráficos"):
    st.markdown("""
    **Acumuladas de producción**
    - Replica la hoja **Watchlist**: suma mensual de `prod_pet`, `prod_gas`, `prod_agua` y `tef` por pozo.
    - Solo ingresan al universo pozos con **TEF acumulado ≥ intervalo elegido** (180d / 365d / 1825d).

    **Cuadrantes P10 / P50 / P90** (convención hidrocarburos)
    | Percentil | Estadístico | Interpretación |
    |-----------|-------------|----------------|
    | **P10**   | Q90         | Optimista — valor alto |
    | **P50**   | Mediana     | Referencia del universo |
    | **P90**   | Q10         | Conservador — valor bajo |

    Líneas **sólidas** = P50 · **guionadas** = P10 · **punteadas** = P90.

    **Modos de visualización**
    - *Gris uniforme*: todos los pozos del universo sin distinción.
    - *Colorear por campaña*: cada año de inicio recibe un color distinto → permite ver evolución histórica.
      Mutuamente exclusivo con el resaltado múltiple.
    - *Normalizar por etapas*: divide caudal y acumulada por la cantidad de fracturas de cada pozo.
      Solo quedan los pozos con datos de fractura disponibles.

    **Filtro de resaltado múltiple**
    - Multiselect en cascada: elegís empresa(s) → las áreas disponibles se filtran → luego las siglas.
    - Podés combinar libremente: varias empresas, varias áreas, varios pozos.
    - Prioridad de resaltado: Siglas > Áreas > Empresas.
    - *Incompatible con 'Colorear por campaña'*: cuando uno está activo el otro se desactiva.

    **Gráfico Qw vs Agua**
    - Eje Y = **Qw pico** (caudal máximo de agua, en m³/d), no el caudal de hidrocarburo.
    - Permite identificar pozos con alta relación agua/hidrocarburo.

    **Auto-zoom en escala log**
    - Al activar log en un eje, el rango visible se ajusta automáticamente al percentil 1–99
      de los valores positivos, evitando que pocos outliers distorsionen la escala.

    **Heatmap de consistencia Alto-Alto**
    - % de pozos de cada Empresa × Área en el cuadrante Alto-Alto.
    - Solo combinaciones con ≥ 2 pozos. Verde intenso = dominancia estructural.
    """)