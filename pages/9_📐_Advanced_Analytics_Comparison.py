"""
9_📐_Análisis_Comparativo_Avanzado.py

Benchmark de Producción y Completación — análisis de cuadrantes P10/P50/P90.

Cambios v2
──────────
- Acumulada calculada igual que Watchlist: TEF acumulado >= intervalo de días.
  El usuario elige 180d, 1 año (365d) o 5 años (1825d).
- Benchmark de Completación incluye acumuladas a los mismos intervalos.
- Filtro cascada: empresa → área (filtrada por empresa) → sigla.
- Al pie de cada cuadrante: tabla de líderes Alto-Alto + heatmap de consistencia.
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — CUMULATIVE PRODUCTION  (same logic as Watchlist)
# ══════════════════════════════════════════════════════════════════════════════

def cumulative_at_interval(df: pd.DataFrame, interval_days: int) -> pd.DataFrame:
    """
    Mirrors Watchlist exactly:
      - sums prod_pet, prod_gas, prod_agua and tef per well
      - keeps wells where tef_sum >= interval_days
    Returns one row per sigla.
    """
    grp = (
        df.groupby("sigla")
        .agg(
            oil_cum       =("prod_pet",        "sum"),
            gas_cum       =("prod_gas",         "sum"),
            water_cum     =("prod_agua",        "sum"),
            tef_cum       =("tef",              "sum"),
            empresaNEW    =("empresaNEW",       "first"),
            areayacimiento=("areayacimiento",   "first"),
            tipopozoNEW   =("tipopozoNEW",      "first"),
        )
        .reset_index()
    )
    return grp[grp["tef_cum"] >= interval_days].copy()


def build_prod_summary(data_filtered: pd.DataFrame, interval_days: int) -> pd.DataFrame:
    """Cumulative volumes + peak rates, one row per sigla."""
    cum   = cumulative_at_interval(data_filtered, interval_days)
    peaks = (
        data_filtered.groupby("sigla")
        .agg(Qo_peak=("oil_rate", "max"), Qg_peak=("gas_rate", "max"))
        .reset_index()
    )
    return cum.merge(peaks, on="sigla", how="left")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — QUADRANT CHART
# ══════════════════════════════════════════════════════════════════════════════

def pct_q(s: pd.Series, q: float) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.quantile(q)) if not clean.empty else np.nan


def _clip(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return df[col]
    lo, hi = np.nanpercentile(s, [0.5, 99.5])
    return df[col].clip(lo, hi)


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
    for x_val, lbl, pos in [(x_p10, f"P10 {xl}", "bottom right"),
                             (x_p90, f"P90 {xl}", "bottom right")]:
        dash = "dash" if "P10" in lbl else "dot"
        fig.add_vline(x=x_val, line_dash=dash, **kw,
                      annotation_text=lbl, annotation_position=pos,
                      annotation_font=dict(size=8, color="rgba(80,80,80,0.68)"))
    for y_val, lbl, pos in [(y_p10, f"P10 {yl}", "top left"),
                             (y_p90, f"P90 {yl}", "top left")]:
        dash = "dash" if "P10" in lbl else "dot"
        fig.add_hline(y=y_val, line_dash=dash, **kw,
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

    df_all = df_all.copy()
    df_all[x_col] = _clip(df_all, x_col)
    df_all[y_col] = _clip(df_all, y_col)

    x_p10 = pct_q(df_all[x_col], 0.90)
    x_p50 = pct_q(df_all[x_col], 0.50)
    x_p90 = pct_q(df_all[x_col], 0.10)
    y_p10 = pct_q(df_all[y_col], 0.90)
    y_p50 = pct_q(df_all[y_col], 0.50)
    y_p90 = pct_q(df_all[y_col], 0.10)
    x_min, x_max = df_all[x_col].min(), df_all[x_col].max()
    y_min, y_max = df_all[y_col].min(), df_all[y_col].max()

    # hover columns always present
    cd_cols = ["sigla", "empresaNEW", "areayacimiento"]
    cd_cols = [c for c in cd_cols if c in df_all.columns]

    htmpl = (
        "<b>%{customdata[0]}</b><br>"
        + (f"Empresa: %{{customdata[1]}}<br>" if len(cd_cols) > 1 else "")
        + (f"Área: %{{customdata[2]}}<br>"    if len(cd_cols) > 2 else "")
        + f"{x_label}: %{{x:,.0f}}<br>{y_label}: %{{y:,.0f}}"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_all[x_col], y=df_all[y_col],
        mode="markers", name="Todos los pozos",
        marker=dict(color="rgba(175,175,175,0.30)", size=5, line=dict(width=0)),
        hovertemplate=htmpl + "<extra>Universo</extra>",
        customdata=df_all[cd_cols].values,
    ))

    if not df_highlight.empty:
        df_hl = df_highlight.copy()
        df_hl[x_col] = _clip(df_hl, x_col)
        df_hl[y_col] = _clip(df_hl, y_col)
        fig.add_trace(go.Scatter(
            x=df_hl[x_col], y=df_hl[y_col],
            mode="markers", name=highlight_label,
            marker=dict(color=highlight_color, size=9,
                        line=dict(width=1.3, color="white")),
            hovertemplate=htmpl + f"<extra>{highlight_label}</extra>",
            customdata=df_hl[cd_cols].values,
        ))

    add_quadrant_lines(fig, x_p50, y_p50, x_p10, y_p10, x_p90, y_p90, x_label, y_label)
    add_quadrant_labels(fig, x_p50, y_p50, x_max, y_max, x_min, y_min)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title=x_label, type="log" if log_x else "linear",
                   showgrid=True, gridcolor="rgba(200,200,200,0.28)", zeroline=False),
        yaxis=dict(title=y_label, type="log" if log_y else "linear",
                   showgrid=True, gridcolor="rgba(200,200,200,0.28)", zeroline=False),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=530,
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
    df_aa  = df_universe[(df_universe[x_col] >= x_p50) & (df_universe[y_col] >= y_p50)].copy()

    if df_aa.empty:
        st.caption("No hay pozos en el cuadrante Alto-Alto.")
        return

    pct_aa_univ = len(df_aa) / len(df_universe) * 100
    st.markdown(
        f"**{len(df_aa)} pozos** ({pct_aa_univ:.1f}% del universo {fluid_label}) "
        f"están en el cuadrante ⭐ Alto–Alto."
    )

    # ── Top wells table ───────────────────────────────────────────────────────
    show_cols  = [c for c in ["sigla", "empresaNEW", "areayacimiento", x_col, y_col] if c in df_aa.columns]
    rename_map = {"sigla": "Pozo", "empresaNEW": "Empresa",
                  "areayacimiento": "Área", x_col: x_label, y_col: y_label}
    st.markdown("**Top 15 pozos del cuadrante Alto–Alto** (ordenados por eje Y)")
    st.dataframe(
        df_aa[show_cols].rename(columns=rename_map)
        .sort_values(y_label, ascending=False).head(15).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

    # ── Consistency heatmap ───────────────────────────────────────────────────
    if "empresaNEW" not in df_aa.columns or "areayacimiento" not in df_aa.columns:
        return

    aa_cnt  = (df_aa.groupby(["empresaNEW", "areayacimiento"])["sigla"].nunique()
               .reset_index(name="aa_n"))
    tot_cnt = (df_universe.groupby(["empresaNEW", "areayacimiento"])["sigla"].nunique()
               .reset_index(name="tot_n"))
    heat    = tot_cnt.merge(aa_cnt, on=["empresaNEW", "areayacimiento"], how="left")
    heat["aa_n"]   = heat["aa_n"].fillna(0)
    heat["pct"]    = (heat["aa_n"] / heat["tot_n"] * 100).round(1)
    heat           = heat[heat["tot_n"] >= 2]

    if heat.empty:
        st.caption("No hay suficientes datos para el heatmap (mínimo 2 pozos por combinación).")
        return

    pivot = heat.pivot_table(index="empresaNEW", columns="areayacimiento",
                              values="pct", fill_value=np.nan)
    row_ord = pivot.mean(axis=1).sort_values(ascending=False).index
    col_ord = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot   = pivot.loc[row_ord, col_ord]
    txt_v   = pivot.map(lambda v: f"{v:.0f}%" if pd.notna(v) else "—").values

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
    st.caption("Verde intenso = alta consistencia. Solo combinaciones con ≥ 2 pozos en el universo.")
    st.plotly_chart(fig_h, use_container_width=True, key=f"heat_{section_key}")

    # ── Empresa ranking ───────────────────────────────────────────────────────
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
# HELPERS — CASCADED FILTER  (empresa → area → sigla)
# ══════════════════════════════════════════════════════════════════════════════

def cascaded_filter(df: pd.DataFrame, key_prefix: str):
    """
    Three-column cascaded selectors.
    Returns (mask: pd.Series, label: str).
    """
    col1, col2, col3 = st.columns(3)

    companies = ["(Todas)"] + sorted(df["empresaNEW"].dropna().unique())
    sel_co    = col1.selectbox("Empresa:", companies, key=f"{key_prefix}_co")

    df_area = df if sel_co == "(Todas)" else df[df["empresaNEW"] == sel_co]
    areas   = ["(Todas)"] + sorted(df_area["areayacimiento"].dropna().unique())
    sel_ar  = col2.selectbox("Área de Yacimiento:", areas, key=f"{key_prefix}_ar")

    df_sig  = df_area if sel_ar == "(Todas)" else df_area[df_area["areayacimiento"] == sel_ar]
    siglas  = ["(Ninguna)"] + sorted(df_sig["sigla"].dropna().unique())
    sel_sig = col3.selectbox("Sigla (pozo):", siglas, key=f"{key_prefix}_sig")

    # Priority: sigla > area > company
    if sel_sig != "(Ninguna)":
        mask  = df["sigla"] == sel_sig
        label = sel_sig
    elif sel_ar != "(Todas)":
        mask  = df["areayacimiento"] == sel_ar
        label = sel_ar
    elif sel_co != "(Todas)":
        mask  = df["empresaNEW"] == sel_co
        label = sel_co
    else:
        mask  = pd.Series(False, index=df.index)
        label = "Selección"

    return mask, label


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE & COMMON DATA
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

# Pre-build cumulative summaries for all intervals (cached via function argument hash)
@st.cache_data
def all_cum_summaries(_df):
    """Returns dict {label: DataFrame with oil_cum, gas_cum, water_cum, tef_cum + meta}"""
    out = {}
    for lbl, days in INTERVALS.items():
        c = cumulative_at_interval(_df, days)
        out[lbl] = c
    return out

# We pass a hashable proxy — use the shape + a sample to bust cache on reload
_cache_key = (len(data_filtered), int(data_filtered["prod_pet"].sum()))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BENCHMARK DE PRODUCCIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🛢️ Benchmark de Producción", divider="blue")

st.markdown("""
Todos los pozos con **TEF acumulado ≥ intervalo seleccionado** integran el universo de referencia
(misma lógica que hoja **Watchlist**). Los cuadrantes P10/P50/P90 se calculan sobre ese universo.
""")

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl1, ctrl2 = st.columns(2)
fluid_prod    = ctrl1.radio("Tipo de Fluido:", ["Petrolífero", "Gasífero"],
                             horizontal=True, key="fluid_prod")
interval_lbl  = ctrl2.selectbox("Intervalo de acumulada:",
                                  list(INTERVALS.keys()), index=1, key="int_prod")
interval_days = INTERVALS[interval_lbl]

is_oil     = fluid_prod == "Petrolífero"
peak_col   = "Qo_peak"   if is_oil else "Qg_peak"
cum_col    = "oil_cum"   if is_oil else "gas_cum"
peak_label = "Qo Pico (m³/d)"             if is_oil else "Qg Pico (km³/d)"
cum_label  = f"Np @ {interval_lbl} (m³)"  if is_oil else f"Gp @ {interval_lbl} (km³)"
water_label = f"Wp @ {interval_lbl} (m³)"

with st.spinner(f"Calculando acumuladas @ {interval_lbl}…"):
    prod_sum = build_prod_summary(data_filtered, interval_days)

summary_fluid = prod_sum[prod_sum["tipopozoNEW"] == fluid_prod].dropna(subset=[peak_col, cum_col])

if summary_fluid.empty:
    st.warning(f"No hay pozos con TEF ≥ {interval_days} días para {fluid_prod}.")
    st.stop()

# ── Cascaded filter ───────────────────────────────────────────────────────────
st.markdown("#### 🎯 Filtro de Resaltado")
hl_mask_p, hl_label_p = cascaded_filter(summary_fluid, "prod")
df_hl_p = summary_fluid[hl_mask_p]

log1, log2 = st.columns(2)
log_x_p = log1.checkbox("Log eje X (acumulada)", key="lx_p")
log_y_p = log2.checkbox("Log eje Y (caudal pico)", key="ly_p")

# ── Chart 1a ──────────────────────────────────────────────────────────────────
st.markdown(f"#### {peak_label} vs {cum_label}")
fig1a = build_quadrant_chart(
    df_all=summary_fluid, df_highlight=df_hl_p,
    x_col=cum_col, y_col=peak_col,
    x_label=cum_label, y_label=peak_label,
    title=f"Benchmark {fluid_prod}: Caudal Pico vs Producción Acumulada ({interval_lbl})",
    highlight_label=hl_label_p,
    highlight_color="#E63946" if is_oil else "#1565C0",
    log_x=log_x_p, log_y=log_y_p,
)
st.plotly_chart(fig1a, use_container_width=True)

if not df_hl_p.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric(peak_label,  f"{df_hl_p[peak_col].median():,.0f}")
    c2.metric(cum_label,   f"{df_hl_p[cum_col].median():,.0f}")
    c3.metric("N° Pozos",  len(df_hl_p))
    rx = (summary_fluid[cum_col].dropna()  < df_hl_p[cum_col].median()).mean()  * 100
    ry = (summary_fluid[peak_col].dropna() < df_hl_p[peak_col].median()).mean() * 100
    st.caption(f"Percentil **{rx:.0f}** en {cum_label} · percentil **{ry:.0f}** en {peak_label} vs universo {fluid_prod}.")

with st.expander("⭐ Líderes del cuadrante Alto-Alto", expanded=False):
    render_leaders(summary_fluid, cum_col, peak_col, cum_label, peak_label,
                   fluid_prod, "p1a")

st.divider()

# ── Chart 1b: Q_pico vs Agua ──────────────────────────────────────────────────
sum_w   = summary_fluid.dropna(subset=[peak_col, "water_cum"])
df_hl_w = sum_w[sum_w["sigla"].isin(df_hl_p["sigla"])]

st.markdown(f"#### {peak_label} vs {water_label}")
fig1b = build_quadrant_chart(
    df_all=sum_w, df_highlight=df_hl_w,
    x_col="water_cum", y_col=peak_col,
    x_label=water_label, y_label=peak_label,
    title=f"Benchmark {fluid_prod}: Caudal Pico vs Agua Acumulada ({interval_lbl})",
    highlight_label=hl_label_p,
    highlight_color="#E63946" if is_oil else "#1565C0",
    log_x=log_x_p, log_y=log_y_p,
)
st.plotly_chart(fig1b, use_container_width=True)

with st.expander("⭐ Líderes del cuadrante Alto-Alto (Q_pico vs Agua)", expanded=False):
    render_leaders(sum_w, "water_cum", peak_col, water_label, peak_label,
                   fluid_prod, "p1b")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BENCHMARK DE COMPLETACIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("🔩 Benchmark de Completación", divider="blue")

st.markdown("""
Compará la completación de cualquier pozo, empresa o área vs el universo con datos de fractura.  
Las **acumuladas de producción** disponibles en los ejes usan la misma lógica de TEF acumulado
que el Benchmark de Producción.
""")

# Build per-well peak rates + metadata
peaks_all = (
    data_filtered.groupby("sigla")
    .agg(
        Qo_peak       =("oil_rate",       "max"),
        Qg_peak       =("gas_rate",       "max"),
        empresaNEW    =("empresaNEW",     "first"),
        areayacimiento=("areayacimiento", "first"),
        tipopozoNEW   =("tipopozoNEW",    "first"),
    )
    .reset_index()
)

# Merge frac + peaks
df_comp_base = df_frac.merge(peaks_all, on="sigla", how="inner").drop_duplicates(subset="sigla")

# Attach cumulative volumes for each interval
for lbl, days in INTERVALS.items():
    c_int = cumulative_at_interval(data_filtered, days)[
        ["sigla", "oil_cum", "gas_cum", "water_cum"]
    ].rename(columns={
        "oil_cum":   f"oil_{lbl}",
        "gas_cum":   f"gas_{lbl}",
        "water_cum": f"wtr_{lbl}",
    })
    df_comp_base = df_comp_base.merge(c_int, on="sigla", how="left")

# Derived completion metrics
df_comp_base["fracspacing"]       = df_comp_base["longitud_rama_horizontal_m"] / df_comp_base["cantidad_fracturas"]
df_comp_base["prop_x_etapa"]      = df_comp_base["arena_total_tn"] / df_comp_base["cantidad_fracturas"]
df_comp_base["proppant_intensity"] = df_comp_base["arena_total_tn"] / df_comp_base["longitud_rama_horizontal_m"]
df_comp_base["AS_x_vol"]          = df_comp_base["arena_total_tn"] / (df_comp_base["agua_inyectada_m3"].replace(0, np.nan) / 1000)
df_comp_base["Qo_peak_x_etapa"]   = df_comp_base["Qo_peak"] / df_comp_base["cantidad_fracturas"]
df_comp_base["Qg_peak_x_etapa"]   = df_comp_base["Qg_peak"] / df_comp_base["cantidad_fracturas"]
df_comp_base = df_comp_base.replace([np.inf, -np.inf], np.nan)

# ── Variable catalog ──────────────────────────────────────────────────────────
COMP_VARS: dict = {
    "Longitud Rama (m)":               "longitud_rama_horizontal_m",
    "Cantidad de Etapas":              "cantidad_fracturas",
    "Arena Total (tn)":                "arena_total_tn",
    "Arena Nacional (tn)":             "arena_bombeada_nacional_tn",
    "Arena Importada (tn)":            "arena_bombeada_importada_tn",
    "Agua Inyectada (m³)":             "agua_inyectada_m3",
    "Fracspacing (m)":                 "fracspacing",
    "Propante x Etapa (tn/etapa)":     "prop_x_etapa",
    "Intensidad Propante (tn/m)":      "proppant_intensity",
    "AS x Vol. Inyectado (tn/1000m³)": "AS_x_vol",
    "Qo Pico (m³/d)":                  "Qo_peak",
    "Qg Pico (km³/d)":                 "Qg_peak",
    "Qo Pico x Etapa (m³/d/etapa)":   "Qo_peak_x_etapa",
    "Qg Pico x Etapa (km³/d/etapa)":  "Qg_peak_x_etapa",
}
for lbl in INTERVALS.keys():
    COMP_VARS[f"Np @ {lbl} (m³)"]  = f"oil_{lbl}"
    COMP_VARS[f"Gp @ {lbl} (km³)"] = f"gas_{lbl}"
    COMP_VARS[f"Wp @ {lbl} (m³)"]  = f"wtr_{lbl}"

# ── Axis + fluid selectors ────────────────────────────────────────────────────
ca1, ca2 = st.columns(2)
xvl = ca1.selectbox("Variable eje X:", list(COMP_VARS.keys()), index=0,  key="xvc")
yvl = ca2.selectbox("Variable eje Y:", list(COMP_VARS.keys()), index=6, key="yvc")
xv  = COMP_VARS[xvl]
yv  = COMP_VARS[yvl]

fluid_c = st.radio("Tipo de Fluido:", ["Todos", "Petrolífero", "Gasífero"],
                    horizontal=True, key="fluid_c")
df_c = df_comp_base.copy()
if fluid_c != "Todos":
    df_c = df_c[df_c["tipopozoNEW"] == fluid_c]
df_c = df_c.dropna(subset=[xv, yv])

if df_c.empty:
    st.warning("No hay datos con ambas variables seleccionadas. Probá otra combinación.")
    st.stop()

# ── Cascaded filter ───────────────────────────────────────────────────────────
st.markdown("#### 🎯 Filtro de Resaltado")
hl_mask_c, hl_label_c = cascaded_filter(df_c, "comp")
df_hl_c = df_c[hl_mask_c]

cl1, cl2 = st.columns(2)
log_xc = cl1.checkbox("Log eje X", key="lxc")
log_yc = cl2.checkbox("Log eje Y", key="lyc")

# ── Chart 2 ───────────────────────────────────────────────────────────────────
fig2 = build_quadrant_chart(
    df_all=df_c, df_highlight=df_hl_c,
    x_col=xv, y_col=yv,
    x_label=xvl, y_label=yvl,
    title=f"Benchmark Completación — {xvl} vs {yvl}",
    highlight_label=hl_label_c,
    highlight_color="#FF6B35",
    log_x=log_xc, log_y=log_yc,
)
st.plotly_chart(fig2, use_container_width=True)

if not df_hl_c.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric(xvl,        f"{df_hl_c[xv].median():,.1f}")
    c2.metric(yvl,        f"{df_hl_c[yv].median():,.1f}")
    c3.metric("N° Pozos", len(df_hl_c))
    rx = (df_c[xv].dropna() < df_hl_c[xv].median()).mean() * 100
    ry = (df_c[yv].dropna() < df_hl_c[yv].median()).mean() * 100
    st.caption(f"Percentil **{rx:.0f}** en {xvl} · percentil **{ry:.0f}** en {yvl} vs universo de completación.")

    if len(df_hl_c) > 1:
        with st.expander("Ver detalle de pozos seleccionados"):
            dcols = [c for c in ["sigla", "empresaNEW", "areayacimiento", xv, yv] if c in df_hl_c.columns]
            ren   = {"sigla": "Pozo", "empresaNEW": "Empresa",
                     "areayacimiento": "Área", xv: xvl, yv: yvl}
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
    - Replica la hoja **Watchlist**: suma de `prod_pet`, `prod_gas`, `prod_agua` y `tef` por pozo.
    - Solo ingresan al universo los pozos con **TEF acumulado ≥ intervalo elegido**.
    - Disponible en 3 horizontes: 180 días, 1 año (365d) y 5 años (1825d).

    **Cuadrantes y percentiles (convención hidrocarburos)**
    | Percentil | Valor estadístico | Interpretación |
    |-----------|-------------------|----------------|
    | **P10**   | Q90 estadístico   | Pozo optimista (valor alto) |
    | **P50**   | Mediana           | Referencia del universo |
    | **P90**   | Q10 estadístico   | Pozo conservador (valor bajo) |

    - Líneas **sólidas** = P50 (cruce de cuadrantes).
    - Líneas **guionadas** = P10. Líneas **punteadas** = P90.

    **Filtro en cascada (Empresa → Área → Sigla)**
    - Al elegir empresa, las áreas disponibles se filtran automáticamente.
    - Al elegir área, las siglas se filtran también.
    - Prioridad de resaltado: Sigla > Área > Empresa.

    **Heatmap de consistencia Alto-Alto**
    - % de pozos de cada combinación Empresa × Área en el cuadrante Alto-Alto.
    - Sólo combinaciones con ≥ 2 pozos en el universo.
    - Verde intenso = el desempeño superior es **estructural**, no un outlier.

    **Benchmark de Completación**
    - Filtros de calidad aplicados (mismos que el resto del reporte):
      longitud rama > 100 m · etapas > 6 · arena total > 100 tn.
    - Las acumuladas disponibles en los ejes siguen el mismo criterio de TEF.
    """)