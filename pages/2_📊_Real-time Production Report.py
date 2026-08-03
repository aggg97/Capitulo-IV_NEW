import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta
from PIL import Image

from utils import BARRELS_PER_M3, COMPANY_REPLACEMENTS


# ── Session state guard ───────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.warning("⚠️ No se han cargado los datos. Por favor, volvé a la Página Principal.")
    st.stop()

data_sorted = st.session_state["df"].copy()
data_sorted["empresaNEW"] = data_sorted["empresa"].replace(COMPANY_REPLACEMENTS)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image(Image.open("Vaca Muerta rig.png"))


# ── Header ────────────────────────────────────────────────────────────────────

st.header(":blue[Producción General — Vaca Muerta]")


# ── Base filter ───────────────────────────────────────────────────────────────

data_filtered = data_sorted[data_sorted["tef"] > 0]


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS POR EMPRESA
# ══════════════════════════════════════════════════════════════════════════════

# ── Company aggregation ───────────────────────────────────────────────────────

company_summary = (
    data_filtered
    .groupby(["empresaNEW", "date"])
    .agg(total_gas_rate=("gas_rate", "sum"), total_oil_rate=("oil_rate", "sum"))
    .reset_index()
)

top_companies = set(
    company_summary.groupby("empresaNEW")["total_oil_rate"].sum().nlargest(10).index
)

company_summary["empresaNEW"] = company_summary["empresaNEW"].apply(
    lambda x: x if x in top_companies else "Otros"
)

company_summary_aggregated = (
    company_summary
    .groupby(["empresaNEW", "date"])
    .agg(total_gas_rate=("total_gas_rate", "sum"), total_oil_rate=("total_oil_rate", "sum"))
    .reset_index()
)

# ── KPIs del período consolidado ──────────────────────────────────────────────

st.subheader("⚡ Producción al Último Período Consolidado")

data_filtered = data_sorted[data_sorted["tef"] > 0]
latest_date_non_official = data_filtered["date"].max()
latest_date = latest_date_non_official - relativedelta(months=1)
latest_data = data_filtered[data_filtered["date"] == latest_date]

total_gas_rate_rounded = round(latest_data["gas_rate"].sum() / 1000, 1)
total_oil_rate_rounded = round(latest_data["oil_rate"].sum() / 1000, 1)
oil_rate_bpd_rounded   = round(total_oil_rate_rounded * BARRELS_PER_M3, 1)

col1, col2, col3 = st.columns(3)
col1.metric(label="🔥 Caudal de Gas (MMm³/d)",     value=total_gas_rate_rounded)
col2.metric(label="🛢️ Caudal de Petróleo (km³/d)", value=total_oil_rate_rounded)
col3.metric(label="🛢️ Caudal de Petróleo (kbpd)",  value=oil_rate_bpd_rounded)

st.divider()

# ── Gas por empresa ───────────────────────────────────────────────────────────

SEMILOG_HELP = (
    "Activar esta escala facilita detectar tendencias lineales en los datos, "
    "lo que equivale a identificar patrones de crecimiento o decline exponencial "
    "en la producción de manera más efectiva."
)

fig_gas_company = px.area(
    company_summary_aggregated,
    x="date", y="total_gas_rate", color="empresaNEW",
    title="Caudal de Gas por Empresa",
    labels={"date": "Fecha", "total_gas_rate": "Caudal de Gas (km³/d)", "empresaNEW": "Empresa"},
)
fig_gas_company.update_layout(
    legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=10))
)

log_scale_gas = st.checkbox("Escala semilog — Gas por Empresa", help=SEMILOG_HELP)
if log_scale_gas:
    fig_gas_company.update_layout(yaxis=dict(type="log", dtick=1))
st.plotly_chart(fig_gas_company, use_container_width=True)

# ── Petróleo por empresa ──────────────────────────────────────────────────────

fig_oil_company = px.area(
    company_summary_aggregated,
    x="date", y="total_oil_rate", color="empresaNEW",
    title="Caudal de Petróleo por Empresa",
    labels={"date": "Fecha", "total_oil_rate": "Caudal de Petróleo (m³/d)", "empresaNEW": "Empresa"},
)
fig_oil_company.update_layout(
    legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=10))
)

log_scale_oil = st.checkbox("Escala semilog — Petróleo por Empresa", help=SEMILOG_HELP)
if log_scale_oil:
    fig_oil_company.update_layout(yaxis=dict(type="log", dtick=1))
st.plotly_chart(fig_oil_company, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS POR CAMPAÑA (en expander)
# ══════════════════════════════════════════════════════════════════════════════

well_start_year = (
    data_filtered.groupby("sigla")["anio"].min()
    .reset_index()
    .rename(columns={"anio": "start_year"})
)

yearly_summary = (
    data_filtered
    .merge(well_start_year, on="sigla")
    .groupby(["start_year", "date"])
    .agg(total_gas_rate=("gas_rate", "sum"), total_oil_rate=("oil_rate", "sum"))
    .reset_index()
    .query("total_gas_rate > 0 and total_oil_rate > 0")
)

with st.expander("Ver producción por campaña", expanded=False):
    fig_gas_year = px.area(
        yearly_summary,
        x="date", y="total_gas_rate", color="start_year",
        title="Caudal de Gas por Campaña",
        labels={"date": "Fecha", "total_gas_rate": "Caudal de Gas (km³/d)", "start_year": "Campaña"},
    )
    fig_gas_year.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=10))
    )
    st.plotly_chart(fig_gas_year, use_container_width=True)

    fig_oil_year = px.area(
        yearly_summary,
        x="date", y="total_oil_rate", color="start_year",
        title="Caudal de Petróleo por Campaña",
        labels={"date": "Fecha", "total_oil_rate": "Caudal de Petróleo (m³/d)", "start_year": "Campaña"},
    )
    fig_oil_year.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=10))
    )
    st.plotly_chart(fig_oil_year, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS OPERATIVAS AVANZADAS
# ══════════════════════════════════════════════════════════════════════════════

LEGEND_BOTTOM = dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5)
COMPANY_PALETTE = px.colors.qualitative.Set2

st.divider()
st.subheader("📊 Métricas Operativas Avanzadas", divider="blue")
st.caption(
    "Semáforo YoY, variación porcentual interanual, evolución de GOR/WOR/WGR "
    "y market share por operador. Se evalúa la variación en caudal respecto al mismo "
    "mes del año anterior."
)


# ── Filtros: modo área o modo empresa ────────────────────────────────────────

with st.expander("🔍 Filtrar datos (opcional)", expanded=False):
    filter_mode = st.radio(
        "Modo de filtrado:",
        ["Sin filtro (toda la cuenca)", "Filtrar por Área", "Filtrar por Empresa"],
        horizontal=True,
        key="adv_filter_mode",
    )

    adv_sel_areas: list     = []
    adv_sel_company: str | None = None
    adv_sel_sub_areas: list = []

    if filter_mode == "Filtrar por Área":
        all_areas = sorted(data_sorted["areayacimiento"].dropna().unique())
        adv_sel_areas = st.multiselect(
            "Seleccioná una o más áreas de yacimiento:",
            options=all_areas,
            default=[],
            key="adv_area_filter",
        )

    elif filter_mode == "Filtrar por Empresa":
        all_companies = sorted(data_sorted["empresaNEW"].dropna().unique())
        adv_sel_company = st.selectbox(
            "Seleccioná una empresa:",
            options=[""] + all_companies,
            index=0,
            key="adv_company_filter",
        )
        if adv_sel_company:
            areas_for_company = sorted(
                data_sorted.loc[
                    data_sorted["empresaNEW"] == adv_sel_company, "areayacimiento"
                ].dropna().unique()
            )
            adv_sel_sub_areas = st.multiselect(
                f"Áreas operadas por {adv_sel_company} (opcional — vacío = todas):",
                options=areas_for_company,
                default=[],
                key="adv_sub_area_filter",
            )

# ── Aplicar filtros ───────────────────────────────────────────────────────────

_adv_base = data_sorted[data_sorted["tef"] > 0].copy()

if filter_mode == "Filtrar por Área" and adv_sel_areas:
    _adv_base = _adv_base[_adv_base["areayacimiento"].isin(adv_sel_areas)]

elif filter_mode == "Filtrar por Empresa" and adv_sel_company:
    _adv_base = _adv_base[_adv_base["empresaNEW"] == adv_sel_company]
    if adv_sel_sub_areas:
        _adv_base = _adv_base[_adv_base["areayacimiento"].isin(adv_sel_sub_areas)]

if _adv_base.empty:
    st.warning("No hay datos para los filtros seleccionados. Ajustá la selección.")
    st.stop()

# Etiqueta para títulos
_adv_scope = ""
if filter_mode == "Filtrar por Área" and adv_sel_areas:
    _adv_scope = f" — {', '.join(adv_sel_areas)}"
elif filter_mode == "Filtrar por Empresa" and adv_sel_company:
    parts = [adv_sel_company]
    if adv_sel_sub_areas:
        parts.append(", ".join(adv_sel_sub_areas))
    _adv_scope = f" — {' · '.join(parts)}"


# ── Base mensual agregada ─────────────────────────────────────────────────────

monthly = (
    _adv_base
    .groupby("date")
    .agg(
        oil_rate    =("oil_rate",   "sum"),
        gas_rate    =("gas_rate",   "sum"),
        water_rate  =("water_rate", "sum"),
        wells_active=("sigla",      "nunique"),
    )
    .sort_index()
    .reset_index()
)

# Ratios instantáneos (usando rates agregados de la cuenca)
monthly["GOR"] = (monthly["gas_rate"]   / monthly["oil_rate"].replace(0, np.nan) * 1000)
monthly["WOR"] = (monthly["water_rate"] / monthly["oil_rate"].replace(0, np.nan))
monthly["WGR"] = (monthly["water_rate"] / monthly["gas_rate"].replace(0, np.nan) * 1000)
for col in ["GOR", "WOR", "WGR"]:
    monthly[col] = monthly[col].replace([np.inf, -np.inf], np.nan)
    monthly[f"{col}_3m"] = monthly[col].rolling(3, min_periods=1).mean()

# Variaciones YoY
for col in ["oil_rate", "gas_rate", "water_rate", "wells_active"]:
    monthly[f"{col}_yoy"] = monthly[col].pct_change(12) * 100


# ── 1. Semáforo YoY ───────────────────────────────────────────────────────────

st.markdown("### 🚦 Indicadores — Último Período Consolidado")

last     = monthly.dropna(subset=["oil_rate"]).iloc[-1]
last_yoy = monthly.dropna(subset=["oil_rate_yoy"]).iloc[-1] if monthly["oil_rate_yoy"].notna().any() else None

def _yoy_delta(row, col: str):
    """Returns (delta_str, delta_color) for a YoY metric."""
    if row is None or pd.isna(row[col]):
        return None, "off"
    val   = row[col]
    sign  = "+" if val >= 0 else ""
    color = "normal" if val >= 0 else "inverse"
    return f"{sign}{val:.1f}% YoY", color

col1, col2, col3, col4 = st.columns(4)
with col1:
    d, dc = _yoy_delta(last_yoy, "oil_rate_yoy")
    st.metric("🛢️ Qo Total (m³/d)", f"{last['oil_rate']:,.0f}", delta=d, delta_color=dc)
with col2:
    d, dc = _yoy_delta(last_yoy, "gas_rate_yoy")
    st.metric("🔥 Qg Total (km³/d)", f"{last['gas_rate']:,.0f}", delta=d, delta_color=dc)
with col3:
    d, dc = _yoy_delta(last_yoy, "water_rate_yoy")
    st.metric("💧 Qw Total (m³/d)", f"{last['water_rate']:,.0f}", delta=d, delta_color=dc)
with col4:
    d, dc = _yoy_delta(last_yoy, "wells_active_yoy")
    st.metric("🛞 Pozos Activos", f"{int(last['wells_active']):,}", delta=d, delta_color=dc)


# ── 2. Variación YoY — Petróleo ───────────────────────────────────────────────

st.divider()
st.markdown("### 📉 Variación Interanual (YoY) — Petróleo")

monthly_yoy = monthly.dropna(subset=["oil_rate_yoy"]).copy()
monthly_yoy["oil_ma3_yoy"] = monthly_yoy["oil_rate_yoy"].rolling(3, min_periods=1).mean()
monthly_yoy["bar_color_oil"] = monthly_yoy["oil_rate_yoy"].apply(
    lambda v: "#2ecc71" if v >= 0 else "#e74c3c"
)

fig_yoy_oil = go.Figure()
fig_yoy_oil.add_trace(go.Bar(
    x=monthly_yoy["date"],
    y=monthly_yoy["oil_rate_yoy"],
    name="YoY Petróleo",
    marker_color=monthly_yoy["bar_color_oil"],
    opacity=0.80,
    hovertemplate="Fecha: %{x}<br>YoY Petróleo: %{y:.1f}%<extra></extra>",
))
fig_yoy_oil.add_trace(go.Scatter(
    x=monthly_yoy["date"],
    y=monthly_yoy["oil_ma3_yoy"],
    mode="lines",
    name="Media móvil 3m",
    line=dict(color="#1a3a1a", width=1.5, dash="dot"),
    marker=dict(size=3),
    hovertemplate="Fecha: %{x}<br>MA 3m: %{y:.1f}%<extra></extra>",
))
fig_yoy_oil.add_hline(y=0, line_color="rgba(0,0,0,0.25)", line_width=1.5)
fig_yoy_oil.update_layout(
    title=f"Variación YoY — Petróleo{_adv_scope}",
    xaxis_title="Fecha",
    yaxis_title="Variación YoY (%)",
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_yoy_oil, use_container_width=True)


# ── 3. Variación YoY — Gas ────────────────────────────────────────────────────

st.markdown("### 📉 Variación Interanual (YoY) — Gas")

monthly_yoy["gas_ma3_yoy"] = monthly_yoy["gas_rate_yoy"].rolling(3, min_periods=1).mean()
monthly_yoy["bar_color_gas"] = monthly_yoy["gas_rate_yoy"].apply(
    lambda v: "#2ecc71" if v >= 0 else "#e74c3c"
)

fig_yoy_gas = go.Figure()
fig_yoy_gas.add_trace(go.Bar(
    x=monthly_yoy["date"],
    y=monthly_yoy["gas_rate_yoy"],
    name="YoY Gas",
    marker_color=monthly_yoy["bar_color_gas"],
    opacity=0.80,
    hovertemplate="Fecha: %{x}<br>YoY Gas: %{y:.1f}%<extra></extra>",
))
fig_yoy_gas.add_trace(go.Scatter(
    x=monthly_yoy["date"],
    y=monthly_yoy["gas_ma3_yoy"],
    mode="lines",
    name="Media móvil 3m",
    line=dict(color="#3a1a1a", width=1.5, dash="dot"),
    marker=dict(size=3),
    hovertemplate="Fecha: %{x}<br>MA 3m: %{y:.1f}%<extra></extra>",
))
fig_yoy_gas.add_hline(y=0, line_color="rgba(0,0,0,0.25)", line_width=1.5)
fig_yoy_gas.update_layout(
    title=f"Variación YoY — Gas{_adv_scope}",
    xaxis_title="Fecha",
    yaxis_title="Variación YoY (%)",
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_yoy_gas, use_container_width=True)


# ── 4. GOR / WOR / WGR ───────────────────────────────────────────────────────

st.divider()
st.markdown("### 🔬 Evolución de Ratios de Fluidos")
st.caption(
    "Los ratios se calculan con los **caudales instantáneos agregados** de toda la cuenca "
    "(o del filtro activo), no con producción acumulada. "
    "GOR = Qg / Qo × 1000 · WOR = Qw / Qo · WGR = Qw / Qg × 1000. "
    "Esto refleja el comportamiento dinámico del período, no la historia acumulativa del pozo."
)

ratio_mode = st.radio(
    "Seleccioná qué ratios visualizar:",
    ["GOR + WOR", "GOR + WGR"],
    horizontal=True,
    key="ratio_mode",
)

COLOR_GOR = "#e74c3c"
COLOR_WOR = "#2980b9"
COLOR_WGR = "#8e44ad"

fig_ratios = go.Figure()

# GOR — siempre presente
fig_ratios.add_trace(go.Scatter(
    x=monthly["date"], y=monthly["GOR"],
    mode="lines", name="GOR mensual",
    line=dict(color=COLOR_GOR, width=1, dash="dot"), opacity=0.4,
    hovertemplate="Fecha: %{x}<br>GOR: %{y:,.1f}<extra></extra>",
))
fig_ratios.add_trace(go.Scatter(
    x=monthly["date"], y=monthly["GOR_3m"],
    mode="lines", name="GOR media 3m",
    line=dict(color=COLOR_GOR, width=2.5),
    hovertemplate="Fecha: %{x}<br>GOR 3m: %{y:,.1f}<extra></extra>",
))

if ratio_mode == "GOR + WOR":
    fig_ratios.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["WOR"],
        mode="lines", name="WOR mensual",
        line=dict(color=COLOR_WOR, width=1, dash="dot"), opacity=0.4,
        yaxis="y2",
        hovertemplate="Fecha: %{x}<br>WOR: %{y:,.3f}<extra></extra>",
    ))
    fig_ratios.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["WOR_3m"],
        mode="lines", name="WOR media 3m",
        line=dict(color=COLOR_WOR, width=2.5),
        yaxis="y2",
        hovertemplate="Fecha: %{x}<br>WOR 3m: %{y:,.3f}<extra></extra>",
    ))
    fig_ratios.update_layout(
        yaxis2=dict(title="WOR (m³agua / m³oil)", overlaying="y", side="right", color=COLOR_WOR)
    )
    y2_label = "WOR"

else:  # GOR + WGR
    fig_ratios.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["WGR"],
        mode="lines", name="WGR mensual",
        line=dict(color=COLOR_WGR, width=1, dash="dot"), opacity=0.4,
        yaxis="y2",
        hovertemplate="Fecha: %{x}<br>WGR: %{y:,.3f}<extra></extra>",
    ))
    fig_ratios.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["WGR_3m"],
        mode="lines", name="WGR media 3m",
        line=dict(color=COLOR_WGR, width=2.5),
        yaxis="y2",
        hovertemplate="Fecha: %{x}<br>WGR 3m: %{y:,.3f}<extra></extra>",
    ))
    fig_ratios.update_layout(
        yaxis2=dict(title="WGR (m³agua / km³gas)", overlaying="y", side="right", color=COLOR_WGR)
    )
    y2_label = "WGR"

fig_ratios.update_layout(
    title=f"Evolución GOR + {y2_label}{_adv_scope}",
    xaxis_title="Fecha",
    yaxis=dict(title="GOR (m³gas / km³oil)", color=COLOR_GOR),
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_ratios, use_container_width=True)


# ── 5. Market Share dinámico por operador ─────────────────────────────────────

st.divider()
st.markdown("### 🏢 Market Share Dinámico por Operador")
st.caption(
    "Participación porcentual de cada empresa en la producción total mensual. "
    "Permite ver cómo se redistribuye el mercado entre operadores a lo largo del tiempo."
)

ms_fluid = st.radio("Fluido para market share", ["Petróleo", "Gas"], horizontal=True, key="ms_fluid")
_ms_rate = "oil_rate" if ms_fluid == "Petróleo" else "gas_rate"
_ms_lbl  = "Petróleo (m³/d)" if ms_fluid == "Petróleo" else "Gas (km³/d)"

ms_n = st.slider("Top N operadores (resto = 'Otros')", min_value=3, max_value=12, value=6, key="ms_n")

company_monthly = (
    _adv_base
    .groupby(["date", "empresaNEW"])[_ms_rate]
    .sum()
    .reset_index()
)
top_cos = (
    company_monthly.groupby("empresaNEW")[_ms_rate].sum().nlargest(ms_n).index.tolist()
)
company_monthly["empresa_label"] = company_monthly["empresaNEW"].apply(
    lambda x: x if x in top_cos else "Otros"
)
total_by_date = company_monthly.groupby("date")[_ms_rate].sum().rename("total")
company_monthly = company_monthly.merge(total_by_date, on="date")
company_monthly["share"] = company_monthly[_ms_rate] / company_monthly["total"] * 100
ms_agg = (
    company_monthly.groupby(["date", "empresa_label"])[["share", _ms_rate]]
    .sum().reset_index()
)
all_labels = [c for c in top_cos if c in ms_agg["empresa_label"].unique()] + (
    ["Otros"] if "Otros" in ms_agg["empresa_label"].unique() else []
)

fig_ms = go.Figure()
for i, co in enumerate(all_labels):
    co_data = ms_agg[ms_agg["empresa_label"] == co].sort_values("date")
    color   = COMPANY_PALETTE[i % len(COMPANY_PALETTE)] if co != "Otros" else "#bdc3c7"
    fig_ms.add_trace(go.Scatter(
        x=co_data["date"], y=co_data["share"], name=co,
        mode="lines", stackgroup="one",
        line=dict(color=color, width=0.5), fillcolor=color,
        hovertemplate=f"{co}<br>Fecha: %{{x}}<br>Share: %{{y:.1f}}%<br>{_ms_lbl}: %{{customdata:,.0f}}",
        customdata=co_data[_ms_rate].values,
    ))
fig_ms.update_layout(
    title=f"Market Share de Producción de {ms_fluid} por Operador{_adv_scope}",
    xaxis_title="Fecha", yaxis_title="Participación (%)",
    hovermode="x unified", template="plotly_white", legend=LEGEND_BOTTOM,
    yaxis=dict(range=[0, 100], ticksuffix="%"),
)
st.plotly_chart(fig_ms, use_container_width=True)
