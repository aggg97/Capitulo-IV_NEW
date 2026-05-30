import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta
from PIL import Image

from utils import BARRELS_PER_M3, COMPANY_REPLACEMENTS, DATASET_URL


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_and_sort_data(dataset_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(dataset_url, usecols=[
            "sigla", "anio", "mes", "prod_pet", "prod_gas", "prod_agua",
            "tef", "empresa", "areayacimiento", "coordenadax", "coordenaday",
            "formprod", "sub_tipo_recurso", "tipopozo",
        ])
        df["date"]       = pd.to_datetime(df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-1")
        df["gas_rate"]   = df["prod_gas"]  / df["tef"]
        df["oil_rate"]   = df["prod_pet"]  / df["tef"]
        df["water_rate"] = df["prod_agua"] / df["tef"]
        df["Np"] = df.groupby("sigla")["prod_pet"].cumsum()
        df["Gp"] = df.groupby("sigla")["prod_gas"].cumsum()
        df["Wp"] = df.groupby("sigla")["prod_agua"].cumsum()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# ── Session state ─────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    with st.spinner("🔄 Sincronizando los últimos datos oficiales de la Secretaría de Energía..."):
        df = load_and_sort_data(DATASET_URL)
        if df.empty:
            st.error(
                "⚠️ No se pudieron cargar los datos (timeout o error de red). "
                "Por favor, recargá la página para intentar nuevamente."
            )
            st.stop()
        st.session_state["df"] = df
        st.success("✅ Datos cargados correctamente. La sesión está activa para todas las páginas.")

# Guard: handle case where a previous session stored an empty DataFrame
if st.session_state["df"].empty:
    del st.session_state["df"]
    st.error(
        "⚠️ Los datos almacenados están vacíos. "
        "Por favor, recargá la página para volver a cargarlos."
    )
    st.stop()

data_sorted = st.session_state["df"]
data_sorted["empresaNEW"] = data_sorted["empresa"].replace(COMPANY_REPLACEMENTS)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.header(":blue[Reporte de Producción No Convencional]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))


# ── Base filter & dates ───────────────────────────────────────────────────────

data_filtered = data_sorted[data_sorted["tef"] > 0]

# Display the in-progress (non-official) date first — intentional
latest_date_non_official = data_filtered["date"].max()
st.write("Fecha de Alocación en Progreso: ", latest_date_non_official.date())

# Consolidated date: one month behind the latest available
latest_date = latest_date_non_official - relativedelta(months=1)
latest_data = data_filtered[data_filtered["date"] == latest_date]


# ── KPI metrics ───────────────────────────────────────────────────────────────

total_gas_rate_rounded = round(latest_data["gas_rate"].sum() / 1000, 1)
total_oil_rate_rounded = round(latest_data["oil_rate"].sum() / 1000, 1)
oil_rate_bpd_rounded   = round(total_oil_rate_rounded * BARRELS_PER_M3, 1)

st.write("Fecha de Última Alocación Finalizada y Consolidada*: ", latest_date.date())
st.caption(
    "*A mediados de cada mes se realiza el cierre oficial de los datos correspondientes "
    "al mes anterior. Para garantizar la precisión y evitar mostrar información incompleta "
    "o no consolidada, este reporte presenta únicamente los datos del mes anterior ya "
    "finalizados, completos y representativos."
)

col1, col2, col3 = st.columns(3)
col1.metric(label=":red[Total Caudal de Gas (MMm³/d)]",       value=total_gas_rate_rounded)
col2.metric(label=":green[Total Caudal de Petróleo (km³/d)]", value=total_oil_rate_rounded)
col3.metric(label=":green[Total Caudal de Petróleo (kbpd)]",  value=oil_rate_bpd_rounded)


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


# ── Vintage (start-year) aggregation ─────────────────────────────────────────

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


# ── Plots ─────────────────────────────────────────────────────────────────────

# Gas by company
fig_gas_company = px.area(
    company_summary_aggregated,
    x="date", y="total_gas_rate", color="empresaNEW",
    title="Caudal de Gas por Empresa",
    labels={"date": "Fecha", "total_gas_rate": "Caudal de Gas (km³/d)", "empresaNEW": "Empresa"},
)
fig_gas_company.update_layout(
    legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=10))
)

log_scale_gas = st.checkbox("Escala semilog Caudal de Gas")
st.caption(
    "Nota: Activar la escala semilog facilita la detección rápida de tendencias lineales "
    "en los datos, permitiendo identificar patrones de crecimiento exponencial en la "
    "producción de manera más efectiva."
)
if log_scale_gas:
    fig_gas_company.update_layout(yaxis=dict(type="log", dtick=1))
st.plotly_chart(fig_gas_company)

# Oil by company
fig_oil_company = px.area(
    company_summary_aggregated,
    x="date", y="total_oil_rate", color="empresaNEW",
    title="Caudal de Petróleo por Empresa",
    labels={"date": "Fecha", "total_oil_rate": "Caudal de Petróleo (m³/d)", "empresaNEW": "Empresa"},
)
fig_oil_company.update_layout(
    legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=10))
)

log_scale_oil = st.checkbox("Escala semilog Caudal de Petróleo")
if log_scale_oil:
    fig_oil_company.update_layout(yaxis=dict(type="log", dtick=1))
st.plotly_chart(fig_oil_company)

# Gas by vintage
fig_gas_year = px.area(
    yearly_summary,
    x="date", y="total_gas_rate", color="start_year",
    title="Caudal de Gas por Campaña",
    labels={"date": "Fecha", "total_gas_rate": "Caudal de Gas (km³/d)", "start_year": "Campaña"},
)
fig_gas_year.update_layout(
    legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=10))
)
st.plotly_chart(fig_gas_year)

# Oil by vintage
fig_oil_year = px.area(
    yearly_summary,
    x="date", y="total_oil_rate", color="start_year",
    title="Caudal de Petróleo por Campaña",
    labels={"date": "Fecha", "total_oil_rate": "Caudal de Petróleo (m³/d)", "start_year": "Campaña"},
)
fig_oil_year.update_layout(
    legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=10))
)
st.plotly_chart(fig_oil_year)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paletas ────────────────────────────────────────────────────────────────────
COMPANY_PALETTE = px.colors.qualitative.Set2
ROLL_COLOR_OIL  = "#27ae60"
ROLL_COLOR_GAS  = "#e74c3c"
ROLL_COLOR_3M   = "#3498db"
ROLL_COLOR_6M   = "#e67e22"
BROWNFIELD_COLOR = "#8e44ad"

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE COMPLETO — pegar dentro de `with tab1:`
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 Métricas Operativas Avanzadas", divider="blue")
st.caption(
    "Variaciones MoM / YoY, producción incremental, market share, "
    "GOR/WOR evolution y señales de madurez brownfield."
)


# ══════════════════════════════════════════════════════════════════════════════
# 0. BASE MENSUAL AGREGADA
# ══════════════════════════════════════════════════════════════════════════════

monthly = (
    data_sorted[data_sorted["tef"] > 0]
    .groupby("date")
    .agg(
        oil_rate  =("oil_rate",   "sum"),
        gas_rate  =("gas_rate",   "sum"),
        water_rate=("water_rate", "sum"),
        prod_pet  =("prod_pet",   "sum"),
        prod_gas  =("prod_gas",   "sum"),
        prod_agua =("prod_agua",  "sum"),
        tef       =("tef",        "sum"),
        wells_active=("sigla",    "nunique"),
    )
    .sort_index()
    .reset_index()
)

# GOR y WOR mensuales totales
monthly["GOR"] = (monthly["prod_gas"] / monthly["prod_pet"].replace(0, np.nan) * 1000)
monthly["WOR"] = (monthly["prod_agua"] / monthly["prod_pet"].replace(0, np.nan))
monthly["GOR"] = monthly["GOR"].replace([np.inf, -np.inf], np.nan)
monthly["WOR"] = monthly["WOR"].replace([np.inf, -np.inf], np.nan)

# Rolling 3m y 6m
for col in ["oil_rate", "gas_rate", "water_rate", "GOR", "WOR"]:
    monthly[f"{col}_3m"] = monthly[col].rolling(3, min_periods=1).mean()
    monthly[f"{col}_6m"] = monthly[col].rolling(6, min_periods=1).mean()

# MoM y YoY
for col in ["oil_rate", "gas_rate", "water_rate", "wells_active"]:
    monthly[f"{col}_mom"] = monthly[col].pct_change(1) * 100
    monthly[f"{col}_yoy"] = monthly[col].pct_change(12) * 100


# ══════════════════════════════════════════════════════════════════════════════
# 1. KPIs SEMÁFORO — MoM y YoY del ÚLTIMO MES
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 🚦 Variaciones MoM / YoY — Último período")

last = monthly.dropna(subset=["oil_rate"]).iloc[-1]

def _delta_str(val):
    if pd.isna(val):
        return "N/D", "off"
    sign = "+" if val >= 0 else ""
    color = "normal" if val >= 0 else "inverse"
    return f"{sign}{val:.1f}%", color

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🛢️ Qo Total (m³/d)", f"{last['oil_rate']:,.0f}",
              delta=f"MoM: {_delta_str(last['oil_rate_mom'])[0]}",
              delta_color=_delta_str(last["oil_rate_mom"])[1])
with col2:
    st.metric("🔥 Qg Total (km³/d)", f"{last['gas_rate']:,.0f}",
              delta=f"MoM: {_delta_str(last['gas_rate_mom'])[0]}",
              delta_color=_delta_str(last["gas_rate_mom"])[1])
with col3:
    st.metric("💧 Qw Total (m³/d)", f"{last['water_rate']:,.0f}",
              delta=f"MoM: {_delta_str(last['water_rate_mom'])[0]}",
              delta_color=_delta_str(last["water_rate_mom"])[1])
with col4:
    st.metric("🛞 Pozos Activos", f"{int(last['wells_active']):,}",
              delta=f"MoM: {_delta_str(last['wells_active_mom'])[0]}",
              delta_color=_delta_str(last["wells_active_mom"])[1])

# Fila YoY
monthly_yoy_row = monthly.dropna(subset=["oil_rate_yoy"]).iloc[-1] if monthly["oil_rate_yoy"].notna().any() else None

if monthly_yoy_row is not None:
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("🛢️ YoY Petróleo", "",
                  delta=f"YoY: {_delta_str(monthly_yoy_row['oil_rate_yoy'])[0]}",
                  delta_color=_delta_str(monthly_yoy_row["oil_rate_yoy"])[1])
    with col6:
        st.metric("🔥 YoY Gas", "",
                  delta=f"YoY: {_delta_str(monthly_yoy_row['gas_rate_yoy'])[0]}",
                  delta_color=_delta_str(monthly_yoy_row["gas_rate_yoy"])[1])
    with col7:
        st.metric("💧 YoY Agua", "",
                  delta=f"YoY: {_delta_str(monthly_yoy_row['water_rate_yoy'])[0]}",
                  delta_color=_delta_str(monthly_yoy_row["water_rate_yoy"])[1])
    with col8:
        st.metric("🛞 YoY Pozos Activos", "",
                  delta=f"YoY: {_delta_str(monthly_yoy_row['wells_active_yoy'])[0]}",
                  delta_color=_delta_str(monthly_yoy_row["wells_active_yoy"])[1])


# ══════════════════════════════════════════════════════════════════════════════
# 2. ROLLING AVERAGES — Petróleo y Gas
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 📈 Producción con Rolling Averages (3m / 6m)")

fluid_roll = st.radio(
    "Fluido",
    ["Petróleo", "Gas", "Agua"],
    horizontal=True,
    key="fluid_roll",
)

_map_roll = {
    "Petróleo": ("oil_rate", "oil_rate_3m", "oil_rate_6m",
                 "Caudal de Petróleo (m³/d)", ROLL_COLOR_OIL),
    "Gas":      ("gas_rate", "gas_rate_3m", "gas_rate_6m",
                 "Caudal de Gas (km³/d)", ROLL_COLOR_GAS),
    "Agua":     ("water_rate", "water_rate_3m", "water_rate_6m",
                 "Caudal de Agua (m³/d)", "#2980b9"),
}
raw_col, col3m, col6m, y_lbl, base_color = _map_roll[fluid_roll]

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(
    x=monthly["date"], y=monthly[raw_col],
    mode="lines", name="Mensual",
    line=dict(color=base_color, width=1, dash="dot"),
    opacity=0.45,
    hovertemplate="Fecha: %{x}<br>Valor: %{y:,.1f}",
))
fig_roll.add_trace(go.Scatter(
    x=monthly["date"], y=monthly[col3m],
    mode="lines", name="Media 3m",
    line=dict(color=ROLL_COLOR_3M, width=2.5),
    hovertemplate="Fecha: %{x}<br>Media 3m: %{y:,.1f}",
))
fig_roll.add_trace(go.Scatter(
    x=monthly["date"], y=monthly[col6m],
    mode="lines", name="Media 6m",
    line=dict(color=ROLL_COLOR_6M, width=2.5),
    hovertemplate="Fecha: %{x}<br>Media 6m: %{y:,.1f}",
))
fig_roll.update_layout(
    title=f"Producción Total de {fluid_roll} — Mensual vs Rolling Average",
    xaxis_title="Fecha",
    yaxis_title=y_lbl,
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_roll, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. VARIACIONES MoM Y YoY — Gráfico de barras waterfall estilo
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 📉 Variación Porcentual MoM y YoY")

var_fluid = st.radio(
    "Fluido para variaciones",
    ["Petróleo", "Gas"],
    horizontal=True,
    key="var_fluid",
)
_var_col = "oil_rate" if var_fluid == "Petróleo" else "gas_rate"
_var_lbl = "Petróleo (m³/d)" if var_fluid == "Petróleo" else "Gas (km³/d)"

monthly_plot = monthly.dropna(subset=[f"{_var_col}_mom"]).copy()
monthly_plot["mom_color"] = monthly_plot[f"{_var_col}_mom"].apply(
    lambda v: "#27ae60" if v >= 0 else "#e74c3c"
)

fig_var = go.Figure()
fig_var.add_trace(go.Bar(
    x=monthly_plot["date"],
    y=monthly_plot[f"{_var_col}_mom"],
    name="MoM %",
    marker_color=monthly_plot["mom_color"],
    hovertemplate="Fecha: %{x}<br>MoM: %{y:.1f}%",
))

monthly_yoy_plot = monthly.dropna(subset=[f"{_var_col}_yoy"]).copy()
fig_var.add_trace(go.Scatter(
    x=monthly_yoy_plot["date"],
    y=monthly_yoy_plot[f"{_var_col}_yoy"],
    mode="lines+markers",
    name="YoY %",
    line=dict(color="#2c3e50", width=2),
    marker=dict(size=5),
    hovertemplate="Fecha: %{x}<br>YoY: %{y:.1f}%",
))

fig_var.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1)
fig_var.update_layout(
    title=f"Variación MoM (barras) y YoY (línea) — {_var_lbl}",
    xaxis_title="Fecha",
    yaxis_title="Variación (%)",
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_var, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PRODUCCIÓN INCREMENTAL MENSUAL NETA
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### ⚡ Producción Incremental Mensual Neta")
st.caption(
    "Diferencia neta entre el caudal del mes actual y el mes anterior. "
    "Positivo = cuenca creciendo orgánicamente. Negativo = decline neto."
)

monthly["oil_increment"] = monthly["oil_rate"].diff()
monthly["gas_increment"] = monthly["gas_rate"].diff()

incr_fluid = st.radio(
    "Fluido incremental",
    ["Petróleo", "Gas"],
    horizontal=True,
    key="incr_fluid",
)
_inc_col = "oil_increment" if incr_fluid == "Petróleo" else "gas_increment"
_inc_lbl = "Incremento Petróleo (m³/d)" if incr_fluid == "Petróleo" else "Incremento Gas (km³/d)"
_inc_color = ROLL_COLOR_OIL if incr_fluid == "Petróleo" else ROLL_COLOR_GAS

monthly_incr = monthly.dropna(subset=[_inc_col]).copy()
monthly_incr["bar_color"] = monthly_incr[_inc_col].apply(
    lambda v: "#27ae60" if v >= 0 else "#e74c3c"
)

fig_incr = go.Figure()
fig_incr.add_trace(go.Bar(
    x=monthly_incr["date"],
    y=monthly_incr[_inc_col],
    marker_color=monthly_incr["bar_color"],
    name="Incremento neto",
    hovertemplate="Fecha: %{x}<br>Δ: %{y:,.0f}",
))
# Rolling 3m del incremento para tendencia
monthly_incr["incr_3m"] = monthly_incr[_inc_col].rolling(3, min_periods=1).mean()
fig_incr.add_trace(go.Scatter(
    x=monthly_incr["date"],
    y=monthly_incr["incr_3m"],
    mode="lines",
    name="Tendencia 3m",
    line=dict(color="#2c3e50", width=2.5, dash="dash"),
    hovertemplate="Fecha: %{x}<br>Tendencia 3m: %{y:,.0f}",
))
fig_incr.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1.5)
fig_incr.update_layout(
    title=f"Producción Incremental Mensual Neta — {incr_fluid}",
    xaxis_title="Fecha",
    yaxis_title=_inc_lbl,
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_incr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. GOR Y WOR EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 🔬 Evolución de GOR y WOR")
st.caption(
    "GOR = Gas-Oil Ratio (m³/m³ × 1000 = m³/km³). "
    "WOR = Water-Oil Ratio (m³/m³). "
    "GOR creciente puede indicar agotamiento de presión o madurez del yacimiento."
)

gorwor_sel = st.radio(
    "Métrica",
    ["GOR", "WOR", "Ambas"],
    horizontal=True,
    key="gorwor_sel",
)

fig_gw = go.Figure()

if gorwor_sel in ["GOR", "Ambas"]:
    fig_gw.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["GOR"],
        mode="lines", name="GOR mensual",
        line=dict(color=ROLL_COLOR_GAS, width=1, dash="dot"),
        opacity=0.4,
    ))
    fig_gw.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["GOR_3m"],
        mode="lines", name="GOR media 3m",
        line=dict(color=ROLL_COLOR_GAS, width=2.5),
        hovertemplate="Fecha: %{x}<br>GOR 3m: %{y:,.1f}",
    ))

if gorwor_sel in ["WOR", "Ambas"]:
    yaxis_id = "y2" if gorwor_sel == "Ambas" else "y"
    fig_gw.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["WOR"],
        mode="lines", name="WOR mensual",
        line=dict(color="#2980b9", width=1, dash="dot"),
        opacity=0.4,
        yaxis=yaxis_id,
    ))
    fig_gw.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["WOR_3m"],
        mode="lines", name="WOR media 3m",
        line=dict(color="#2980b9", width=2.5),
        hovertemplate="Fecha: %{x}<br>WOR 3m: %{y:,.3f}",
        yaxis=yaxis_id,
    ))

layout_gw = dict(
    title="Evolución GOR y WOR — Cuenca Vaca Muerta",
    xaxis_title="Fecha",
    yaxis_title="GOR (m³/km³)" if gorwor_sel != "WOR" else "WOR (m³/m³)",
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
if gorwor_sel == "Ambas":
    layout_gw["yaxis2"] = dict(
        title="WOR (m³/m³)",
        overlaying="y",
        side="right",
    )
fig_gw.update_layout(**layout_gw)
st.plotly_chart(fig_gw, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MARKET SHARE DINÁMICO POR OPERADOR — STACKED AREA
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 🏢 Market Share Dinámico por Operador")
st.caption(
    "Participación porcentual de cada empresa en la producción total mensual. "
    "Permite ver cómo se redistribuye el mercado entre operadores a lo largo del tiempo."
)

ms_fluid = st.radio(
    "Fluido para market share",
    ["Petróleo", "Gas"],
    horizontal=True,
    key="ms_fluid",
)
_ms_rate = "oil_rate" if ms_fluid == "Petróleo" else "gas_rate"
_ms_lbl  = "Petróleo (m³/d)" if ms_fluid == "Petróleo" else "Gas (km³/d)"

ms_n = st.slider(
    "Top N operadores a mostrar (resto agrupado como 'Otros')",
    min_value=3, max_value=12, value=6, key="ms_n",
)

company_monthly = (
    data_sorted[data_sorted["tef"] > 0]
    .groupby(["date", "empresaNEW"])[_ms_rate]
    .sum()
    .reset_index()
)

# Top N por producción total
top_cos = (
    company_monthly.groupby("empresaNEW")[_ms_rate]
    .sum()
    .nlargest(ms_n)
    .index.tolist()
)
company_monthly["empresa_label"] = company_monthly["empresaNEW"].apply(
    lambda x: x if x in top_cos else "Otros"
)

# Share calculation
total_monthly = company_monthly.groupby("date")[_ms_rate].sum().rename("total")
company_monthly = company_monthly.merge(total_monthly, on="date")
company_monthly["share"] = company_monthly[_ms_rate] / company_monthly["total"] * 100

ms_agg = (
    company_monthly.groupby(["date", "empresa_label"])[["share", _ms_rate]]
    .sum()
    .reset_index()
)

fig_ms = go.Figure()
all_labels = [c for c in top_cos if c in ms_agg["empresa_label"].unique()] + (
    ["Otros"] if "Otros" in ms_agg["empresa_label"].unique() else []
)

for i, co in enumerate(all_labels):
    co_data = ms_agg[ms_agg["empresa_label"] == co].sort_values("date")
    color = COMPANY_PALETTE[i % len(COMPANY_PALETTE)] if co != "Otros" else "#bdc3c7"
    fig_ms.add_trace(go.Scatter(
        x=co_data["date"],
        y=co_data["share"],
        name=co,
        mode="lines",
        stackgroup="one",
        groupnorm="",
        line=dict(color=color, width=0.5),
        fillcolor=color,
        hovertemplate=f"{co}<br>Fecha: %{{x}}<br>Share: %{{y:.1f}}%<br>{_ms_lbl}: %{{customdata:,.0f}}",
        customdata=co_data[_ms_rate].values,
    ))

fig_ms.update_layout(
    title=f"Market Share de Producción de {ms_fluid} por Operador",
    xaxis_title="Fecha",
    yaxis_title="Participación (%)",
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
    yaxis=dict(range=[0, 100], ticksuffix="%"),
)
st.plotly_chart(fig_ms, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SEÑALES DE MADUREZ BROWNFIELD
#    "La ventaja del campo maduro: más por activo existente"
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 🏗️ Vaca Muerta como Brownfield Consolidado")
st.caption(
    "En unconventional, la cuenca madura se comporta como brownfield de clase mundial: "
    "mayor producción por pozo activo, menor dispersión de resultados y eficiencia "
    "creciente del capital. Estas métricas muestran esa transición greenfield → brownfield."
)

# 7a. PRODUCCIÓN POR POZO ACTIVO — proxy de eficiencia de capital
monthly_bf = monthly.copy()
monthly_bf["oil_per_well"] = monthly_bf["oil_rate"] / monthly_bf["wells_active"].replace(0, np.nan)
monthly_bf["gas_per_well"] = monthly_bf["gas_rate"] / monthly_bf["wells_active"].replace(0, np.nan)
monthly_bf["oil_per_well_3m"] = monthly_bf["oil_per_well"].rolling(3, min_periods=1).mean()
monthly_bf["gas_per_well_3m"] = monthly_bf["gas_per_well"].rolling(3, min_periods=1).mean()

bf_fluid = st.radio(
    "Fluido",
    ["Petróleo", "Gas"],
    horizontal=True,
    key="bf_fluid",
)
_bf_col  = "oil_per_well"   if bf_fluid == "Petróleo" else "gas_per_well"
_bf_3m   = "oil_per_well_3m" if bf_fluid == "Petróleo" else "gas_per_well_3m"
_bf_lbl  = "m³/d por pozo activo" if bf_fluid == "Petróleo" else "km³/d por pozo activo"
_bf_clr  = ROLL_COLOR_OIL if bf_fluid == "Petróleo" else ROLL_COLOR_GAS

fig_bf1 = go.Figure()
fig_bf1.add_trace(go.Scatter(
    x=monthly_bf["date"], y=monthly_bf[_bf_col],
    mode="lines", name="Mensual",
    line=dict(color=_bf_clr, width=1, dash="dot"),
    opacity=0.4,
))
fig_bf1.add_trace(go.Scatter(
    x=monthly_bf["date"], y=monthly_bf[_bf_3m],
    mode="lines", name="Media 3m",
    line=dict(color=_bf_clr, width=3),
    hovertemplate="Fecha: %{x}<br>%{y:,.2f} " + _bf_lbl,
))
fig_bf1.update_layout(
    title=f"Productividad por Pozo Activo — {bf_fluid} (señal de madurez brownfield)",
    xaxis_title="Fecha",
    yaxis_title=_bf_lbl,
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_bf1, use_container_width=True)

# 7b. POZOS ACTIVOS Y TASA TOTAL — crecimiento de base instalada
fig_bf2 = go.Figure()
fig_bf2.add_trace(go.Scatter(
    x=monthly_bf["date"], y=monthly_bf["wells_active"],
    mode="lines", name="Pozos activos",
    line=dict(color=BROWNFIELD_COLOR, width=2.5),
    hovertemplate="Fecha: %{x}<br>Pozos activos: %{y:,}",
))
fig_bf2.add_trace(go.Scatter(
    x=monthly_bf["date"], y=monthly_bf["oil_rate"],
    mode="lines", name="Qo total (m³/d)",
    line=dict(color=ROLL_COLOR_OIL, width=2),
    yaxis="y2",
    hovertemplate="Fecha: %{x}<br>Qo: %{y:,.0f} m³/d",
))
fig_bf2.update_layout(
    title="Crecimiento de Base Instalada: Pozos Activos vs Producción Total",
    xaxis_title="Fecha",
    yaxis=dict(title="Pozos Activos", color=BROWNFIELD_COLOR),
    yaxis2=dict(title="Qo Total (m³/d)", overlaying="y", side="right", color=ROLL_COLOR_OIL),
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_bf2, use_container_width=True)

# 7c. EFICIENCIA DE PERFORACIÓN POR CAMPAÑA (IP30 proxy como Qo pico / etapas)
#     Muestra si la cuenca mejora su retorno por unidad de capital invertido

if "Qo_peak" in df_vmut.columns and "cantidad_fracturas" in df_vmut.columns:
    df_eff = df_vmut[
        (df_vmut["tipopozoNEW"] == "Petrolífero") &
        (df_vmut["Qo_peak"] > 0) &
        (df_vmut["cantidad_fracturas"] > 0)
    ].copy()
    df_eff["Qo_peak_x_etapa"] = df_eff["Qo_peak"] / df_eff["cantidad_fracturas"]

    eff_agg = (
        df_eff.groupby("start_year")["Qo_peak_x_etapa"]
        .agg(p50="median", p10=lambda x: x.quantile(0.9), p90=lambda x: x.quantile(0.1), count="count")
        .reset_index()
        .dropna(subset=["p50"])
    )

    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(
        x=eff_agg["start_year"], y=eff_agg["p10"],
        mode="lines", name="P10",
        line=dict(color=ROLL_COLOR_OIL, width=1, dash="dot"),
        showlegend=True,
    ))
    fig_eff.add_trace(go.Scatter(
        x=eff_agg["start_year"], y=eff_agg["p90"],
        mode="lines", name="P90",
        line=dict(color=ROLL_COLOR_OIL, width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(39,174,96,0.12)",
        showlegend=True,
    ))
    fig_eff.add_trace(go.Scatter(
        x=eff_agg["start_year"], y=eff_agg["p50"],
        mode="lines+markers+text",
        name="P50 (mediana)",
        line=dict(color=ROLL_COLOR_OIL, width=3),
        marker=dict(size=8),
        text=eff_agg["p50"].round(1).astype(str),
        textposition="top center",
        textfont=dict(size=9, color=ROLL_COLOR_OIL),
    ))
    for _, row in eff_agg.iterrows():
        fig_eff.add_annotation(
            x=row["start_year"], y=row["p50"],
            text=f"n={int(row['count'])}",
            showarrow=False, yshift=-18,
            font=dict(size=8, color="grey"),
        )
    fig_eff.update_layout(
        title="Eficiencia de Capital: Qo Pico por Etapa por Campaña — Petrolífero (Proxy de Retorno/Inversión)",
        xaxis_title="Campaña de Perforación",
        yaxis_title="Qo Pico / Etapa (m³/d/etapa)",
        template="plotly_white",
        legend=LEGEND_BOTTOM,
        annotations=[dict(
            x=0.01, y=0.97,
            xref="paper", yref="paper",
            text="📌 Tendencia creciente = mayor retorno por unidad de capital (brownfield advantage)",
            showarrow=False,
            font=dict(size=10, color=BROWNFIELD_COLOR),
            bgcolor="rgba(142,68,173,0.08)",
            bordercolor=BROWNFIELD_COLOR,
            borderwidth=1,
        )],
    )
    st.plotly_chart(fig_eff, use_container_width=True)

# 7d. INFOGRAPHIC — contexto brownfield
st.info(
    """
    **🏗️ ¿Por qué Vaca Muerta es ya un Brownfield de clase mundial?**

    En la industria upstream, **brownfield casi siempre supera a greenfield** en 
    rentabilidad y velocidad de monetización. La ventaja del campo maduro:

    - ✅ **Infraestructura existente** → menores costos de transporte y procesamiento  
    - ✅ **Curva de aprendizaje acumulada** → completaciones más eficientes por campaña  
    - ✅ **Base de datos de subsuelo** → menor riesgo de exploración  
    - ✅ **Cadena de valor local consolidada** → proveedores, arena nacional, crews especializados  
    - ✅ **Productividad por pozo activo en crecimiento** → más producción con base de pozos madura  

    La excepción histórica al dominio brownfield fue **Guyana** (greenfield de primer orden). 
    Vaca Muerta, por contraste, combina escala de greenfield con ventajas operacionales 
    de un activo maduro — una combinación muy poco común a nivel global.
    """
)