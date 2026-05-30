import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from dateutil.relativedelta import relativedelta
from PIL import Image

from utils import (
    BARRELS_PER_M3, COMPANY_REPLACEMENTS, DATASET_URL,
    calculate_rates, calculate_gor_wc, to_km3_per_day, to_kbbl_per_day,
    calculate_monthly_incremental, calculate_yoy_metrics, calculate_operator_metrics,
    calculate_productivity_by_vintage, calculate_base_decline_contribution,
)


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_and_sort_data(dataset_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(dataset_url, usecols=[
            "sigla", "anio", "mes", "prod_pet", "prod_gas", "prod_agua",
            "tef", "empresa", "areayacimiento", "coordenadax", "coordenaday",
            "formprod", "sub_tipo_recurso", "tipopozo",
        ])
        df["date"] = pd.to_datetime(df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-1")
        
        # Use safe rate calculation with division-by-zero protection
        df = calculate_rates(df)
        df = calculate_gor_wc(df)
        
        # Cumulative production
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

# Create a working copy to avoid mutating cached session_state
data_sorted = st.session_state["df"].copy()
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

# Draft - testing mode

from dateutil.relativedelta import relativedelta

# ── VARIACIÓN INTERANUAL ───────────────────────────────────────────────
# Fecha del mismo mes del año anterior
previous_year_date = latest_date - relativedelta(years=1)
previous_year_data = data_filtered[data_filtered['date'] == previous_year_date]

# Caudales del año anterior
prev_gas_rate  = previous_year_data['gas_rate'].sum() / 1000
prev_oil_rate  = previous_year_data['oil_rate'].sum() / 1000

# Deltas para el widget metric()
gas_yoy_delta  = round(total_gas_rate_rounded - round(prev_gas_rate, 1), 1)
oil_yoy_delta  = round(total_oil_rate_rounded - round(prev_oil_rate, 1), 1)
bpd_yoy_delta  = round(oil_rate_bpd_rounded - round(prev_oil_rate * 6.28981, 1), 1)

# Actualizar las métricas con deltas
col1.metric(
    label=":red[Total Caudal de Gas (MMm³/d)]",
    value=total_gas_rate_rounded,
    delta=f"{gas_yoy_delta:+.1f} vs año anterior"
)
col2.metric(
    label=":green[Total Caudal de Petróleo (km³/d)]",
    value=total_oil_rate_rounded,
    delta=f"{oil_yoy_delta:+.1f} vs año anterior"
)
col3.metric(
    label=":green[Total Caudal de Petróleo (kbpd)]",
    value=oil_rate_bpd_rounded,
    delta=f"{bpd_yoy_delta:+.1f} vs año anterior"
)

# ── MÉTRICAS OPERATIVAS (segunda fila) ────────────────────────────────
active_wells  = latest_data['sigla'].nunique()
water_cut     = round(
    latest_data['prod_agua'].sum() /
    (latest_data['prod_pet'].sum() + latest_data['prod_agua'].sum()) * 100, 1
)
gor           = round(
    latest_data['prod_gas'].sum() / latest_data['prod_pet'].sum(), 1
) if latest_data['prod_pet'].sum() > 0 else 0

oil_per_well  = round(total_oil_rate_rounded * 1000 / active_wells, 1) if active_wells > 0 else 0

# Año anterior para pozos activos
prev_wells    = previous_year_data['sigla'].nunique()
wells_delta   = active_wells - prev_wells

col4, col5, col6, col7 = st.columns(4)
col4.metric(
    label="🛢️ Pozos Activos",
    value=active_wells,
    delta=f"{wells_delta:+d} vs año anterior"
)
col5.metric(
    label="💧 Water Cut (%)",
    value=water_cut
)
col6.metric(
    label="⚗️ GOR (m³gas/m³oil)",
    value=gor
)
col7.metric(
    label="📊 Productividad/Pozo (m³/d)",
    value=oil_per_well
)
# ── GRÁFICO YoY & INCREMENTAL PRODUCTION ────────────────────────────────

monthly_totals = (
    data_filtered
    .groupby('date')
    .agg(oil_rate=('oil_rate', 'sum'), gas_rate=('gas_rate', 'sum'))
    .reset_index()
    .sort_values('date')
)

# Use robust YoY calculation
yoy = calculate_yoy_metrics(monthly_totals)

# Calculate incremental changes month-over-month
monthly_incremental = calculate_monthly_incremental(monthly_totals)

# YoY plot
fig_yoy = go.Figure()
fig_yoy.add_bar(
    x=yoy['date'], y=yoy['oil_yoy_pct'],
    name='Petróleo YoY %',
    marker_color=yoy['oil_yoy_pct'].apply(lambda v: '#27ae60' if v >= 0 else '#e74c3c')
)
fig_yoy.update_layout(
    title="Variación Interanual de Petróleo (%)",
    xaxis_title="Fecha",
    yaxis_title="Variación YoY (%)",
    hovermode="x unified"
)
st.plotly_chart(fig_yoy, use_container_width=True)

# Monthly incremental production
st.subheader("📈 Producción Incremental Mensual")
st.caption("Cambio neto mes-a-mes. Identifica crecimiento orgánico vs. estabilización/declinación.")

fig_incremental = go.Figure()
fig_incremental.add_bar(
    x=monthly_incremental['date'], y=monthly_incremental['oil_rate_change'],
    name='Cambio Neto de Petróleo (m³/d)',
    marker_color=monthly_incremental['oil_rate_change'].apply(
        lambda v: '#27ae60' if v >= 0 else '#e74c3c'
    )
)
fig_incremental.update_layout(
    title="Incremento Neto de Producción (m³/d)",
    xaxis_title="Fecha",
    yaxis_title="Cambio en Caudal (m³/d)",
    hovermode="x unified"
)
st.plotly_chart(fig_incremental, use_container_width=True)


# ── BASE DECLINE vs NEW WELLS CONTRIBUTION ──────────────────────────────────

st.subheader("🔄 Análisis de Base Decline vs Aporte de Pozos Nuevos")
st.caption(
    "Separa la producción de pozos existentes (base decline) del aporte "
    "de nuevos pozos. Crítico para entender trajectoria del portafolio."
)

decline_analysis = calculate_base_decline_contribution(data_filtered, latest_date)

col_a1, col_a2, col_a3, col_a4 = st.columns(4)
col_a1.metric(
    "Producción Base (m³/d)",
    f"{decline_analysis['base_production']/1000:.1f}k",
    f"{decline_analysis['base_production']/decline_analysis['total_production']*100:.1f}%"
)
col_a2.metric(
    "Aporte Pozos Nuevos (m³/d)",
    f"{decline_analysis['new_well_contribution']/1000:.1f}k",
    f"{decline_analysis['new_well_contribution']/decline_analysis['total_production']*100:.1f}%"
)
col_a3.metric(
    "Pozos Nuevos Agregados",
    f"{decline_analysis['new_wells_added']}",
    f"Activos: {decline_analysis['active_wells_current']}"
)
col_a4.metric(
    "Productividad Prom. (m³/d/pozo)",
    f"{decline_analysis['total_production']/decline_analysis['active_wells_current']:.1f}",
    f"vs {decline_analysis['base_production']/max(decline_analysis['active_wells_previous'], 1):.1f} mes anterior"
)


# ── OPERATOR BENCHMARKING ────────────────────────────────────────────────────

st.subheader("🏢 Benchmarking por Operador")
st.caption("Productividad, market share y eficiencia operacional por empresa.")

operator_metrics = calculate_operator_metrics(data_filtered, latest_date, company_col="empresaNEW")

# Heatmap: Operator productivity over time
operator_over_time = (
    data_filtered
    .groupby(['date', 'empresaNEW'])
    .agg(
        oil_rate=('oil_rate', 'sum'),
        active_wells=('sigla', 'nunique')
    )
    .reset_index()
)
operator_over_time['productivity'] = operator_over_time['oil_rate'] / operator_over_time['active_wells']

# Pivot for heatmap
heatmap_data = operator_over_time.pivot_table(
    index='empresaNEW', 
    columns='date', 
    values='productivity',
    aggfunc='mean'
)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        colorbar=dict(title="m³/d/pozo")
    )
)
fig_heatmap.update_layout(
    title="Productividad por Operador y Período (m³/d/pozo)",
    xaxis_title="Período",
    yaxis_title="Operador",
    height=400
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Market share treemap
fig_treemap = px.treemap(
    operator_metrics,
    labels="empresaNEW",
    parents=[""] * len(operator_metrics),
    values="total_oil_rate",
    color="market_share_oil_pct",
    color_continuous_scale="Blues",
    title="Market Share de Producción de Petróleo"
)
fig_treemap.update_layout(height=400)
st.plotly_chart(fig_treemap, use_container_width=True)


# ── PRODUCTIVITY BY VINTAGE (COHORT ANALYSIS) ────────────────────────────────

st.subheader("📊 Análisis de Productividad por Campaña (Cohort)")
st.caption(
    "Compara rendimiento de diferentes vintages. Permite medir aprendizaje operacional, "
    "mejoras en completaciones, y quality de landing zone."
)

# Add start_year to data
data_with_vintage = data_filtered.copy()
well_start_year = (
    data_with_vintage.groupby("sigla")["anio"].min()
    .reset_index()
    .rename(columns={"anio": "start_year"})
)
data_with_vintage = data_with_vintage.merge(well_start_year, on="sigla")

cohort_productivity = calculate_productivity_by_vintage(data_with_vintage)

# Filter to recent cohorts for readability
recent_cohorts = cohort_productivity[cohort_productivity['start_year'] >= 2015].copy()

fig_cohort = go.Figure()
fig_cohort.add_scatter(
    x=recent_cohorts['start_year'],
    y=recent_cohorts['median_oil_rate'],
    mode='lines+markers',
    name='Mediana Caudal Petróleo',
    marker=dict(size=10)
)
fig_cohort.add_scatter(
    x=recent_cohorts['start_year'],
    y=recent_cohorts['avg_oil_rate'],
    mode='lines+markers',
    name='Promedio Caudal Petróleo',
    marker=dict(size=8, symbol='diamond')
)
fig_cohort.update_layout(
    title="Productividad Promedio por Campaña (Vintage)",
    xaxis_title="Año de Inicio",
    yaxis_title="Caudal de Petróleo (m³/d)",
    hovermode="x unified",
    height=400
)
st.plotly_chart(fig_cohort, use_container_width=True)

# Cohort table
st.write("**Detalle por Campaña:**")
cohort_table = recent_cohorts[[
    'start_year', 'wells_count', 'avg_oil_rate', 'median_oil_rate', 
    'avg_gas_rate', 'total_np', 'total_gp'
]].copy()
cohort_table.columns = ['Campaña', 'Nº Pozos', 'Promedio Crudo (m³/d)', 'Mediana Crudo (m³/d)',
                        'Promedio Gas (m³/d)', 'Total Np (m³)', 'Total Gp (m³)']
cohort_table = cohort_table.sort_values('Campaña', ascending=False)
st.dataframe(cohort_table, use_container_width=True)


# ── OIL/WATER EVOLUTION & GOR TRACKING ──────────────────────────────────────

st.subheader("💧 Evolución de Oil vs Water & GOR Temporal")
st.caption(
    "Monitorea madurez de pozos, breakthrough de agua, interferencias, y comportamiento "
    "de fluido. Crítico para identificar issues operacionales."
)

monthly_fluid = (
    data_filtered
    .groupby('date')
    .agg(
        prod_oil=('prod_pet', 'sum'),
        prod_water=('prod_agua', 'sum'),
        prod_gas=('prod_gas', 'sum')
    )
    .reset_index()
)

# Calculate ratios
monthly_fluid['water_oil_ratio'] = np.where(
    monthly_fluid['prod_oil'] > 0,
    monthly_fluid['prod_water'] / monthly_fluid['prod_oil'],
    0
)
monthly_fluid['GOR'] = np.where(
    monthly_fluid['prod_oil'] > 0,
    monthly_fluid['prod_gas'] / monthly_fluid['prod_oil'],
    0
)

# Stacked area: Oil + Water
fig_oil_water = go.Figure()
fig_oil_water.add_fill(
    x=monthly_fluid['date'], 
    y=monthly_fluid['prod_oil']/1000,
    name='Petróleo (m³/d)',
    mode='lines',
    line=dict(color='green')
)
fig_oil_water.add_fill(
    x=monthly_fluid['date'],
    y=monthly_fluid['prod_water']/1000,
    name='Agua (m³/d)',
    mode='lines',
    line=dict(color='blue'),
    stackgroup='one'
)
fig_oil_water.update_layout(
    title="Evolución de Petróleo vs Agua",
    xaxis_title="Fecha",
    yaxis_title="Producción (m³/d)",
    hovermode="x unified",
    height=400
)
st.plotly_chart(fig_oil_water, use_container_width=True)

# GOR evolution
fig_gor = px.line(
    monthly_fluid,
    x='date',
    y='GOR',
    title="Evolución Temporal del GOR (m³gas/m³oil)",
    labels={'date': 'Fecha', 'GOR': 'GOR (m³/m³)'},
)
fig_gor.update_layout(height=400, hovermode='x unified')
st.plotly_chart(fig_gor, use_container_width=True)