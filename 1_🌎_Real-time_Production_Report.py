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
# ── GRÁFICO YoY ────────────────────────────────────────────────────────
# Agregar totales mensuales
monthly_totals = (
    data_filtered
    .groupby('date')
    .agg(oil_rate=('oil_rate', 'sum'), gas_rate=('gas_rate', 'sum'))
    .reset_index()
    .sort_values('date')
)
monthly_totals['oil_rate_km3'] = monthly_totals['oil_rate'] / 1000
monthly_totals['gas_rate_mm3'] = monthly_totals['gas_rate'] / 1000
monthly_totals['anio']         = monthly_totals['date'].dt.year
monthly_totals['mes']          = monthly_totals['date'].dt.month

# Merge con año anterior
yoy = monthly_totals.merge(
    monthly_totals[['anio', 'mes', 'oil_rate_km3', 'gas_rate_mm3']],
    left_on=['anio', 'mes'],
    right_on=[monthly_totals['anio'] + 1, monthly_totals['mes']],
    suffixes=('', '_prev')
)
yoy['oil_yoy_pct'] = (yoy['oil_rate_km3'] - yoy['oil_rate_km3_prev']) / yoy['oil_rate_km3_prev'] * 100
yoy['gas_yoy_pct'] = (yoy['gas_rate_mm3'] - yoy['gas_rate_mm3_prev']) / yoy['gas_rate_mm3_prev'] * 100

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
st.plotly_chart(fig_yoy)