import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from PIL import Image

from utils import (
    BARRELS_PER_M3,
    COMPANY_REPLACEMENTS,
    DATASET_URL,
)


# ── Data loading ──────────────────────────────────────────────────────────────

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

if st.session_state["df"].empty:
    del st.session_state["df"]
    st.error(
        "⚠️ Los datos almacenados están vacíos. "
        "Por favor, recargá la página para volver a cargarlos."
    )
    st.stop()

data_sorted = st.session_state["df"]
data_sorted["empresaNEW"] = data_sorted["empresa"].replace(COMPANY_REPLACEMENTS)

data_filtered = data_sorted[data_sorted["tef"] > 0]
latest_date_non_official = data_filtered["date"].max()
latest_date = latest_date_non_official - relativedelta(months=1)
latest_data = data_filtered[data_filtered["date"] == latest_date]

total_gas_rate_rounded = round(latest_data["gas_rate"].sum() / 1000, 1)
total_oil_rate_rounded = round(latest_data["oil_rate"].sum() / 1000, 1)
oil_rate_bpd_rounded   = round(total_oil_rate_rounded * BARRELS_PER_M3, 1)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image(Image.open("Vaca Muerta rig.png"))


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🌎 Dashboard de Producción No Convencional")
st.caption("Vaca Muerta · Datos oficiales de la Secretaría de Energía de Argentina")
st.divider()


# ── Fechas y estado del reporte ───────────────────────────────────────────────

st.subheader("📅 Estado del Reporte")

col_a, col_b = st.columns(2)
with col_a:
    st.metric("Fecha de Alocación en Progreso", str(latest_date_non_official.date()))
with col_b:
    st.metric("Última Alocación Consolidada", str(latest_date.date()))

st.info(
    "**¿Cuál es la diferencia entre estas dos fechas?**\n\n"
    "- **Alocación en Progreso**: es el mes más reciente con datos disponibles, "
    "pero aún no fue cerrado oficialmente. Los valores pueden cambiar hasta el cierre.\n\n"
    "- **Última Alocación Consolidada**: a mediados de cada mes se realiza el cierre "
    "oficial de los datos del mes anterior. Esta fecha representa el último mes con "
    "información completa, verificada y representativa. El reporte utiliza esta fecha "
    "para garantizar precisión y evitar mostrar información incompleta."
)


# ── KPIs del período consolidado ──────────────────────────────────────────────

st.subheader("⚡ Producción al Último Período Consolidado")

col1, col2, col3 = st.columns(3)
col1.metric(label="🔥 Caudal de Gas (MMm³/d)",     value=total_gas_rate_rounded)
col2.metric(label="🛢️ Caudal de Petróleo (km³/d)", value=total_oil_rate_rounded)
col3.metric(label="🛢️ Caudal de Petróleo (kbpd)",  value=oil_rate_bpd_rounded)

st.divider()


# ── Clasificación de fluidos: criterio McCain ─────────────────────────────────

st.subheader("🧪 Clasificación de Fluidos: Criterio McCain")
st.write(
    "A lo largo del dashboard, los pozos son clasificados como **Petrolíferos** o **Gasíferos** "
    "utilizando el criterio de GOR acumulado según McCain. Los pozos originalmente etiquetados "
    "como *'Otro tipo'* son reclasificados automáticamente con este criterio."
)
st.write(
    "**Regla**: si el GOR acumulado del pozo supera los **3.000 m³gas / m³oil**, "
    "el pozo se clasifica como Gasífero; de lo contrario, como Petrolífero."
)

st.image(Image.open("McCain.png"), caption="Clasificación de fluidos según McCain (GOR)", use_container_width=True)

st.divider()


# ── Índice de páginas ─────────────────────────────────────────────────────────

st.subheader("🗂️ Índice de Páginas")

pages = [
    ("📊 Producción General", "pages/2_📊_Real-time Production Report.py",
     "Evolución de caudales de gas y petróleo por empresa y por campaña (vintage). "
     "Incluye métricas operativas avanzadas: semáforo YoY, variación porcentual, "
     "evolución de GOR/WOR/WGR y market share por operador."),
    ("🧮 Análisis de Producción", "pages/2_🧮_Production_Analysis.py",
     "Análisis de producción agregada con filtros por empresa, área y tipo de recurso."),
    ("📊 Análisis por Pozo Individual", "pages/3_📊_Single-well_Analysis.py",
     "Curvas de decline, producción acumulada y métricas por pozo. "
     "Ideal para evaluar el desempeño individual."),
    ("📊 Comparación Multi-Pozo", "pages/4_📊_Multi-well_Comparison.py",
     "Comparación de curvas de producción entre múltiples pozos seleccionados."),
    ("🏆 Ranking y Records", "pages/5_🏆_Ranking.py",
     "Pozos de mayor producción pico y acumulada. Benchmarks de completación "
     "(lateral length, etapas, proppant) por campaña y empresa."),
    ("👩‍🔧 Reporte FracData en Tiempo Real", "pages/6_👩‍🔧_Real-time FracData Report.py",
     "Evolución P10/P50/P90 de indicadores de completación y productividad. "
     "Análisis por área con overlay comparativo."),
    ("🚨 Watchlist", "pages/7_🚨_Watchlist.py",
     "Pozos con comportamiento atípico o señales de alerta operativa."),
    ("🗂️ Gestión de Datos", "pages/8_🗂️_Data_Management.py",
     "Exploración, filtrado y exportación del dataset completo de producción."),
    ("📐 Análisis Avanzado Comparativo", "pages/9_📐_Advanced_Analytics_Comparison.py",
     "Benchmarks de cuadrantes de producción y completación con highlights "
     "por empresa, área o pozo individual."),
    ("🗺️ Análisis Geoespacial", "pages/10_🗺️ _Geospacial_Analysis.py",
     "Mapa de pozos con capa de producción, empresa y campaña."),
]

for name, _, description in pages:
    with st.container():
        st.markdown(f"**{name}**")
        st.caption(description)
        st.write("")
