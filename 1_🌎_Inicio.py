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
    "utilizando el criterio de GOR según McCain, adaptado para pozos no convencionales. "
    "Los pozos originalmente etiquetados como *'Otro tipo'* son reclasificados automáticamente "
    "con este criterio."
)
st.write(
    "**Metodología**: en pozos convencionales McCain propone usar el GOR inicial. "
    "En no convencional, el GOR inicial es artificialmente bajo durante el flujo transiente "
    "temprano y no representa la composición del fluido del reservorio. "
    "Por eso se usa el **GOR instantáneo a los 180 días** desde el primer registro de producción "
    "— punto donde el pozo ya superó el período de limpieza y el GOR refleja mejor el fluido "
    "del yacimiento. Si el pozo tiene menos de 180 días de historia, se toma el último registro disponible."
)
st.write(
    "**Regla**: GOR instantáneo > **3.000 m³gas / m³oil** a los 180 días → Gasífero. "
    "Si `Qo = 0` en ese punto → Gasífero directamente. Caso contrario → Petrolífero."
)

st.image(Image.open("McCain.png"), caption="Clasificación de fluidos según McCain (GOR)", use_container_width=True)

st.divider()


# ── Índice de páginas ─────────────────────────────────────────────────────────

st.subheader("🗂️ Índice de Páginas")

pages = [
    ("📊 Real-time Production Report",
     "Visualización de la evolución de la producción de petróleo, gas y agua por empresa, área y campaña. "
     "Incluye indicadores clave de desempeño (KPIs) como variación interanual (YoY), semáforo de tendencias, "
     "evolución de GOR, WOR y WGR, participación de mercado (market share) y métricas operativas agregadas."),
    ("🧮 Análisis de Producción",
     "Análisis detallado mediante gráficos de diagnóstico para evaluar tendencias productivas. "
     "Permite segmentar la información por empresa, área, formación, tipo de recurso y campaña "
     "para identificar patrones de desempeño."),
    ("📈 Análisis por Pozo Individual",
     "Evaluación del desempeño de cada pozo mediante curvas de producción, análisis de declinación, "
     "producción acumulada e indicadores operativos. Ideal para estudiar el comportamiento y la productividad individual."),
    ("📊 Comparación Multi-Pozo",
     "Comparación simultánea de múltiples pozos mediante curvas de producción normalizadas y acumuladas, "
     "facilitando el análisis de desempeño entre campañas, operadores o áreas."),
    ("🏆 Ranking y Récords",
     "Ranking de los pozos con mayor producción pico y acumulada. Incluye benchmarks de completación "
     "(longitud lateral, número de etapas, volumen de proppant, intensidad de estimulación y otros parámetros) "
     "segmentados por campaña, empresa y área."),
    ("👩‍🔧 Reporte FracData en Tiempo Real",
     "Seguimiento en tiempo real de indicadores de completación utilizando estadísticas P10, P50 y P90. "
     "Permite monitorear la evolución de las estrategias de fractura y su impacto en la productividad de los pozos."),
    ("🚨 Watchlist",
     "Monitoreo automático de los pozos con mejor desempeño productivo en la actualidad. "
     "Destaca nuevos récords, incrementos significativos de producción y pozos que requieren seguimiento."),
    ("🗂️ Gestión de Datos",
     "Herramientas para explorar, filtrar y exportar el conjunto completo de datos de producción. "
     "Incluye análisis de calidad de datos, detección de información faltante o inconsistente, "
     "cobertura temporal y completitud por empresa y área."),
    ("📐 Análisis Avanzado Comparativo",
     "Benchmarks avanzados mediante gráficos de cuadrantes y análisis estadísticos. "
     "Incluye comparación de desempeño entre empresas, áreas y pozos individuales, análisis por percentiles, "
     "métricas normalizadas (por etapas, longitud lateral u otras variables) y evaluación de productividad "
     "versus parámetros de completación."),
    ("🗺️ Análisis Geoespacial",
     "Evaluación espacial de la producción mediante agrupación de pozos en pads utilizando GeoPandas. "
     "Permite reducir sesgos asociados a la interferencia entre pozos (parent-child wells), analizar la "
     "influencia del desarrollo del pad y realizar comparaciones más robustas entre áreas y operadores."),
]

for name, description in pages:
    with st.container():
        st.markdown(f"**{name}**")
        st.caption(description)
        st.write("")
