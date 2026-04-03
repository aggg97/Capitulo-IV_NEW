"""
Watchlist page - Top performing wells in Vaca Muerta.
Displays current top 5 gas and oil producers with rankings.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from utils import COMPANY_REPLACEMENTS


# ── Constants ─────────────────────────────────────────────────────────────────

TOP_N_WELLS = 5
DATA_MEMORY_KEY = "df"
IMAGE_PATH = "Vaca Muerta rig.png"


# ── Data loading ───────────────────────────────────────────────────────────────

def get_data_from_session() -> pd.DataFrame | None:
    """Retrieve and process data from session state."""
    if DATA_MEMORY_KEY not in st.session_state:
        return None
    
    df = st.session_state[DATA_MEMORY_KEY].copy()
    df["date"] = pd.to_datetime(
        df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-1"
    )
    df["gas_rate"] = df["prod_gas"] / df["tef"]
    df["oil_rate"] = df["prod_pet"] / df["tef"]
    
    return df.sort_values(by=["sigla", "date"], ascending=True)


# ── UI Components ──────────────────────────────────────────────────────────────

def render_header():
    """Render page header and sidebar."""
    st.header(":blue[🚨 Watchlist - Nuevos Pozos en Vaca Muerta]")
    st.sidebar.image(Image.open(IMAGE_PATH))


def render_data_status(df: pd.DataFrame | None) -> pd.DataFrame:
    """Handle data loading status and return valid dataframe."""
    if df is None:
        st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la Página Principal.")
        st.stop()
    
    st.info("Utilizando datos recuperados de la memoria.")
    return df


# ── Data Processing ────────────────────────────────────────────────────────────

def get_latest_valid_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Filter valid data and get latest date metrics."""
    valid_data = df[df["tef"] > 0]
    latest_date = valid_data["date"].max()
    latest_data = valid_data[valid_data["date"] == latest_date]
    
    return latest_data, latest_date


def get_top_producers(data: pd.DataFrame, n: int = TOP_N_WELLS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get top N gas and oil producers."""
    top_gas = data.nlargest(n, "gas_rate")
    top_oil = data.nlargest(n, "oil_rate")
    return top_gas, top_oil


def standardize_company_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply company name standardization."""
    df["empresaNEW"] = df["empresa"].replace(COMPANY_REPLACEMENTS)
    return df


# ── Visualization ──────────────────────────────────────────────────────────────

def create_production_bar_chart(
    data: pd.DataFrame,
    metric: str,
    title: str,
    x_label: str,
    color_col: str = "empresaNEW"
) -> px.Figure:
    """Create horizontal bar chart for production ranking."""
    sorted_data = data.sort_values(by=metric)
    
    fig = px.bar(
        sorted_data,
        y="sigla",
        x=metric,
        color=color_col,
        orientation="h",
        labels={
            metric: x_label,
            "sigla": "Pozo",
            color_col: "Empresa",
            "areayacimiento": "Bloque",
        },
        text=metric,
        hover_data=[color_col, "areayacimiento"],
    )
    
    fig.update_traces(texttemplate="%{text:.2f}", textposition="inside")
    fig.update_layout(
        title=title,
        yaxis=dict(categoryorder="total ascending"),
        yaxis_title=None,
    )
    
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    render_header()
    
    df = get_data_from_session()
    df = render_data_status(df)
    df = standardize_company_names(df)
    
    latest_data, latest_date = get_latest_valid_data(df)
    st.write("Fecha de Alocación en Progreso: ", latest_date.date())
    
    top_gas, top_oil = get_top_producers(latest_data)
    
    # Gas ranking
    st.subheader("🔥 Ranking actual de los 5 pozos de gas más productivos de la Cuenca")
    fig_gas = create_production_bar_chart(
        top_gas,
        metric="gas_rate",
        title="Producción de Gas (km³/d)",
        x_label="Producción de Gas (m³/día)",
    )
    st.plotly_chart(fig_gas, use_container_width=True)
    
    # Oil ranking
    st.subheader("🔥 Ranking actual de los 5 pozos de petróleo más productivos de la Cuenca")
    fig_oil = create_production_bar_chart(
        top_oil,
        metric="oil_rate",
        title="Producción de Petróleo (m³/d)",
        x_label="Producción de Petróleo (m³/día)",
    )
    st.plotly_chart(fig_oil, use_container_width=True)
    
    # Disclaimer
    st.markdown("""
    ⚠️ **Nota:** Al evaluar la productividad en Vaca Muerta, es importante tener precaución 
    con los pozos considerados "más productivos", únicamente por su caudal máximo.
    
    El caudal máximo registrado suele estar influenciado por el *choke management*, así también 
    como la interferencia de pozos en un mismo PAD. Este fenómeno está relacionado con el 
    concepto del SRV *(Stimulated Rock Volume)*.
    
    Por lo tanto, una evaluación más representativa de la productividad debería realizarse 
    a nivel de PAD y no de manera individual por pozo.
    """)


if __name__ == "__main__":
    main()