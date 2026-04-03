"""
Single Well Analysis Page
=========================
Interactive production analysis for individual wells.
Data is shared from the main page via session state.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image


# ── Constants ────────────────────────────────────────────────────────────────

PLOT_CONFIG = {
    "gas": {"color": "red", "title": "Gas", "unit": "km³/d"},
    "oil": {"color": "green", "title": "Petróleo", "unit": "m³/d"},
    "water": {"color": "blue", "title": "Agua", "unit": "m³/d"},
}

COLUMN_RENAMES = {
    "sigla": "Sigla",
    "date": "Fecha",
    "oil_rate": "Caudal de petróleo (m³/d)",
    "gas_rate": "Caudal de gas (m³/d)",
    "water_rate": "Caudal de agua (m³/d)",
    "Np": "Acumulada de Petróleo (m³)",
    "Gp": "Acumulada de Gas (m³)",
    "Wp": "Acumulada de Agua (m³)",
    "tef": "TEF",
    "tipoextraccion": "Tipo de Extracción",
    "tipopozo": "Tipo de Pozo",
    "empresa": "Empresa",
    "formacion": "Formación",
    "areayacimiento": "Área yacimiento",
}

MAX_RATE_THRESHOLD = 1_000_000


# ── Data Loading ─────────────────────────────────────────────────────────────

def get_data_from_session() -> pd.DataFrame | None:
    """Retrieve and prepare data from session state."""
    if "df" not in st.session_state:
        return None
    
    df = st.session_state["df"].copy()
    
    # Ensure date column exists
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(
            df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-1"
        )
    
    # Ensure rate columns exist
    if "gas_rate" not in df.columns:
        df["gas_rate"] = df["prod_gas"] / df["tef"]
    if "oil_rate" not in df.columns:
        df["oil_rate"] = df["prod_pet"] / df["tef"]
    if "water_rate" not in df.columns:
        df["water_rate"] = df["prod_agua"] / df["tef"]
    
    return df.sort_values(by=["sigla", "date"])


# ── UI Components ─────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> tuple:
    """Render sidebar filters and return selected values."""
    st.sidebar.image(Image.open("Vaca Muerta rig.png"))
    st.sidebar.title("Filtros")
    
    # Well type filter
    well_types = df["tipopozo"].unique()
    selected_types = st.sidebar.multiselect(
        "Tipo de pozo:",
        options=well_types,
        default=well_types[0] if len(well_types) > 0 else None,
    )
    
    # Company filter
    companies = df["empresa"].unique()
    selected_company = st.sidebar.selectbox("Operadora:", options=companies)
    
    # Well filter (dependent on company and type)
    available_wells = df[
        (df["tipopozo"].isin(selected_types)) & 
        (df["empresa"] == selected_company)
    ]["sigla"].unique()
    
    selected_well = st.sidebar.selectbox(
        "Sigla del pozo:",
        options=available_wells if len(available_wells) > 0 else ["No disponible"],
        disabled=len(available_wells) == 0,
    )
    
    return selected_types, selected_company, selected_well


def render_metrics(well_data: pd.DataFrame):
    """Display key production metrics for the selected well."""
    # Calculate max rates (capped at threshold to filter outliers)
    max_rates = {
        fluid: well_data.loc[well_data[f"{fluid}_rate"] <= MAX_RATE_THRESHOLD, f"{fluid}_rate"].max()
        for fluid in ["gas", "oil", "water"]
    }
    
    cols = st.columns(3)
    metrics = [
        (":red[Caudal Máximo de Gas]", max_rates["gas"], "km³/d"),
        (":green[Caudal Máximo de Petróleo]", max_rates["oil"], "m³/d"),
        (":blue[Caudal Máximo de Agua]", max_rates["water"], "m³/d"),
    ]
    
    for col, (label, value, unit) in zip(cols, metrics):
        col.metric(label=label, value=f"{value:,.1f} {unit}" if pd.notna(value) else "N/A")


def create_rate_plot(well_data: pd.DataFrame, fluid: str, well_name: str) -> go.Figure:
    """Create a production rate plot for the specified fluid."""
    config = PLOT_CONFIG[fluid]
    rate_col = f"{fluid}_rate"
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=well_data["date"],
            y=well_data[rate_col],
            mode="lines+markers",
            name=f"{config['title']} Rate",
            line=dict(color=config["color"]),
        )
    )
    
    fig.update_layout(
        title=f"Historia de Producción de {config['title']} - {well_name}",
        xaxis_title="Fecha",
        yaxis_title=f"Caudal de {config['title']} ({config['unit']})",
        yaxis=dict(rangemode="tozero"),
        hovermode="x unified",
    )
    
    return fig


def render_download_section(well_data: pd.DataFrame, well_name: str):
    """Render data table and CSV download option."""
    display_df = well_data.rename(columns=COLUMN_RENAMES)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar datos como CSV",
        data=csv,
        file_name=f"{well_name}_production.csv",
        mime="text/csv",
    )


# ── Main Application ───────────────────────────────────────────────────────────

def main():
    st.title(":blue[Análisis de Pozo Individual]")
    st.caption("Capítulo IV - Producción No Convencional")
    
    # Data retrieval
    df = get_data_from_session()
    if df is None:
        st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la **Página Principal**.")
        st.stop()
    
    st.success("✅ Datos cargados desde memoria")
    
    # Sidebar filters
    selected_types, selected_company, selected_well = render_sidebar(df)
    
    if selected_well == "No disponible":
        st.error("No hay pozos disponibles para los filtros seleccionados.")
        st.stop()
    
    # Filter data for selected well
    well_data = df[
        (df["empresa"] == selected_company) & 
        (df["sigla"] == selected_well)
    ].copy()
    
    if well_data.empty:
        st.error(f"No se encontraron datos para el pozo {selected_well}")
        st.stop()
    
    # Header and metrics
    st.header(selected_well)
    render_metrics(well_data)
    
    # Production plots
    st.subheader("Históricos de Producción")
    for fluid in ["gas", "oil", "water"]:
        st.plotly_chart(create_rate_plot(well_data, fluid, selected_well), use_container_width=True)
    
    # Data export
    st.subheader("Datos Detallados")
    render_download_section(well_data, selected_well)


if __name__ == "__main__":
    main()