"""
Single Well Analysis Page
=========================
Interactive production analysis for individual wells.
Data is shared from the main page via session state.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from typing import Optional, Tuple, Dict, List


# ── Constants ────────────────────────────────────────────────────────────────

PLOT_CONFIG = {
    "oil": {"color": "#00CC96", "title": "Petróleo", "unit": "m³/d", "cumulative": "Np", "cum_unit": "Mm³"},
    "gas": {"color": "#FF4B4B", "title": "Gas", "unit": "km³/d", "cumulative": "Gp", "cum_unit": "Mm³"},
    "water": {"color": "#4B9CD3", "title": "Agua", "unit": "m³/d", "cumulative": "Wp", "cum_unit": "Mm³"},
}

COLUMN_RENAMES = {
    "sigla": "Sigla",
    "date": "Fecha",
    "oil_rate": "Caudal Petróleo (m³/d)",
    "gas_rate": "Caudal Gas (km³/d)",
    "water_rate": "Caudal Agua (m³/d)",
    "Np": "Petroléo Acum (m³)",
    "Gp": "Gas Acum (m³)",
    "Wp": "Agua Acum (m³)",
    "tef": "TEF",
    "tipopozo": "Tipo Pozo",
    "empresa": "Empresa",
    "areayacimiento": "Área Yacimiento",
    "formprod": "Formación",
    "sub_tipo_recurso": "Recurso",
}

MAX_RATE_THRESHOLD = 1_000_000


# ── Data Preparation ─────────────────────────────────────────────────────────

def get_data_from_session() -> Optional[pd.DataFrame]:
    """Retrieve and prepare data from session state."""
    if "df" not in st.session_state:
        return None
    
    df = st.session_state["df"].copy()
    
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(
            df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-1"
        )
    
    # Calculate rates if missing
    for fluid, prod_col in [("gas", "prod_gas"), ("oil", "prod_pet"), ("water", "prod_agua")]:
        rate_col = f"{fluid}_rate"
        if rate_col not in df.columns:
            df[rate_col] = df[prod_col] / df["tef"]
    
    # Calculate ratios safely
    df["gor"] = np.where(df["oil_rate"] > 0, df["gas_rate"] / df["oil_rate"], np.nan)
    df["wor"] = np.where(df["oil_rate"] > 0, df["water_rate"] / df["oil_rate"], np.nan)
    
    return df.sort_values(by=["sigla", "date"])


def calculate_months_on_production(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized time column (months since first production)."""
    df = df.copy()
    min_date = df["date"].min()
    df["months_on_prod"] = ((df["date"] - min_date).dt.days / 30.44).round().astype(int) + 1
    return df


def get_benchmark_data(df: pd.DataFrame, well_types: List[str]) -> pd.DataFrame:
    """Calculate P50 and average curves for all wells of same type."""
    if not well_types:
        return pd.DataFrame()
        
    type_data = df[df["tipopozo"].isin(well_types)].copy()
    
    if type_data.empty:
        return pd.DataFrame()
    
    # Normalize time for each well
    normalized_data = []
    for well in type_data["sigla"].unique():
        well_data = type_data[type_data["sigla"] == well].copy()
        well_data = calculate_months_on_production(well_data)
        normalized_data.append(well_data)
    
    combined = pd.concat(normalized_data, ignore_index=True)
    
    # Calculate statistics by month
    stats = (
        combined.groupby("months_on_prod")
        .agg({
            "oil_rate": ["mean", "median", "count"],
            "gas_rate": ["mean", "median", "count"],
            "water_rate": ["mean", "median"],
        })
        .reset_index()
    )
    
    # Flatten column names
    stats.columns = [
        "months_on_prod",
        "oil_mean", "oil_p50", "oil_count",
        "gas_mean", "gas_p50", "gas_count",
        "water_mean", "water_p50",
    ]
    
    return stats[stats["oil_count"] >= 3]


# ── Visualization Components ─────────────────────────────────────────────────

def create_rate_plot_with_benchmark(
    well_data: pd.DataFrame, 
    benchmark: pd.DataFrame, 
    fluid: str, 
    well_name: str
) -> go.Figure:
    """Create production rate plot with benchmark overlay."""
    config = PLOT_CONFIG[fluid]
    fig = go.Figure()
    
    well_data_norm = calculate_months_on_production(well_data)
    
    # Selected well
    fig.add_trace(go.Scatter(
        x=well_data_norm["months_on_prod"],
        y=well_data_norm[f"{fluid}_rate"],
        mode="lines+markers",
        name=well_name,
        line=dict(color=config["color"], width=3),
        marker=dict(size=6),
    ))
    
    # Benchmark
    if not benchmark.empty and f"{fluid}_mean" in benchmark.columns:
        fig.add_trace(go.Scatter(
            x=benchmark["months_on_prod"],
            y=benchmark[f"{fluid}_mean"],
            mode="lines",
            name=f"Promedio Tipo",
            line=dict(color="gray", width=2, dash="dash"),
            opacity=0.7,
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark["months_on_prod"],
            y=benchmark[f"{fluid}_p50"],
            mode="lines",
            name=f"P50 Tipo",
            line=dict(color="orange", width=2, dash="dot"),
            opacity=0.7,
        ))
    
    fig.update_layout(
        title=f"Caudal de {config['title']} vs Benchmark",
        xaxis_title="Meses en Producción",
        yaxis_title=f"Caudal ({config['unit']})",
        yaxis=dict(rangemode="tozero"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def create_cumulative_plot(well_data: pd.DataFrame, fluid: str) -> go.Figure:
    """Create cumulative production plot."""
    config = PLOT_CONFIG[fluid]
    cum_col = config["cumulative"]
    
    if cum_col not in well_data.columns:
        return go.Figure()
    
    fig = go.Figure()
    cum_values = well_data[cum_col] / 1_000_000  # Convert to millions
    
    fig.add_trace(go.Scatter(
        x=well_data["date"],
        y=cum_values,
        mode="lines",
        fill="tozeroy",
        line=dict(color=config["color"]),
        name=f"{config['title']} Acumulado",
    ))
    
    fig.update_layout(
        title=f"Producción Acumulada de {config['title']}",
        xaxis_title="Fecha",
        yaxis_title=f"Volumen ({config['cum_unit']})",
        hovermode="x unified",
    )
    
    return fig


def create_ratio_plot(well_data: pd.DataFrame) -> go.Figure:
    """Create GOR and WOR trend plot."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Gas-Oil Ratio (GOR)", "Water-Oil Ratio (WOR)"),
        vertical_spacing=0.12,
    )
    
    # GOR
    valid_gor = well_data["gor"].notna() & (well_data["gor"] > 0) & np.isfinite(well_data["gor"])
    if valid_gor.any():
        fig.add_trace(
            go.Scatter(
                x=well_data.loc[valid_gor, "date"],
                y=well_data.loc[valid_gor, "gor"],
                mode="lines+markers",
                line=dict(color="orange", width=2),
                name="GOR",
            ),
            row=1, col=1,
        )
    
    # WOR
    valid_wor = well_data["wor"].notna() & (well_data["wor"] > 0) & np.isfinite(well_data["wor"])
    if valid_wor.any():
        fig.add_trace(
            go.Scatter(
                x=well_data.loc[valid_wor, "date"],
                y=well_data.loc[valid_wor, "wor"],
                mode="lines+markers",
                line=dict(color="purple", width=2),
                name="WOR",
            ),
            row=2, col=1,
        )
    
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="GOR (km³/m³)", row=1, col=1)
    fig.update_yaxes(title_text="WOR (m³/m³)", row=2, col=1)
    fig.update_layout(height=600, showlegend=False, hovermode="x unified")
    
    return fig


# ── UI Components ─────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> Tuple[List[str], str, str]:
    """Render sidebar filters."""
    st.sidebar.image(Image.open("Vaca Muerta rig.png"))
    st.sidebar.title("Filtros")
    
    well_types = sorted(df["tipopozo"].unique())
    selected_types = st.sidebar.multiselect(
        "Tipo de pozo:",
        options=well_types,
        default=well_types[:1] if well_types else None,
    )
    
    companies = sorted(df["empresa"].unique())
    selected_company = st.sidebar.selectbox("Operadora:", options=companies)
    
    mask = (df["tipopozo"].isin(selected_types)) & (df["empresa"] == selected_company)
    available_wells = sorted(df.loc[mask, "sigla"].unique())
    
    selected_well = st.sidebar.selectbox(
        "Sigla del pozo:",
        options=available_wells if available_wells else ["No disponible"],
        disabled=len(available_wells) == 0,
    )
    
    return selected_types, selected_company, selected_well


def render_metrics(well_data: pd.DataFrame):
    """Display production metrics."""
    max_rates = {}
    for fluid in ["oil", "gas", "water"]:
        rate_col = f"{fluid}_rate"
        valid_data = well_data[well_data[rate_col] <= MAX_RATE_THRESHOLD][rate_col]
        max_rates[fluid] = valid_data.max() if not valid_data.empty else 0
    
    latest = well_data.iloc[-1] if not well_data.empty else None
    latest_gor = latest["gor"] if latest is not None and pd.notna(latest["gor"]) else 0
    latest_wor = latest["wor"] if latest is not None and pd.notna(latest["wor"]) else 0
    
    cols = st.columns(5)
    metrics = [
        (":green[Petróleo Max]", max_rates["oil"], "m³/d"),
        (":red[Gas Max]", max_rates["gas"], "km³/d"),
        (":blue[Agua Max]", max_rates["water"], "m³/d"),
        (":orange[GOR Actual]", latest_gor, "km³/m³"),
        (":violet[WOR Actual]", latest_wor, "m³/m³"),
    ]
    
    for col, (label, value, unit) in zip(cols, metrics):
        display_val = f"{value:,.1f}" if pd.notna(value) and value > 0 else "N/A"
        col.metric(label=label, value=f"{display_val} {unit}")


# ── Main Application ─────────────────────────────────────────────────────────

def main():
    st.title(":blue[Análisis de Pozo Individual]")
    st.caption("Capítulo IV - Benchmarking & Análisis de Fluidos")
    
    df = get_data_from_session()
    if df is None:
        st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la **Página Principal**.")
        st.stop()
    
    st.success("✅ Datos cargados desde memoria")
    
    selected_types, selected_company, selected_well = render_sidebar(df)
    
    if selected_well == "No disponible" or not selected_well:
        st.error("Selecciona un pozo válido.")
        st.stop()
    
    # Filter data
    well_data = df[
        (df["empresa"] == selected_company) & 
        (df["sigla"] == selected_well)
    ].copy()
    
    if well_data.empty:
        st.error(f"No se encontraron datos para {selected_well}")
        st.stop()
    
    well_data = calculate_months_on_production(well_data)
    benchmark_data = get_benchmark_data(df, selected_types)
    
    # Header
    st.header(selected_well)
    tipo_pozo = well_data["tipopozo"].iloc[0]
    area = well_data["areayacimiento"].iloc[0]
    formacion = well_data["formprod"].iloc[0]
    st.write(f"**Tipo:** {tipo_pozo} | **Área:** {area} | **Formación:** {formacion}")
    render_metrics(well_data)
    
    # Tabs (sin DCA)
    tab_rates, tab_cumul, tab_ratios = st.tabs([
        "📈 Producción vs Benchmark", 
        "📊 Acumuladas", 
        "⚗️ Relaciones de Fluidos"
    ])
    
    with tab_rates:
        st.subheader("Comparación con Pozos Similares")
        st.caption("Línea sólida: Pozo seleccionado | Gris: Promedio | Naranja: P50")
        
        for fluid in ["oil", "gas", "water"]:
            fig = create_rate_plot_with_benchmark(well_data, benchmark_data, fluid, selected_well)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_cumul:
        st.subheader("Producción Acumulada")
        cols = st.columns(3)
        for idx, fluid in enumerate(["oil", "gas", "water"]):
            cum_col = PLOT_CONFIG[fluid]["cumulative"]
            if cum_col in well_data.columns:
                total = well_data[cum_col].iloc[-1] / 1_000_000
                with cols[idx]:
                    st.metric(
                        label=f"Total {PLOT_CONFIG[fluid]['title']}", 
                        value=f"{total:,.1f} {PLOT_CONFIG[fluid]['cum_unit']}"
                    )
        
        for fluid in ["oil", "gas", "water"]:
            fig = create_cumulative_plot(well_data, fluid)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_ratios:
        st.subheader("Tendencias de Relaciones de Fluidos")
        c1, c2 = st.columns(2)
        with c1:
            avg_gor = well_data["gor"].mean()
            st.metric("GOR Promedio", f"{avg_gor:.1f}" if pd.notna(avg_gor) else "N/A")
        with c2:
            avg_wor = well_data["wor"].mean()
            st.metric("WOR Promedio", f"{avg_wor:.2f}" if pd.notna(avg_wor) else "N/A")
        
        st.plotly_chart(create_ratio_plot(well_data), use_container_width=True)
        
        with st.expander("ℹ️ Interpretación"):
            st.markdown("""
            - **GOR ↑**: Caída de presión, conificación de gas, o drenaje de gas en solución
            - **WOR ↑**: Breakthrough de agua, conificación, o comunicación con acuífero
            - **Tendencia estable**: Comportamiento de drenaje uniforme
            """)
    
    # Export
    st.divider()
    with st.expander("📥 Descargar Datos"):
        display_df = well_data.rename(columns=COLUMN_RENAMES)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name=f"{selected_well}_analysis.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
