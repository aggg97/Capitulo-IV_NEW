"""
Single Well Analysis Page - Enhanced Edition
============================================
Interactive production analysis with benchmarking, decline curves,
cumulative tracking, and fluid ratio analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from scipy.optimize import curve_fit
from typing import Optional, Tuple, Dict, List


# ── Constants ────────────────────────────────────────────────────────────────

PLOT_CONFIG = {
    "gas": {"color": "#FF4B4B", "title": "Gas", "unit": "km³/d", "cumulative": "Gp", "cum_unit": "Mm³"},
    "oil": {"color": "#00CC96", "title": "Petróleo", "unit": "m³/d", "cumulative": "Np", "cum_unit": "Mm³"},
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
MIN_DATA_POINTS = 6  # Minimum points for decline curve fitting


# ── Data Preparation ─────────────────────────────────────────────────────────

def get_data_from_session() -> Optional[pd.DataFrame]:
    """Retrieve and prepare data from session state."""
    if "df" not in st.session_state:
        return None
    
    df = st.session_state["df"].copy()
    
    # Ensure derived columns exist
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(
            df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-1"
        )
    
    if "gas_rate" not in df.columns:
        df["gas_rate"] = df["prod_gas"] / df["tef"]
    if "oil_rate" not in df.columns:
        df["oil_rate"] = df["prod_pet"] / df["tef"]
    if "water_rate" not in df.columns:
        df["water_rate"] = df["prod_agua"] / df["tef"]
    
    # Calculate ratios safely
    df["gor"] = np.where(df["oil_rate"] > 0, df["gas_rate"] / df["oil_rate"], np.nan)
    df["wor"] = np.where(df["oil_rate"] > 0, df["water_rate"] / df["oil_rate"], np.nan)
    df["wgr"] = np.where(df["gas_rate"] > 0, df["water_rate"] / df["gas_rate"], np.nan)
    
    return df.sort_values(by=["sigla", "date"])


def calculate_months_on_production(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized time column (months since first production)."""
    df = df.copy()
    df["months_on_prod"] = (
        (df["date"] - df["date"].min()) / np.timedelta64(1, "M")
    ).round().astype(int) + 1
    return df


def get_benchmark_data(df: pd.DataFrame, well_types: List[str]) -> pd.DataFrame:
    """Calculate P50 and average curves for all wells of same type."""
    type_data = df[df["tipopozo"].isin(well_types)].copy()
    
    # Normalize time for each well
    normalized_data = []
    for well in type_data["sigla"].unique():
        well_data = type_data[type_data["sigla"] == well].copy()
        well_data = calculate_months_on_production(well_data)
        normalized_data.append(well_data)
    
    if not normalized_data:
        return pd.DataFrame()
    
    combined = pd.concat(normalized_data)
    
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
    
    return stats[stats["oil_count"] >= 3]  # At least 3 wells for statistics


# ── Decline Curve Analysis ───────────────────────────────────────────────────

def arps_exponential(t: np.ndarray, qi: float, D: float) -> np.ndarray:
    """Exponential decline: q(t) = qi * exp(-D*t)"""
    return qi * np.exp(-D * t)


def arps_hyperbolic(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    """Hyperbolic decline: q(t) = qi / (1 + b*Di*t)^(1/b)"""
    # Avoid division issues
    with np.errstate(divide='ignore', invalid='ignore'):
        result = qi / np.power(1 + b * Di * t, 1/b)
    return np.where(np.isfinite(result), result, 0)


def fit_decline_curve(time: np.ndarray, rate: np.ndarray, model: str = "exponential"):
    """Fit decline curve to production data."""
    if len(time) < MIN_DATA_POINTS or np.all(rate <= 0):
        return None, None
    
    # Normalize time to start at 0
    t_normalized = time - time.min()
    valid_idx = rate > 0
    t_valid = t_normalized[valid_idx]
    q_valid = rate[valid_idx]
    
    if len(t_valid) < MIN_DATA_POINTS:
        return None, None
    
    try:
        if model == "exponential":
            p0 = [q_valid[0], 0.1]
            bounds = ([0, 0], [q_valid[0]*10, 5])
            popt, _ = curve_fit(arps_exponential, t_valid, q_valid, p0=p0, bounds=bounds, maxfev=5000)
            return popt, lambda t: arps_exponential(t, *popt)
        else:  # hyperbolic
            p0 = [q_valid[0], 0.1, 0.5]
            bounds = ([0, 0, 0.1], [q_valid[0]*10, 5, 2.0])
            popt, _ = curve_fit(arps_hyperbolic, t_valid, q_valid, p0=p0, bounds=bounds, maxfev=5000)
            return popt, lambda t: arps_hyperbolic(t, *popt)
    except (RuntimeError, ValueError):
        return None, None


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
    
    available_wells = df[
        (df["tipopozo"].isin(selected_types)) & 
        (df["empresa"] == selected_company)
    ]["sigla"].unique()
    
    selected_well = st.sidebar.selectbox(
        "Sigla del pozo:",
        options=sorted(available_wells),
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
    
    latest_gor = well_data["gor"].iloc[-1] if not well_data.empty and pd.notna(well_data["gor"].iloc[-1]) else 0
    latest_wor = well_data["wor"].iloc[-1] if not well_data.empty and pd.notna(well_data["wor"].iloc[-1]) else 0
    
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


# ── Plotting Functions ───────────────────────────────────────────────────────

def create_rate_plot_with_benchmark(
    well_data: pd.DataFrame, 
    benchmark: pd.DataFrame, 
    fluid: str, 
    well_name: str
) -> go.Figure:
    """Create production rate plot with benchmark overlay."""
    config = PLOT_CONFIG[fluid]
    fig = go.Figure()
    
    # Selected well data
    well_data_norm = calculate_months_on_production(well_data)
    
    fig.add_trace(go.Scatter(
        x=well_data_norm["months_on_prod"],
        y=well_data_norm[f"{fluid}_rate"],
        mode="lines+markers",
        name=f"{well_name}",
        line=dict(color=config["color"], width=3),
        marker=dict(size=6),
    ))
    
    # Benchmark data if available
    if not benchmark.empty and f"{fluid}_mean" in benchmark.columns:
        fig.add_trace(go.Scatter(
            x=benchmark["months_on_prod"],
            y=benchmark[f"{fluid}_mean"],
            mode="lines",
            name=f"Promedio {config['title']}",
            line=dict(color="gray", width=2, dash="dash"),
            opacity=0.7,
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark["months_on_prod"],
            y=benchmark[f"{fluid}_p50"],
            mode="lines",
            name=f"P50 {config['title']}",
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
    
    # Convert to millions for readability
    cum_values = well_data[cum_col] / 1_000_000
    
    fig.add_trace(go.Scatter(
        x=well_data["date"],
        y=cum_values,
        mode="lines+markers",
        fill="tozeroy",
        line=dict(color=config["color"]),
        name=f"{config['title']} Acumulado",
    ))
    
    fig.update_layout(
        title=f"Producción Acumulada de {config['title']}",
        xaxis_title="Fecha",
        yaxis_title=f"Volumen Acumulado ({config['cum_unit']})",
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
    
    # GOR plot
    fig.add_trace(
        go.Scatter(
            x=well_data["date"],
            y=well_data["gor"],
            mode="lines+markers",
            line=dict(color="orange", width=2),
            name="GOR",
        ),
        row=1, col=1,
    )
    
    # WOR plot
    fig.add_trace(
        go.Scatter(
            x=well_data["date"],
            y=well_data["wor"],
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


def create_decline_analysis_plot(well_data: pd.DataFrame, fluid: str) -> Tuple[go.Figure, Dict]:
    """Create decline curve analysis with forecast."""
    config = PLOT_CONFIG[fluid]
    rate_col = f"{fluid}_rate"
    
    fig = go.Figure()
    well_norm = calculate_months_on_production(well_data)
    
    months = well_norm["months_on_prod"].values
    rates = well_norm[rate_col].values
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=months,
        y=rates,
        mode="markers",
        name="Datos Históricos",
        marker=dict(color=config["color"], size=8),
    ))
    
    results = {"fluid": fluid, "model": None, "params": None, "euro": 0}
    
    if len(months) >= MIN_DATA_POINTS:
        # Fit both models
        for model_name, model_label, color in [("exponential", "Exp", "red"), ("hyperbolic", "Hyp", "blue")]:
            params, func = fit_decline_curve(months, rates, model_name)
            
            if func is not None:
                # Generate smooth curve
                t_smooth = np.linspace(months.min(), months.max() + 24, 200)  # Extend 24 months for forecast
                q_smooth = func(t_smooth - months.min())  # Normalize to start at 0
                
                fig.add_trace(go.Scatter(
                    x=t_smooth,
                    y=q_smooth,
                    mode="lines",
                    name=f"Ajuste {model_label}",
                    line=dict(color=color, dash="dash" if model_name == "exponential" else "dot"),
                ))
                
                if results["model"] is None:  # Store first successful fit
                    results["model"] = model_name
                    results["params"] = params
                    # Calculate EUR (Estimated Ultimate Recovery) - simplified
                    if model_name == "exponential" and len(params) == 2:
                        qi, D = params
                        # EUR = qi/D for exponential (theoretical infinite time)
                        results["euro"] = qi / D if D > 0 else 0
    
    fig.update_layout(
        title=f"Análisis de Declinación - {config['title']}",
        xaxis_title="Meses en Producción",
        yaxis_title=f"Caudal ({config['unit']})",
        yaxis=dict(type="log" if st.session_state.get("log_scale_decline", False) else "linear"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig, results


# ── Main Application ─────────────────────────────────────────────────────────

def main():
    st.title(":blue[Análisis Avanzado de Pozo Individual]")
    st.caption("Capítulo IV - Producción No Convencional | Benchmarking & Declinación")
    
    df = get_data_from_session()
    if df is None:
        st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la **Página Principal**.")
        st.stop()
    
    st.success("✅ Datos cargados desde memoria")
    
    # Filters
    selected_types, selected_company, selected_well = render_sidebar(df)
    
    if not selected_well:
        st.error("Selecciona un pozo válido.")
        st.stop()
    
    # Data preparation
    well_data = df[
        (df["empresa"] == selected_company) & 
        (df["sigla"] == selected_well)
    ].copy()
    
    if well_data.empty:
        st.error(f"No se encontraron datos para {selected_well}")
        st.stop()
    
    well_data = calculate_months_on_production(well_data)
    benchmark_data = get_benchmark_data(df, selected_types)
    
    # Header and metrics
    st.header(selected_well)
    st.write(f"**Tipo:** {well_data['tipopozo'].iloc[0]} | **Área:** {well_data['areayacimiento'].iloc[0]} | **Formación:** {well_data['formprod'].iloc[0]}")
    render_metrics(well_data)
    
    # Tabs for different analyses
    tab_rates, tab_cumul, tab_ratios, tab_decline = st.tabs([
        "📈 Producción vs Benchmark", 
        "📊 Acumuladas", 
        "⚗️ Relaciones de Fluidos", 
        "📉 Análisis de Declinación"
    ])
    
    # Tab 1: Rates with Benchmark
    with tab_rates:
        st.subheader("Comparación con Pozos Similares")
        st.caption("Línea sólida: Pozo seleccionado | Línea gris: Promedio | Línea naranja: P50 (mediana)")
        
        for fluid in ["oil", "gas", "water"]:
            st.plotly_chart(
                create_rate_plot_with_benchmark(well_data, benchmark_data, fluid, selected_well),
                use_container_width=True,
            )
    
    # Tab 2: Cumulative Production
    with tab_cumul:
        st.subheader("Producción Acumulada vs Tiempo")
        cols = st.columns(3)
        for idx, fluid in enumerate(["oil", "gas", "water"]):
            with cols[idx]:
                cum_col = PLOT_CONFIG[fluid]["cumulative"]
                if cum_col in well_data.columns:
                    total = well_data[cum_col].iloc[-1] / 1_000_000
                    st.metric(
                        label=f"Total {PLOT_CONFIG[fluid]['title']}", 
                        value=f"{total:,.1f} {PLOT_CONFIG[fluid]['cum_unit']}"
                    )
        
        for fluid in ["oil", "gas", "water"]:
            st.plotly_chart(create_cumulative_plot(well_data, fluid), use_container_width=True)
    
    # Tab 3: Ratios (GOR/WOR)
    with tab_ratios:
        st.subheader("Tendencias de Relaciones de Fluidos")
        col1, col2 = st.columns(2)
        with col1:
            avg_gor = well_data["gor"].mean()
            st.metric("GOR Promedio", f"{avg_gor:.1f} km³/m³" if pd.notna(avg_gor) else "N/A")
        with col2:
            avg_wor = well_data["wor"].mean()
            st.metric("WOR Promedio", f"{avg_wor:.2f} m³/m³" if pd.notna(avg_wor) else "N/A")
        
        st.plotly_chart(create_ratio_plot(well_data), use_container_width=True)
        
        # Add interpretation
        with st.expander("ℹ️ Interpretación de Relaciones"):
            st.markdown("""
            - **GOR (Gas-Oil Ratio)**: Incrementos indican posible caída de presión o conificación de gas
            - **WOR (Water-Oil Ratio)**: Incrementos sugieren breakthrough de agua o conificación
            - **Tendencia estable**: Comportamiento de drenaje uniforme
            """)
    
    # Tab 4: Decline Curve Analysis
    with tab_decline:
        st.subheader("Curvas de Declinación (Arps)")
        st.session_state["log_scale_decline"] = st.checkbox("Escala Logarítmica", value=False)
        
        decline_results = {}
        for fluid in ["oil", "gas"]:
            fig, results = create_decline_analysis_plot(well_data, fluid)
            st.plotly_chart(fig, use_container_width=True)
            decline_results[fluid] = results
            
            if results["model"]:
                with st.expander(f"📋 Detalles del Ajuste - {PLOT_CONFIG[fluid]['title']}"):
                    cols = st.columns(3)
                    with cols[0]:
                        st.write(f"**Modelo:** {results['model'].title()}")
                    with cols[1]:
                        if results["params"]:
                            st.write(f"**qi:** {results['params'][0]:.2f}")
                    with cols[2]:
                        if len(results["params"]) > 1:
                            st.write(f"**Di:** {results['params'][1]:.4f}")
                        if len(results["params"]) > 2:
                            st.write(f"**b:** {results['params'][2]:.3f}")
    
    # Data Export Section
    st.divider()
    with st.expander("📥 Ver Datos Completos y Descargar"):
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