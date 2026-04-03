"""
Single Well Analysis Page - Production Ready
============================================
Interactive production analysis with shale-optimized DCA.
Fixed: Duong boundary condition, Modified Hyperbolic switching logic, type safety.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from scipy.optimize import curve_fit
from typing import Optional, Tuple, Callable, Dict, List


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
MIN_DATA_POINTS = 6
FORECAST_YEARS = 30


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
    df["wgr"] = np.where(df["gas_rate"] > 0, df["water_rate"] / df["gas_rate"], np.nan)
    
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


# ── Shale DCA Models (Fixed) ─────────────────────────────────────────────────

def modified_hyperbolic(
    t: np.ndarray, 
    qi: float, 
    Di: float, 
    b: float, 
    Dmin: float
) -> np.ndarray:
    """
    Modified Hyperbolic: Hyperbolic until D <= Dmin, then exponential at Dmin.
    Fixed: Handles edge cases where Dmin >= Di.
    """
    # Input validation
    b = max(b, 0.01)  # Avoid division by zero
    Di = max(Di, 1e-6)
    Dmin = max(Dmin, 1e-6)
    qi = max(qi, 0)
    
    # If Dmin >= Di, pure exponential from start
    if Dmin >= Di:
        return qi * np.exp(-Di * t)
    
    # Switch time: when hyperbolic decline rate hits Dmin
    # D(t) = Di / (1 + b*Di*t) = Dmin  =>  t_switch = (Di/Dmin - 1)/(b*Di)
    t_switch = (Di / Dmin - 1) / (b * Di)
    
    result = np.zeros_like(t, dtype=float)
    
    # Hyperbolic portion (t <= t_switch)
    mask_hyp = (t <= t_switch) & (t >= 0)
    if np.any(mask_hyp):
        t_hyp = t[mask_hyp]
        result[mask_hyp] = qi / np.power(1 + b * Di * t_hyp, 1/b)
    
    # Exponential portion (t > t_switch)
    mask_exp = t > t_switch
    if np.any(mask_exp):
        q_switch = qi / np.power(1 + b * Di * t_switch, 1/b)
        t_exp = t[mask_exp]
        result[mask_exp] = q_switch * np.exp(-Dmin * (t_exp - t_switch))
    
    # Handle negative times (shouldn't happen but safety first)
    result[t < 0] = qi
    
    return np.maximum(result, 0)  # Ensure non-negative


def power_law_exponential(
    t: np.ndarray, 
    qi: float, 
    D: float, 
    n: float
) -> np.ndarray:
    """
    Power-Law Exponential (PLE): q(t) = qi * exp(-D * t^n)
    Fixed: Handles t=0 correctly (0^n = 0, exp(0) = 1, so q(0) = qi).
    """
    qi = max(qi, 0)
    D = max(D, 0)
    n = np.clip(n, 0.01, 1.0)  # Constrain to valid shale range
    
    # Ensure t is non-negative
    t = np.maximum(t, 0)
    
    with np.errstate(over='ignore', under='ignore'):
        result = qi * np.exp(-D * np.power(t, n))
    
    return np.where(np.isfinite(result), result, 0)


def duong_model(
    t: np.ndarray, 
    qi: float, 
    m: float, 
    a: float
) -> np.ndarray:
    """
    Duong model: q(t) = qi * t^(-m) * exp(a*(t^(1-m) - 1)/(1-m))
    Fixed: Handles t=0 by starting at t=1 (first month).
    """
    qi = max(qi, 0)
    m = np.clip(m, 0.01, 0.99)  # Ensure 0 < m < 1
    a = max(a, 0.01)
    
    # Shift time to avoid t=0 singularity (start at month 1)
    t_adj = np.maximum(t, 1.0)
    
    with np.errstate(over='ignore', under='ignore', divide='ignore'):
        power_term = np.power(t_adj, -m)
        exp_exponent = a * (np.power(t_adj, 1 - m) - 1) / (1 - m)
        exp_term = np.exp(exp_exponent)
        result = qi * power_term * exp_term
    
    # Ensure finite and non-negative
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-10)
    
    # For t=0 (original), set to initial production approximation
    result[t <= 0] = qi
    
    return result


def fit_shale_decline(
    time: np.ndarray, 
    rate: np.ndarray, 
    model: str = "ple"
) -> Tuple[Optional[np.ndarray], Optional[Callable], Optional[float]]:
    """
    Fit decline curve model to shale well data.
    Returns: (parameters, function, EUR_30yr)
    """
    if len(time) < MIN_DATA_POINTS or np.all(rate <= 0):
        return None, None, None
    
    # Normalize time to start at 0
    t_norm = time - time.min()
    valid = rate > 0
    t_valid = t_norm[valid]
    q_valid = rate[valid]
    
    if len(t_valid) < MIN_DATA_POINTS:
        return None, None, None
    
    qi_init = q_valid[0]
    if qi_init <= 0:
        qi_init = np.median(q_valid[:3])  # Use median of first 3 points
    
    try:
        if model == "modified_hyperbolic":
            p0 = [qi_init, 0.5, 1.0, 0.05]
            bounds = ([qi_init * 0.05, 0.001, 0.1, 0.0001], 
                     [qi_init * 5, 20.0, 2.0, 1.0])
            
            popt, _ = curve_fit(
                modified_hyperbolic, t_valid, q_valid, 
                p0=p0, bounds=bounds, maxfev=15000, method='trf'
            )
            func = lambda t: modified_hyperbolic(t, *popt)
            
        elif model == "ple":
            p0 = [qi_init, 0.1, 0.5]
            bounds = ([qi_init * 0.05, 0.0001, 0.05], 
                     [qi_init * 5, 10.0, 1.0])
            
            popt, _ = curve_fit(
                power_law_exponential, t_valid, q_valid, 
                p0=p0, bounds=bounds, maxfev=15000, method='trf'
            )
            func = lambda t: power_law_exponential(t, *popt)
            
        elif model == "duong":
            p0 = [qi_init, 0.7, 1.5]
            bounds = ([qi_init * 0.05, 0.1, 0.01], 
                     [qi_init * 5, 0.99, 20.0])
            
            popt, _ = curve_fit(
                duong_model, t_valid, q_valid, 
                p0=p0, bounds=bounds, maxfev=15000, method='trf'
            )
            func = lambda t: duong_model(t, *popt)
        else:
            return None, None, None
            
        # Calculate 30-year EUR (monthly steps, shifted to start at 0)
        t_forecast = np.arange(0, FORECAST_YEARS * 12 + 1, 1)
        q_forecast = func(t_forecast)
        
        # Only integrate positive, finite values
        valid_forecast = (q_forecast > 0) & np.isfinite(q_forecast)
        if np.any(valid_forecast):
            eur = np.trapz(q_forecast[valid_forecast], t_forecast[valid_forecast])
        else:
            eur = 0
            
        return popt, func, eur
        
    except (RuntimeError, ValueError, Warning) as e:
        # Log error for debugging but don't crash
        print(f"DCA fit failed for {model}: {str(e)}")
        return None, None, None


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


def create_shale_dca_plot(
    well_data: pd.DataFrame, 
    fluid: str, 
    selected_well: str
) -> Tuple[go.Figure, Dict]:
    """Create DCA plot with shale-specific models."""
    config = PLOT_CONFIG[fluid]
    rate_col = f"{fluid}_rate"
    
    well_norm = calculate_months_on_production(well_data)
    months = well_norm["months_on_prod"].values
    rates = well_norm[rate_col].values
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=months, y=rates,
        mode="markers",
        name="Datos Históricos",
        marker=dict(color="black", size=8, opacity=0.7),
    ))
    
    results = {
        "fluid": fluid,
        "models": {},
        "best_model": None,
        "best_r2": -np.inf
    }
    
    if len(months) >= MIN_DATA_POINTS:
        # Fit all models
        max_month = max(months.max() + 60, 120)
        forecast_months = np.linspace(0, max_month, 300)
        
        model_configs = [
            ("ple", "Power-Law Exponential", "blue", "solid"),
            ("modified_hyperbolic", "Modified Hyperbolic", "red", "dash"),
            ("duong", "Duong", "purple", "dot")
        ]
        
        for model_key, model_name, color, dash in model_configs:
            params, func, eur = fit_shale_decline(months, rates, model=model_key)
            
            if func is not None:
                # Calculate R² on historical data only
                q_pred = func(months - months.min())
                
                # Safe R² calculation
                ss_res = np.sum((rates - q_pred) ** 2)
                ss_tot = np.sum((rates - np.mean(rates)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
                
                # Only accept positive R²
                if r2 > 0:
                    results["models"][model_key] = {
                        "params": params,
                        "r2": r2,
                        "eur_30yr": eur,
                        "name": model_name
                    }
                    
                    if r2 > results["best_r2"]:
                        results["best_r2"] = r2
                        results["best_model"] = model_key
                    
                    # Forecast
                    q_forecast = func(forecast_months)
                    
                    # Clip unrealistic values for display
                    q_forecast = np.clip(q_forecast, 0, rates.max() * 3)
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_months, y=q_forecast,
                        mode="lines",
                        name=f"{model_name} (R²={r2:.3f})",
                        line=dict(color=color, dash=dash, width=2),
                        opacity=0.8,
                    ))
    
    # Highlight best model
    if results["best_model"]:
        best_name = results["models"][results["best_model"]]["name"]
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"🏆 Mejor ajuste: {best_name}",
            showarrow=False,
            font=dict(size=12, color="green"),
            bgcolor="white", opacity=0.9,
            align="left", valign="top",
            bordercolor="green", borderwidth=1
        )
    
    use_log = st.session_state.get("log_scale_dca", True)
    
    fig.update_layout(
        title=f"DCA Shale - {config['title']} ({selected_well})",
        xaxis_title="Meses en Producción",
        yaxis_title=f"Caudal ({config['unit']})",
        yaxis_type="log" if use_log else "linear",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    
    # Vertical line at end of history
    fig.add_vline(
        x=months.max(), 
        line_dash="dash", 
        line_color="gray", 
        opacity=0.5,
        annotation_text="Fin Historia", 
        annotation_position="top"
    )
    
    return fig, results


def render_dca_details(results: Dict):
    """Render DCA results."""
    if not results["models"]:
        st.warning("No se pudieron ajustar los modelos (insuficientes datos o calidad)")
        return
    
    st.subheader("Resultados del Ajuste")
    
    cols = st.columns(len(results["models"]))
    model_order = ["ple", "modified_hyperbolic", "duong"]
    
    for idx, model_key in enumerate(model_order):
        if model_key not in results["models"]:
            continue
            
        data = results["models"][model_key]
        is_best = model_key == results["best_model"]
        
        with cols[idx]:
            border = "2px solid #4CAF50" if is_best else "1px solid #ddd"
            bg_color = "#f1f8e9" if is_best else "white"
            
            st.markdown(f"""
            <div style="border: {border}; padding: 10px; border-radius: 5px; background-color: {bg_color};">
                <h4>{'🏆 ' if is_best else ''}{data['name']}</h4>
                <p><b>R²:</b> {data['r2']:.4f}</p>
                <p><b>EUR (30a):</b> {data['eur_30yr']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            params = data["params"]
            if model_key == "ple" and params is not None:
                st.caption(f"n={params[2]:.3f} (transitorio si <0.8)")
            elif model_key == "modified_hyperbolic" and params is not None:
                st.caption(f"b={params[2]:.2f}, Dmin={params[3]:.4f}")
            elif model_key == "duong" and params is not None:
                st.caption(f"m={params[1]:.3f}")


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
    st.title(":blue[Análisis Avanzado de Pozo Individual]")
    st.caption("Capítulo IV - Shale DCA & Benchmarking")
    
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
    
    # Tabs
    tab_rates, tab_cumul, tab_ratios, tab_decline = st.tabs([
        "📈 Producción vs Benchmark", 
        "📊 Acumuladas", 
        "⚗️ Relaciones de Fluidos", 
        "📉 DCA Shale"
    ])
    
    with tab_rates:
        st.subheader("Comparación con Pozos Similares")
        st.caption("Línea sólida: Pozo | Gris: Promedio | Naranja: P50")
        
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
            - **Estable**: Drenaje uniforme del yacimiento
            """)
    
    with tab_decline:
        st.subheader("Análisis de Declinación para Shale")
        st.caption("Modelos: PLE (transitorio), Modified Hyperbolic (terminal), Duong (fracturas)")
        
        st.session_state["log_scale_dca"] = st.toggle("Escala Log", value=True)
        
        for fluid in ["oil", "gas"]:
            st.markdown(f"### {PLOT_CONFIG[fluid]['title']}")
            fig, results = create_shale_dca_plot(well_data, fluid, selected_well)
            st.plotly_chart(fig, use_container_width=True)
            render_dca_details(results)
        
        with st.expander("ℹ️ Guía de Modelos Shale"):
            st.markdown("""
            **Power-Law Exponential (PLE)**:
            - q(t) = qi·exp(-D·tⁿ)
            - n≈0.5: Flujo lineal (fractura infinita)
            - n→1.0: Flujo radial (boundary-dominated)
            - **Mejor para**: Primeros 2-3 años de producción
            
            **Modified Hyperbolic**:
            - Hiperbólica hasta Dmin, luego exponencial
            - Dmin típico: 5-15% anual (shale oil)
            - **Mejor para**: Pronósticos económicos (evita sobreestimación)
            
            **Duong**:
            - Especializado en pozos con flujo lineal dominante
            - Popular en gas shale (Marcellus, Haynesville)
            - **Mejor para**: Pozos con >1 año de flujo lineal claro
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