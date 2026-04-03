"""
Enhanced DCA Module for Unconventional/Shale Wells
Supports: Modified Hyperbolic, Power-Law Exponential (PLE), Duong
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from typing import Optional, Tuple, Callable
import streamlit as st


# ── Shale DCA Models ─────────────────────────────────────────────────────────

def modified_hyperbolic(
    t: np.ndarray, 
    qi: float, 
    Di: float, 
    b: float, 
    Dmin: float
) -> np.ndarray:
    """
    Modified Hyperbolic: Hyperbolic until D = Dmin, then exponential.
    
    Standard hyperbolic decline rate: D(t) = Di / (1 + b*Di*t)
    Switch time: t_switch = (Di/Dmin - 1) / (b*Di)
    """
    if b == 0:
        return qi * np.exp(-Di * t)
    
    # Calculate when decline rate hits Dmin
    t_switch = (Di / Dmin - 1) / (b * Di) if Dmin < Di else 0
    
    result = np.zeros_like(t, dtype=float)
    
    # Hyperbolic portion (t <= t_switch)
    hyperbolic_mask = t <= t_switch
    t_hyp = t[hyperbolic_mask]
    result[hyperbolic_mask] = qi / np.power(1 + b * Di * t_hyp, 1/b)
    
    # Exponential portion (t > t_switch) with rate Dmin
    exponential_mask = t > t_switch
    t_exp = t[exponential_mask]
    if len(t_exp) > 0:
        q_switch = qi / np.power(1 + b * Di * t_switch, 1/b)
        result[exponential_mask] = q_switch * np.exp(-Dmin * (t_exp - t_switch))
    
    return result


def power_law_exponential(
    t: np.ndarray, 
    qi: float, 
    D: float, 
    n: float
) -> np.ndarray:
    """
    Power-Law Exponential (PLE): q(t) = qi * exp(-D * t^n)
    Best for transient flow in shale (0 < n < 1)
    n ≈ 0.5 often indicates linear flow
    """
    with np.errstate(over='ignore'):
        result = qi * np.exp(-D * np.power(t, n))
    return np.where(np.isfinite(result), result, 0)


def duong_model(
    t: np.ndarray, 
    qi: float, 
    m: float, 
    a: float
) -> np.ndarray:
    """
    Duong model for shale gas: q(t) = qi * t^(-m) * exp(a/(1-m)*(t^(1-m)-1))
    where m = 1 - b (decline exponent)
    Often better than Arps for shale with long linear flow
    """
    if m >= 1:
        m = 0.99
    
    with np.errstate(over='ignore', divide='ignore'):
        exp_term = np.exp(a / (1 - m) * (np.power(t, 1 - m) - 1))
        power_term = np.power(t, -m)
        result = qi * power_term * exp_term
    
    return np.where((np.isfinite(result)) & (result > 0), result, 1e-10)


# ── Curve Fitting Functions ──────────────────────────────────────────────────

def fit_shale_decline(
    time: np.ndarray, 
    rate: np.ndarray, 
    model: str = "ple",
    bounds: Optional[dict] = None
) -> Tuple[Optional[np.ndarray], Optional[Callable], Optional[float]]:
    """
    Fit decline curve model to shale well data.
    
    Returns: (parameters, function, EUR_30yr)
    """
    if len(time) < 6 or np.all(rate <= 0):
        return None, None, None
    
    t_norm = time - time.min()
    valid = rate > 0
    t_valid = t_norm[valid]
    q_valid = rate[valid]
    
    if len(t_valid) < 6:
        return None, None, None
    
    # Initial guesses and bounds
    qi_init = q_valid[0]
    
    try:
        if model == "modified_hyperbolic":
            # params: [qi, Di, b, Dmin]
            p0 = [qi_init, 0.5, 1.0, 0.05]
            bounds = ([qi_init * 0.1, 0.01, 0.1, 0.001], [qi_init * 3, 10, 2.0, 0.5])
            
            popt, _ = curve_fit(
                lambda t, qi, Di, b, Dmin: modified_hyperbolic(t, qi, Di, b, Dmin),
                t_valid, q_valid, p0=p0, bounds=bounds, maxfev=10000
            )
            func = lambda t: modified_hyperbolic(t, *popt)
            
        elif model == "ple":
            # params: [qi, D, n]
            p0 = [qi_init, 0.1, 0.5]
            bounds = ([qi_init * 0.1, 0.001, 0.1], [qi_init * 3, 5, 1.0])
            
            popt, _ = curve_fit(
                power_law_exponential, t_valid, q_valid, 
                p0=p0, bounds=bounds, maxfev=10000
            )
            func = lambda t: power_law_exponential(t, *popt)
            
        elif model == "duong":
            # params: [qi, m, a]
            p0 = [qi_init, 0.8, 2.0]
            bounds = ([qi_init * 0.1, 0.1, 0.1], [qi_init * 3, 0.99, 10.0])
            
            popt, _ = curve_fit(
                duong_model, t_valid, q_valid, 
                p0=p0, bounds=bounds, maxfev=10000
            )
            func = lambda t: duong_model(t, *popt)
        
        else:
            return None, None, None
            
        # Calculate 30-year EUR (monthly steps)
        t_30yr = np.arange(0, 30*12, 1)  # Monthly for 30 years
        q_forecast = func(t_30yr)
        # Trapezoidal integration
        eur = np.trapz(q_forecast, t_30yr) / 1_000  # Convert to appropriate units
        
        return popt, func, eur
        
    except (RuntimeError, ValueError, Warning):
        return None, None, None


# ── Visualization Component ───────────────────────────────────────────────────

def create_shale_dca_plot(
    well_data: pd.DataFrame, 
    fluid: str, 
    selected_well: str
) -> Tuple[go.Figure, dict]:
    """Create interactive DCA plot with shale-specific models."""
    config = {
        "oil": {"color": "#00CC96", "title": "Petróleo", "unit": "m³/d"},
        "gas": {"color": "#FF4B4B", "title": "Gas", "unit": "km³/d"}
    }[fluid]
    
    rate_col = f"{fluid}_rate"
    
    # Prepare data
    well_norm = calculate_months_on_production(well_data)
    months = well_norm["months_on_prod"].values
    rates = well_norm[rate_col].values
    
    fig = go.Figure()
    
    # Historical production
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
    
    # Fit all three models
    forecast_months = np.linspace(months.min(), max(months.max() + 60, 120), 200)
    
    model_configs = [
        ("ple", "Power-Law Exponential", "blue", "solid"),
        ("modified_hyperbolic", "Modified Hyperbolic", "red", "dash"),
        ("duong", "Duong", "purple", "dot")
    ]
    
    for model_key, model_name, color, dash in model_configs:
        params, func, eur = fit_shale_decline(months, rates, model=model_key)
        
        if func is not None:
            # Calculate R²
            q_pred = func(months - months.min())
            ss_res = np.sum((rates - q_pred) ** 2)
            ss_tot = np.sum((rates - np.mean(rates)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results["models"][model_key] = {
                "params": params,
                "r2": r2,
                "eur_30yr": eur,
                "name": model_name
            }
            
            if r2 > results["best_r2"]:
                results["best_r2"] = r2
                results["best_model"] = model_key
            
            # Plot forecast
            q_forecast = func(forecast_months)
            
            fig.add_trace(go.Scatter(
                x=forecast_months, y=q_forecast,
                mode="lines",
                name=f"{model_name} (R²={r2:.3f})",
                line=dict(color=color, dash=dash, width=2),
                opacity=0.8,
            ))
    
    # Highlight best model
    if results["best_model"]:
        best_config = next(c for c in model_configs if c[0] == results["best_model"])
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"🏆 Mejor ajuste: {results['models'][results['best_model']]['name']}",
            showarrow=False,
            font=dict(size=12, color=best_config[2]),
            bgcolor="white", opacity=0.8,
            align="left", valign="top"
        )
    
    fig.update_layout(
        title=f"Análisis de Declinación Shale - {config['title']} ({selected_well})",
        xaxis_title="Meses en Producción",
        yaxis_title=f"Caudal ({config['unit']})",
        yaxis_type="log" if st.session_state.get("log_scale_dca", True) else "linear",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    
    # Add vertical line at end of history
    fig.add_vline(x=months.max(), line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="Fin de Historia", annotation_position="top")
    
    return fig, results


def render_dca_details(results: dict):
    """Render detailed DCA results in Streamlit."""
    if not results["models"]:
        st.warning("No se pudieron ajustar los modelos. Datos insuficientes.")
        return
    
    st.subheader("📊 Resultados del Ajuste")
    
    cols = st.columns(len(results["models"]))
    for idx, (model_key, data) in enumerate(results["models"].items()):
        with cols[idx]:
            is_best = model_key == results["best_model"]
            border = "2px solid gold" if is_best else "1px solid gray"
            
            st.markdown(f"""
            <div style="border: {border}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <h4>{'🏆 ' if is_best else ''}{data['name']}</h4>
                <p><b>R²:</b> {data['r2']:.4f}</p>
                <p><b>EUR (30a):</b> {data['eur_30yr']:,.0f} units</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Parameter details
            params = data["params"]
            if model_key == "ple" and len(params) == 3:
                st.caption(f"qi={params[0]:.1f}, D={params[1]:.3f}, n={params[2]:.3f}")
            elif model_key == "modified_hyperbolic" and len(params) == 4:
                st.caption(f"qi={params[0]:.1f}, Di={params[1]:.3f}, b={params[2]:.2f}, Dmin={params[3]:.4f}")
            elif model_key == "duong" and len(params) == 3:
                st.caption(f"qi={params[0]:.1f}, m={params[1]:.3f}, a={params[2]:.3f}")


# ── Integration with Main App ────────────────────────────────────────────────

# In your main() function, replace the decline tab with:

with tab_decline:
    st.subheader("Análisis de Declinación para Shale/Tight")
    st.caption("""
    Modelos implementados:
    - **Power-Law Exponential (PLE)**: Mejor para flujo transitorio prolongado
    - **Modified Hyperbolic**: Transición de hiperbólica a exponencial (Dmin)
    - **Duong**: Especializado en pozos fracturados con flujo lineal largo
    """)
    
    st.session_state["log_scale_dca"] = st.toggle("Escala Logarítmica (recomendado)", value=True)
    
    for fluid in ["oil", "gas"]:
        st.markdown(f"### {PLOT_CONFIG[fluid]['title']}")
        fig, results = create_shale_dca_plot(well_data, fluid, selected_well)
        st.plotly_chart(fig, use_container_width=True)
        render_dca_details(results)
        
        with st.expander("ℹ️ Interpretación para Shale"):
            st.markdown("""
            **Power-Law Exponential (PLE)**:
            - n ≈ 0.5 indica flujo lineal (típico en shale)
            - n ≈ 1.0 indica flujo radial/boundary-dominated
            - Mejor para pozos con < 2-3 años de historia
            
            **Modified Hyperbolic**:
            - Dmin típico: 5-10% anual para shale oil
            - Switch time indica cuándo alcanza declinación terminal
            
            **Duong**:
            - Parámetro 'm' relacionado con geometría de fractura
            - Popular en shale gas (Marcellus, Eagle Ford)
            """)