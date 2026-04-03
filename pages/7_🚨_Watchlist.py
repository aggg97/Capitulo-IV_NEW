"""
Watchlist page - Top performing wells by cumulative production at key time intervals.
Analyzes well performance @ 180 days, 365 days, and 5 years to identify sustained productivity.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils import COMPANY_REPLACEMENTS


# ── Constants ─────────────────────────────────────────────────────────────────

DATA_MEMORY_KEY = "df"
IMAGE_PATH = "Vaca Muerta rig.png"

# Time intervals for cumulative analysis (days)
INTERVALS = {
    "180d": 180,
    "1y": 365,
    "5y": 365 * 5,
}

TOP_N_WELLS = 10


# ── Data Loading ──────────────────────────────────────────────────────────────

def get_data_from_session() -> pd.DataFrame | None:
    """Retrieve and process data from session state."""
    if DATA_MEMORY_KEY not in st.session_state:
        return None
    
    df = st.session_state[DATA_MEMORY_KEY].copy()
    df["date"] = pd.to_datetime(
        df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-1"
    )
    df["days_online"] = df.groupby("sigla")["tef"].cumsum()
    
    return df.sort_values(by=["sigla", "date"])


# ── Cumulative Production Calculations ─────────────────────────────────────────

def calculate_cumulative_at_interval(df: pd.DataFrame, interval_days: int) -> pd.DataFrame:
    """
    Calculate cumulative production for each well at exactly N days online.
    Uses interpolation if exact day not available.
    """
    # Get production data up to target days for each well
    mask = df["days_online"] <= interval_days
    filtered = df[mask].copy()
    
    # Calculate cumulative sums per well
    cumulative = (
        filtered.groupby("sigla")
        .agg({
            "prod_pet": "sum",
            "prod_gas": "sum",
            "days_online": "max",
            "empresa": "first",
            "areayacimiento": "first",
            "coordenadax": "first",
            "coordenaday": "first",
            "formprod": "first",
            "anio": "min",  # Start year
        })
        .reset_index()
    )
    
    # Only include wells that reached at least 90% of target interval
    min_days_required = interval_days * 0.9
    cumulative = cumulative[cumulative["days_online"] >= min_days_required]
    
    cumulative.columns = [
        "sigla", f"oil_cum_{interval_days}d", f"gas_cum_{interval_days}d",
        "days_online_actual", "empresa", "areayacimiento", "coordenadax",
        "coordenaday", "formprod", "start_year"
    ]
    
    return cumulative


def get_all_intervals_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merge cumulative data for all time intervals."""
    # Start with 180 days as base
    base = calculate_cumulative_at_interval(df, INTERVALS["180d"])
    
    # Merge 1 year data
    yearly = calculate_cumulative_at_interval(df, INTERVALS["1y"])
    base = base.merge(
        yearly[["sigla", "oil_cum_365d", "gas_cum_365d"]],
        on="sigla",
        how="outer"
    )
    
    # Merge 5 year data
    five_year = calculate_cumulative_at_interval(df, INTERVALS["5y"])
    base = base.merge(
        five_year[["sigla", "oil_cum_1825d", "gas_cum_1825d"]],
        on="sigla",
        how="outer"
    )
    
    # Standardize company names
    base["empresaNEW"] = base["empresa"].replace(COMPANY_REPLACEMENTS)
    
    return base


# ── Ranking Functions ───────────────────────────────────────────────────────────

def get_top_producers(df: pd.DataFrame, metric: str, n: int = TOP_N_WELLS) -> pd.DataFrame:
    """Get top N wells for a specific cumulative metric."""
    valid_data = df[df[metric].notna()].copy()
    return valid_data.nlargest(n, metric)


# ── Visualization ──────────────────────────────────────────────────────────────

def create_comparison_bar_chart(
    data: pd.DataFrame,
    metric: str,
    title: str,
    unit: str
) -> go.Figure:
    """Create horizontal bar chart for cumulative production ranking."""
    sorted_data = data.sort_values(by=metric, ascending=True)
    
    fig = px.bar(
        sorted_data,
        y="sigla",
        x=metric,
        color="empresaNEW",
        orientation="h",
        labels={
            metric: f"Cumulativo ({unit})",
            "sigla": "Pozo",
            "empresaNEW": "Operadora",
            "start_year": "Año Inicio",
            "areayacimiento": "Bloque",
        },
        text=metric,
        hover_data=["empresaNEW", "areayacimiento", "start_year", "days_online_actual"],
    )
    
    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="inside",
        marker_line_width=1,
        marker_line_color="white",
    )
    
    fig.update_layout(
        title=dict(text=title, font_size=14),
        yaxis=dict(categoryorder="total ascending", title=None),
        xaxis=dict(title=f"Cumulativo ({unit})"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=9)
        ),
        height=400,
        margin=dict(l=100, r=20, t=50, b=100),
    )
    
    return fig


def create_interval_comparison_scatter(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str
) -> go.Figure:
    """Scatter plot comparing performance across two time intervals."""
    # Get wells with data for both intervals
    valid = df[(df[x_metric].notna()) & (df[y_metric].notna())].copy()
    
    fig = px.scatter(
        valid,
        x=x_metric,
        y=y_metric,
        color="empresaNEW",
        size="days_online_actual",
        hover_data=["sigla", "areayacimiento", "start_year"],
        labels={
            x_metric: x_label,
            y_metric: y_label,
            "empresaNEW": "Operadora",
            "days_online_actual": "Días Online",
        },
        title=title,
    )
    
    # Add diagonal reference line (y=x)
    max_val = max(valid[x_metric].max(), valid[y_metric].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            name="Línea 1:1",
            showlegend=False,
        )
    )
    
    fig.update_layout(
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=9)
        ),
    )
    
    return fig


def create_summary_table(top_data: pd.DataFrame) -> pd.DataFrame:
    """Create formatted summary table for display."""
    display_cols = [
        "sigla", "empresaNEW", "areayacimiento", "start_year",
        "oil_cum_180d", "oil_cum_365d", "oil_cum_1825d",
        "gas_cum_180d", "gas_cum_365d", "gas_cum_1825d",
    ]
    
    available_cols = [c for c in display_cols if c in top_data.columns]
    summary = top_data[available_cols].copy()
    
    # Rename columns for display
    col_mapping = {
        "sigla": "Pozo",
        "empresaNEW": "Operadora",
        "areayacimiento": "Bloque",
        "start_year": "Año Inicio",
        "oil_cum_180d": "NP @ 180d",
        "oil_cum_365d": "NP @ 1año",
        "oil_cum_1825d": "NP @ 5años",
        "gas_cum_180d": "GP @ 180d",
        "gas_cum_365d": "GP @ 1año",
        "gas_cum_1825d": "GP @ 5años",
    }
    
    summary = summary.rename(columns=col_mapping)
    
    # Format numeric columns
    for col in summary.columns:
        if "NP" in col or "GP" in col:
            summary[col] = summary[col].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
            )
    
    return summary


# ── UI Components ─────────────────────────────────────────────────────────────

def render_header():
    """Render page header and sidebar."""
    st.set_page_config(
        page_title="Watchlist - Cumulative Production",
        page_icon="🚨",
        layout="wide"
    )
    st.header(":blue[🚨 Watchlist - Análisis de Producción Acumulada]")
    st.sidebar.image(Image.open(IMAGE_PATH))


def render_data_status(df: pd.DataFrame | None) -> pd.DataFrame:
    """Handle data loading status."""
    if df is None:
        st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la Página Principal.")
        st.stop()
    
    st.success("✅ Datos cargados. Analizando producción acumulada por pozo.")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    render_header()
    
    df = get_data_from_session()
    df = render_data_status(df)
    
    # Calculate all cumulative intervals
    with st.spinner("Calculando producción acumulada @ 180d, 1año, 5años..."):
        cumulative_data = get_all_intervals_data(df)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    total_wells_180d = cumulative_data["oil_cum_180d"].notna().sum()
    total_wells_1y = cumulative_data["oil_cum_365d"].notna().sum()
    total_wells_5y = cumulative_data["oil_cum_1825d"].notna().sum()
    
    col1.metric("Pozos con >180 días", f"{total_wells_180d}")
    col2.metric("Pozos con >1 año", f"{total_wells_1y}")
    col3.metric("Pozos con >5 años", f"{total_wells_5y}")
    
    st.divider()
    
    # ── Oil Cumulative Rankings ────────────────────────────────────────────────
    st.subheader("⛽ Rankings: Producción Acumulada de Petróleo")
    
    tab_oil_180d, tab_oil_1y, tab_oil_5y = st.tabs(["@ 180 días", "@ 1 año", "@ 5 años"])
    
    with tab_oil_180d:
        top_oil_180d = get_top_producers(cumulative_data, "oil_cum_180d")
        if not top_oil_180d.empty:
            fig = create_comparison_bar_chart(
                top_oil_180d, "oil_cum_180d",
                f"Top {TOP_N_WELLS} - Petróleo Acumulado @ 180 días", "m³"
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Ver tabla de datos"):
                st.dataframe(create_summary_table(top_oil_180d), use_container_width=True)
        else:
            st.info("No hay suficientes pozos con 180 días de producción.")
    
    with tab_oil_1y:
        top_oil_1y = get_top_producers(cumulative_data, "oil_cum_365d")
        if not top_oil_1y.empty:
            fig = create_comparison_bar_chart(
                top_oil_1y, "oil_cum_365d",
                f"Top {TOP_N_WELLS} - Petróleo Acumulado @ 1 año", "m³"
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Ver tabla de datos"):
                st.dataframe(create_summary_table(top_oil_1y), use_container_width=True)
        else:
            st.info("No hay suficientes pozos con 1 año de producción.")
    
    with tab_oil_5y:
        top_oil_5y = get_top_producers(cumulative_data, "oil_cum_1825d")
        if not top_oil_5y.empty:
            fig = create_comparison_bar_chart(
                top_oil_5y, "oil_cum_1825d",
                f"Top {TOP_N_WELLS} - Petróleo Acumulado @ 5 años", "m³"
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Ver tabla de datos"):
                st.dataframe(create_summary_table(top_oil_5y), use_container_width=True)
        else:
            st.info("No hay suficientes pozos con 5 años de producción.")
    
    st.divider()
    
    # ── Gas Cumulative Rankings ────────────────────────────────────────────────
    st.subheader("🔥 Rankings: Producción Acumulada de Gas")
    
    tab_gas_180d, tab_gas_1y, tab_gas_5y = st.tabs(["@ 180 días", "@ 1 año", "@ 5 años"])
    
    with tab_gas_180d:
        top_gas_180d = get_top_producers(cumulative_data, "gas_cum_180d")
        if not top_gas_180d.empty:
            fig = create_comparison_bar_chart(
                top_gas_180d, "gas_cum_180d",
                f"Top {TOP_N_WELLS} - Gas Acumulado @ 180 días", "m³"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay suficientes pozos con 180 días de producción.")
    
    with tab_gas_1y:
        top_gas_1y = get_top_producers(cumulative_data, "gas_cum_365d")
        if not top_gas_1y.empty:
            fig = create_comparison_bar_chart(
                top_gas_1y, "gas_cum_365d",
                f"Top {TOP_N_WELLS} - Gas Acumulado @ 1 año", "m³"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay suficientes pozos con 1 año de producción.")
    
    with tab_gas_5y:
        top_gas_5y = get_top_producers(cumulative_data, "gas_cum_1825d")
        if not top_gas_5y.empty:
            fig = create_comparison_bar_chart(
                top_gas_5y, "gas_cum_1825d",
                f"Top {TOP_N_WELLS} - Gas Acumulado @ 5 años", "m³"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay suficientes pozos con 5 años de producción.")
    
    st.divider()
    
    # ── Cross-Interval Analysis ────────────────────────────────────────────────
    st.subheader("📊 Análisis Comparativo entre Intervalos")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        # 180d vs 1y comparison
        valid_comparison = cumulative_data[
            (cumulative_data["oil_cum_180d"].notna()) & 
            (cumulative_data["oil_cum_365d"].notna())
        ].copy()
        
        if len(valid_comparison) >= 5:
            fig_compare = create_interval_comparison_scatter(
                valid_comparison,
                "oil_cum_180d", "oil_cum_365d",
                "NP @ 180 días (m³)", "NP @ 1 año (m³)",
                "Comparación: 180 días vs 1 año (Petróleo)"
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Calculate decline indicators
            valid_comparison["decline_factor"] = (
                valid_comparison["oil_cum_365d"] / valid_comparison["oil_cum_180d"]
            )
            avg_decline = valid_comparison["decline_factor"].mean()
            st.caption(f"**Factor promedio:** {avg_decline:.2f}x (1 año / 180 días)")
        else:
            st.info("Datos insuficientes para comparación 180d vs 1año.")
    
    with col_right:
        # 1y vs 5y comparison (if data available)
        valid_long = cumulative_data[
            (cumulative_data["oil_cum_365d"].notna()) & 
            (cumulative_data["oil_cum_1825d"].notna())
        ].copy()
        
        if len(valid_long) >= 3:
            fig_long = create_interval_comparison_scatter(
                valid_long,
                "oil_cum_365d", "oil_cum_1825d",
                "NP @ 1 año (m³)", "NP @ 5 años (m³)",
                "Comparación: 1 año vs 5 años (Petróleo)"
            )
            st.plotly_chart(fig_long, use_container_width=True)
        else:
            st.info("Datos insuficientes para comparación 1año vs 5años.")
    
    st.divider()
    
    # ── Methodology Note ───────────────────────────────────────────────────────
    with st.expander("📋 Metodología y Notas"):
        st.markdown("""
        **Cálculo de Producción Acumulada:**
        - Se calcula la suma acumulada de producción desde el primer mes reportado
        - Solo se incluyen pozos con al menos 90% del intervalo objetivo (ej: 162+ días para @180d)
        - La producción se suma mensualmente según datos oficiales de la Secretaría de Energía
        
        **Interpretación:**
        - **@ 180 días:** Indica productividad inicial post-frac (IP/Early time)
        - **@ 1 año:** Rendimiento sostenido a mediano plazo
        - **@ 5 años:** Productividad de largo plazo y EUR estimable
        
        **Limitaciones:**
        - No ajusta por choke management o estrategias de producción diferentes
        - No considera interferencia de pozos cercanos (PAD effects)
        - Diferentes espaciamientos y diseños de fractura afectan comparabilidad
        """)


if __name__ == "__main__":
    main()