"""
Watchlist page - Top performing wells in Vaca Muerta.
Tab 1: Current production rates (latest allocation)
Tab 2: Cumulative production analysis @ 180d, 1y, 5y
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from utils import COMPANY_REPLACEMENTS, BARRELS_PER_M3


# ── Constants ─────────────────────────────────────────────────────────────────

TOP_N_WELLS = 5
DATA_MEMORY_KEY = "df"
IMAGE_PATH = "Vaca Muerta rig.png"

# Time intervals for cumulative analysis (days)
INTERVALS = {
    "180d": 180,
    "1y": 365,
    "5y": 365 * 5,
}


# ── Data Loading ───────────────────────────────────────────────────────────────

def get_data_from_session():
    """Retrieve and process data from session state."""
    if DATA_MEMORY_KEY not in st.session_state:
        return None
    
    df = st.session_state[DATA_MEMORY_KEY].copy()
    df["date"] = pd.to_datetime(
        df["anio"].astype(str) + "-" + df["mes"].astype(str) + "-1"
    )
    df["gas_rate"] = df["prod_gas"] / df["tef"]
    df["oil_rate"] = df["prod_pet"] / df["tef"]
    df["water_rate"] = df["prod_agua"] / df["tef"]
    df["days_online"] = df.groupby("sigla")["tef"].cumsum()
    
    return df.sort_values(by=["sigla", "date"], ascending=True)


# ── UI Components ──────────────────────────────────────────────────────────────

def render_header():
    """Render page header and sidebar."""
    st.set_page_config(
        page_title="Watchlist - Vaca Muerta",
        page_icon="🚨",
        layout="wide"
    )
    st.header(":blue[🚨 Watchlist - Análisis de Pozos]")
    st.sidebar.image(Image.open(IMAGE_PATH))


def render_data_status(df):
    """Handle data loading status."""
    if df is None:
        st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la Página Principal.")
        st.stop()
    
    st.success("✅ Datos cargados correctamente.")
    return df


# ── Cumulative Production Logic ────────────────────────────────────────────────

def calculate_cumulative_at_interval(df, interval_days):
    """
    Calculate cumulative production for each well at exactly N days online.
    """
    mask = df["days_online"] <= interval_days
    filtered = df[mask].copy()
    
    cumulative = (
        filtered.groupby("sigla")
        .agg({
            "prod_pet": "sum",
            "prod_gas": "sum",
            "days_online": "max",
            "empresa": "first",
            "areayacimiento": "first",
            "formprod": "first",
            "anio": "min",
        })
        .reset_index()
    )
    
    min_days_required = interval_days * 0.9
    cumulative = cumulative[cumulative["days_online"] >= min_days_required]
    
    return cumulative


def get_cumulative_data(df):
    """Get cumulative production data for all intervals."""
    data_180d = calculate_cumulative_at_interval(df, INTERVALS["180d"])
    data_1y = calculate_cumulative_at_interval(df, INTERVALS["1y"])
    data_5y = calculate_cumulative_at_interval(df, INTERVALS["5y"])
    
    # Merge all intervals
    result = data_180d.rename(columns={
        "prod_pet": "oil_cum_180d",
        "prod_gas": "gas_cum_180d"
    })[["sigla", "oil_cum_180d", "gas_cum_180d", "empresa", "areayacimiento", "anio"]]
    
    if not data_1y.empty:
        result = result.merge(
            data_1y[["sigla", "prod_pet", "prod_gas"]].rename(columns={
                "prod_pet": "oil_cum_1y",
                "prod_gas": "gas_cum_1y"
            }),
            on="sigla",
            how="outer"
        )
    
    if not data_5y.empty:
        result = result.merge(
            data_5y[["sigla", "prod_pet", "prod_gas"]].rename(columns={
                "prod_pet": "oil_cum_5y",
                "prod_gas": "gas_cum_5y"
            }),
            on="sigla",
            how="outer"
        )
    
    result["empresaNEW"] = result["empresa"].replace(COMPANY_REPLACEMENTS)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    render_header()
    
    df = get_data_from_session()
    df = render_data_status(df)
    
    # Standardize company names for the entire dataset
    df["empresaNEW"] = df["empresa"].replace(COMPANY_REPLACEMENTS)
    
    # Create tabs
    tab_rates, tab_cumulative = st.tabs([
        "📊 Caudales Actuales", 
        "📈 Producción Acumulada (180d / 1año / 5años)"
    ])
    
    # ── TAB 1: Current Rates (Original Watchlist) ─────────────────────────────
    with tab_rates:
        # Filter valid data
        data_filtered = df[df["tef"] > 0]
        latest_date = data_filtered["date"].max()
        latest_data = data_filtered[data_filtered["date"] == latest_date]
        
        st.write("**Fecha de Alocación en Progreso:** ", latest_date.date())
        
        # Top producers
        top_gas = latest_data.nlargest(TOP_N_WELLS, "gas_rate")
        top_oil = latest_data.nlargest(TOP_N_WELLS, "oil_rate")
        
        # Gas ranking
        st.subheader("🔥 Ranking actual: Top 5 pozos de gas más productivos")
        fig_gas = px.bar(
            top_gas.sort_values(by="gas_rate"),
            y="sigla",
            x="gas_rate",
            color="empresaNEW",
            orientation="h",
            labels={
                "gas_rate": "Producción de Gas (m³/día)", 
                "sigla": "Pozo", 
                "empresaNEW": "Empresa",
                "areayacimiento": "Bloque"
            },
            text="gas_rate",
            hover_data=["empresaNEW", "areayacimiento"],
        )
        fig_gas.update_traces(texttemplate="%{text:.2f}", textposition="inside")
        fig_gas.update_layout(
            title="Producción de Gas (km³/d)",
            yaxis=dict(categoryorder="total ascending"),
            yaxis_title=None,
            height=400
        )
        st.plotly_chart(fig_gas, use_container_width=True)
        
        # Oil ranking
        st.subheader("🔥 Ranking actual: Top 5 pozos de petróleo más productivos")
        fig_oil = px.bar(
            top_oil.sort_values(by="oil_rate"),
            y="sigla",
            x="oil_rate",
            color="empresaNEW",
            orientation="h",
            labels={
                "oil_rate": "Producción de Petróleo (m³/día)", 
                "sigla": "Pozo", 
                "empresaNEW": "Empresa",
                "areayacimiento": "Bloque"
            },
            text="oil_rate",
            hover_data=["empresaNEW", "areayacimiento"],
        )
        fig_oil.update_traces(texttemplate="%{text:.2f}", textposition="inside")
        fig_oil.update_layout(
            title="Producción de Petróleo (m³/d)",
            yaxis=dict(categoryorder="total ascending"),
            yaxis_title=None,
            height=400
        )
        st.plotly_chart(fig_oil, use_container_width=True)
        
        # Disclaimer
        st.info("""
        **Nota:** Al evaluar la productividad en Vaca Muerta, es importante tener precaución 
        con los pozos considerados "más productivos" únicamente por su caudal máximo. 
        El caudal máximo está influenciado por el *choke management* e interferencia de pozos (SRV). 
        Una evaluación más representativa considera la producción acumulada en el tiempo (ver pestaña "Producción Acumulada").
        """)
    
    # ── TAB 2: Cumulative Production (New Enhancement) ────────────────────────
    with tab_cumulative:
        st.subheader("📈 Análisis de Producción Acumulada por Pozo")
        st.caption("Comparación de productividad acumulada a 180 días, 1 año y 5 años")
        
        with st.spinner("Calculando producción acumulada..."):
            cum_data = get_cumulative_data(df)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        wells_180d = cum_data["oil_cum_180d"].notna().sum()
        wells_1y = cum_data["oil_cum_1y"].notna().sum() if "oil_cum_1y" in cum_data.columns else 0
        wells_5y = cum_data["oil_cum_5y"].notna().sum() if "oil_cum_5y" in cum_data.columns else 0
        
        col1.metric("Pozos con >180 días", wells_180d)
        col2.metric("Pozos con >1 año", wells_1y)
        col3.metric("Pozos con >5 años", wells_5y)
        
        # Oil Cumulative Rankings
        st.markdown("---")
        st.markdown("### ⛽ Petróleo Acumulado")
        
        col_oil_180d, col_oil_1y = st.columns(2)
        
        with col_oil_180d:
            st.markdown("**@ 180 días**")
            if wells_180d > 0:
                top_oil_180d = cum_data.nlargest(TOP_N_WELLS, "oil_cum_180d")
                fig_oil_180d = px.bar(
                    top_oil_180d.sort_values(by="oil_cum_180d"),
                    y="sigla",
                    x="oil_cum_180d",
                    color="empresaNEW",
                    orientation="h",
                    labels={"oil_cum_180d": "m³", "sigla": "Pozo", "empresaNEW": "Empresa"},
                    text="oil_cum_180d",
                    hover_data=["areayacimiento"],
                    height=350
                )
                fig_oil_180d.update_traces(texttemplate="%{text:,.0f}", textposition="inside")
                fig_oil_180d.update_layout(yaxis_title=None, showlegend=False)
                st.plotly_chart(fig_oil_180d, use_container_width=True, key="oil_180d")
            else:
                st.info("No hay suficientes datos")
        
        with col_oil_1y:
            st.markdown("**@ 1 año**")
            if wells_1y > 0:
                top_oil_1y = cum_data.nlargest(TOP_N_WELLS, "oil_cum_1y")
                fig_oil_1y = px.bar(
                    top_oil_1y.sort_values(by="oil_cum_1y"),
                    y="sigla",
                    x="oil_cum_1y",
                    color="empresaNEW",
                    orientation="h",
                    labels={"oil_cum_1y": "m³", "sigla": "Pozo", "empresaNEW": "Empresa"},
                    text="oil_cum_1y",
                    hover_data=["areayacimiento"],
                    height=350
                )
                fig_oil_1y.update_traces(texttemplate="%{text:,.0f}", textposition="inside")
                fig_oil_1y.update_layout(yaxis_title=None, showlegend=False)
                st.plotly_chart(fig_oil_1y, use_container_width=True, key="oil_1y")
            else:
                st.info("No hay suficientes datos")
        
        # 5 years oil (full width if available)
        if wells_5y > 0:
            st.markdown("**@ 5 años**")
            top_oil_5y = cum_data.nlargest(TOP_N_WELLS, "oil_cum_5y")
            fig_oil_5y = px.bar(
                top_oil_5y.sort_values(by="oil_cum_5y"),
                y="sigla",
                x="oil_cum_5y",
                color="empresaNEW",
                orientation="h",
                labels={"oil_cum_5y": "m³", "sigla": "Pozo", "empresaNEW": "Empresa"},
                text="oil_cum_5y",
                hover_data=["areayacimiento"],
                height=350
            )
            fig_oil_5y.update_traces(texttemplate="%{text:,.0f}", textposition="inside")
            fig_oil_5y.update_layout(yaxis_title=None, showlegend=False)
            st.plotly_chart(fig_oil_5y, use_container_width=True, key="oil_5y")
        
        # Gas Cumulative Rankings
        st.markdown("---")
        st.markdown("### 🔥 Gas Acumulado")
        
        col_gas_180d, col_gas_1y = st.columns(2)
        
        with col_gas_180d:
            st.markdown("**@ 180 días**")
            if wells_180d > 0:
                top_gas_180d = cum_data.nlargest(TOP_N_WELLS, "gas_cum_180d")
                fig_gas_180d = px.bar(
                    top_gas_180d.sort_values(by="gas_cum_180d"),
                    y="sigla",
                    x="gas_cum_180d",
                    color="empresaNEW",
                    orientation="h",
                    labels={"gas_cum_180d": "m³", "sigla": "Pozo", "empresaNEW": "Empresa"},
                    text="gas_cum_180d",
                    hover_data=["areayacimiento"],
                    height=350
                )
                fig_gas_180d.update_traces(texttemplate="%{text:,.0f}", textposition="inside")
                fig_gas_180d.update_layout(yaxis_title=None, showlegend=False)
                st.plotly_chart(fig_gas_180d, use_container_width=True, key="gas_180d")
            else:
                st.info("No hay suficientes datos")
        
        with col_gas_1y:
            st.markdown("**@ 1 año**")
            if wells_1y > 0:
                top_gas_1y = cum_data.nlargest(TOP_N_WELLS, "gas_cum_1y")
                fig_gas_1y = px.bar(
                    top_gas_1y.sort_values(by="gas_cum_1y"),
                    y="sigla",
                    x="gas_cum_1y",
                    color="empresaNEW",
                    orientation="h",
                    labels={"gas_cum_1y": "m³", "sigla": "Pozo", "empresaNEW": "Empresa"},
                    text="gas_cum_1y",
                    hover_data=["areayacimiento"],
                    height=350
                )
                fig_gas_1y.update_traces(texttemplate="%{text:,.0f}", textposition="inside")
                fig_gas_1y.update_layout(yaxis_title=None, showlegend=False)
                st.plotly_chart(fig_gas_1y, use_container_width=True, key="gas_1y")
            else:
                st.info("No hay suficientes datos")
        
        # 5 years gas
        if wells_5y > 0:
            st.markdown("**@ 5 años**")
            top_gas_5y = cum_data.nlargest(TOP_N_WELLS, "gas_cum_5y")
            fig_gas_5y = px.bar(
                top_gas_5y.sort_values(by="gas_cum_5y"),
                y="sigla",
                x="gas_cum_5y",
                color="empresaNEW",
                orientation="h",
                labels={"gas_cum_5y": "m³", "sigla": "Pozo", "empresaNEW": "Empresa"},
                text="gas_cum_5y",
                hover_data=["areayacimiento"],
                height=350
            )
            fig_gas_5y.update_traces(texttemplate="%{text:,.0f}", textposition="inside")
            fig_gas_5y.update_layout(yaxis_title=None, showlegend=False)
            st.plotly_chart(fig_gas_5y, use_container_width=True, key="gas_5y")
        
        # Methodology
        with st.expander("📋 Metodología"):
            st.markdown("""
            **Cálculo de Producción Acumulada:**
            - Se suma la producción mes a mes desde el inicio de operaciones
            - Solo se incluyen pozos con ≥90% del período objetivo completado
            - @180d: Pozos con al menos 162 días de producción
            - @1año: Pozos con al menos 328 días de producción  
            - @5años: Pozos con al menos 1642 días de producción
            
            **Interpretación:**
            - **180 días:** Productividad inicial post-frac (temprana)
            - **1 año:** Performance sostenida mediano plazo
            - **5 años:** Productividad de largo plazo (EUR)
            """)


if __name__ == "__main__":
    main()