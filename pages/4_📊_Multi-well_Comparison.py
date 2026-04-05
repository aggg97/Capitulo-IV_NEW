import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils import COMPANY_REPLACEMENTS, get_fluid_classification


# ── Session state ─────────────────────────────────────────────────────────────

if "df" in st.session_state:
    data_sorted = st.session_state["df"]
    data_sorted["date"]       = pd.to_datetime(data_sorted["anio"].astype(str) + "-" + data_sorted["mes"].astype(str) + "-1")
    data_sorted["gas_rate"]   = data_sorted["prod_gas"] / data_sorted["tef"]
    data_sorted["oil_rate"]   = data_sorted["prod_pet"] / data_sorted["tef"]
    data_sorted["water_rate"] = data_sorted["prod_agua"] / data_sorted["tef"]
    data_sorted               = data_sorted.sort_values(by=["sigla", "date"], ascending=True)
    data_sorted["empresaNEW"] = data_sorted["empresa"].replace(COMPANY_REPLACEMENTS)
    data_sorted               = get_fluid_classification(data_sorted)
    st.info("Utilizando datos recuperados de la memoria.")
else:
    st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la Página Principal.")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.header(":blue[Comparación Multi-Pozo]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))
st.sidebar.title("Por favor filtrar aquí:")

# Fluid filter drives the sigla multiselect
available_fluids = sorted(data_sorted["tipopozoNEW"].unique())
selected_fluido  = st.sidebar.selectbox(
    "Seleccionar tipo de fluido (McCain):",
    options=available_fluids,
)

siglas_for_fluid = (
    data_sorted[data_sorted["tipopozoNEW"] == selected_fluido]["sigla"]
    .unique()
)

selected_siglas = st.sidebar.multiselect(
    "Seleccionar pozos a comparar:",
    options=sorted(siglas_for_fluid),
)

filtered_data = data_sorted[data_sorted["sigla"].isin(selected_siglas)]


# ── Color palettes ────────────────────────────────────────────────────────────

GAS_PALETTE   = ["#FF0000", "#FFA07A", "#FA8072", "#E9967A", "#F08080",
                 "#CD5C5C", "#DC143C", "#B22222", "#8B0000"]
OIL_PALETTE   = ["#008000", "#006400", "#90EE90", "#98FB98", "#8FBC8F",
                 "#3CB371", "#2E8B57", "#808000", "#556B2F", "#6B8E23"]
WATER_PALETTE = ["#0000FF", "#0000CD", "#00008B", "#000080", "#191970",
                 "#7B68EE", "#6A5ACD", "#483D8B", "#B0E0E6", "#ADD8E6",
                 "#87CEFA", "#87CEEB", "#00BFFF", "#B0C4DE", "#1E90FF", "#6495ED"]


# ── Time-zero normalisation ───────────────────────────────────────────────────

def add_time_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'month_number' column counting from the first month where any
    production (oil or gas) > 0. Month 1 = first producing month.
    """
    df = df.copy()
    first_prod = (
        df[df[["oil_rate", "gas_rate"]].max(axis=1) > 0]
        .groupby("sigla")["date"]
        .min()
        .rename("first_prod_date")
    )
    df = df.merge(first_prod, on="sigla", how="left")
    df["month_number"] = (
        (df["date"].dt.year  - df["first_prod_date"].dt.year) * 12 +
        (df["date"].dt.month - df["first_prod_date"].dt.month) + 1
    )
    return df[df["month_number"] >= 1]


# ── Chart helpers ─────────────────────────────────────────────────────────────

def build_rate_chart(
    data: pd.DataFrame,
    wells: list,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    palette: list,
) -> go.Figure:
    fig = go.Figure()
    for i, sigla in enumerate(wells):
        wd = data[data["sigla"] == sigla].sort_values(x_col)
        fig.add_trace(go.Scatter(
            x=wd[x_col],
            y=wd[y_col],
            mode="lines+markers",
            name=sigla,
            line=dict(color=palette[i % len(palette)]),
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}",
        ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Pozo",
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def build_diagnostic_chart(
    data: pd.DataFrame,
    wells: list,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    palette: list,
) -> go.Figure:
    fig = go.Figure()
    for i, sigla in enumerate(wells):
        wd = data[data["sigla"] == sigla].dropna(subset=[x_col, y_col]).sort_values(x_col)
        if wd.empty:
            continue
        fig.add_trace(go.Scatter(
            x=wd[x_col],
            y=wd[y_col],
            mode="lines+markers",
            name=sigla,
            line=dict(color=palette[i % len(palette)]),
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}",
        ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Pozo",
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


# ── Main content ──────────────────────────────────────────────────────────────

if not selected_siglas:
    st.info("👈 Seleccioná al menos un pozo en el panel lateral para comenzar.")
    st.stop()

# Apply time-zero and keep only producing months
plot_data = add_time_zero(filtered_data)

# Time-axis toggle
time_axis = st.radio(
    "Eje temporal",
    options=["📅 Fecha calendario", "⏱️ Tiempo cero (mes de producción)"],
    horizontal=True,
)
use_time_zero = time_axis == "⏱️ Tiempo cero (mes de producción)"
x_col   = "month_number" if use_time_zero else "date"
x_label = "Mes de Producción" if use_time_zero else "Fecha"


# ── Rate history charts ───────────────────────────────────────────────────────

st.subheader("Historia de Producción")

st.plotly_chart(build_rate_chart(
    plot_data, selected_siglas, x_col,
    "gas_rate", x_label, "Caudal de Gas (km3/d)",
    "Historia de Producción de Gas",
    GAS_PALETTE,
), use_container_width=True)

st.plotly_chart(build_rate_chart(
    plot_data, selected_siglas, x_col,
    "oil_rate", x_label, "Caudal de Petróleo (m3/d)",
    "Historia de Producción de Petróleo",
    OIL_PALETTE,
), use_container_width=True)

st.plotly_chart(build_rate_chart(
    plot_data, selected_siglas, x_col,
    "water_rate", x_label, "Caudal de Agua (m3/d)",
    "Historia de Producción de Agua",
    WATER_PALETTE,
), use_container_width=True)


# ── Diagnostic plots ──────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 Gráficos Diagnóstico")

# Per-row ratios
diag_data = plot_data.copy()
diag_data["GOR"] = (diag_data["Gp"] / diag_data["Np"] * 1000).replace([float("inf"), -float("inf")], None)
diag_data["WOR"] = (diag_data["Wp"] / diag_data["Np"]).replace([float("inf"), -float("inf")], None)
diag_data["WGR"] = (diag_data["Wp"] / diag_data["Gp"] * 1000).replace([float("inf"), -float("inf")], None)

GAS_PLOTS = {
    "Qg vs Gp":  ("Gp", "gas_rate", "Gp (km3)",  "Qg (km3/d)",   GAS_PALETTE),
    "WGR vs Gp": ("Gp", "WGR",      "Gp (km3)",  "WGR (m3/km3)", WATER_PALETTE),
    "GOR vs Gp": ("Gp", "GOR",      "Gp (km3)",  "GOR (m3/km3)", OIL_PALETTE),
}
OIL_PLOTS = {
    "Qo vs Np":  ("Np", "oil_rate", "Np (m3)",   "Qo (m3/d)",    OIL_PALETTE),
    "WOR vs Np": ("Np", "WOR",      "Np (m3)",   "WOR (m3/m3)",  WATER_PALETTE),
    "GOR vs Np": ("Np", "GOR",      "Np (m3)",   "GOR (m3/m3)",  GAS_PALETTE),
}

available_plots = GAS_PLOTS if selected_fluido == "Gasífero" else OIL_PLOTS

selected_diag = st.multiselect(
    f"Seleccionar gráficos diagnóstico ({selected_fluido}):",
    options=list(available_plots.keys()),
    default=[],
)

if selected_diag:
    for plot_name in selected_diag:
        x_col_d, y_col_d, x_label_d, y_label_d, palette_d = available_plots[plot_name]
        st.plotly_chart(
            build_diagnostic_chart(
                diag_data, selected_siglas,
                x_col_d, y_col_d, x_label_d, y_label_d,
                f"{selected_fluido} — {plot_name}",
                palette_d,
            ),
            use_container_width=True,
        )
else:
    st.caption("Seleccione al menos un gráfico diagnóstico para visualizarlo.")