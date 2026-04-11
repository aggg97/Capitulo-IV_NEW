import numpy as np
import pandas as pd
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


# ── Sidebar filters ───────────────────────────────────────────────────────────

st.title(":blue[Análisis de Pozo Individual]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))
st.sidebar.title("Por favor filtrar aquí:")

# dropna() prevents TypeError when tipopozoNEW contains NaN
selected_tipo = st.sidebar.multiselect(
    "Seleccionar tipo de pozo:",
    options=sorted(data_sorted["tipopozoNEW"].dropna().unique()),
)

selected_company = st.sidebar.selectbox(
    "Seleccionar operadora:",
    options=sorted(data_sorted["empresaNEW"].unique()),
)

# Filter available siglas based on tipo and company
filtered_siglas = data_sorted[
    (data_sorted["tipopozoNEW"].isin(selected_tipo)) &
    (data_sorted["empresaNEW"] == selected_company)
]["sigla"].unique()

# Guard: nothing to show until the user picks at least one well type
if len(selected_tipo) == 0 or len(filtered_siglas) == 0:
    st.info("👈 Seleccioná al menos un tipo de pozo y una operadora para comenzar.")
    st.stop()

selected_sigla = st.sidebar.selectbox(
    "Seleccionar sigla del pozo:",
    options=sorted(filtered_siglas),
)

# Well data
well_data = data_sorted[
    (data_sorted["empresaNEW"] == selected_company) &
    (data_sorted["sigla"] == selected_sigla)
].copy()

# Guard: empty well data should never happen but protects downstream
if well_data.empty:
    st.warning("No se encontraron datos para el pozo seleccionado.")
    st.stop()


# ── Time-zero normalisation ───────────────────────────────────────────────────

first_prod_date = well_data.loc[
    well_data[["oil_rate", "gas_rate"]].max(axis=1) > 0, "date"
].min()

# Guard: well has no production at all
if pd.isna(first_prod_date):
    st.warning("El pozo seleccionado no registra producción.")
    st.stop()

well_data["month_number"] = (
    (well_data["date"].dt.year  - first_prod_date.year) * 12 +
    (well_data["date"].dt.month - first_prod_date.month) + 1
)
well_data = well_data[well_data["month_number"] >= 1]


# ── KPI metrics ───────────────────────────────────────────────────────────────

max_gas_rate   = round(well_data["gas_rate"].clip(upper=1_000_000).max(), 1)
max_oil_rate   = round(well_data["oil_rate"].clip(upper=1_000_000).max(), 1)
max_water_rate = round(well_data["water_rate"].clip(upper=1_000_000).max(), 1)

st.header(selected_sigla)
col1, col2, col3 = st.columns(3)
col1.metric(label=":red[Caudal Máximo de Gas (km3/d)]",       value=max_gas_rate)
col2.metric(label=":green[Caudal Máximo de Petróleo (m3/d)]", value=max_oil_rate)
col3.metric(label=":blue[Caudal Máximo de Agua (m3/d)]",      value=max_water_rate)


# ── Time-axis toggle ──────────────────────────────────────────────────────────

time_axis = st.radio(
    "Eje temporal",
    options=["📅 Fecha calendario", "⏱️ Tiempo cero (mes de producción)"],
    horizontal=True,
)
use_time_zero = time_axis == "⏱️ Tiempo cero (mes de producción)"
x_col   = "month_number" if use_time_zero else "date"
x_label = "Mes de Producción" if use_time_zero else "Fecha"


# ── Shared y-axis scaler ──────────────────────────────────────────────────────

def robust_yaxis_range(series: pd.Series, margin: float = 0.10) -> list:
    """
    Returns [y_min, y_max] based on 1st and 99th percentile,
    with a margin above the upper bound. Ignores NaN and inf.
    Falls back to [0, None] if data is empty.
    """
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return [0, None]
    y_min = max(0, np.percentile(clean, 1))
    y_max = np.percentile(clean, 99)
    return [y_min, y_max * (1 + margin)]


# ── Production history charts ─────────────────────────────────────────────────

def build_rate_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    color: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode="lines",
        name=y_col,
        line=dict(color=color),
        hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}",
    ))
    y_range = robust_yaxis_range(data[y_col])
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_range=y_range,
    )
    return fig


st.plotly_chart(build_rate_chart(
    well_data, x_col, "gas_rate", x_label,
    "Caudal de Gas (km3/d)",
    f"Historia de Producción de Gas — {selected_sigla}",
    "red",
), use_container_width=True)

st.plotly_chart(build_rate_chart(
    well_data, x_col, "oil_rate", x_label,
    "Caudal de Petróleo (m3/d)",
    f"Historia de Producción de Petróleo — {selected_sigla}",
    "green",
), use_container_width=True)

st.plotly_chart(build_rate_chart(
    well_data, x_col, "water_rate", x_label,
    "Caudal de Agua (m3/d)",
    f"Historia de Producción de Agua — {selected_sigla}",
    "blue",
), use_container_width=True)


# ── Diagnostic plots ──────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 Gráficos Diagnóstico")

# Per-row ratios
diag_data = well_data.copy()
diag_data["GOR"] = (diag_data["Gp"] / diag_data["Np"] * 1000).replace([float("inf"), -float("inf")], np.nan)
diag_data["WOR"] = (diag_data["Wp"] / diag_data["Np"]).replace([float("inf"), -float("inf")], np.nan)
diag_data["WGR"] = (diag_data["Wp"] / diag_data["Gp"] * 1000).replace([float("inf"), -float("inf")], np.nan)

GAS_PLOTS = {
    "Qg vs Gp":  ("Gp", "gas_rate", "Gp (km3)",  "Qg (km3/d)"),
    "WGR vs Gp": ("Gp", "WGR",      "Gp (km3)",  "WGR (m3/km3)"),
    "GOR vs Gp": ("Gp", "GOR",      "Gp (km3)",  "GOR (m3/km3)"),
}
OIL_PLOTS = {
    "Qo vs Np":  ("Np", "oil_rate", "Np (m3)",   "Qo (m3/d)"),
    "WOR vs Np": ("Np", "WOR",      "Np (m3)",   "WOR (m3/m3)"),
    "GOR vs Np": ("Np", "GOR",      "Np (m3)",   "GOR (m3/m3)"),
}

# Show only plots relevant to the well's fluid type
well_fluid = well_data["tipopozoNEW"].iloc[0]

if well_fluid == "Gasífero":
    available_plots = GAS_PLOTS
    fluid_label     = "Gasífero"
elif well_fluid == "Petrolífero":
    available_plots = OIL_PLOTS
    fluid_label     = "Petrolífero"
else:
    available_plots = {**GAS_PLOTS, **OIL_PLOTS}
    fluid_label     = "Todos"

selected_diag_plots = st.multiselect(
    f"Seleccionar gráficos diagnóstico ({fluid_label})",
    options=list(available_plots.keys()),
    default=[],
)

PLOT_COLORS = {
    "Qg vs Gp":  "red",
    "WGR vs Gp": "blue",
    "GOR vs Gp": "orange",
    "Qo vs Np":  "green",
    "WOR vs Np": "blue",
    "GOR vs Np": "orange",
}


def build_diagnostic_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    color: str = "#1f77b4",
) -> go.Figure:
    plot_data = data.dropna(subset=[x_col, y_col]).sort_values(x_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_data[x_col],
        y=plot_data[y_col],
        mode="lines",
        line=dict(color=color),
        hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}",
    ))
    y_range = robust_yaxis_range(plot_data[y_col])
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_range=y_range,
    )
    return fig


if selected_diag_plots:
    for plot_name in selected_diag_plots:
        x_col_d, y_col_d, x_label_d, y_label_d = available_plots[plot_name]
        st.plotly_chart(
            build_diagnostic_chart(
                diag_data, x_col_d, y_col_d, x_label_d, y_label_d,
                f"{selected_sigla} — {plot_name}",
                color=PLOT_COLORS.get(plot_name, "#1f77b4"),
            ),
            use_container_width=True,
        )
else:
    st.caption("Seleccione al menos un gráfico diagnóstico para visualizarlo.")


# ── Data table & download ─────────────────────────────────────────────────────

st.divider()

COLUMN_RENAME = {
    "sigla":          "Sigla",
    "date":           "Fecha",
    "oil_rate":       "Caudal de petróleo (m3/d)",
    "gas_rate":       "Caudal de gas (km3/d)",
    "water_rate":     "Caudal de agua (m3/d)",
    "Np":             "Acumulada de Petróleo (m3)",
    "Gp":             "Acumulada de Gas (m3)",
    "Wp":             "Acumulada de Agua (m3)",
    "tef":            "TEF",
    "tipoextraccion": "Tipo de Extracción",
    "tipopozoNEW":    "Tipo de Pozo",
    "empresaNEW":     "Empresa",
    "formacion":      "Formación",
    "areayacimiento": "Área yacimiento",
}

display_cols  = [c for c in COLUMN_RENAME if c in well_data.columns]
download_data = well_data[display_cols].rename(columns=COLUMN_RENAME)

st.write(download_data)

st.download_button(
    label="⬇️ Descargar tabla como CSV",
    data=download_data.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_sigla}.csv",
    mime="text/csv",
)