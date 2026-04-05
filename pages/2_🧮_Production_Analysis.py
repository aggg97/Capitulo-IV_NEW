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
    data_sorted               = data_sorted.sort_values(by=["sigla", "date"], ascending=True)
    data_sorted["empresaNEW"] = data_sorted["empresa"].replace(COMPANY_REPLACEMENTS)
    data_sorted               = get_fluid_classification(data_sorted)
    st.info("Utilizando datos recuperados de la memoria.")
else:
    st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la Página Principal.")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.header(":blue[Análisis de Producción No Convencional]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))
st.sidebar.title("Por favor filtrar aquí:")

selected_company = st.sidebar.selectbox(
    "Seleccione la empresa",
    options=sorted(data_sorted["empresaNEW"].unique()),
)

company_data = data_sorted[data_sorted["empresaNEW"] == selected_company]

color_palette = px.colors.qualitative.Set3


# ── Company-level stacked area charts ────────────────────────────────────────

summary_df = (
    company_data
    .groupby(["areayacimiento", "date"])
    .agg(total_gas_rate=("gas_rate", "sum"), total_oil_rate=("oil_rate", "sum"))
    .reset_index()
)


def build_stacked_area(summary: pd.DataFrame, y_col: str, y_label: str, title: str) -> go.Figure:
    fig = go.Figure()
    for i, area in enumerate(summary["areayacimiento"].unique()):
        area_data = summary[summary["areayacimiento"] == area]
        fig.add_trace(go.Scatter(
            x=area_data["date"],
            y=area_data[y_col],
            mode="lines",
            name=area,
            stackgroup="one",
            line=dict(color=color_palette[i % len(color_palette)]),
            hovertemplate=f"Fecha: %{{x}}<br>{y_label}: %{{y:.2f}}",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Área de Yacimiento",
    )
    return fig


st.plotly_chart(build_stacked_area(
    summary_df, "total_oil_rate", "Caudal de Petróleo (m3/d)",
    "Producción Total de Petróleo por Área de Yacimiento",
), use_container_width=True)

st.plotly_chart(build_stacked_area(
    summary_df, "total_gas_rate", "Caudal de Gas (km3/d)",
    "Producción Total de Gas por Área de Yacimiento",
), use_container_width=True)


# ── Top-10 well filters ───────────────────────────────────────────────────────

st.divider()

selected_area = st.selectbox(
    "Seleccione el área de yacimiento",
    options=sorted(company_data["areayacimiento"].unique()),
)

selected_year = st.number_input(
    "Ingrese el año",
    min_value=int(data_sorted["anio"].min()),
    max_value=int(data_sorted["anio"].max()),
    value=int(data_sorted["anio"].max()),
    step=1,
)

area_year_data = company_data[
    (company_data["areayacimiento"] == selected_area) &
    (company_data["anio"] == selected_year)
]

top_10_oil_wells = area_year_data.sort_values("oil_rate", ascending=False).head(10)["sigla"].unique()
top_10_gas_wells = area_year_data.sort_values("gas_rate", ascending=False).head(10)["sigla"].unique()

top_10_oil_data = company_data[company_data["sigla"].isin(top_10_oil_wells)].copy()
top_10_gas_data = company_data[company_data["sigla"].isin(top_10_gas_wells)].copy()


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


top_10_oil_data = add_time_zero(top_10_oil_data)
top_10_gas_data = add_time_zero(top_10_gas_data)


# ── Time-axis toggle ──────────────────────────────────────────────────────────

time_axis = st.radio(
    "Eje temporal",
    options=["📅 Fecha calendario", "⏱️ Tiempo cero (mes de producción)"],
    horizontal=True,
)
use_time_zero = time_axis == "⏱️ Tiempo cero (mes de producción)"


# ── Top-10 well production profiles ──────────────────────────────────────────

def build_top10_chart(
    well_data: pd.DataFrame,
    wells: list,
    y_col: str,
    y_label: str,
    title: str,
    use_time_zero: bool,
) -> go.Figure:
    fig     = go.Figure()
    x_col   = "month_number" if use_time_zero else "date"
    x_label = "Mes de Producción" if use_time_zero else "Fecha"

    for i, well in enumerate(wells):
        wd = well_data[well_data["sigla"] == well].sort_values(x_col)
        fig.add_trace(go.Scatter(
            x=wd[x_col],
            y=wd[y_col],
            mode="lines+markers",
            name=well,
            line=dict(color=color_palette[i % len(color_palette)]),
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}",
        ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Pozos",
    )
    return fig


st.plotly_chart(build_top10_chart(
    top_10_oil_data, top_10_oil_wells,
    "oil_rate", "Caudal de Petróleo (m3/d)",
    f"Top 10 Pozos — Perfil de Producción de Petróleo ({selected_area}, {selected_year})",
    use_time_zero,
), use_container_width=True)

st.plotly_chart(build_top10_chart(
    top_10_gas_data, top_10_gas_wells,
    "gas_rate", "Caudal de Gas (km3/d)",
    f"Top 10 Pozos — Perfil de Producción de Gas ({selected_area}, {selected_year})",
    use_time_zero,
), use_container_width=True)


# ── Diagnostic plots ──────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 Gráficos Diagnóstico")

# Compute per-row ratios
diag_data = company_data.copy()
diag_data["GOR"] = (diag_data["Gp"] / diag_data["Np"] * 1000).replace([float("inf"), -float("inf")], None)
diag_data["WOR"] = (diag_data["Wp"] / diag_data["Np"]).replace([float("inf"), -float("inf")], None)
diag_data["WGR"] = (diag_data["Wp"] / diag_data["Gp"] * 1000).replace([float("inf"), -float("inf")], None)

gasifero_data    = diag_data[diag_data["tipopozoNEW"] == "Gasífero"]
petrolifero_data = diag_data[diag_data["tipopozoNEW"] == "Petrolífero"]

# Plot definitions: display name → (x_col, y_col, x_label, y_label)
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

col_left, col_right = st.columns(2)
with col_left:
    selected_gas_plots = st.multiselect(
        "Gráficos Gasífero",
        options=list(GAS_PLOTS.keys()),
        default=[],
    )
with col_right:
    selected_oil_plots = st.multiselect(
        "Gráficos Petrolífero",
        options=list(OIL_PLOTS.keys()),
        default=[],
    )

all_selected = (
    [("gas", name, GAS_PLOTS[name]) for name in selected_gas_plots] +
    [("oil", name, OIL_PLOTS[name]) for name in selected_oil_plots]
)


def build_diagnostic_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for i, well in enumerate(data["sigla"].unique()):
        wd = data[data["sigla"] == well].dropna(subset=[x_col, y_col]).sort_values(x_col)
        if wd.empty:
            continue
        fig.add_trace(go.Scatter(
            x=wd[x_col],
            y=wd[y_col],
            mode="lines+markers",
            name=well,
            line=dict(color=color_palette[i % len(color_palette)]),
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}",
        ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Pozo",
    )
    return fig


if all_selected:
    for fluid, plot_name, (x_col, y_col, x_label, y_label) in all_selected:
        source      = gasifero_data if fluid == "gas" else petrolifero_data
        fluid_label = "Gasífero" if fluid == "gas" else "Petrolífero"
        st.plotly_chart(
            build_diagnostic_chart(
                source, x_col, y_col, x_label, y_label,
                f"{fluid_label} — {plot_name}",
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
    "tipopozo":       "Tipo de Pozo",
    "empresa":        "Empresa",
    "formacion":      "Formación",
    "areayacimiento": "Área yacimiento",
}

download_data = (
    pd.concat([top_10_oil_data, top_10_gas_data])
    .drop_duplicates()
    .rename(columns=COLUMN_RENAME)
)

st.write(download_data)

st.download_button(
    label="⬇️ Descargar tabla como CSV",
    data=download_data.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_company}_{selected_area}_{selected_year}_top10.csv",
    mime="text/csv",
)