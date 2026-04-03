import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils import COMPANY_REPLACEMENTS


# ── Session state ─────────────────────────────────────────────────────────────

if "df" in st.session_state:
    data_sorted = st.session_state["df"]
    data_sorted["date"]       = pd.to_datetime(data_sorted["anio"].astype(str) + "-" + data_sorted["mes"].astype(str) + "-1")
    data_sorted["gas_rate"]   = data_sorted["prod_gas"] / data_sorted["tef"]
    data_sorted["oil_rate"]   = data_sorted["prod_pet"] / data_sorted["tef"]
    data_sorted               = data_sorted.sort_values(by=["sigla", "date"], ascending=True)
    data_sorted["empresaNEW"] = data_sorted["empresa"].replace(COMPANY_REPLACEMENTS)
    st.info("Utilizando datos recuperados de la memoria.")
else:
    st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la Página Principal.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.header(":blue[Análisis de Producción No Convencional]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))
st.sidebar.title("Por favor filtrar aquí:")

selected_company = st.sidebar.selectbox(
    "Seleccione la empresa",
    options=sorted(data_sorted["empresaNEW"].unique()),
)

company_data = data_sorted[data_sorted["empresaNEW"] == selected_company]


# ── Company-level stacked area charts ────────────────────────────────────────

summary_df = (
    company_data
    .groupby(["areayacimiento", "date"])
    .agg(total_gas_rate=("gas_rate", "sum"), total_oil_rate=("oil_rate", "sum"))
    .reset_index()
)

color_palette = px.colors.qualitative.Set3


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
    Adds a 'month_number' column counting from the first month where any
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


# ── Data table & download ─────────────────────────────────────────────────────

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

# Download reflects the current top-10 selection (oil + gas wells combined)
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