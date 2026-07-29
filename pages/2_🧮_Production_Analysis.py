import numpy as np
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

company_data  = data_sorted[data_sorted["empresaNEW"] == selected_company]
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

# ── Filtros del análisis por área ─────────────────────────────────────────────

col_area, col_years = st.columns([1, 2])

with col_area:
    selected_area = st.selectbox(
        "Área de yacimiento",
        options=sorted(company_data["areayacimiento"].unique()),
    )

all_years_available = sorted(
    company_data[company_data["areayacimiento"] == selected_area]["anio"].unique(),
    reverse=True,
)

with col_years:
    selected_years = st.multiselect(
        "Años de inicio del pozo (Top 10 por año, cada año = un color)",
        options=all_years_available,
        default=all_years_available[:3] if len(all_years_available) >= 3 else all_years_available,
        help="Podés elegir varios años. Cada año tendrá su propio color en los gráficos.",
    )

if not selected_years:
    st.warning("Seleccioná al menos un año para ver los gráficos.")
    st.stop()

# Palette fija: un color por año (consistente entre todos los gráficos)
YEAR_PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
year_color_map = {yr: YEAR_PALETTE[i % len(YEAR_PALETTE)] for i, yr in enumerate(sorted(selected_years))}

# Top-10 por año: tomamos los 10 mejores pozos de cada año seleccionado
def get_top_wells_multi_year(df_area: pd.DataFrame, rate_col: str, years: list, top_n: int = 10) -> list:
    """Devuelve la unión de los top_n pozos de cada año, por peak rate en ese año."""
    wells = set()
    for yr in years:
        yr_data = df_area[df_area["anio"] == yr]
        top = yr_data.sort_values(rate_col, ascending=False).head(top_n)["sigla"].unique()
        wells.update(top)
    return list(wells)

area_data = company_data[company_data["areayacimiento"] == selected_area]

top_oil_wells = get_top_wells_multi_year(area_data, "oil_rate", selected_years)
top_gas_wells = get_top_wells_multi_year(area_data, "gas_rate", selected_years)

top_10_oil_data = company_data[company_data["sigla"].isin(top_oil_wells)].copy()
top_10_gas_data = company_data[company_data["sigla"].isin(top_gas_wells)].copy()

# Asignar el año de inicio de cada pozo (primer año con producción > 0) para colorear
def assign_start_year(df: pd.DataFrame) -> pd.DataFrame:
    start = (
        df[df[["oil_rate", "gas_rate"]].max(axis=1) > 0]
        .groupby("sigla")["anio"]
        .min()
        .rename("start_year")
    )
    return df.merge(start, on="sigla", how="left")

top_10_oil_data = assign_start_year(top_10_oil_data)
top_10_gas_data = assign_start_year(top_10_gas_data)


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


# ── Shared y-axis scaler ──────────────────────────────────────────────────────

def robust_yaxis_range(series: pd.Series, margin: float = 0.10) -> list:
    """
    Returns [y_min, y_max] based on the 1st and 99th percentile of the
    series, with a margin added above the upper bound.
    Ignores NaN and inf values. Falls back to [0, None] if data is empty.
    """
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return [0, None]
    y_min = max(0, np.percentile(clean, 1))
    y_max = np.percentile(clean, 99)
    return [y_min, y_max * (1 + margin)]


# ── Top-10 well production profiles ──────────────────────────────────────────

def build_top10_chart(
    well_data: pd.DataFrame,
    wells: list,
    y_col: str,
    y_label: str,
    title: str,
    use_time_zero: bool,
) -> go.Figure:
    """
    Gráfico de perfiles de producción para los pozos indicados.
    Cada pozo se colorea según su año de inicio, usando year_color_map.
    Pozos del mismo año comparten color pero cada línea tiene su propio nombre.
    """
    fig     = go.Figure()
    x_col   = "month_number" if use_time_zero else "date"
    x_label = "Mes de Producción" if use_time_zero else "Fecha"

    # Agrupar por año para leyenda coherente
    year_shown = set()   # para no duplicar entradas de leyenda por año
    for well in wells:
        wd = well_data[well_data["sigla"] == well].sort_values(x_col)
        if wd.empty:
            continue
        yr = int(wd["start_year"].iloc[0]) if "start_year" in wd.columns and pd.notna(wd["start_year"].iloc[0]) else "Sin dato"
        color = year_color_map.get(yr, "#888888")
        show_legend = yr not in year_shown
        year_shown.add(yr)
        fig.add_trace(go.Scatter(
            x=wd[x_col],
            y=wd[y_col],
            mode="lines+markers",
            name=str(yr),
            legendgroup=str(yr),
            showlegend=show_legend,
            line=dict(color=color),
            hovertemplate=f"<b>{well}</b> ({yr})<br>{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
        ))

    y_range = robust_yaxis_range(well_data[y_col])
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_range=y_range,
        hovermode="x unified",
        legend_title="Año de inicio",
    )
    return fig


years_label = ", ".join(str(y) for y in sorted(selected_years))

st.plotly_chart(build_top10_chart(
    top_10_oil_data, top_oil_wells,
    "oil_rate", "Caudal de Petróleo (m3/d)",
    f"Top 10 Pozos — Perfil de Producción de Petróleo ({selected_area} | {years_label})",
    use_time_zero,
), use_container_width=True)

st.plotly_chart(build_top10_chart(
    top_10_gas_data, top_gas_wells,
    "gas_rate", "Caudal de Gas (km3/d)",
    f"Top 10 Pozos — Perfil de Producción de Gas ({selected_area} | {years_label})",
    use_time_zero,
), use_container_width=True)


# ── Diagnostic plots — scoped to top-10 wells ────────────────────────────────

st.divider()
st.subheader("📊 Gráficos Diagnóstico")
st.caption("Los gráficos diagnóstico muestran únicamente los Top 10 pozos seleccionados arriba.")

# Per-row ratios computed on top-10 data only
# Recompute clean monotonic cumulative per well from monthly production volumes.
# The source dataset sometimes has corrections that make raw Gp/Np non-monotonic
# — recomputing per well guarantees a smooth x-axis for every curve.
def prepare_diag_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sigla", "date"]).copy()
    df["Gp_clean"] = df.groupby("sigla")["prod_gas"].cumsum()
    df["Np_clean"] = df.groupby("sigla")["prod_pet"].cumsum()
    df["Wp_clean"] = df.groupby("sigla")["prod_agua"].cumsum()
    df["GOR"] = (df["Gp_clean"] / df["Np_clean"] * 1000).replace([float("inf"), -float("inf")], np.nan)
    df["WOR"] = (df["Wp_clean"] / df["Np_clean"]).replace([float("inf"), -float("inf")], np.nan)
    df["WGR"] = (df["Wp_clean"] / df["Gp_clean"] * 1000).replace([float("inf"), -float("inf")], np.nan)
    return df

diag_oil_data = prepare_diag_data(top_10_oil_data)
diag_gas_data = prepare_diag_data(top_10_gas_data)

GAS_PLOTS = {
    "Qg vs Gp":  ("Gp_clean", "gas_rate", "Gp (km3)",  "Qg (km3/d)"),
    "WGR vs Gp": ("Gp_clean", "WGR",      "Gp (km3)",  "WGR (m3/km3)"),
    "GOR vs Gp": ("Gp_clean", "GOR",      "Gp (km3)",  "GOR (m3/km3)"),
}
OIL_PLOTS = {
    "Qo vs Np":  ("Np_clean", "oil_rate", "Np (m3)",   "Qo (m3/d)"),
    "WOR vs Np": ("Np_clean", "WOR",      "Np (m3)",   "WOR (m3/m3)"),
    "GOR vs Np": ("Np_clean", "GOR",      "Np (m3)",   "GOR (m3/m3)"),
}

col_left, col_right = st.columns(2)
with col_left:
    selected_gas_plots = st.multiselect(
        "Gráficos Gasífero (Top 10 pozos de gas)",
        options=list(GAS_PLOTS.keys()),
        default=[],
    )
with col_right:
    selected_oil_plots = st.multiselect(
        "Gráficos Petrolífero (Top 10 pozos de petróleo)",
        options=list(OIL_PLOTS.keys()),
        default=[],
    )

all_selected = (
    [("gas", name, GAS_PLOTS[name]) for name in selected_gas_plots] +
    [("oil", name, OIL_PLOTS[name]) for name in selected_oil_plots]
)


def build_diagnostic_chart(
    data: pd.DataFrame,
    wells: list,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    all_y_values = []

    for i, well in enumerate(wells):
        # Data is pre-sorted by [sigla, date] and uses clean cumulative columns
        wd = data[data["sigla"] == well].dropna(subset=[x_col, y_col])
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
        all_y_values.extend(wd[y_col].tolist())

    y_range = robust_yaxis_range(pd.Series(all_y_values))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_range=y_range,
        hovermode="x unified",
        legend_title="Pozo",
    )
    return fig


if all_selected:
    for fluid, plot_name, (x_col, y_col, x_label, y_label) in all_selected:
        source      = diag_gas_data if fluid == "gas" else diag_oil_data
        wells       = top_gas_wells if fluid == "gas" else top_oil_wells
        fluid_label = "Gasífero" if fluid == "gas" else "Petrolífero"
        st.plotly_chart(
            build_diagnostic_chart(
                source, wells, x_col, y_col, x_label, y_label,
                f"{fluid_label} — {plot_name} (Top 10, {selected_area} | {years_label})",
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
    file_name=f"{selected_company}_{selected_area}_{years_label}_top10.csv",
    mime="text/csv",
)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — COMPARACIÓN MULTI-ÁREA / MULTI-EMPRESA
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("🔀 Comparación Multi-Área / Multi-Empresa")
st.caption(
    "Comparar el perfil de producción promedio (P50) de diferentes áreas, "
    "incluso de distintas empresas. Cada área/empresa queda con su propio color."
)

# ── Selección de combinaciones área + empresa ─────────────────────────────────
all_companies   = sorted(data_sorted["empresaNEW"].unique())
all_areas_global = sorted(data_sorted["areayacimiento"].unique())

comp_years = st.multiselect(
    "Años a incluir en la comparación:",
    options=sorted(data_sorted["anio"].unique(), reverse=True),
    default=sorted(data_sorted["anio"].unique(), reverse=True)[:3],
    key="comp_years",
)

st.markdown("**Agregar áreas para comparar** — podés mezclar empresas:")

# Hasta 6 combinaciones empresa + área
MAX_COMBOS = 6
if "comp_combos" not in st.session_state:
    st.session_state["comp_combos"] = 1

col_add, col_reset = st.columns([1, 1])
with col_add:
    if st.button("➕ Agregar área", disabled=st.session_state["comp_combos"] >= MAX_COMBOS):
        st.session_state["comp_combos"] += 1
with col_reset:
    if st.button("🗑️ Reiniciar selección"):
        st.session_state["comp_combos"] = 1

combos = []
COMBO_PALETTE = px.colors.qualitative.Dark24
for idx in range(st.session_state["comp_combos"]):
    c1, c2 = st.columns(2)
    with c1:
        emp = st.selectbox(
            f"Empresa #{idx+1}",
            options=all_companies,
            key=f"comp_emp_{idx}",
            index=0,
        )
    with c2:
        areas_for_emp = sorted(data_sorted[data_sorted["empresaNEW"] == emp]["areayacimiento"].unique())
        area = st.selectbox(
            f"Área #{idx+1}",
            options=areas_for_emp,
            key=f"comp_area_{idx}",
        )
    combos.append((emp, area, COMBO_PALETTE[idx % len(COMBO_PALETTE)]))

# ── Helpers de comparación ────────────────────────────────────────────────────

def median_profile_time_zero(df: pd.DataFrame, rate_col: str) -> pd.DataFrame:
    """
    Normaliza cada pozo a tiempo cero y calcula P50 mensual del rate.
    Devuelve DataFrame con columnas [month_number, p50, p10, p90].
    """
    df = df.copy()
    first_prod = (
        df[df[rate_col] > 0]
        .groupby("sigla")["date"]
        .min()
        .rename("first_prod_date")
    )
    df = df.merge(first_prod, on="sigla", how="left").dropna(subset=["first_prod_date"])
    df["month_number"] = (
        (df["date"].dt.year  - df["first_prod_date"].dt.year) * 12 +
        (df["date"].dt.month - df["first_prod_date"].dt.month) + 1
    )
    df = df[df["month_number"] >= 1]
    grp = df.groupby("month_number")[rate_col]
    result = grp.median().rename("p50").reset_index()
    result["p10"] = grp.quantile(0.10).values
    result["p90"] = grp.quantile(0.90).values
    return result


comp_fluid = st.radio(
    "Fluido a comparar:",
    ["Petróleo", "Gas"],
    horizontal=True,
    key="comp_fluid",
)
comp_rate_col  = "oil_rate" if comp_fluid == "Petróleo" else "gas_rate"
comp_rate_lbl  = "Caudal de Petróleo (m3/d)" if comp_fluid == "Petróleo" else "Caudal de Gas (km3/d)"
show_p10_p90   = st.checkbox("Mostrar banda P10–P90", value=True, key="comp_band")
comp_time_zero = st.checkbox("Usar tiempo cero (mes de producción)", value=True, key="comp_tz")

fig_comp = go.Figure()

for emp, area, color in combos:
    base = data_sorted[
        (data_sorted["empresaNEW"] == emp) &
        (data_sorted["areayacimiento"] == area) &
        (data_sorted["anio"].isin(comp_years)) &
        (data_sorted["tef"] > 0)
    ]
    if base.empty:
        continue

    label = f"{emp} — {area}"

    if comp_time_zero:
        profile = median_profile_time_zero(base, comp_rate_col)
        x_vals  = profile["month_number"]
        x_lbl   = "Mes de Producción"
    else:
        profile_cal = (
            base.groupby("date")[comp_rate_col]
            .median()
            .reset_index()
            .sort_values("date")
        )
        profile = profile_cal.rename(columns={comp_rate_col: "p50"})
        profile["p10"] = base.groupby("date")[comp_rate_col].quantile(0.10).values
        profile["p90"] = base.groupby("date")[comp_rate_col].quantile(0.90).values
        x_vals = profile["date"]
        x_lbl  = "Fecha"

    # Banda P10-P90
    if show_p10_p90:
        fig_comp.add_trace(go.Scatter(
            x=pd.concat([x_vals, x_vals[::-1]]),
            y=pd.concat([profile["p90"], profile["p10"][::-1]]),
            fill="toself",
            fillcolor=color,
            opacity=0.12,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Línea P50
    fig_comp.add_trace(go.Scatter(
        x=x_vals,
        y=profile["p50"],
        mode="lines",
        name=label,
        line=dict(color=color, width=2.5),
        hovertemplate=f"<b>{label}</b><br>{x_lbl}: %{{x}}<br>P50 {comp_rate_lbl}: %{{y:.1f}}<extra></extra>",
    ))

fig_comp.update_layout(
    title=f"Comparación P50 — {comp_rate_lbl}",
    xaxis_title=x_lbl if combos else "Mes",
    yaxis_title=comp_rate_lbl,
    hovermode="x unified",
    template="plotly_white",
    legend_title="Área / Empresa",
    height=500,
)
st.plotly_chart(fig_comp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — RANKING DE MEJORES POZOS + ENVÍO A WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("🏆 Ranking de Mejores Pozos")
st.caption(
    "Ranking por caudal pico en el área/años seleccionados arriba. "
    "Podés enviar pozos a la Watchlist para hacer seguimiento entre páginas."
)

# ── Inicializar watchlist en session_state ────────────────────────────────────
if "watchlist_wells" not in st.session_state:
    st.session_state["watchlist_wells"] = []

# ── Calcular caudal pico por pozo en el área/años elegidos ───────────────────
rank_data = data_sorted[
    (data_sorted["empresaNEW"] == selected_company) &
    (data_sorted["areayacimiento"] == selected_area) &
    (data_sorted["anio"].isin(selected_years)) &
    (data_sorted["tef"] > 0)
].copy()

peak_df = (
    rank_data.groupby("sigla")
    .agg(
        Qo_pico    =("oil_rate",       "max"),
        Qg_pico    =("gas_rate",       "max"),
        Np_total   =("prod_pet",       "sum"),
        Gp_total   =("prod_gas",       "sum"),
        meses_prod =("tef",            "count"),
        area       =("areayacimiento", "first"),
        empresa    =("empresaNEW",     "first"),
        año_inicio =("anio",           "min"),
    )
    .reset_index()
    .rename(columns={"sigla": "Pozo"})
)

rank_fluid = st.radio(
    "Ordenar ranking por:",
    ["Qo pico (m³/d)", "Qg pico (km³/d)", "Np total (m³)", "Gp total (km³)"],
    horizontal=True,
    key="rank_fluid",
)

sort_col_map = {
    "Qo pico (m³/d)":   "Qo_pico",
    "Qg pico (km³/d)":  "Qg_pico",
    "Np total (m³)":    "Np_total",
    "Gp total (km³)":   "Gp_total",
}
rank_sort_col = sort_col_map[rank_fluid]
rank_top_n = st.slider("Mostrar top N pozos:", 5, 50, 20, key="rank_top_n")

peak_df_sorted = peak_df.nlargest(rank_top_n, rank_sort_col).reset_index(drop=True)
peak_df_sorted.index += 1   # ranking desde 1

# ── Gráfico de barras del ranking ─────────────────────────────────────────────
fig_rank = px.bar(
    peak_df_sorted.sort_values(rank_sort_col),
    x=rank_sort_col,
    y="Pozo",
    orientation="h",
    color="año_inicio",
    color_discrete_map={yr: year_color_map.get(yr, "#888") for yr in peak_df_sorted["año_inicio"].unique()},
    text=rank_sort_col,
    hover_data={"empresa": True, "area": True, "meses_prod": True},
    labels={rank_sort_col: rank_fluid, "año_inicio": "Año inicio"},
    height=max(350, rank_top_n * 26),
    title=f"Ranking — {rank_fluid} ({selected_area} | {years_label})",
)
fig_rank.update_traces(texttemplate="%{text:,.0f}", textposition="inside")
fig_rank.update_layout(template="plotly_white", yaxis_title=None)
st.plotly_chart(fig_rank, use_container_width=True)

# ── Tabla interactiva ─────────────────────────────────────────────────────────
st.markdown("**Tabla de Ranking**")
display_rank = peak_df_sorted.copy()
display_rank["Qo_pico"]  = display_rank["Qo_pico"].map("{:,.1f}".format)
display_rank["Qg_pico"]  = display_rank["Qg_pico"].map("{:,.1f}".format)
display_rank["Np_total"] = display_rank["Np_total"].map("{:,.0f}".format)
display_rank["Gp_total"] = display_rank["Gp_total"].map("{:,.0f}".format)
display_rank = display_rank.rename(columns={
    "Qo_pico":    "Qo pico (m³/d)",
    "Qg_pico":    "Qg pico (km³/d)",
    "Np_total":   "Np total (m³)",
    "Gp_total":   "Gp total (km³)",
    "meses_prod": "Meses prod.",
    "area":       "Área",
    "empresa":    "Empresa",
    "año_inicio": "Año inicio",
})
st.dataframe(display_rank, use_container_width=True)

st.download_button(
    label="⬇️ Descargar ranking como CSV",
    data=peak_df_sorted.to_csv(index=False).encode("utf-8"),
    file_name=f"ranking_{selected_company}_{selected_area}_{years_label}.csv",
    mime="text/csv",
    key="dl_ranking",
)

# ── Enviar pozos a la Watchlist ───────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🚨 Enviar pozos a la Watchlist")
st.caption(
    "Seleccioná pozos del ranking para agregarlos a la Watchlist. "
    "Quedan guardados en la sesión y podés verlos en la página Watchlist."
)

wells_to_add = st.multiselect(
    "Seleccionar pozos a agregar:",
    options=peak_df_sorted["Pozo"].tolist(),
    default=[],
    key="wells_to_watchlist",
    help="Los pozos se guardan en memoria de sesión — disponibles en todas las páginas.",
)

col_add_wl, col_clear_wl = st.columns([1, 1])
with col_add_wl:
    if st.button("➕ Agregar a Watchlist", disabled=not wells_to_add):
        current = set(st.session_state["watchlist_wells"])
        new_wells = [w for w in wells_to_add if w not in current]
        st.session_state["watchlist_wells"].extend(new_wells)
        if new_wells:
            st.success(f"Agregados {len(new_wells)} pozo(s): {', '.join(new_wells)}")
        else:
            st.info("Todos los pozos seleccionados ya estaban en la Watchlist.")

with col_clear_wl:
    if st.button("🗑️ Limpiar Watchlist completa"):
        st.session_state["watchlist_wells"] = []
        st.success("Watchlist limpiada.")

# Mostrar estado actual de la Watchlist
if st.session_state["watchlist_wells"]:
    st.markdown(f"**Watchlist actual ({len(st.session_state['watchlist_wells'])} pozos):**")

    wl_data = data_sorted[
        data_sorted["sigla"].isin(st.session_state["watchlist_wells"]) &
        (data_sorted["tef"] > 0)
    ]

    # Perfiles de producción de la watchlist — tiempo cero
    if not wl_data.empty:
        wl_data = assign_start_year(wl_data)

        # Paleta independiente de la watchlist
        WL_PALETTE = px.colors.qualitative.Vivid
        wl_color_map = {
            w: WL_PALETTE[i % len(WL_PALETTE)]
            for i, w in enumerate(st.session_state["watchlist_wells"])
        }

        for rate_col, rate_lbl in [("oil_rate", "Caudal de Petróleo (m3/d)"), ("gas_rate", "Caudal de Gas (km3/d)")]:
            fig_wl = go.Figure()
            for well in st.session_state["watchlist_wells"]:
                wd = wl_data[wl_data["sigla"] == well].copy()
                if wd.empty:
                    continue
                first = wd[wd[rate_col] > 0]["date"].min()
                if pd.isna(first):
                    continue
                wd["month_number"] = (
                    (wd["date"].dt.year  - first.year)  * 12 +
                    (wd["date"].dt.month - first.month) + 1
                )
                wd = wd[wd["month_number"] >= 1].sort_values("month_number")
                yr = int(wd["start_year"].iloc[0]) if pd.notna(wd["start_year"].iloc[0]) else "?"
                fig_wl.add_trace(go.Scatter(
                    x=wd["month_number"],
                    y=wd[rate_col],
                    mode="lines+markers",
                    name=f"{well} ({yr})",
                    line=dict(color=wl_color_map[well]),
                    hovertemplate=f"<b>{well}</b><br>Mes: %{{x}}<br>{rate_lbl}: %{{y:.1f}}<extra></extra>",
                ))
            y_range = robust_yaxis_range(wl_data[rate_col])
            fig_wl.update_layout(
                title=f"Watchlist — {rate_lbl} (tiempo cero)",
                xaxis_title="Mes de Producción",
                yaxis_title=rate_lbl,
                yaxis_range=y_range,
                hovermode="x unified",
                template="plotly_white",
                legend_title="Pozo (año inicio)",
                height=420,
            )
            st.plotly_chart(fig_wl, use_container_width=True)

    # Tabla resumen de la watchlist
    wl_summary = (
        wl_data.groupby("sigla")
        .agg(
            Qo_pico    =("oil_rate",       "max"),
            Qg_pico    =("gas_rate",       "max"),
            Np_total   =("prod_pet",       "sum"),
            Gp_total   =("prod_gas",       "sum"),
            Empresa    =("empresaNEW",     "first"),
            Area       =("areayacimiento", "first"),
            Año_inicio =("anio",           "min"),
        )
        .reset_index()
        .rename(columns={
            "sigla":     "Pozo",
            "Qo_pico":   "Qo pico (m³/d)",
            "Qg_pico":   "Qg pico (km³/d)",
            "Np_total":  "Np (m³)",
            "Gp_total":  "Gp (km³)",
        })
        .sort_values("Qo pico (m³/d)", ascending=False)
    )
    st.dataframe(wl_summary, use_container_width=True, hide_index=True)

    # Botón para eliminar pozos individuales
    remove_wells = st.multiselect(
        "Eliminar pozos de la Watchlist:",
        options=st.session_state["watchlist_wells"],
        default=[],
        key="remove_from_watchlist",
    )
    if st.button("🗑️ Eliminar seleccionados", disabled=not remove_wells):
        st.session_state["watchlist_wells"] = [
            w for w in st.session_state["watchlist_wells"] if w not in remove_wells
        ]
        st.success(f"Eliminados: {', '.join(remove_wells)}")
        st.rerun()
else:
    st.info("La Watchlist está vacía. Seleccioná pozos del ranking y presioná 'Agregar a Watchlist'.")