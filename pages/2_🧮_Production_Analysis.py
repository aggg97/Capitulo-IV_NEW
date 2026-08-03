import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils import BARRELS_PER_M3, COMPANY_REPLACEMENTS, get_fluid_classification


# ── Session state ─────────────────────────────────────────────────────────────

if "df" in st.session_state:
    data_sorted = st.session_state["df"]
    data_sorted["date"]       = pd.to_datetime(data_sorted["anio"].astype(str) + "-" + data_sorted["mes"].astype(str) + "-1")
    data_sorted["gas_rate"]   = data_sorted["prod_gas"]  / data_sorted["tef"]
    data_sorted["oil_rate"]   = data_sorted["prod_pet"]  / data_sorted["tef"]
    data_sorted["water_rate"] = data_sorted["prod_agua"] / data_sorted["tef"]
    data_sorted               = data_sorted.sort_values(by=["sigla", "date"], ascending=True)
    data_sorted["empresaNEW"] = data_sorted["empresa"].replace(COMPANY_REPLACEMENTS)
    data_sorted               = get_fluid_classification(data_sorted)
    st.info("Utilizando datos recuperados de la memoria.")
else:
    st.warning("⚠️ No se han cargado los datos. Por favor, volvé a la Página Principal.")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image(Image.open("Vaca Muerta rig.png"))

st.sidebar.markdown("### Filtros")

company_options = ["Todas las empresas"] + sorted(data_sorted["empresaNEW"].dropna().unique())
selected_company = st.sidebar.selectbox(
    "Empresa (opcional)",
    options=company_options,
    help=(
        "El filtro de empresa es opcional. Si querés analizar un área sin restricción "
        "de operador — por ejemplo para ver toda la historia productiva de un área que "
        "cambió de manos — dejá 'Todas las empresas'. Aplicar empresa y área en simultáneo "
        "excluye los pozos de operadoras anteriores en esa área."
    ),
)

st.sidebar.divider()
st.sidebar.markdown("### Visualización")

use_semilog = st.sidebar.checkbox(
    "Escala semilog (eje Y)",
    value=False,
    help="Aplica escala logarítmica al eje Y en todos los gráficos de producción y diagnóstico.",
)

st.sidebar.divider()
st.sidebar.markdown("### 🔖 Guardar pozos para comparar")
st.sidebar.caption(
    "Escribí una o más siglas para guardarlas en sesión y usarlas en otras páginas."
)

all_siglas = sorted(data_sorted["sigla"].dropna().unique().tolist())
if "comparison_wells" not in st.session_state:
    st.session_state["comparison_wells"] = []

selected_comparison = st.sidebar.multiselect(
    "Seleccioná siglas:",
    options=all_siglas,
    default=st.session_state["comparison_wells"],
    key="comparison_wells_widget",
)
st.session_state["comparison_wells"] = selected_comparison

if selected_comparison:
    st.sidebar.success(f"{len(selected_comparison)} pozo(s) guardado(s) en sesión.")


# ── Apply company filter ──────────────────────────────────────────────────────

if selected_company == "Todas las empresas":
    company_data = data_sorted.copy()
else:
    company_data = data_sorted[data_sorted["empresaNEW"] == selected_company].copy()

color_palette = px.colors.qualitative.Set3


# ── Header ────────────────────────────────────────────────────────────────────

st.header(":blue[Análisis de Producción]")


# ── Stacked area charts by area ───────────────────────────────────────────────

summary_df = (
    company_data
    .groupby(["areayacimiento", "date"])
    .agg(total_gas_rate=("gas_rate", "sum"), total_oil_rate=("oil_rate", "sum"))
    .reset_index()
)


def apply_semilog(fig: go.Figure) -> go.Figure:
    if use_semilog:
        fig.update_layout(yaxis=dict(type="log"))
    return fig


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
    return apply_semilog(fig)


st.plotly_chart(build_stacked_area(
    summary_df, "total_oil_rate", "Caudal de Petróleo (m³/d)",
    "Producción Total de Petróleo por Área de Yacimiento",
), use_container_width=True)

st.plotly_chart(build_stacked_area(
    summary_df, "total_gas_rate", "Caudal de Gas (km³/d)",
    "Producción Total de Gas por Área de Yacimiento",
), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOP-10 WELL PROFILES
# ══════════════════════════════════════════════════════════════════════════════

st.divider()

col_area, col_years = st.columns([1, 2])

with col_area:
    selected_area = st.selectbox(
        "Área de yacimiento",
        options=sorted(company_data["areayacimiento"].dropna().unique()),
    )

all_years_available = sorted(
    company_data[company_data["areayacimiento"] == selected_area]["anio"].unique(),
    reverse=True,
)

with col_years:
    selected_years = st.multiselect(
        "Años de inicio del pozo",
        options=all_years_available,
        default=all_years_available[:3] if len(all_years_available) >= 3 else all_years_available,
        help="Cada año tendrá su propio color. Los top 10 se destacan sobre el fondo de todos los pozos.",
    )

if not selected_years:
    st.warning("Seleccioná al menos un año para ver los gráficos.")
    st.stop()

rank_criterion = st.radio(
    "Rankear top 10 por:",
    ["Caudal pico", "Acumulada a 1 año"],
    horizontal=True,
    key="rank_criterion",
)

YEAR_PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
year_color_map = {yr: YEAR_PALETTE[i % len(YEAR_PALETTE)] for i, yr in enumerate(sorted(selected_years))}

area_data = company_data[company_data["areayacimiento"] == selected_area].copy()


def cum_at_tef(df: pd.DataFrame, prod_col: str, tef_limit: int) -> pd.Series:
    df = df.copy()
    df["cum_tef"] = df.groupby("sigla")["tef"].cumsum()
    within = df[df["cum_tef"] <= tef_limit]
    return within.groupby("sigla")[prod_col].sum()


def get_top_wells_by_criterion(df_area: pd.DataFrame, rate_col: str, prod_col: str,
                                years: list, criterion: str, top_n: int = 10) -> list:
    wells = set()
    for yr in years:
        yr_data = df_area[df_area["anio"] == yr]
        if criterion == "Caudal pico":
            ranked = (
                yr_data[yr_data[rate_col] > 0]
                .sort_values(rate_col, ascending=False)
                .head(top_n)["sigla"]
                .unique()
            )
        else:  # Acumulada a 1 año
            well_cum = cum_at_tef(yr_data, prod_col, 365)
            ranked = well_cum[well_cum > 0].nlargest(top_n).index.tolist()
        wells.update(ranked)
    return list(wells)


def get_all_new_wells(df_area: pd.DataFrame, years: list) -> list:
    df_prod = df_area[df_area[["oil_rate", "gas_rate"]].max(axis=1) > 0]
    start_yr = df_prod.groupby("sigla")["anio"].min()
    return start_yr[start_yr.isin(years)].index.tolist()


def assign_start_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["start_year"], errors="ignore", inplace=True)
    start = (
        df[df[["oil_rate", "gas_rate"]].max(axis=1) > 0]
        .groupby("sigla")["anio"]
        .min()
        .rename("start_year")
        .reset_index()
    )
    return df.merge(start, on="sigla", how="left")


top_oil_wells = get_top_wells_by_criterion(
    area_data, "oil_rate", "prod_pet", selected_years, rank_criterion
)
top_gas_wells = get_top_wells_by_criterion(
    area_data, "gas_rate", "prod_gas", selected_years, rank_criterion
)

all_new_wells = get_all_new_wells(area_data, selected_years)

top_10_oil_data = assign_start_year(company_data[company_data["sigla"].isin(top_oil_wells)].copy())
top_10_gas_data = assign_start_year(company_data[company_data["sigla"].isin(top_gas_wells)].copy())
all_oil_data    = assign_start_year(company_data[company_data["sigla"].isin(all_new_wells)].copy())
all_gas_data    = all_oil_data.copy()


# ── Time-zero normalisation ───────────────────────────────────────────────────

def add_time_zero(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["first_prod_date", "month_number"], errors="ignore", inplace=True)
    first_prod = (
        df[df[["oil_rate", "gas_rate"]].max(axis=1) > 0]
        .groupby("sigla")["date"]
        .min()
        .rename("first_prod_date")
        .reset_index()
    )
    df = df.merge(first_prod, on="sigla", how="left")
    df["month_number"] = (
        (df["date"].dt.year  - df["first_prod_date"].dt.year) * 12 +
        (df["date"].dt.month - df["first_prod_date"].dt.month) + 1
    )
    return df[df["month_number"] >= 1]


top_10_oil_data = add_time_zero(top_10_oil_data)
top_10_gas_data = add_time_zero(top_10_gas_data)
all_oil_data    = add_time_zero(all_oil_data)
all_gas_data    = add_time_zero(all_gas_data)


# ── Time-axis toggle ──────────────────────────────────────────────────────────

time_axis = st.radio(
    "Eje temporal",
    options=["📅 Fecha calendario", "⏱️ Tiempo cero (mes de producción)"],
    horizontal=True,
)
use_time_zero = time_axis == "⏱️ Tiempo cero (mes de producción)"


# ── Shared helpers ────────────────────────────────────────────────────────────

def robust_yaxis_range(series: pd.Series, margin: float = 0.10) -> list:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return [0, None]
    y_min = max(0, np.percentile(clean, 1))
    y_max = np.percentile(clean, 99)
    return [y_min, y_max * (1 + margin)]


def build_top10_chart(
    well_data: pd.DataFrame,
    highlight_wells: list,
    y_col: str,
    y_label: str,
    title: str,
    use_time_zero: bool,
    all_well_data: pd.DataFrame | None = None,
) -> go.Figure:
    fig   = go.Figure()
    x_col = "month_number" if use_time_zero else "date"
    x_label = "Mes de Producción" if use_time_zero else "Fecha"

    # Grey background: all new wells that are NOT in top
    if all_well_data is not None and not all_well_data.empty:
        non_top = [w for w in all_well_data["sigla"].unique() if w not in set(highlight_wells)]
        grey_shown = False
        for well in non_top:
            wd = all_well_data[all_well_data["sigla"] == well].sort_values(x_col)
            if wd.empty or wd[y_col].max() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=wd[x_col], y=wd[y_col],
                mode="lines",
                name="Otros pozos",
                legendgroup="grey_bg",
                showlegend=not grey_shown,
                line=dict(color="lightgrey", width=1),
                opacity=0.55,
                hovertemplate=f"<b>{well}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y:.1f}}<extra>Otros</extra>",
            ))
            grey_shown = True

    # Colored top wells by start year
    year_shown = set()
    for well in highlight_wells:
        wd = well_data[well_data["sigla"] == well].sort_values(x_col)
        if wd.empty:
            continue
        yr = int(wd["start_year"].iloc[0]) if pd.notna(wd["start_year"].iloc[0]) else "Sin dato"
        color = year_color_map.get(yr, "#888888")
        show_legend = yr not in year_shown
        year_shown.add(yr)
        fig.add_trace(go.Scatter(
            x=wd[x_col], y=wd[y_col],
            mode="lines+markers",
            name=str(yr),
            legendgroup=str(yr),
            showlegend=show_legend,
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate=f"<b>{well}</b> ({yr})<br>{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
        ))

    y_range = robust_yaxis_range(well_data[y_col] if not well_data.empty else pd.Series(dtype=float))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_range=y_range,
        hovermode="x unified",
        legend_title="Año de inicio",
    )
    return apply_semilog(fig)


years_label = ", ".join(str(y) for y in sorted(selected_years))

st.plotly_chart(build_top10_chart(
    top_10_oil_data, top_oil_wells,
    "oil_rate", "Caudal de Petróleo (m³/d)",
    f"Top 10 Pozos — Petróleo ({selected_area} | {years_label})",
    use_time_zero, all_well_data=all_oil_data,
), use_container_width=True)

st.plotly_chart(build_top10_chart(
    top_10_gas_data, top_gas_wells,
    "gas_rate", "Caudal de Gas (km³/d)",
    f"Top 10 Pozos — Gas ({selected_area} | {years_label})",
    use_time_zero, all_well_data=all_gas_data,
), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("📊 Gráficos Diagnóstico")
st.caption(
    "Todos los pozos nuevos del período aparecen en gris. "
    "Los top 10 se destacan en color por año de inicio. "
    "Los ratios se calculan con caudales instantáneos: "
    "GOR = Qg/Qo × 1000, WOR = Qw/Qo, WGR = Qw/Qg × 1000."
)


def prepare_diag_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adds clean cumulative columns and instantaneous ratios."""
    df = df.sort_values(["sigla", "date"]).copy()
    df["Gp_clean"] = df.groupby("sigla")["prod_gas"].cumsum()
    df["Np_clean"] = df.groupby("sigla")["prod_pet"].cumsum()
    df["Wp_clean"] = df.groupby("sigla")["prod_agua"].cumsum()
    # Instantaneous ratios using rates
    oil_safe = df["oil_rate"].replace(0, np.nan)
    gas_safe = df["gas_rate"].replace(0, np.nan)
    df["GOR"] = (df["gas_rate"]   / oil_safe * 1000).replace([np.inf, -np.inf], np.nan)
    df["WOR"] = (df["water_rate"] / oil_safe).replace([np.inf, -np.inf], np.nan)
    df["WGR"] = (df["water_rate"] / gas_safe * 1000).replace([np.inf, -np.inf], np.nan)
    return df


# Prepare diag data for both top wells and all wells
diag_oil_top = prepare_diag_data(top_10_oil_data)
diag_gas_top = prepare_diag_data(top_10_gas_data)
diag_oil_all = prepare_diag_data(all_oil_data)
diag_gas_all = prepare_diag_data(all_gas_data)

GAS_PLOTS = {
    "Qg vs Gp":  ("Gp_clean", "gas_rate",   "Gp (km³)",  "Qg (km³/d)"),
    "WGR vs Gp": ("Gp_clean", "WGR",         "Gp (km³)",  "WGR (m³/km³)"),
    "GOR vs Gp": ("Gp_clean", "GOR",         "Gp (km³)",  "GOR (m³/km³)"),
}
OIL_PLOTS = {
    "Qo vs Np":  ("Np_clean", "oil_rate",   "Np (m³)",   "Qo (m³/d)"),
    "WOR vs Np": ("Np_clean", "WOR",         "Np (m³)",   "WOR (m³/m³)"),
    "GOR vs Np": ("Np_clean", "GOR",         "Np (m³)",   "GOR (m³/m³)"),
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
    top_data: pd.DataFrame,
    all_data: pd.DataFrame,
    top_wells: list,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()

    # Grey background: all wells not in top
    non_top = [w for w in all_data["sigla"].unique() if w not in set(top_wells)]
    grey_shown = False
    for well in non_top:
        wd = all_data[all_data["sigla"] == well].dropna(subset=[x_col, y_col])
        if wd.empty:
            continue
        fig.add_trace(go.Scatter(
            x=wd[x_col], y=wd[y_col],
            mode="lines",
            name="Otros pozos",
            legendgroup="grey_bg",
            showlegend=not grey_shown,
            line=dict(color="lightgrey", width=1),
            opacity=0.5,
            hovertemplate=f"<b>{well}</b><br>{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.2f}}<extra>Otros</extra>",
        ))
        grey_shown = True

    # Colored top wells
    all_y = []
    for i, well in enumerate(top_wells):
        wd = top_data[top_data["sigla"] == well].dropna(subset=[x_col, y_col])
        if wd.empty:
            continue
        fig.add_trace(go.Scatter(
            x=wd[x_col], y=wd[y_col],
            mode="lines+markers",
            name=well,
            line=dict(color=color_palette[i % len(color_palette)]),
            marker=dict(size=4),
            hovertemplate=f"<b>{well}</b><br>{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
        ))
        all_y.extend(wd[y_col].tolist())

    y_range = robust_yaxis_range(pd.Series(all_y))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_range=y_range,
        hovermode="x unified",
        legend_title="Pozo",
    )
    return apply_semilog(fig)


if all_selected:
    for fluid, plot_name, (x_col, y_col, x_label, y_label) in all_selected:
        top_src  = diag_gas_top  if fluid == "gas" else diag_oil_top
        all_src  = diag_gas_all  if fluid == "gas" else diag_oil_all
        top_ws   = top_gas_wells if fluid == "gas" else top_oil_wells
        fluid_lbl = "Gasífero" if fluid == "gas" else "Petrolífero"
        st.plotly_chart(
            build_diagnostic_chart(
                top_src, all_src, top_ws,
                x_col, y_col, x_label, y_label,
                f"{fluid_lbl} — {plot_name} ({selected_area} | {years_label})",
            ),
            use_container_width=True,
        )
else:
    st.caption("Seleccioná al menos un gráfico diagnóstico para visualizarlo.")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — COMPARACIÓN MULTI-ÁREA / MULTI-EMPRESA
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("🔀 Comparación Multi-Área / Multi-Empresa")
st.caption(
    "Comparar el perfil de producción (P50) y los indicadores de acumulada "
    "de diferentes áreas, incluso de distintas empresas."
)

all_companies    = sorted(data_sorted["empresaNEW"].dropna().unique())
all_areas_global = sorted(data_sorted["areayacimiento"].dropna().unique())

comp_years = st.multiselect(
    "Años a incluir en la comparación:",
    options=sorted(data_sorted["anio"].unique(), reverse=True),
    default=sorted(data_sorted["anio"].unique(), reverse=True)[:3],
    key="comp_years",
)

st.markdown("**Agregar áreas para comparar** — podés mezclar empresas:")

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
        areas_for_emp = sorted(data_sorted[data_sorted["empresaNEW"] == emp]["areayacimiento"].dropna().unique())
        area = st.selectbox(
            f"Área #{idx+1}",
            options=areas_for_emp,
            key=f"comp_area_{idx}",
        )
    combos.append((emp, area, COMBO_PALETTE[idx % len(COMBO_PALETTE)]))


def get_combo_base(emp: str, area: str) -> pd.DataFrame:
    return data_sorted[
        (data_sorted["empresaNEW"] == emp) &
        (data_sorted["areayacimiento"] == area) &
        (data_sorted["anio"].isin(comp_years)) &
        (data_sorted["tef"] > 0)
    ].copy()


def median_profile_time_zero(df: pd.DataFrame, rate_col: str) -> pd.DataFrame:
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


# ── TAB de comparación ────────────────────────────────────────────────────────

comp_tab_rate, comp_tab_cum = st.tabs([
    "📈 Perfiles de Producción (P50)",
    "📊 Acumulada por Intervalo",
])

# ── TAB A: Perfiles ───────────────────────────────────────────────────────────
with comp_tab_rate:
    comp_fluid = st.radio(
        "Fluido:",
        ["Petróleo", "Gas"],
        horizontal=True,
        key="comp_fluid",
    )
    comp_rate_col = "oil_rate" if comp_fluid == "Petróleo" else "gas_rate"
    comp_rate_lbl = "Caudal de Petróleo (m³/d)" if comp_fluid == "Petróleo" else "Caudal de Gas (km³/d)"
    show_p10_p90  = st.checkbox("Mostrar banda P10–P90", value=True, key="comp_band")
    comp_time_zero = st.checkbox("Usar tiempo cero", value=True, key="comp_tz")

    fig_comp = go.Figure()
    x_lbl = "Mes de Producción"

    for emp, area, color in combos:
        base = get_combo_base(emp, area)
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

        fig_comp.add_trace(go.Scatter(
            x=x_vals,
            y=profile["p50"],
            mode="lines",
            name=label,
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{label}</b><br>{x_lbl}: %{{x}}<br>P50: %{{y:.1f}}<extra></extra>",
        ))

    fig_comp.update_layout(
        title=f"Comparación P50 — {comp_rate_lbl}",
        xaxis_title=x_lbl,
        yaxis_title=comp_rate_lbl,
        hovermode="x unified",
        template="plotly_white",
        legend_title="Área / Empresa",
        height=520,
    )
    apply_semilog(fig_comp)
    st.plotly_chart(fig_comp, use_container_width=True)

# ── TAB B: Acumulada por intervalo ────────────────────────────────────────────
with comp_tab_cum:
    st.caption(
        "Acumulada P50 por pozo en cada área, a 180 días, 1 año y 5 años de TEF. "
        "Solo se incluyen pozos que alcanzaron ese umbral de producción."
    )

    INTERVALS_CUM = {"180 días": 180, "1 año": 365, "5 años": 365 * 5}

    cum_fluid = st.radio(
        "Fluido:",
        ["Petróleo", "Gas"],
        horizontal=True,
        key="cum_fluid",
    )
    cum_prod_col = "prod_pet" if cum_fluid == "Petróleo" else "prod_gas"
    cum_lbl      = "Petróleo Acumulado (m³)" if cum_fluid == "Petróleo" else "Gas Acumulado (km³)"

    rows_cum = []
    for emp, area, color in combos:
        base = get_combo_base(emp, area)
        if base.empty:
            continue
        label = f"{emp} — {area}"
        n_total = base["sigla"].nunique()
        row = {"Área / Empresa": label, "Pozos totales": n_total, "_color": color}
        for interval_lbl, days in INTERVALS_CUM.items():
            well_cum = cum_at_tef(base, cum_prod_col, days)
            eligible = well_cum[well_cum > 0]
            row[f"N pozos ≥{interval_lbl}"] = len(eligible)
            row[f"P10 @ {interval_lbl}"]    = eligible.quantile(0.10) if not eligible.empty else np.nan
            row[f"P50 @ {interval_lbl}"]    = eligible.median()        if not eligible.empty else np.nan
            row[f"P90 @ {interval_lbl}"]    = eligible.quantile(0.90) if not eligible.empty else np.nan
        rows_cum.append(row)

    if not rows_cum:
        st.info("Seleccioná al menos una combinación área/empresa con datos.")
    else:
        cum_df = pd.DataFrame(rows_cum)

        st.markdown("#### Comparación P50 Acumulada por Intervalo")
        for interval_lbl in INTERVALS_CUM:
            p50_col = f"P50 @ {interval_lbl}"
            p10_col = f"P10 @ {interval_lbl}"
            p90_col = f"P90 @ {interval_lbl}"
            n_col   = f"N pozos ≥{interval_lbl}"

            plot_cum = cum_df.dropna(subset=[p50_col]).sort_values(p50_col, ascending=True)
            if plot_cum.empty:
                continue

            fig_cum_bar = go.Figure()
            for _, r in plot_cum.iterrows():
                p50v = r[p50_col]
                p10v = r[p10_col] if not np.isnan(r[p10_col]) else p50v
                p90v = r[p90_col] if not np.isnan(r[p90_col]) else p50v
                fig_cum_bar.add_trace(go.Bar(
                    x=[r["Área / Empresa"]],
                    y=[p50v],
                    name=r["Área / Empresa"],
                    marker_color=r["_color"],
                    error_y=dict(
                        type="data", symmetric=False,
                        array=[p90v - p50v], arrayminus=[p50v - p10v],
                        visible=True, color="rgba(50,50,50,0.6)", thickness=2, width=6,
                    ),
                    text=f"{p50v:,.0f}",
                    textposition="outside",
                    hovertemplate=(
                        f"<b>%{{x}}</b><br>P50: %{{y:,.0f}}<br>"
                        f"P10: {p10v:,.0f}<br>P90: {p90v:,.0f}<br>"
                        f"N pozos: {int(r[n_col])}<extra></extra>"
                    ),
                    showlegend=False,
                ))
            fig_cum_bar.update_layout(
                title=f"@ {interval_lbl} — P50 {cum_lbl} (barras de error = P10–P90)",
                xaxis_title=None, yaxis_title=cum_lbl,
                template="plotly_white", height=380, bargap=0.35,
            )
            st.plotly_chart(fig_cum_bar, use_container_width=True)

        # Boxplot distribución completa
        st.markdown("#### Distribución completa — boxplot por área")
        st.caption("Distribución de la acumulada individual de cada pozo en cada área.")

        box_interval = st.selectbox("Intervalo:", list(INTERVALS_CUM.keys()), key="box_interval")
        box_days = INTERVALS_CUM[box_interval]

        box_rows = []
        for emp, area, color in combos:
            base = get_combo_base(emp, area)
            if base.empty:
                continue
            label = f"{emp} — {area}"
            well_cum = cum_at_tef(base, cum_prod_col, box_days)
            for val in well_cum[well_cum > 0]:
                box_rows.append({"Área / Empresa": label, cum_lbl: val, "_color": color})

        if box_rows:
            box_df = pd.DataFrame(box_rows)
            color_map_box = {r["Área / Empresa"]: r["_color"] for r in rows_cum}
            fig_box = px.box(
                box_df,
                x="Área / Empresa", y=cum_lbl,
                color="Área / Empresa",
                color_discrete_map=color_map_box,
                points="outliers",
                template="plotly_white",
                height=420,
                title=f"Distribución acumulada @ {box_interval}",
            )
            fig_box.update_layout(showlegend=False, xaxis_title=None)
            apply_semilog(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

        # Tabla resumen
        st.markdown("#### Tabla Resumen")
        display_cum = cum_df.drop(columns=["_color"]).copy()
        for col in display_cum.columns:
            if col.startswith("P") and "@" in col:
                display_cum[col] = display_cum[col].map(
                    lambda v: f"{v:,.0f}" if pd.notna(v) else "—"
                )
        st.dataframe(display_cum, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — RANKING DE MEJORES POZOS
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.header("🏆 Ranking de Mejores Pozos")
st.caption(
    "Solo se consideran pozos que iniciaron en los años seleccionados arriba."
)

_well_start = (
    data_sorted[
        (data_sorted["empresaNEW"] == selected_company if selected_company != "Todas las empresas" else pd.Series(True, index=data_sorted.index)) &
        (data_sorted["areayacimiento"] == selected_area) &
        (data_sorted["tef"] > 0)
    ]
    .groupby("sigla")["anio"]
    .min()
    .rename("start_year")
    .reset_index()
)
_wells_in_years = _well_start[_well_start["start_year"].isin(selected_years)]["sigla"]

rank_data = data_sorted[
    (data_sorted["sigla"].isin(_wells_in_years)) &
    (data_sorted["tef"] > 0)
].copy()

# Pozos nuevos por año
new_wells_by_year = (
    _well_start[_well_start["start_year"].isin(selected_years)]
    .groupby("start_year")["sigla"].nunique()
    .rename("Pozos nuevos").reset_index()
    .rename(columns={"start_year": "Año"})
    .sort_values("Año")
)

st.markdown("#### 📅 Pozos nuevos por año de inicio")
fig_new_wells = px.bar(
    new_wells_by_year,
    x="Año", y="Pozos nuevos",
    text="Pozos nuevos",
    color="Año",
    color_discrete_map={yr: year_color_map.get(yr, "#888") for yr in new_wells_by_year["Año"]},
    labels={"Año": "Año de inicio", "Pozos nuevos": "N° Pozos"},
    height=320,
    title=f"Pozos nuevos por año — {selected_area}",
)
fig_new_wells.update_traces(textposition="outside")
fig_new_wells.update_layout(template="plotly_white", showlegend=False, xaxis=dict(type="category"))
st.plotly_chart(fig_new_wells, use_container_width=True)

# Peak metrics
peak_df = (
    rank_data.groupby("sigla")
    .agg(
        Qo_pico   =("oil_rate",       "max"),
        Qg_pico   =("gas_rate",       "max"),
        Np_total  =("prod_pet",       "sum"),
        Gp_total  =("prod_gas",       "sum"),
        meses_prod=("tef",            "count"),
        area      =("areayacimiento", "first"),
        empresa   =("empresaNEW",     "first"),
        año_inicio=("anio",           "min"),
    )
    .reset_index()
    .rename(columns={"sigla": "Pozo"})
)
peak_df = peak_df[(peak_df["Qo_pico"] > 0) | (peak_df["Qg_pico"] > 0)].reset_index(drop=True)

rank_fluid = st.radio(
    "Ordenar ranking por:",
    ["Qo pico (m³/d)", "Qg pico (km³/d)", "Np total (m³)", "Gp total (km³)"],
    horizontal=True,
    key="rank_fluid",
)
sort_col_map = {
    "Qo pico (m³/d)":  "Qo_pico",
    "Qg pico (km³/d)": "Qg_pico",
    "Np total (m³)":   "Np_total",
    "Gp total (km³)":  "Gp_total",
}
rank_sort_col = sort_col_map[rank_fluid]
rank_top_n = st.slider("Mostrar top N pozos:", 5, 50, 20, key="rank_top_n")

peak_df_sorted = peak_df.nlargest(rank_top_n, rank_sort_col).reset_index(drop=True)
peak_df_sorted.index += 1

fig_rank = px.bar(
    peak_df_sorted.sort_values(rank_sort_col),
    x=rank_sort_col, y="Pozo",
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

# Ranking table
st.markdown("**Tabla de Ranking**")
display_rank = peak_df_sorted.rename(columns={
    "Qo_pico":    "Qo pico (m³/d)",
    "Qg_pico":    "Qg pico (km³/d)",
    "Np_total":   "Np total (m³)",
    "Gp_total":   "Gp total (km³)",
    "meses_prod": "Meses prod.",
    "area":       "Área",
    "empresa":    "Empresa",
    "año_inicio": "Año inicio",
}).copy()
for col in ["Qo pico (m³/d)", "Qg pico (km³/d)"]:
    display_rank[col] = display_rank[col].map("{:,.1f}".format)
for col in ["Np total (m³)", "Gp total (km³)"]:
    display_rank[col] = display_rank[col].map("{:,.0f}".format)
st.dataframe(display_rank, use_container_width=True)
