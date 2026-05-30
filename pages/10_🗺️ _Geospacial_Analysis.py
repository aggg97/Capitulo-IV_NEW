"""
10_🗺️_Análisis_Geoespacial.py

Análisis geoespacial de pozos en Vaca Muerta:
  - Mapa interactivo de pozos con filtros
  - Agrupación automática de pozos en pads (buffer 30 m, POSGAR 2007)
  - Análisis de producción agregada por pad
  - Rankings y comparativas entre pads
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

from utils import COMPANY_REPLACEMENTS, get_fluid_classification


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Buffer distance in metres (POSGAR 2007 / EPSG:5344)
PAD_BUFFER_M = 30

# Approximate degree-to-metre conversion near Neuquén (~38°S)
# Used as a fallback when geopandas is unavailable
DEG_PER_METRE_LAT = 1 / 111_320
DEG_PER_METRE_LON = 1 / (111_320 * np.cos(np.radians(38)))

MAPBOX_STYLE = "carto-positron"

FLUID_COLORS = {
    "Petrolífero": "#2ecc71",
    "Gasífero":    "#e74c3c",
    "Otro":        "#95a5a6",
}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — load & preprocess
# ══════════════════════════════════════════════════════════════════════════════

if "df" in st.session_state:
    _raw = st.session_state["df"]
    _raw["date"]       = pd.to_datetime(_raw["anio"].astype(str) + "-" + _raw["mes"].astype(str) + "-1")
    _raw["gas_rate"]   = _raw["prod_gas"]  / _raw["tef"]
    _raw["oil_rate"]   = _raw["prod_pet"]  / _raw["tef"]
    _raw["water_rate"] = _raw["prod_agua"] / _raw["tef"]
    _raw               = _raw.sort_values(by=["sigla", "date"], ascending=True)
    _raw["empresaNEW"] = _raw["empresa"].replace(COMPANY_REPLACEMENTS)
    _raw               = get_fluid_classification(_raw)
    data_sorted        = _raw
    st.info("Utilizando datos recuperados de la memoria.")
else:
    st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la Página Principal.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — PAD DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_well_number(sigla: pd.Series) -> pd.Series:
    """Extract the first numeric sequence from the well name."""
    return pd.to_numeric(sigla.str.extract(r"(\d+)", expand=False), errors="coerce")


def _bbox_overlap(ax, ay, bx, by, tol_lon, tol_lat) -> bool:
    return abs(ax - bx) <= tol_lon and abs(ay - by) <= tol_lat


def assign_pads_pure_python(df_wells: pd.DataFrame) -> pd.DataFrame:
    """
    Pure-Python pad assignment using a simple union-find on point proximity.

    Works on WGS-84 coordinates; converts the 30-metre buffer to approximate
    degree tolerances centred at 38°S (Neuquén basin).

    Returns df_wells with a new 'pad_name' column.
    """
    tol_lat = PAD_BUFFER_M * DEG_PER_METRE_LAT
    tol_lon = PAD_BUFFER_M * DEG_PER_METRE_LON

    coords = list(zip(df_wells["x"].values, df_wells["y"].values))
    n = len(coords)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_overlap(coords[i][0], coords[i][1],
                             coords[j][0], coords[j][1],
                             tol_lon, tol_lat):
                union(i, j)

    df_out = df_wells.copy()
    df_out["pad_id"] = [find(i) for i in range(n)]

    # Name each pad after the well with the largest numeric ID in the group
    df_out["nro_pozo"] = extract_well_number(df_out["sigla"])
    best = (
        df_out.sort_values("nro_pozo", ascending=False)
        .drop_duplicates("pad_id")[["pad_id", "sigla"]]
        .rename(columns={"sigla": "pad_name"})
    )
    best["pad_name"] = "pad_" + best["pad_name"].astype(str)
    df_out = df_out.merge(best, on="pad_id", how="left")
    return df_out


def assign_pads_geopandas(df_wells: pd.DataFrame) -> pd.DataFrame:
    """
    Geopandas-based pad assignment: project to POSGAR 2007, buffer 30 m,
    dissolve, explode, spatial-join back to wells.
    """
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        df_wells,
        geometry=gpd.points_from_xy(df_wells["x"], df_wells["y"]),
        crs="EPSG:4326",
    ).to_crs(epsg=5344)

    buffers     = gdf.buffer(PAD_BUFFER_M)
    union_geom  = buffers.union_all()
    pads_series = gpd.GeoSeries([union_geom], crs="EPSG:5344").explode(
        ignore_index=True, index_parts=False
    )
    pads_gdf = gpd.GeoDataFrame(geometry=pads_series)

    joined = gpd.sjoin(gdf, pads_gdf, how="left").rename(
        columns={"index_right": "pad_id"}
    )

    joined["nro_pozo"] = extract_well_number(joined["sigla"])
    best = (
        joined.sort_values("nro_pozo", ascending=False)
        .drop_duplicates("pad_id")[["pad_id", "sigla"]]
        .rename(columns={"sigla": "pad_name"})
    )
    best["pad_name"] = "pad_" + best["pad_name"].astype(str)
    joined = joined.merge(best, on="pad_id", how="left")
    return pd.DataFrame(joined.drop(columns="geometry", errors="ignore"))


@st.cache_data(show_spinner="Detectando pads de perforación…")
def compute_pads(df_prod: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-well table with coordinates + pad assignment.
    Tries geopandas; falls back to pure-Python.
    """
    coord_cols = ["coordenadax", "coordenaday"]
    has_coords = all(c in df_prod.columns for c in coord_cols)

    if not has_coords:
        st.warning(
            "Las columnas 'coordenadax' / 'coordenaday' no están presentes en el dataset. "
            "El análisis geoespacial requiere coordenadas. "
            "Mostrando mapa sin agrupación por pads."
        )
        return pd.DataFrame()

    wells = (
        df_prod[["sigla", "coordenadax", "coordenaday", "anio",
                  "empresaNEW", "tipopozoNEW", "areayacimiento"]]
        .sort_values("anio", ascending=False)
        .drop_duplicates("sigla")
        .rename(columns={"coordenadax": "x", "coordenaday": "y"})
        .dropna(subset=["x", "y"])
        .reset_index(drop=True)
    )

    # Drop wells with obviously invalid coords (0,0 or far outside Argentina)
    wells = wells[
        (wells["x"].between(-75, -55)) &
        (wells["y"].between(-42, -30))
    ].reset_index(drop=True)

    if wells.empty:
        return pd.DataFrame()

    try:
        import geopandas  # noqa: F401
        return assign_pads_geopandas(wells)
    except ImportError:
        return assign_pads_pure_python(wells)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — PRODUCTION AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def build_pad_production(
    df_prod: pd.DataFrame,
    df_pads: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join pad assignments to the monthly production table and aggregate
    total and peak production per pad.
    """
    prod = df_prod.merge(df_pads[["sigla", "pad_name", "pad_id"]], on="sigla", how="inner")

    agg = (
        prod.groupby("pad_name")
        .agg(
            n_wells         =("sigla",      "nunique"),
            total_oil_m3    =("prod_pet",   "sum"),
            total_gas_km3   =("prod_gas",   "sum"),
            total_water_m3  =("prod_agua",  "sum"),
            peak_oil_rate   =("oil_rate",   "max"),
            peak_gas_rate   =("gas_rate",   "max"),
            empresa         =("empresaNEW", lambda s: s.mode().iloc[0] if not s.empty else ""),
            area            =("areayacimiento", lambda s: s.mode().iloc[0] if not s.empty else ""),
            fluid           =("tipopozoNEW", lambda s: s.mode().iloc[0] if not s.empty else ""),
        )
        .reset_index()
    )

    # Centroid of each pad (mean lat/lon of member wells)
    centroids = (
        df_pads.groupby("pad_name")
        .agg(lat=("y", "mean"), lon=("x", "mean"))
        .reset_index()
    )
    return agg.merge(centroids, on="pad_name", how="left")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.header(":blue[🗺️ Análisis Geoespacial — Pozos y Pads]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))
st.sidebar.title("Filtros")

# ── Sidebar filters ───────────────────────────────────────────────────────────

all_companies = sorted(data_sorted["empresaNEW"].dropna().unique())
sel_companies = st.sidebar.multiselect(
    "Empresa:", all_companies, default=[]
)

all_areas = sorted(data_sorted["areayacimiento"].dropna().unique())
sel_areas = st.sidebar.multiselect("Área de yacimiento:", all_areas, default=[])

all_fluids = sorted(data_sorted["tipopozoNEW"].dropna().unique())
sel_fluids = st.sidebar.multiselect("Tipo de pozo:", all_fluids, default=[])

# Apply filters
mask = pd.Series(True, index=data_sorted.index)
if sel_companies:
    mask &= data_sorted["empresaNEW"].isin(sel_companies)
if sel_areas:
    mask &= data_sorted["areayacimiento"].isin(sel_areas)
if sel_fluids:
    mask &= data_sorted["tipopozoNEW"].isin(sel_fluids)

df_filtered = data_sorted[mask]

# ── Compute pads ──────────────────────────────────────────────────────────────

df_pads = compute_pads(df_filtered)

has_pads = not df_pads.empty

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_map, tab_pads, tab_prod, tab_export = st.tabs([
    "🗺️ Mapa de Pozos",
    "🔵 Análisis de Pads",
    "📊 Producción por Pad",
    "⬇️ Exportar",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAPA DE POZOS
# ══════════════════════════════════════════════════════════════════════════════

with tab_map:

    coord_cols = ["coordenadax", "coordenaday"]
    if not all(c in df_filtered.columns for c in coord_cols):
        st.warning(
            "El dataset no contiene columnas de coordenadas ('coordenadax', 'coordenaday'). "
            "Verificá que el archivo de producción incluya la geolocalización de los pozos."
        )
        st.stop()

    # One row per well
    wells_map = (
        df_filtered[["sigla", "coordenadax", "coordenaday",
                      "empresaNEW", "tipopozoNEW", "areayacimiento"]]
        .drop_duplicates("sigla")
        .dropna(subset=coord_cols)
        .rename(columns={"coordenadax": "lon", "coordenaday": "lat"})
    )
    wells_map = wells_map[
        wells_map["lon"].between(-75, -55) & wells_map["lat"].between(-42, -30)
    ]

    if has_pads:
        wells_map = wells_map.merge(
            df_pads[["sigla", "pad_name"]], on="sigla", how="left"
        )

    st.markdown(f"**{len(wells_map):,} pozos** visualizados (según filtros activos).")

    color_by = st.radio(
        "Colorear por:",
        ["Tipo de fluido", "Empresa", "Pad"] if has_pads else ["Tipo de fluido", "Empresa"],
        horizontal=True,
    )

    color_col = {
        "Tipo de fluido": "tipopozoNEW",
        "Empresa":        "empresaNEW",
        "Pad":            "pad_name",
    }[color_by]

    fig_map = px.scatter_mapbox(
        wells_map,
        lat="lat",
        lon="lon",
        color=color_col,
        hover_name="sigla",
        hover_data={
            "empresaNEW":     True,
            "tipopozoNEW":    True,
            "areayacimiento": True,
            "lat":            ":.4f",
            "lon":            ":.4f",
        },
        zoom=8,
        height=620,
        mapbox_style=MAPBOX_STYLE,
        title="Mapa de Pozos — Vaca Muerta",
    )
    fig_map.update_traces(marker=dict(size=6, opacity=0.80))
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend_title=color_by)
    st.plotly_chart(fig_map, use_container_width=True)

    # KPI strip
    c1, c2, c3 = st.columns(3)
    c1.metric("Pozos en mapa",    f"{len(wells_map):,}")
    c2.metric("Empresas",         f"{wells_map['empresaNEW'].nunique():,}")
    c3.metric("Áreas",            f"{wells_map['areayacimiento'].nunique():,}")

    # Mini-breakdown
    st.markdown("#### Distribución por tipo de fluido")
    fluid_counts = (
        wells_map["tipopozoNEW"]
        .value_counts()
        .reset_index()
        .rename(columns={"tipopozoNEW": "Tipo", "count": "Pozos"})
    )
    fig_fluid = px.bar(
        fluid_counts, x="Tipo", y="Pozos", color="Tipo",
        color_discrete_map=FLUID_COLORS,
        text="Pozos",
    )
    fig_fluid.update_traces(textposition="outside")
    fig_fluid.update_layout(
        template="plotly_white", showlegend=False, height=300,
        yaxis_title="N° Pozos", xaxis_title=None,
    )
    st.plotly_chart(fig_fluid, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISIS DE PADS
# ══════════════════════════════════════════════════════════════════════════════

with tab_pads:

    if not has_pads:
        st.info(
            "La detección de pads requiere coordenadas válidas en el dataset. "
            "Verificá que las columnas 'coordenadax' / 'coordenaday' estén presentes."
        )
        st.stop()

    n_pads  = df_pads["pad_name"].nunique()
    n_wells = df_pads["sigla"].nunique()
    solo    = (df_pads.groupby("pad_name")["sigla"].nunique() == 1).sum()
    multi   = n_pads - solo

    st.markdown(f"""
    El algoritmo utilizó un **buffer de {PAD_BUFFER_M} m** proyectado en POSGAR 2007 
    para agrupar pozos que comparten plataforma de perforación.
    """)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pads detectados",      f"{n_pads:,}")
    m2.metric("Pozos con pad",        f"{n_wells:,}")
    m3.metric("Pads multi-pozo",      f"{multi:,}")
    m4.metric("Pozos solitarios",     f"{solo:,}")

    # Mapa coloreado por pad_name
    st.markdown("#### Mapa de Pads")

    pad_map_df = df_pads.rename(columns={"x": "lon", "y": "lat"})

    fig_pads = px.scatter_mapbox(
        pad_map_df,
        lat="lat",
        lon="lon",
        color="pad_name",
        hover_name="sigla",
        hover_data={
            "pad_name":       True,
            "empresaNEW":     True,
            "tipopozoNEW":    True,
            "lat":            ":.4f",
            "lon":            ":.4f",
        },
        zoom=8,
        height=580,
        mapbox_style=MAPBOX_STYLE,
        title="Agrupación de Pozos por Pad (buffer 30 m)",
    )
    fig_pads.update_traces(marker=dict(size=7, opacity=0.85))
    fig_pads.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_pads, use_container_width=True)
    st.caption(
        "Cada color representa un pad distinto. "
        "Pozos del mismo color comparten plataforma según el criterio de proximidad."
    )

    # Distribution of wells per pad
    st.markdown("#### Distribución: Pozos por Pad")
    wells_per_pad = (
        df_pads.groupby("pad_name")["sigla"]
        .nunique()
        .reset_index(name="n_wells")
    )

    fig_dist = px.histogram(
        wells_per_pad, x="n_wells",
        nbins=max(1, int(wells_per_pad["n_wells"].max())),
        labels={"n_wells": "Pozos por Pad", "count": "N° Pads"},
        color_discrete_sequence=["#3498db"],
        title="Histograma: ¿cuántos pozos tiene cada pad?",
    )
    fig_dist.update_layout(template="plotly_white", height=320)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Top pads by well count
    st.markdown("#### Top Pads con Más Pozos")
    top_pads = (
        wells_per_pad.sort_values("n_wells", ascending=False)
        .head(20)
    )
    fig_top = px.bar(
        top_pads.sort_values("n_wells"),
        x="n_wells", y="pad_name", orientation="h",
        labels={"n_wells": "N° Pozos", "pad_name": "Pad"},
        color="n_wells", color_continuous_scale="Blues",
        text="n_wells",
    )
    fig_top.update_traces(textposition="outside")
    fig_top.update_layout(
        template="plotly_white", height=480,
        coloraxis_showscale=False,
        yaxis_title=None,
    )
    st.plotly_chart(fig_top, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRODUCCIÓN POR PAD
# ══════════════════════════════════════════════════════════════════════════════

with tab_prod:

    if not has_pads:
        st.info("Requiere coordenadas para detectar pads.")
        st.stop()

    pad_prod = build_pad_production(
        df_filtered[df_filtered["tef"] > 0],
        df_pads,
    )

    if pad_prod.empty:
        st.warning("No hay datos de producción para los pads detectados.")
        st.stop()

    # Selector: fluid type for ranking
    fluid_sel = st.radio(
        "Tipo de pozo dominante:", ["Petrolífero", "Gasífero", "Todos"],
        horizontal=True,
    )
    top_n = st.slider("Top N pads:", min_value=5, max_value=30, value=15)

    if fluid_sel != "Todos":
        pad_prod_f = pad_prod[pad_prod["fluid"] == fluid_sel]
    else:
        pad_prod_f = pad_prod

    # ── Bubble map: Pad productivity ──────────────────────────────────────────
    st.markdown("#### Mapa de Burbujas — Producción Acumulada por Pad")

    metric_bubble = st.selectbox(
        "Métrica de tamaño:",
        ["total_oil_m3", "total_gas_km3", "n_wells"],
        format_func=lambda c: {
            "total_oil_m3":  "Petróleo Acumulado (m³)",
            "total_gas_km3": "Gas Acumulado (km³)",
            "n_wells":       "N° Pozos",
        }[c],
    )

    fig_bubble_map = px.scatter_mapbox(
        pad_prod_f.dropna(subset=["lat", "lon"]),
        lat="lat", lon="lon",
        size=metric_bubble,
        color="fluid",
        color_discrete_map=FLUID_COLORS,
        hover_name="pad_name",
        hover_data={
            "n_wells":       True,
            "empresa":       True,
            "area":          True,
            "total_oil_m3":  ":,.0f",
            "total_gas_km3": ":,.1f",
            "lat": False, "lon": False,
        },
        zoom=8, height=580,
        mapbox_style=MAPBOX_STYLE,
        size_max=35,
        title="Producción Acumulada por Pad",
    )
    fig_bubble_map.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title="Fluido",
    )
    st.plotly_chart(fig_bubble_map, use_container_width=True)

    # ── Ranking bar charts ────────────────────────────────────────────────────
    st.markdown(f"#### Ranking Top {top_n} Pads — Producción Acumulada")

    col_oil, col_gas = st.columns(2)

    with col_oil:
        st.markdown("**⛽ Petróleo Acumulado (m³)**")
        top_oil = pad_prod_f.nlargest(top_n, "total_oil_m3")
        fig_r_oil = px.bar(
            top_oil.sort_values("total_oil_m3"),
            x="total_oil_m3", y="pad_name", orientation="h",
            color="empresa", text="total_oil_m3",
            labels={"total_oil_m3": "m³", "pad_name": "Pad", "empresa": "Empresa"},
            height=420,
        )
        fig_r_oil.update_traces(texttemplate="%{text:,.0f}", textposition="inside")
        fig_r_oil.update_layout(template="plotly_white", yaxis_title=None)
        st.plotly_chart(fig_r_oil, use_container_width=True)

    with col_gas:
        st.markdown("**🔥 Gas Acumulado (km³)**")
        top_gas = pad_prod_f.nlargest(top_n, "total_gas_km3")
        fig_r_gas = px.bar(
            top_gas.sort_values("total_gas_km3"),
            x="total_gas_km3", y="pad_name", orientation="h",
            color="empresa", text="total_gas_km3",
            labels={"total_gas_km3": "km³", "pad_name": "Pad", "empresa": "Empresa"},
            height=420,
        )
        fig_r_gas.update_traces(texttemplate="%{text:,.0f}", textposition="inside")
        fig_r_gas.update_layout(template="plotly_white", yaxis_title=None)
        st.plotly_chart(fig_r_gas, use_container_width=True)

    # ── Scatter: wells vs production ──────────────────────────────────────────
    st.markdown("#### Eficiencia: Pozos vs Producción Acumulada")
    st.caption(
        "Pads en la esquina superior-izquierda logran alta producción con pocos pozos "
        "— candidatos a benchmark de completación."
    )

    metric_y = st.radio(
        "Eje Y:", ["total_oil_m3", "total_gas_km3"],
        format_func=lambda c: "Petróleo (m³)" if c == "total_oil_m3" else "Gas (km³)",
        horizontal=True, key="eff_y",
    )

    fig_eff = px.scatter(
        pad_prod_f,
        x="n_wells", y=metric_y,
        color="empresa", size="n_wells", size_max=22,
        hover_name="pad_name",
        hover_data={"empresa": True, "area": True, "fluid": True},
        labels={
            "n_wells":       "N° Pozos en el Pad",
            "total_oil_m3":  "Petróleo Acumulado (m³)",
            "total_gas_km3": "Gas Acumulado (km³)",
            "empresa":       "Empresa",
        },
        title="Pozos por Pad vs Producción Acumulada",
        template="plotly_white",
        height=480,
    )

    # Reference lines at medians
    med_x = pad_prod_f["n_wells"].median()
    med_y = pad_prod_f[metric_y].median()
    fig_eff.add_vline(x=med_x, line_dash="dash", line_color="grey",
                      annotation_text=f"P50 pozos ({med_x:.0f})",
                      annotation_position="top right",
                      annotation_font=dict(size=9))
    fig_eff.add_hline(y=med_y, line_dash="dash", line_color="grey",
                      annotation_text="P50 producción",
                      annotation_position="top left",
                      annotation_font=dict(size=9))
    st.plotly_chart(fig_eff, use_container_width=True)

    # ── Summary table ──────────────────────────────────────────────────────────
    st.markdown("#### Tabla Resumen por Pad")

    display_df = (
        pad_prod_f
        .sort_values("total_oil_m3", ascending=False)
        .rename(columns={
            "pad_name":      "Pad",
            "n_wells":       "Pozos",
            "total_oil_m3":  "Petróleo (m³)",
            "total_gas_km3": "Gas (km³)",
            "total_water_m3":"Agua (m³)",
            "peak_oil_rate": "Q pico petróleo (m³/d)",
            "peak_gas_rate": "Q pico gas (km³/d)",
            "empresa":       "Empresa dominante",
            "area":          "Área",
            "fluid":         "Fluido",
        })
        .reset_index(drop=True)
    )
    for col in ["Petróleo (m³)", "Agua (m³)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map("{:,.0f}".format)
    for col in ["Gas (km³)", "Q pico petróleo (m³/d)", "Q pico gas (km³/d)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map("{:,.1f}".format)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORTAR
# ══════════════════════════════════════════════════════════════════════════════

with tab_export:

    st.markdown("### ⬇️ Descargar resultados")

    if has_pads:
        # Well → pad mapping
        st.markdown("#### Listado Pozo → Pad")
        pozo_pad_df = df_pads[["sigla", "pad_name", "empresaNEW", "tipopozoNEW", "areayacimiento"]].copy()
        pozo_pad_df.columns = ["Sigla", "Pad", "Empresa", "Tipo de Pozo", "Área"]

        st.dataframe(pozo_pad_df, use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇️ Descargar pozo_pad.csv",
            data=pozo_pad_df.to_csv(index=False).encode("utf-8"),
            file_name="pozo_pad.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # Pad production summary
        st.markdown("#### Producción Acumulada por Pad")
        pad_exp = build_pad_production(
            data_sorted[data_sorted["tef"] > 0], df_pads
        ).rename(columns={
            "pad_name":      "Pad",
            "n_wells":       "Pozos",
            "total_oil_m3":  "Petróleo_m3",
            "total_gas_km3": "Gas_km3",
            "total_water_m3":"Agua_m3",
            "peak_oil_rate": "Qo_pico_m3d",
            "peak_gas_rate": "Qg_pico_km3d",
            "empresa":       "Empresa_dominante",
            "area":          "Area",
            "fluid":         "Fluido_dominante",
            "lat":           "Lat",
            "lon":           "Lon",
        })

        st.download_button(
            label="⬇️ Descargar produccion_por_pad.csv",
            data=pad_exp.to_csv(index=False).encode("utf-8"),
            file_name="produccion_por_pad.csv",
            mime="text/csv",
        )
    else:
        st.info("No hay pads calculados para exportar. Verificá que el dataset tenga coordenadas.")