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

# The maps below use Plotly's SVG geo renderer rather than MapLibre.  This is
# intentional: MapLibre maps need WebGL and external raster tiles, either of
# which can result in a blank chart inside some Streamlit/browser setups.

FLUID_COLORS = {
    "Petrolífero": "#2ecc71",
    "Gasífero":    "#e74c3c",
    "Otro":        "#95a5a6",
}


def normalise_wgs84_coordinates(
    df: pd.DataFrame, lon_col: str, lat_col: str
) -> pd.DataFrame:
    """Return only valid WGS-84 coordinates, converting strings when needed.

    The production files occasionally contain coordinates as text.  This also
    detects the common X/Y inversion (latitude stored in X and longitude in Y).
    It deliberately does *not* try to convert projected coordinates in metres:
    those need their EPSG code before they can be mapped correctly.
    """
    out = df.copy()
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")

    normal = (
        out[lon_col].between(-75, -55)
        & out[lat_col].between(-42, -30)
    )
    swapped = (
        out[lon_col].between(-42, -30)
        & out[lat_col].between(-75, -55)
    )
    if swapped.sum() > normal.sum():
        out[[lon_col, lat_col]] = out[[lat_col, lon_col]].to_numpy()
        normal = (
            out[lon_col].between(-75, -55)
            & out[lat_col].between(-42, -30)
        )

    return out.loc[normal].copy()


def map_center(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> dict:
    """Center a MapLibre map over the currently displayed records."""
    return {"lat": float(df[lat_col].mean()), "lon": float(df[lon_col].mean())}


def configure_geo_map(fig: go.Figure) -> go.Figure:
    """Style an SVG geographic map that works without MapLibre tiles/WebGL."""
    fig.update_geos(
        fitbounds="locations",
        projection_type="mercator",
        showland=True,
        landcolor="#edf2f7",
        showocean=True,
        oceancolor="#dbeafe",
        showcountries=True,
        countrycolor="#94a3b8",
        showcoastlines=True,
        coastlinecolor="#64748b",
        lataxis_showgrid=True,
        lonaxis_showgrid=True,
    )
    return fig


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
        # A well can have no location in its earliest production record but
        # have one in a later record; discard blank locations *before*
        # choosing the representative row for that well.
        .dropna(subset=["coordenadax", "coordenaday"])
        .sort_values("anio", ascending=False)
        .drop_duplicates("sigla")
        .rename(columns={"coordenadax": "x", "coordenaday": "y"})
        .dropna(subset=["x", "y"])
        .reset_index(drop=True)
    )

    # Keep only usable geographic coordinates (and repair inverted X/Y files).
    wells = normalise_wgs84_coordinates(wells, "x", "y").reset_index(drop=True)

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

def _safe_mode(s: pd.Series, default: str = "Sin dato"):
    """
    Returns the most frequent non-null value in s, or `default` if s has
    no non-null values at all (s.mode() drops NaNs, so it can come back
    empty even when s itself isn't — this guards against that case).
    """
    m = s.mode()
    return m.iloc[0] if not m.empty else default


def build_pad_production(
    df_prod: pd.DataFrame,
    df_pads: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join pad assignments to the monthly production table and aggregate
    total and peak production per pad.

    Caudal pico promedio por pad:
        - Para cada pozo se calcula su caudal pico individual (máximo mensual).
        - El caudal pico del pad = suma de caudales pico de sus pozos / n_wells.
    Esto refleja la productividad media de un pozo representativo del pad.
    """
    prod = df_prod.merge(df_pads[["sigla", "pad_name", "pad_id"]], on="sigla", how="inner")

    # Per-well peak rates, then average across wells in the same pad
    well_peaks = (
        prod.groupby(["pad_name", "sigla"])
        .agg(
            well_peak_oil  =("oil_rate",  "max"),
            well_peak_gas  =("gas_rate",  "max"),
        )
        .reset_index()
    )
    pad_avg_peaks = (
        well_peaks.groupby("pad_name")
        .agg(
            avg_peak_oil_rate =("well_peak_oil", "mean"),
            avg_peak_gas_rate =("well_peak_gas", "mean"),
        )
        .reset_index()
    )

    agg = (
        prod.groupby("pad_name")
        .agg(
            n_wells         =("sigla",          "nunique"),
            total_oil_m3    =("prod_pet",       "sum"),
            total_gas_km3   =("prod_gas",       "sum"),
            total_water_m3  =("prod_agua",      "sum"),
            empresa         =("empresaNEW",     _safe_mode),
            area            =("areayacimiento", _safe_mode),
            fluid           =("tipopozoNEW",    _safe_mode),
        )
        .reset_index()
    )

    agg = agg.merge(pad_avg_peaks, on="pad_name", how="left")

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
        # Do this before drop_duplicates: data_sorted is chronological and
        # the first record of a well can legitimately lack coordinates.
        .dropna(subset=coord_cols)
        .drop_duplicates("sigla")
        .rename(columns={"coordenadax": "lon", "coordenaday": "lat"})
    )
    wells_map = normalise_wgs84_coordinates(wells_map, "lon", "lat")

    if has_pads:
        wells_map = wells_map.merge(
            df_pads[["sigla", "pad_name"]], on="sigla", how="left"
        )

    if wells_map.empty:
        st.error(
            "No hay coordenadas geográficas válidas para mostrar. Se esperan "
            "longitudes entre -75 y -55 y latitudes entre -42 y -30. "
            "Si los valores son números grandes (coordenadas POSGAR en metros), "
            "primero hay que convertirlos a WGS-84 (EPSG:4326)."
        )
        st.stop()

    # ── Filtro por área (dentro del tab) ─────────────────────────────────────
    all_areas_map = sorted(wells_map["areayacimiento"].dropna().unique())
    sel_areas_map = st.multiselect(
        "Filtrar por área de yacimiento:",
        all_areas_map,
        default=[],
        key="map_area_filter",
        help="Seleccioná una o más áreas para enfocar el mapa. Sin selección se muestran todas.",
    )
    if sel_areas_map:
        wells_map = wells_map[wells_map["areayacimiento"].isin(sel_areas_map)]
        if wells_map.empty:
            st.warning("No hay pozos con coordenadas válidas para las áreas seleccionadas.")
            st.stop()

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

    fig_map = px.scatter_geo(
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
        scope="south america",
        height=620,
        title="Mapa de Pozos — Vaca Muerta",
    )
    fig_map.update_traces(marker=dict(size=6, opacity=0.80))
    configure_geo_map(fig_map)
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

    fig_pads = px.scatter_geo(
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
        scope="south america",
        height=580,
        title="Agrupación de Pozos por Pad (buffer 30 m)",
    )
    fig_pads.update_traces(marker=dict(size=7, opacity=0.85))
    configure_geo_map(fig_pads)
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

    # ── Filtros de Tab 3 ──────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 1])

    with fc1:
        fluid_sel = st.radio(
            "Tipo de pozo dominante:", ["Petrolífero", "Gasífero", "Todos"],
            horizontal=True,
        )
    with fc2:
        all_areas_prod = sorted(pad_prod["area"].dropna().unique())
        sel_areas_prod = st.multiselect(
            "Filtrar por área:",
            all_areas_prod,
            default=[],
            key="prod_area_filter",
            help="Sin selección se muestran todas las áreas.",
        )
    with fc3:
        top_n = st.slider("Top N pads:", min_value=5, max_value=30, value=15)

    # Apply filters
    pad_prod_f = pad_prod.copy()
    if fluid_sel != "Todos":
        pad_prod_f = pad_prod_f[pad_prod_f["fluid"] == fluid_sel]
    if sel_areas_prod:
        pad_prod_f = pad_prod_f[pad_prod_f["area"].isin(sel_areas_prod)]

    if pad_prod_f.empty:
        st.warning("No hay pads con datos para los filtros seleccionados.")
        st.stop()

    # ── Mapa 1: burbujas coloreadas por ÁREA ─────────────────────────────────
    st.markdown("#### Mapa de Burbujas — Producción Acumulada por Pad (color por Área)")

    metric_bubble = st.selectbox(
        "Métrica de tamaño:",
        ["total_oil_m3", "total_gas_km3", "avg_peak_oil_rate", "avg_peak_gas_rate", "n_wells"],
        format_func=lambda c: {
            "total_oil_m3":       "Petróleo Acumulado (m³)",
            "total_gas_km3":      "Gas Acumulado (km³)",
            "avg_peak_oil_rate":  "Caudal Pico Prom. Petróleo (m³/d)",
            "avg_peak_gas_rate":  "Caudal Pico Prom. Gas (km³/d)",
            "n_wells":            "N° Pozos",
        }[c],
    )

    bubble_map_df = pad_prod_f.dropna(subset=["lat", "lon"])
    if bubble_map_df.empty:
        st.warning("No hay pads con coordenadas válidas para el mapa de producción.")
        st.stop()

    fig_bubble_map = px.scatter_geo(
        bubble_map_df,
        lat="lat", lon="lon",
        size=metric_bubble,
        color="area",                          # ← color por área
        hover_name="pad_name",
        hover_data={
            "n_wells":            True,
            "empresa":            True,
            "area":               True,
            "total_oil_m3":       ":,.0f",
            "total_gas_km3":      ":,.1f",
            "avg_peak_oil_rate":  ":,.1f",
            "avg_peak_gas_rate":  ":,.1f",
            "lat": False, "lon": False,
        },
        scope="south america", height=580,
        size_max=35,
        title="Producción Acumulada por Pad — color por Área",
    )
    configure_geo_map(fig_bubble_map)
    fig_bubble_map.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title="Área",
    )
    st.plotly_chart(fig_bubble_map, use_container_width=True)

    # ── Mapa 2: cut-off — pads destacados vs grises ───────────────────────────
    st.markdown("#### Mapa de Cut-off — Pads que superan un umbral")
    st.caption(
        "Los pads en color superan el umbral ingresado en la métrica seleccionada. "
        "Los demás aparecen en gris claro."
    )

    cutoff_col_options = {
        "Petróleo Acumulado (m³)":          "total_oil_m3",
        "Gas Acumulado (km³)":              "total_gas_km3",
        "Qo pico prom. (m³/d/pozo)":        "avg_peak_oil_rate",
        "Qg pico prom. (km³/d/pozo)":       "avg_peak_gas_rate",
    }
    co1, co2 = st.columns([2, 1])
    with co1:
        cutoff_metric_lbl = st.selectbox(
            "Métrica para el cut-off:",
            list(cutoff_col_options.keys()),
            key="cutoff_metric",
        )
    cutoff_metric_col = cutoff_col_options[cutoff_metric_lbl]

    valid_vals = bubble_map_df[cutoff_metric_col].dropna()
    default_cutoff = float(valid_vals.quantile(0.75)) if not valid_vals.empty else 0.0

    with co2:
        cutoff_value = st.number_input(
            f"Umbral mínimo ({cutoff_metric_lbl}):",
            min_value=0.0,
            value=round(default_cutoff, 1),
            step=max(1.0, round(default_cutoff * 0.05, 1)),
            format="%.1f",
            key="cutoff_value",
        )

    # Separar pads que superan y no superan el umbral
    above_df = bubble_map_df[bubble_map_df[cutoff_metric_col] >= cutoff_value]
    below_df = bubble_map_df[bubble_map_df[cutoff_metric_col] <  cutoff_value]

    fig_cutoff = go.Figure()

    # Pads por debajo: marcadores grises sin leyenda por pad
    if not below_df.empty:
        fig_cutoff.add_trace(go.Scattergeo(
            lat=below_df["lat"],
            lon=below_df["lon"],
            mode="markers",
            marker=dict(size=6, color="lightgrey", opacity=0.55),
            name="Bajo umbral",
            text=below_df["pad_name"],
            customdata=below_df[["n_wells", "empresa", "area",
                                  cutoff_metric_col]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "N° pozos: %{customdata[0]}<br>"
                "Empresa: %{customdata[1]}<br>"
                "Área: %{customdata[2]}<br>"
                f"{cutoff_metric_lbl}: %{{customdata[3]:,.1f}}<extra>Bajo umbral</extra>"
            ),
        ))

    # Pads por encima: un trace por área para tener leyenda de colores por área
    if not above_df.empty:
        area_palette = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
        for i, area_val in enumerate(sorted(above_df["area"].unique())):
            sub = above_df[above_df["area"] == area_val]
            fig_cutoff.add_trace(go.Scattergeo(
                lat=sub["lat"],
                lon=sub["lon"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=area_palette[i % len(area_palette)],
                    opacity=0.9,
                    line=dict(width=0.5, color="white"),
                ),
                name=area_val,
                text=sub["pad_name"],
                customdata=sub[["n_wells", "empresa",
                                 cutoff_metric_col]].values,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "N° pozos: %{customdata[0]}<br>"
                    "Empresa: %{customdata[1]}<br>"
                    f"{cutoff_metric_lbl}: %{{customdata[2]:,.1f}}<extra>{area_val}</extra>"
                ),
            ))

    configure_geo_map(fig_cutoff)
    fig_cutoff.update_layout(
        title=f"Pads con {cutoff_metric_lbl} ≥ {cutoff_value:,.1f} — {len(above_df)} pads destacados",
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title="Área (sobre umbral)",
        height=560,
    )
    st.plotly_chart(fig_cutoff, use_container_width=True)
    st.caption(
        f"**{len(above_df)}** pads superan el umbral · "
        f"**{len(below_df)}** pads en gris (bajo umbral)"
    )

    # ── Ranking: Producción Acumulada ─────────────────────────────────────────
    st.markdown(f"#### Ranking Top {top_n} Pads — Producción Acumulada")

    # Petróleo
    st.markdown("**⛽ Petróleo Acumulado (m³)**")
    top_oil = pad_prod_f.nlargest(top_n, "total_oil_m3")
    fig_r_oil = px.bar(
        top_oil.sort_values("total_oil_m3"),
        x="total_oil_m3", y="pad_name", orientation="h",
        color="empresa", text="total_oil_m3",
        labels={"total_oil_m3": "m³", "pad_name": "Pad", "empresa": "Empresa"},
        height=max(380, top_n * 28),
        custom_data=["n_wells"],
    )
    fig_r_oil.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Petróleo: %{x:,.0f} m³<br>N° pozos: %{customdata[0]}<extra></extra>",
    )
    fig_r_oil.update_layout(template="plotly_white", yaxis_title=None)
    st.plotly_chart(fig_r_oil, use_container_width=True)

    # Gas
    st.markdown("**🔥 Gas Acumulado (km³)**")
    top_gas = pad_prod_f.nlargest(top_n, "total_gas_km3")
    fig_r_gas = px.bar(
        top_gas.sort_values("total_gas_km3"),
        x="total_gas_km3", y="pad_name", orientation="h",
        color="empresa", text="total_gas_km3",
        labels={"total_gas_km3": "km³", "pad_name": "Pad", "empresa": "Empresa"},
        height=max(380, top_n * 28),
        custom_data=["n_wells"],
    )
    fig_r_gas.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Gas: %{x:,.0f} km³<br>N° pozos: %{customdata[0]}<extra></extra>",
    )
    fig_r_gas.update_layout(template="plotly_white", yaxis_title=None)
    st.plotly_chart(fig_r_gas, use_container_width=True)

    # ── Ranking: Caudal Pico Promedio por Pad ─────────────────────────────────
    st.markdown(f"#### 💧 Ranking Top {top_n} Pads — Caudal Pico Promedio")
    st.caption(
        "Caudal pico promedio = suma del caudal pico individual de cada pozo del pad "
        "dividido la cantidad de pozos. Refleja la productividad media de un pozo representativo."
    )

    # Qo
    st.markdown("**⛽ Qo pico promedio (m³/d/pozo)**")
    top_qo = pad_prod_f.nlargest(top_n, "avg_peak_oil_rate")
    fig_qo = px.bar(
        top_qo.sort_values("avg_peak_oil_rate"),
        x="avg_peak_oil_rate", y="pad_name", orientation="h",
        color="area", text="avg_peak_oil_rate",
        labels={"avg_peak_oil_rate": "m³/d", "pad_name": "Pad", "area": "Área"},
        height=max(380, top_n * 28),
        title="Ranking por Qo pico prom.",
        custom_data=["n_wells"],
    )
    fig_qo.update_traces(
        texttemplate="%{text:,.1f}",
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Qo pico prom.: %{x:,.1f} m³/d<br>N° pozos: %{customdata[0]}<extra></extra>",
    )
    fig_qo.update_layout(template="plotly_white", yaxis_title=None)
    st.plotly_chart(fig_qo, use_container_width=True)

    # Qg
    st.markdown("**🔥 Qg pico promedio (km³/d/pozo)**")
    top_qg = pad_prod_f.nlargest(top_n, "avg_peak_gas_rate")
    fig_qg = px.bar(
        top_qg.sort_values("avg_peak_gas_rate"),
        x="avg_peak_gas_rate", y="pad_name", orientation="h",
        color="area", text="avg_peak_gas_rate",
        labels={"avg_peak_gas_rate": "km³/d", "pad_name": "Pad", "area": "Área"},
        height=max(380, top_n * 28),
        title="Ranking por Qg pico prom.",
        custom_data=["n_wells"],
    )
    fig_qg.update_traces(
        texttemplate="%{text:,.1f}",
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Qg pico prom.: %{x:,.1f} km³/d<br>N° pozos: %{customdata[0]}<extra></extra>",
    )
    fig_qg.update_layout(template="plotly_white", yaxis_title=None)
    st.plotly_chart(fig_qg, use_container_width=True)

    # ── Scatter: caudal pico vs producción acumulada ──────────────────────────
    st.markdown("#### Caudal Pico vs Producción Acumulada")
    st.caption(
        "Compara la tasa inicial (caudal pico prom.) con la producción acumulada total. "
        "Pads en la esquina superior-derecha son los de mayor calidad de completación."
    )

    sc1, sc2 = st.columns([1, 2])
    with sc1:
        peak_fluid = st.radio(
            "Fluido:",
            ["Petróleo", "Gas"],
            horizontal=True,
            key="peak_scatter_fluid",
        )
    with sc2:
        all_areas_scatter = sorted(pad_prod_f["area"].dropna().unique())
        sel_areas_scatter = st.multiselect(
            "Filtrar áreas en el scatter:",
            all_areas_scatter,
            default=[],
            key="scatter_area_filter",
            help="Sin selección se muestran todas.",
        )

    if peak_fluid == "Petróleo":
        x_col, y_col = "avg_peak_oil_rate", "total_oil_m3"
        x_lbl, y_lbl = "Qo pico prom. (m³/d)", "Petróleo Acumulado (m³)"
    else:
        x_col, y_col = "avg_peak_gas_rate", "total_gas_km3"
        x_lbl, y_lbl = "Qg pico prom. (km³/d)", "Gas Acumulado (km³)"

    scatter_df = pad_prod_f.dropna(subset=[x_col, y_col])
    if sel_areas_scatter:
        scatter_df = scatter_df[scatter_df["area"].isin(sel_areas_scatter)]

    if not scatter_df.empty:
        fig_pk_scatter = px.scatter(
            scatter_df,
            x=x_col, y=y_col,
            color="area",
            size="n_wells", size_max=22,
            hover_name="pad_name",
            hover_data={
                "empresa": True,
                "area":    True,
                "n_wells": True,
                x_col:     ":,.1f",
                y_col:     ":,.0f",
            },
            labels={x_col: x_lbl, y_col: y_lbl, "area": "Área"},
            title=f"{x_lbl} vs {y_lbl}",
            template="plotly_white",
            height=500,
        )
        med_x_pk = scatter_df[x_col].median()
        med_y_pk = scatter_df[y_col].median()
        fig_pk_scatter.add_vline(
            x=med_x_pk, line_dash="dash", line_color="grey",
            annotation_text=f"P50 caudal ({med_x_pk:,.1f})",
            annotation_position="top right",
            annotation_font=dict(size=9),
        )
        fig_pk_scatter.add_hline(
            y=med_y_pk, line_dash="dash", line_color="grey",
            annotation_text="P50 acumulado",
            annotation_position="top left",
            annotation_font=dict(size=9),
        )
        st.plotly_chart(fig_pk_scatter, use_container_width=True)
    else:
        st.info("No hay datos con los filtros seleccionados.")

    # ── Summary table ──────────────────────────────────────────────────────────
    st.markdown("#### Tabla Resumen por Pad")

    display_df = (
        pad_prod_f
        .sort_values("avg_peak_oil_rate", ascending=False)
        .rename(columns={
            "pad_name":           "Pad",
            "n_wells":            "Pozos",
            "total_oil_m3":       "Petróleo (m³)",
            "total_gas_km3":      "Gas (km³)",
            "total_water_m3":     "Agua (m³)",
            "avg_peak_oil_rate":  "Qo pico prom. (m³/d)",
            "avg_peak_gas_rate":  "Qg pico prom. (km³/d)",
            "empresa":            "Empresa dominante",
            "area":               "Área",
            "fluid":              "Fluido",
        })
        .reset_index(drop=True)
    )
    for col in ["Petróleo (m³)", "Agua (m³)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map("{:,.0f}".format)
    for col in ["Gas (km³)", "Qo pico prom. (m³/d)", "Qg pico prom. (km³/d)"]:
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
            "pad_name":           "Pad",
            "n_wells":            "Pozos",
            "total_oil_m3":       "Petroleo_m3",
            "total_gas_km3":      "Gas_km3",
            "total_water_m3":     "Agua_m3",
            "avg_peak_oil_rate":  "Qo_pico_prom_m3d",
            "avg_peak_gas_rate":  "Qg_pico_prom_km3d",
            "empresa":            "Empresa_dominante",
            "area":               "Area",
            "fluid":              "Fluido_dominante",
            "lat":                "Lat",
            "lon":                "Lon",
        })

        st.download_button(
            label="⬇️ Descargar produccion_por_pad.csv",
            data=pad_exp.to_csv(index=False).encode("utf-8"),
            file_name="produccion_por_pad.csv",
            mime="text/csv",
        )
    else:
        st.info("No hay pads calculados para exportar. Verificá que el dataset tenga coordenadas.")
