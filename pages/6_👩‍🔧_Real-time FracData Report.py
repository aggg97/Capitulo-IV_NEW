import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from utils import (
    COMPANY_REPLACEMENTS,
    get_fluid_classification,
    load_frac_data,
    create_summary_dataframe,
)


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
    st.warning("No se han cargado los datos. Por favor, vuelve a la Pagina Principal.")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.header(":blue[Reporte Extensivo de Completacion y Produccion en Vaca Muerta]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))
st.sidebar.image(Image.open("McCain.png"))
st.sidebar.caption(
    "Los pozos clasificados como Otro tipo son reclasificados como "
    "Gasíferos o Petrolíferos usando el criterio de GOR segun McCain."
)


# ── Data preparation ──────────────────────────────────────────────────────────

data_filtered = data_sorted[data_sorted["tef"] > 0]

df_frac    = load_frac_data()
summary_df = create_summary_dataframe(data_filtered)

df_merged = (
    pd.merge(df_frac, summary_df, on="sigla", how="outer")
    .drop_duplicates()
)

# Vaca Muerta Shale only
df_vmut = df_merged[
    (df_merged["formprod"]         == "VMUT") &
    (df_merged["sub_tipo_recurso"] == "SHALE")
].copy()

# Derived completion metrics
df_vmut["fracspacing"]        = df_vmut["longitud_rama_horizontal_m"] / df_vmut["cantidad_fracturas"]
df_vmut["prop_x_etapa"]       = df_vmut["arena_total_tn"] / df_vmut["cantidad_fracturas"]
df_vmut["proppant_intensity"]  = df_vmut["arena_total_tn"] / df_vmut["longitud_rama_horizontal_m"]
df_vmut["AS_x_vol"]           = df_vmut["arena_total_tn"] / (df_vmut["agua_inyectada_m3"] / 1000)
df_vmut["Qo_peak_x_etapa"]    = df_vmut["Qo_peak"] / df_vmut["cantidad_fracturas"]
df_vmut["Qg_peak_x_etapa"]    = df_vmut["Qg_peak"] / df_vmut["cantidad_fracturas"]
df_vmut = df_vmut.replace([np.inf, -np.inf], np.nan)

# One row per well for completion stats
df_vmut_dedup = df_vmut[df_vmut["longitud_rama_horizontal_m"] > 0].drop_duplicates(subset="sigla")

# Constantes de layout 
LEGEND_BOTTOM = dict(orientation="h", yanchor="top",   y=-0.20, xanchor="center", x=0.5)
FLUID_COLORS = {"Petrolífero": "green", "Gasífero": "red"}
X_AXIS_LABEL = "Campaña de Perforación"


# ── Chart helper ──────────────────────────────────────────────────────────────

def build_evolution_chart(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    group_col: str = "start_year",
    split_by_fluid: bool = False,
    invert_percentiles: bool = False,
) -> go.Figure:
    """
    P10 / P50 / P90 evolution line chart grouped by start_year.
    Optionally splits by tipopozoNEW (Petrolífero / Gasífero).

    invert_percentiles=True  → convención producción de hidrocarburos:
        P10 = optimista (valor alto, cuantil 0.90)
        P90 = conservador (valor bajo, cuantil 0.10)
    invert_percentiles=False → percentiles estadísticos directos (completación).
    """
    fig = go.Figure()

    if split_by_fluid:
        splits = [
            ("Petrolífero", df[df["tipopozoNEW"] == "Petrolífero"], "green"),
            ("Gasífero",    df[df["tipopozoNEW"] == "Gasífero"],    "red"),
        ]
    else:
        splits = [("Todos", df, "#1f77b4")]

    # Asignación de cuantiles según convención
    if invert_percentiles:
        q_p10, q_p50, q_p90 = 0.90, 0.50, 0.10   # P10 optimista arriba
    else:
        q_p10, q_p50, q_p90 = 0.10, 0.50, 0.90   # P10 conservador abajo

    # Colores de banda con transparencia
    BAND_COLORS = {
        "green":   "rgba(0,128,0,0.15)",
        "red":     "rgba(255,0,0,0.15)",
        "#1f77b4": "rgba(31,119,180,0.15)",
    }

    for label, sub, color in splits:
        stats = (
            sub.groupby(group_col)[metric_col]
            .agg(
                p10=lambda x: x.quantile(q_p10),
                p50=lambda x: x.quantile(q_p50),
                p90=lambda x: x.quantile(q_p90),
            )
            .reset_index()
            .dropna(subset=["p50"])
        )
        if stats.empty:
            continue

        suffix     = f" — {label}" if split_by_fluid else ""
        band_color = BAND_COLORS.get(color, "rgba(128,128,128,0.15)")

        # Línea superior de la banda (P10 optimista = valor más alto)
        fig.add_trace(go.Scatter(
            x=stats[group_col],
            y=stats["p10"],
            mode="lines",
            name=f"P10{suffix}",
            line=dict(color=color, width=1, dash="dot"),
            legendgroup=label,
            showlegend=True,
            hovertemplate=f"Campaña: %{{x}}<br>P10: %{{y:.0f}}<extra>{label}</extra>",
        ))
        # Línea inferior de la banda (P90 conservador = valor más bajo)
        # fill="tonexty" rellena desde esta traza hasta la anterior (P10)
        fig.add_trace(go.Scatter(
            x=stats[group_col],
            y=stats["p90"],
            mode="lines",
            name=f"P90{suffix}",
            line=dict(color=color, width=1, dash="dot"),
            fill="tonexty",
            fillcolor=band_color,
            legendgroup=label,
            showlegend=True,
            hovertemplate=f"Campaña: %{{x}}<br>P90: %{{y:.0f}}<extra>{label}</extra>",
        ))
        # P50 línea sólida encima
        fig.add_trace(go.Scatter(
            x=stats[group_col],
            y=stats["p50"],
            mode="lines+markers",
            name=f"P50{suffix}",
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
            legendgroup=label,
            showlegend=True,
            hovertemplate=f"Campaña: %{{x}}<br>P50: %{{y:.0f}}<extra>{label}</extra>",
        ))

        # Anotaciones solo en P50
        for _, row in stats.iterrows():
            fig.add_annotation(
                x=row[group_col], y=row["p50"],
                text=f"{row['p50']:.0f}",
                showarrow=False, yshift=12,
                font=dict(color=color, size=9),
            )

    fig.update_layout(
        title=title,
        xaxis_title=X_AXIS_LABEL,
        yaxis_title=y_label,
        template="plotly_white",
        legend=LEGEND_BOTTOM,
    )
    return fig


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "Indicadores de Actividad",
    "Estrategia de Completacion",
    "Productividad",
])


# ── Tab 1: Activity indicators ────────────────────────────────────────────────

with tab1:

    wells_by_year = (
        df_vmut.groupby(["start_year", "tipopozoNEW"])["sigla"]
        .nunique()
        .reset_index(name="count")
        .pivot_table(index="start_year", columns="tipopozoNEW", values="count", fill_value=0)
        .drop(columns=["Inyeccion de Agua", "Inyeccion de Gas"], errors="ignore")
    )

    fig_wells = go.Figure()
    for fluid, color in [("Petrolífero", "green"), ("Gasífero", "red")]:
        if fluid not in wells_by_year.columns:
            continue
        fig_wells.add_trace(go.Scatter(
            x=wells_by_year.index, y=wells_by_year[fluid],
            mode="lines+markers", name=fluid,
            line=dict(color=color), marker=dict(size=8),
        ))
        for x, y in zip(wells_by_year.index, wells_by_year[fluid]):
            fig_wells.add_annotation(
                x=x, y=y, text=str(int(y)),
                showarrow=False, yshift=15, font=dict(size=10, color=color),
            )

    fig_wells.update_layout(
        title="Pozos enganchados por campaña (Fm. Vaca Muerta)",
        xaxis_title=X_AXIS_LABEL,
        yaxis_title="Cantidad de Pozos",
        template="plotly_white",
        legend=LEGEND_BOTTOM,
    )
    st.plotly_chart(fig_wells, use_container_width=True)

    st.divider()

    # Arena evolution
    df_con_frac = df_vmut[df_vmut["id_base_fractura_adjiv"].notna()].copy()
    arena_by_year = (
        df_con_frac.groupby("start_year")
        .agg(
            arena_total    =("arena_total_tn",              "sum"),
            arena_importada=("arena_bombeada_importada_tn", "sum"),
        )
        .reset_index()
    )
    arena_by_year["perc_importada"] = (
        arena_by_year["arena_importada"] / arena_by_year["arena_total"] * 100
    ).round(1)
    arena_by_year["start_year"] = arena_by_year["start_year"].astype(int).astype(str)

    fig_arena = go.Figure()
    fig_arena.add_trace(go.Scatter(
        x=arena_by_year["start_year"], y=arena_by_year["arena_total"],
        mode="lines+markers", name="Arena Total (tn)", line=dict(width=3),
    ))
    fig_arena.add_trace(go.Scatter(
        x=arena_by_year["start_year"], y=arena_by_year["perc_importada"],
        mode="lines+markers", name="% Arena Importada",
        line=dict(color="green", width=3), yaxis="y2",
    ))
    fig_arena.update_layout(
        title="Total Arena Bombeada vs % Arena Importada por Año",
        xaxis_title=X_AXIS_LABEL,
        yaxis_title="Arena Bombeada (tn)",
        yaxis2=dict(title="% Arena Importada", overlaying="y", side="right"),
        template="plotly_white",
        legend=LEGEND_BOTTOM,
    )
    st.write("### Evolucion de Arena Bombeada")
    st.plotly_chart(fig_arena, use_container_width=True)


# ── Tab 2: Completion strategy ────────────────────────────────────────────────

with tab2:

    split_completion = st.checkbox(
        "Separar por tipo de fluido (Petrolífero / Gasífero)",
        key="split_completion",
    )

    COMPLETION_CHARTS = {
        "Longitud de Rama Lateral": (
            df_vmut_dedup, "longitud_rama_horizontal_m",
            "Evolución de la Rama Lateral (Fm. Vaca Muerta)", "Longitud de Rama (m)",
        ),
        "Cantidad de Etapas": (
            df_vmut_dedup, "cantidad_fracturas",
            "Evolución de Cantidad de Etapas (Fm. Vaca Muerta)", "Cantidad de Etapas",
        ),
        "Arena Bombeada": (
            df_vmut_dedup[df_vmut_dedup["arena_total_tn"] > 0],
            "arena_total_tn",
            "Evolución de Arena Bombeada (Fm. Vaca Muerta)", "Arena Total (tn)",
        ),
        "Fracspacing": (
            df_vmut_dedup[df_vmut_dedup["fracspacing"] > 0],
            "fracspacing",
            "Evolución del Fracspacing (Fm. Vaca Muerta)", "Fracspacing (m)",
        ),
        "Agua Inyectada": (
            df_vmut[df_vmut["agua_inyectada_m3"].notna()],
            "agua_inyectada_m3",
            "Evolución del Agua Inyectada (Fm. Vaca Muerta)", "Agua Inyectada (m3)",
        ),
        "Propante por Etapa": (
            df_vmut[df_vmut["prop_x_etapa"] > 0],
            "prop_x_etapa",
            "Evolución de Propante por Etapa (Fm. Vaca Muerta)", "Prop x Etapa (tn/etapa)",
        ),
        "Concentracion AS por Vol. Inyectado": (
            df_vmut[df_vmut["AS_x_vol"] > 0],
            "AS_x_vol",
            "Evolución de la Concentración de Agente de Sostén (Fm. Vaca Muerta)",
            "Arena por Vol. Inyectado (tn/1000m3)",
        ),
    }

    selected_completion = st.multiselect(
        "Seleccionar indicadores de completacion a visualizar:",
        options=list(COMPLETION_CHARTS.keys()),
        default=list(COMPLETION_CHARTS.keys())[:2],
    )

    for chart_name in selected_completion:
        df_c, metric, title, y_label = COMPLETION_CHARTS[chart_name]
        st.plotly_chart(
            build_evolution_chart(
                df_c, metric, title, y_label,
                split_by_fluid=split_completion,
                invert_percentiles=True,
            ),
            use_container_width=True,
        )
        st.divider()


# ── Tab 3: Productivity ───────────────────────────────────────────────────────

with tab3:

    split_productivity = st.checkbox(
        "Separar por tipo de fluido (Petrolífero / Gasífero)",
        key="split_productivity",
    )

    PRODUCTIVITY_CHARTS = {
        "Qo Pico": (
            df_vmut[df_vmut["tipopozoNEW"] == "Petrolífero"],
            "Qo_peak",
            "Evolución de Caudal Pico de Petróleo (Fm. Vaca Muerta)",
            "Caudal de Petróleo (m3/d)",
        ),
        "Qg Pico": (
            df_vmut[df_vmut["tipopozoNEW"] == "Gasífero"],
            "Qg_peak",
            "Evolución de Caudal Pico de Gas (Fm. Vaca Muerta)",
            "Caudal de Gas (km3/d)",
        ),
        "Qo Pico x Etapa": (
            df_vmut[(df_vmut["tipopozoNEW"] == "Petrolífero") & (df_vmut["start_year"] > 2012)],
            "Qo_peak_x_etapa",
            "Evolución de Caudal Pico por Etapa — Petróleo (Fm. Vaca Muerta)",
            "Caudal de Petróleo (m3/d/etapa)",
        ),
        "Qg Pico x Etapa": (
            df_vmut[(df_vmut["tipopozoNEW"] == "Gasífero") & (df_vmut["start_year"] > 2012)],
            "Qg_peak_x_etapa",
            "Evolución de Caudal Pico por Etapa — Gas (Fm. Vaca Muerta)",
            "Caudal de Gas (km3/d/etapa)",
        ),
    }

    selected_productivity = st.multiselect(
        "Seleccionar indicadores de productividad a visualizar:",
        options=list(PRODUCTIVITY_CHARTS.keys()),
        default=list(PRODUCTIVITY_CHARTS.keys())[:2],
    )

    for chart_name in selected_productivity:
        df_p, metric, title, y_label = PRODUCTIVITY_CHARTS[chart_name]
        st.plotly_chart(
            build_evolution_chart(
                df_p, metric, title, y_label,
                split_by_fluid=split_productivity,
                invert_percentiles=True,
            ),
            use_container_width=True,
        )
        st.divider()

        """
SECCIÓN DE MÉTRICAS OPERATIVAS — TAB 1 (Indicadores de Actividad)
=================================================================
Pegar este bloque AL FINAL del bloque `with tab1:` en
pages/6_👩‍🔧_Real-time FracData Report.py

Requiere que el contexto ya tenga definido:
  df_vmut        — DataFrame filtrado a VMUT + SHALE
  df_con_frac    — df_vmut con id_base_fractura_adjiv not-null
  data_sorted    — DataFrame completo de producción mensual con columnas:
                     date, anio, mes, tef, prod_pet, prod_gas, prod_agua,
                     gas_rate, oil_rate, water_rate, empresaNEW, sigla,
                     tipopozoNEW, areayacimiento
  FLUID_COLORS, LEGEND_BOTTOM, X_AXIS_LABEL  — constantes ya definidas
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paletas ────────────────────────────────────────────────────────────────────
COMPANY_PALETTE = px.colors.qualitative.Set2
ROLL_COLOR_OIL  = "#27ae60"
ROLL_COLOR_GAS  = "#e74c3c"
ROLL_COLOR_3M   = "#3498db"
ROLL_COLOR_6M   = "#e67e22"
BROWNFIELD_COLOR = "#8e44ad"

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE COMPLETO — pegar dentro de `with tab1:`
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 Métricas Operativas Avanzadas", divider="blue")
st.caption(
    "Variaciones MoM / YoY, producción incremental, market share, "
    "GOR/WOR evolution y señales de madurez brownfield."
)


# ══════════════════════════════════════════════════════════════════════════════
# 0. BASE MENSUAL AGREGADA
# ══════════════════════════════════════════════════════════════════════════════

monthly = (
    data_sorted[data_sorted["tef"] > 0]
    .groupby("date")
    .agg(
        oil_rate  =("oil_rate",   "sum"),
        gas_rate  =("gas_rate",   "sum"),
        water_rate=("water_rate", "sum"),
        prod_pet  =("prod_pet",   "sum"),
        prod_gas  =("prod_gas",   "sum"),
        prod_agua =("prod_agua",  "sum"),
        tef       =("tef",        "sum"),
        wells_active=("sigla",    "nunique"),
    )
    .sort_index()
    .reset_index()
)

# GOR y WOR mensuales totales
monthly["GOR"] = (monthly["prod_gas"] / monthly["prod_pet"].replace(0, np.nan) * 1000)
monthly["WOR"] = (monthly["prod_agua"] / monthly["prod_pet"].replace(0, np.nan))
monthly["GOR"] = monthly["GOR"].replace([np.inf, -np.inf], np.nan)
monthly["WOR"] = monthly["WOR"].replace([np.inf, -np.inf], np.nan)

# Rolling 3m y 6m
for col in ["oil_rate", "gas_rate", "water_rate", "GOR", "WOR"]:
    monthly[f"{col}_3m"] = monthly[col].rolling(3, min_periods=1).mean()
    monthly[f"{col}_6m"] = monthly[col].rolling(6, min_periods=1).mean()

# MoM y YoY
for col in ["oil_rate", "gas_rate", "water_rate", "wells_active"]:
    monthly[f"{col}_mom"] = monthly[col].pct_change(1) * 100
    monthly[f"{col}_yoy"] = monthly[col].pct_change(12) * 100


# ══════════════════════════════════════════════════════════════════════════════
# 1. KPIs SEMÁFORO — MoM y YoY del ÚLTIMO MES
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 🚦 Variaciones MoM / YoY — Último período")

last = monthly.dropna(subset=["oil_rate"]).iloc[-1]

def _delta_str(val):
    if pd.isna(val):
        return "N/D", "off"
    sign = "+" if val >= 0 else ""
    color = "normal" if val >= 0 else "inverse"
    return f"{sign}{val:.1f}%", color

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🛢️ Qo Total (m³/d)", f"{last['oil_rate']:,.0f}",
              delta=f"MoM: {_delta_str(last['oil_rate_mom'])[0]}",
              delta_color=_delta_str(last["oil_rate_mom"])[1])
with col2:
    st.metric("🔥 Qg Total (km³/d)", f"{last['gas_rate']:,.0f}",
              delta=f"MoM: {_delta_str(last['gas_rate_mom'])[0]}",
              delta_color=_delta_str(last["gas_rate_mom"])[1])
with col3:
    st.metric("💧 Qw Total (m³/d)", f"{last['water_rate']:,.0f}",
              delta=f"MoM: {_delta_str(last['water_rate_mom'])[0]}",
              delta_color=_delta_str(last["water_rate_mom"])[1])
with col4:
    st.metric("🛞 Pozos Activos", f"{int(last['wells_active']):,}",
              delta=f"MoM: {_delta_str(last['wells_active_mom'])[0]}",
              delta_color=_delta_str(last["wells_active_mom"])[1])

# Fila YoY
monthly_yoy_row = monthly.dropna(subset=["oil_rate_yoy"]).iloc[-1] if monthly["oil_rate_yoy"].notna().any() else None

if monthly_yoy_row is not None:
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("🛢️ YoY Petróleo", "",
                  delta=f"YoY: {_delta_str(monthly_yoy_row['oil_rate_yoy'])[0]}",
                  delta_color=_delta_str(monthly_yoy_row["oil_rate_yoy"])[1])
    with col6:
        st.metric("🔥 YoY Gas", "",
                  delta=f"YoY: {_delta_str(monthly_yoy_row['gas_rate_yoy'])[0]}",
                  delta_color=_delta_str(monthly_yoy_row["gas_rate_yoy"])[1])
    with col7:
        st.metric("💧 YoY Agua", "",
                  delta=f"YoY: {_delta_str(monthly_yoy_row['water_rate_yoy'])[0]}",
                  delta_color=_delta_str(monthly_yoy_row["water_rate_yoy"])[1])
    with col8:
        st.metric("🛞 YoY Pozos Activos", "",
                  delta=f"YoY: {_delta_str(monthly_yoy_row['wells_active_yoy'])[0]}",
                  delta_color=_delta_str(monthly_yoy_row["wells_active_yoy"])[1])


# ══════════════════════════════════════════════════════════════════════════════
# 2. ROLLING AVERAGES — Petróleo y Gas
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 📈 Producción con Rolling Averages (3m / 6m)")

fluid_roll = st.radio(
    "Fluido",
    ["Petróleo", "Gas", "Agua"],
    horizontal=True,
    key="fluid_roll",
)

_map_roll = {
    "Petróleo": ("oil_rate", "oil_rate_3m", "oil_rate_6m",
                 "Caudal de Petróleo (m³/d)", ROLL_COLOR_OIL),
    "Gas":      ("gas_rate", "gas_rate_3m", "gas_rate_6m",
                 "Caudal de Gas (km³/d)", ROLL_COLOR_GAS),
    "Agua":     ("water_rate", "water_rate_3m", "water_rate_6m",
                 "Caudal de Agua (m³/d)", "#2980b9"),
}
raw_col, col3m, col6m, y_lbl, base_color = _map_roll[fluid_roll]

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(
    x=monthly["date"], y=monthly[raw_col],
    mode="lines", name="Mensual",
    line=dict(color=base_color, width=1, dash="dot"),
    opacity=0.45,
    hovertemplate="Fecha: %{x}<br>Valor: %{y:,.1f}",
))
fig_roll.add_trace(go.Scatter(
    x=monthly["date"], y=monthly[col3m],
    mode="lines", name="Media 3m",
    line=dict(color=ROLL_COLOR_3M, width=2.5),
    hovertemplate="Fecha: %{x}<br>Media 3m: %{y:,.1f}",
))
fig_roll.add_trace(go.Scatter(
    x=monthly["date"], y=monthly[col6m],
    mode="lines", name="Media 6m",
    line=dict(color=ROLL_COLOR_6M, width=2.5),
    hovertemplate="Fecha: %{x}<br>Media 6m: %{y:,.1f}",
))
fig_roll.update_layout(
    title=f"Producción Total de {fluid_roll} — Mensual vs Rolling Average",
    xaxis_title="Fecha",
    yaxis_title=y_lbl,
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_roll, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. VARIACIONES MoM Y YoY — Gráfico de barras waterfall estilo
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 📉 Variación Porcentual MoM y YoY")

var_fluid = st.radio(
    "Fluido para variaciones",
    ["Petróleo", "Gas"],
    horizontal=True,
    key="var_fluid",
)
_var_col = "oil_rate" if var_fluid == "Petróleo" else "gas_rate"
_var_lbl = "Petróleo (m³/d)" if var_fluid == "Petróleo" else "Gas (km³/d)"

monthly_plot = monthly.dropna(subset=[f"{_var_col}_mom"]).copy()
monthly_plot["mom_color"] = monthly_plot[f"{_var_col}_mom"].apply(
    lambda v: "#27ae60" if v >= 0 else "#e74c3c"
)

fig_var = go.Figure()
fig_var.add_trace(go.Bar(
    x=monthly_plot["date"],
    y=monthly_plot[f"{_var_col}_mom"],
    name="MoM %",
    marker_color=monthly_plot["mom_color"],
    hovertemplate="Fecha: %{x}<br>MoM: %{y:.1f}%",
))

monthly_yoy_plot = monthly.dropna(subset=[f"{_var_col}_yoy"]).copy()
fig_var.add_trace(go.Scatter(
    x=monthly_yoy_plot["date"],
    y=monthly_yoy_plot[f"{_var_col}_yoy"],
    mode="lines+markers",
    name="YoY %",
    line=dict(color="#2c3e50", width=2),
    marker=dict(size=5),
    hovertemplate="Fecha: %{x}<br>YoY: %{y:.1f}%",
))

fig_var.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1)
fig_var.update_layout(
    title=f"Variación MoM (barras) y YoY (línea) — {_var_lbl}",
    xaxis_title="Fecha",
    yaxis_title="Variación (%)",
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_var, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PRODUCCIÓN INCREMENTAL MENSUAL NETA
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### ⚡ Producción Incremental Mensual Neta")
st.caption(
    "Diferencia neta entre el caudal del mes actual y el mes anterior. "
    "Positivo = cuenca creciendo orgánicamente. Negativo = decline neto."
)

monthly["oil_increment"] = monthly["oil_rate"].diff()
monthly["gas_increment"] = monthly["gas_rate"].diff()

incr_fluid = st.radio(
    "Fluido incremental",
    ["Petróleo", "Gas"],
    horizontal=True,
    key="incr_fluid",
)
_inc_col = "oil_increment" if incr_fluid == "Petróleo" else "gas_increment"
_inc_lbl = "Incremento Petróleo (m³/d)" if incr_fluid == "Petróleo" else "Incremento Gas (km³/d)"
_inc_color = ROLL_COLOR_OIL if incr_fluid == "Petróleo" else ROLL_COLOR_GAS

monthly_incr = monthly.dropna(subset=[_inc_col]).copy()
monthly_incr["bar_color"] = monthly_incr[_inc_col].apply(
    lambda v: "#27ae60" if v >= 0 else "#e74c3c"
)

fig_incr = go.Figure()
fig_incr.add_trace(go.Bar(
    x=monthly_incr["date"],
    y=monthly_incr[_inc_col],
    marker_color=monthly_incr["bar_color"],
    name="Incremento neto",
    hovertemplate="Fecha: %{x}<br>Δ: %{y:,.0f}",
))
# Rolling 3m del incremento para tendencia
monthly_incr["incr_3m"] = monthly_incr[_inc_col].rolling(3, min_periods=1).mean()
fig_incr.add_trace(go.Scatter(
    x=monthly_incr["date"],
    y=monthly_incr["incr_3m"],
    mode="lines",
    name="Tendencia 3m",
    line=dict(color="#2c3e50", width=2.5, dash="dash"),
    hovertemplate="Fecha: %{x}<br>Tendencia 3m: %{y:,.0f}",
))
fig_incr.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1.5)
fig_incr.update_layout(
    title=f"Producción Incremental Mensual Neta — {incr_fluid}",
    xaxis_title="Fecha",
    yaxis_title=_inc_lbl,
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_incr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. GOR Y WOR EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 🔬 Evolución de GOR y WOR")
st.caption(
    "GOR = Gas-Oil Ratio (m³/m³ × 1000 = m³/km³). "
    "WOR = Water-Oil Ratio (m³/m³). "
    "GOR creciente puede indicar agotamiento de presión o madurez del yacimiento."
)

gorwor_sel = st.radio(
    "Métrica",
    ["GOR", "WOR", "Ambas"],
    horizontal=True,
    key="gorwor_sel",
)

fig_gw = go.Figure()

if gorwor_sel in ["GOR", "Ambas"]:
    fig_gw.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["GOR"],
        mode="lines", name="GOR mensual",
        line=dict(color=ROLL_COLOR_GAS, width=1, dash="dot"),
        opacity=0.4,
    ))
    fig_gw.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["GOR_3m"],
        mode="lines", name="GOR media 3m",
        line=dict(color=ROLL_COLOR_GAS, width=2.5),
        hovertemplate="Fecha: %{x}<br>GOR 3m: %{y:,.1f}",
    ))

if gorwor_sel in ["WOR", "Ambas"]:
    yaxis_id = "y2" if gorwor_sel == "Ambas" else "y"
    fig_gw.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["WOR"],
        mode="lines", name="WOR mensual",
        line=dict(color="#2980b9", width=1, dash="dot"),
        opacity=0.4,
        yaxis=yaxis_id,
    ))
    fig_gw.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["WOR_3m"],
        mode="lines", name="WOR media 3m",
        line=dict(color="#2980b9", width=2.5),
        hovertemplate="Fecha: %{x}<br>WOR 3m: %{y:,.3f}",
        yaxis=yaxis_id,
    ))

layout_gw = dict(
    title="Evolución GOR y WOR — Cuenca Vaca Muerta",
    xaxis_title="Fecha",
    yaxis_title="GOR (m³/km³)" if gorwor_sel != "WOR" else "WOR (m³/m³)",
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
if gorwor_sel == "Ambas":
    layout_gw["yaxis2"] = dict(
        title="WOR (m³/m³)",
        overlaying="y",
        side="right",
    )
fig_gw.update_layout(**layout_gw)
st.plotly_chart(fig_gw, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MARKET SHARE DINÁMICO POR OPERADOR — STACKED AREA
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 🏢 Market Share Dinámico por Operador")
st.caption(
    "Participación porcentual de cada empresa en la producción total mensual. "
    "Permite ver cómo se redistribuye el mercado entre operadores a lo largo del tiempo."
)

ms_fluid = st.radio(
    "Fluido para market share",
    ["Petróleo", "Gas"],
    horizontal=True,
    key="ms_fluid",
)
_ms_rate = "oil_rate" if ms_fluid == "Petróleo" else "gas_rate"
_ms_lbl  = "Petróleo (m³/d)" if ms_fluid == "Petróleo" else "Gas (km³/d)"

ms_n = st.slider(
    "Top N operadores a mostrar (resto agrupado como 'Otros')",
    min_value=3, max_value=12, value=6, key="ms_n",
)

company_monthly = (
    data_sorted[data_sorted["tef"] > 0]
    .groupby(["date", "empresaNEW"])[_ms_rate]
    .sum()
    .reset_index()
)

# Top N por producción total
top_cos = (
    company_monthly.groupby("empresaNEW")[_ms_rate]
    .sum()
    .nlargest(ms_n)
    .index.tolist()
)
company_monthly["empresa_label"] = company_monthly["empresaNEW"].apply(
    lambda x: x if x in top_cos else "Otros"
)

# Share calculation
total_monthly = company_monthly.groupby("date")[_ms_rate].sum().rename("total")
company_monthly = company_monthly.merge(total_monthly, on="date")
company_monthly["share"] = company_monthly[_ms_rate] / company_monthly["total"] * 100

ms_agg = (
    company_monthly.groupby(["date", "empresa_label"])[["share", _ms_rate]]
    .sum()
    .reset_index()
)

fig_ms = go.Figure()
all_labels = [c for c in top_cos if c in ms_agg["empresa_label"].unique()] + (
    ["Otros"] if "Otros" in ms_agg["empresa_label"].unique() else []
)

for i, co in enumerate(all_labels):
    co_data = ms_agg[ms_agg["empresa_label"] == co].sort_values("date")
    color = COMPANY_PALETTE[i % len(COMPANY_PALETTE)] if co != "Otros" else "#bdc3c7"
    fig_ms.add_trace(go.Scatter(
        x=co_data["date"],
        y=co_data["share"],
        name=co,
        mode="lines",
        stackgroup="one",
        groupnorm="",
        line=dict(color=color, width=0.5),
        fillcolor=color,
        hovertemplate=f"{co}<br>Fecha: %{{x}}<br>Share: %{{y:.1f}}%<br>{_ms_lbl}: %{{customdata:,.0f}}",
        customdata=co_data[_ms_rate].values,
    ))

fig_ms.update_layout(
    title=f"Market Share de Producción de {ms_fluid} por Operador",
    xaxis_title="Fecha",
    yaxis_title="Participación (%)",
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
    yaxis=dict(range=[0, 100], ticksuffix="%"),
)
st.plotly_chart(fig_ms, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SEÑALES DE MADUREZ BROWNFIELD
#    "La ventaja del campo maduro: más por activo existente"
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 🏗️ Vaca Muerta como Brownfield Consolidado")
st.caption(
    "En unconventional, la cuenca madura se comporta como brownfield de clase mundial: "
    "mayor producción por pozo activo, menor dispersión de resultados y eficiencia "
    "creciente del capital. Estas métricas muestran esa transición greenfield → brownfield."
)

# 7a. PRODUCCIÓN POR POZO ACTIVO — proxy de eficiencia de capital
monthly_bf = monthly.copy()
monthly_bf["oil_per_well"] = monthly_bf["oil_rate"] / monthly_bf["wells_active"].replace(0, np.nan)
monthly_bf["gas_per_well"] = monthly_bf["gas_rate"] / monthly_bf["wells_active"].replace(0, np.nan)
monthly_bf["oil_per_well_3m"] = monthly_bf["oil_per_well"].rolling(3, min_periods=1).mean()
monthly_bf["gas_per_well_3m"] = monthly_bf["gas_per_well"].rolling(3, min_periods=1).mean()

bf_fluid = st.radio(
    "Fluido",
    ["Petróleo", "Gas"],
    horizontal=True,
    key="bf_fluid",
)
_bf_col  = "oil_per_well"   if bf_fluid == "Petróleo" else "gas_per_well"
_bf_3m   = "oil_per_well_3m" if bf_fluid == "Petróleo" else "gas_per_well_3m"
_bf_lbl  = "m³/d por pozo activo" if bf_fluid == "Petróleo" else "km³/d por pozo activo"
_bf_clr  = ROLL_COLOR_OIL if bf_fluid == "Petróleo" else ROLL_COLOR_GAS

fig_bf1 = go.Figure()
fig_bf1.add_trace(go.Scatter(
    x=monthly_bf["date"], y=monthly_bf[_bf_col],
    mode="lines", name="Mensual",
    line=dict(color=_bf_clr, width=1, dash="dot"),
    opacity=0.4,
))
fig_bf1.add_trace(go.Scatter(
    x=monthly_bf["date"], y=monthly_bf[_bf_3m],
    mode="lines", name="Media 3m",
    line=dict(color=_bf_clr, width=3),
    hovertemplate="Fecha: %{x}<br>%{y:,.2f} " + _bf_lbl,
))
fig_bf1.update_layout(
    title=f"Productividad por Pozo Activo — {bf_fluid} (señal de madurez brownfield)",
    xaxis_title="Fecha",
    yaxis_title=_bf_lbl,
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_bf1, use_container_width=True)

# 7b. POZOS ACTIVOS Y TASA TOTAL — crecimiento de base instalada
fig_bf2 = go.Figure()
fig_bf2.add_trace(go.Scatter(
    x=monthly_bf["date"], y=monthly_bf["wells_active"],
    mode="lines", name="Pozos activos",
    line=dict(color=BROWNFIELD_COLOR, width=2.5),
    hovertemplate="Fecha: %{x}<br>Pozos activos: %{y:,}",
))
fig_bf2.add_trace(go.Scatter(
    x=monthly_bf["date"], y=monthly_bf["oil_rate"],
    mode="lines", name="Qo total (m³/d)",
    line=dict(color=ROLL_COLOR_OIL, width=2),
    yaxis="y2",
    hovertemplate="Fecha: %{x}<br>Qo: %{y:,.0f} m³/d",
))
fig_bf2.update_layout(
    title="Crecimiento de Base Instalada: Pozos Activos vs Producción Total",
    xaxis_title="Fecha",
    yaxis=dict(title="Pozos Activos", color=BROWNFIELD_COLOR),
    yaxis2=dict(title="Qo Total (m³/d)", overlaying="y", side="right", color=ROLL_COLOR_OIL),
    hovermode="x unified",
    template="plotly_white",
    legend=LEGEND_BOTTOM,
)
st.plotly_chart(fig_bf2, use_container_width=True)

# 7c. EFICIENCIA DE PERFORACIÓN POR CAMPAÑA (IP30 proxy como Qo pico / etapas)
#     Muestra si la cuenca mejora su retorno por unidad de capital invertido

if "Qo_peak" in df_vmut.columns and "cantidad_fracturas" in df_vmut.columns:
    df_eff = df_vmut[
        (df_vmut["tipopozoNEW"] == "Petrolífero") &
        (df_vmut["Qo_peak"] > 0) &
        (df_vmut["cantidad_fracturas"] > 0)
    ].copy()
    df_eff["Qo_peak_x_etapa"] = df_eff["Qo_peak"] / df_eff["cantidad_fracturas"]

    eff_agg = (
        df_eff.groupby("start_year")["Qo_peak_x_etapa"]
        .agg(p50="median", p10=lambda x: x.quantile(0.9), p90=lambda x: x.quantile(0.1), count="count")
        .reset_index()
        .dropna(subset=["p50"])
    )

    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(
        x=eff_agg["start_year"], y=eff_agg["p10"],
        mode="lines", name="P10",
        line=dict(color=ROLL_COLOR_OIL, width=1, dash="dot"),
        showlegend=True,
    ))
    fig_eff.add_trace(go.Scatter(
        x=eff_agg["start_year"], y=eff_agg["p90"],
        mode="lines", name="P90",
        line=dict(color=ROLL_COLOR_OIL, width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(39,174,96,0.12)",
        showlegend=True,
    ))
    fig_eff.add_trace(go.Scatter(
        x=eff_agg["start_year"], y=eff_agg["p50"],
        mode="lines+markers+text",
        name="P50 (mediana)",
        line=dict(color=ROLL_COLOR_OIL, width=3),
        marker=dict(size=8),
        text=eff_agg["p50"].round(1).astype(str),
        textposition="top center",
        textfont=dict(size=9, color=ROLL_COLOR_OIL),
    ))
    for _, row in eff_agg.iterrows():
        fig_eff.add_annotation(
            x=row["start_year"], y=row["p50"],
            text=f"n={int(row['count'])}",
            showarrow=False, yshift=-18,
            font=dict(size=8, color="grey"),
        )
    fig_eff.update_layout(
        title="Eficiencia de Capital: Qo Pico por Etapa por Campaña — Petrolífero (Proxy de Retorno/Inversión)",
        xaxis_title="Campaña de Perforación",
        yaxis_title="Qo Pico / Etapa (m³/d/etapa)",
        template="plotly_white",
        legend=LEGEND_BOTTOM,
        annotations=[dict(
            x=0.01, y=0.97,
            xref="paper", yref="paper",
            text="📌 Tendencia creciente = mayor retorno por unidad de capital (brownfield advantage)",
            showarrow=False,
            font=dict(size=10, color=BROWNFIELD_COLOR),
            bgcolor="rgba(142,68,173,0.08)",
            bordercolor=BROWNFIELD_COLOR,
            borderwidth=1,
        )],
    )
    st.plotly_chart(fig_eff, use_container_width=True)

# 7d. INFOGRAPHIC — contexto brownfield
st.info(
    """
    **🏗️ ¿Por qué Vaca Muerta es ya un Brownfield de clase mundial?**

    En la industria upstream, **brownfield casi siempre supera a greenfield** en 
    rentabilidad y velocidad de monetización. La ventaja del campo maduro:

    - ✅ **Infraestructura existente** → menores costos de transporte y procesamiento  
    - ✅ **Curva de aprendizaje acumulada** → completaciones más eficientes por campaña  
    - ✅ **Base de datos de subsuelo** → menor riesgo de exploración  
    - ✅ **Cadena de valor local consolidada** → proveedores, arena nacional, crews especializados  
    - ✅ **Productividad por pozo activo en crecimiento** → más producción con base de pozos madura  

    La excepción histórica al dominio brownfield fue **Guyana** (greenfield de primer orden). 
    Vaca Muerta, por contraste, combina escala de greenfield con ventajas operacionales 
    de un activo maduro — una combinación muy poco común a nivel global.
    """
)