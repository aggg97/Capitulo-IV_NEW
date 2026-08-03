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

# Garantizar que areayacimiento esté presente (puede no venir del join si
# create_summary_dataframe fue compilado sin ella en versiones anteriores)
if "areayacimiento" not in df_vmut.columns or df_vmut["areayacimiento"].isna().all():
    area_map = (
        data_sorted[["sigla", "areayacimiento"]]
        .drop_duplicates("sigla")
        .set_index("sigla")["areayacimiento"]
    )
    df_vmut["areayacimiento"] = df_vmut["sigla"].map(area_map)

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

    # ══════════════════════════════════════════════════════════════════════════
    # ANÁLISIS POR ÁREA — mismo set de gráficos con filtro multiselect
    # ══════════════════════════════════════════════════════════════════════════

    st.divider()
    st.subheader("🗺️ Análisis de Productividad por Área", divider="blue")
    st.caption(
        "Repetición del análisis anterior filtrado por una o más áreas de yacimiento. "
        "Podés comparar el P50 de múltiples áreas en el mismo gráfico activando "
        "'Superponer áreas en el mismo gráfico'."
    )

    # Áreas disponibles en el dataset de fractura
    all_prod_areas = sorted(df_vmut["areayacimiento"].dropna().unique())

    sel_prod_areas = st.multiselect(
        "Seleccionar áreas de yacimiento:",
        options=all_prod_areas,
        default=[],
        key="prod_area_multiselect",
        help="Sin selección no se muestra este análisis.",
    )

    if not sel_prod_areas:
        st.info("Seleccioná al menos un área para activar el análisis por área.")
    else:
        selected_productivity_area = st.multiselect(
            "Indicadores a visualizar (por área):",
            options=list(PRODUCTIVITY_CHARTS.keys()),
            default=list(PRODUCTIVITY_CHARTS.keys())[:2],
            key="productivity_area_charts",
        )

        split_area = st.checkbox(
            "Separar por tipo de fluido",
            key="split_productivity_area",
        )

        overlay_areas = st.checkbox(
            "Superponer áreas en el mismo gráfico (P50 por área)",
            value=False,
            key="overlay_areas",
            help=(
                "Activado: un gráfico con una línea P50 por área. "
                "Desactivado: un gráfico separado por área."
            ),
        )

        # Definición de métricas base (sin filtro de área, lo aplicamos abajo)
        PROD_METRICS = {
            "Qo Pico":         ("Qo_peak",         "Caudal de Petróleo (m3/d)",      "Petrolífero"),
            "Qg Pico":         ("Qg_peak",          "Caudal de Gas (km3/d)",          "Gasífero"),
            "Qo Pico x Etapa": ("Qo_peak_x_etapa",  "Caudal de Petróleo (m3/d/etapa)","Petrolífero"),
            "Qg Pico x Etapa": ("Qg_peak_x_etapa",  "Caudal de Gas (km3/d/etapa)",    "Gasífero"),
        }

        AREA_PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Pastel

        for chart_name in selected_productivity_area:
            metric_col, y_label, fluid_type = PROD_METRICS[chart_name]

            # Filtro base: tipo de pozo + año mínimo según métrica
            base_mask = df_vmut["tipopozoNEW"] == fluid_type
            if "x Etapa" in chart_name:
                base_mask &= df_vmut["start_year"] > 2012

            if overlay_areas:
                # ── Un gráfico con P50 por área ───────────────────────────────
                fig_overlay = go.Figure()

                for i, area in enumerate(sel_prod_areas):
                    df_area = df_vmut[
                        base_mask &
                        (df_vmut["areayacimiento"] == area)
                    ].dropna(subset=[metric_col, "start_year"])

                    if df_area.empty:
                        continue

                    stats = (
                        df_area.groupby("start_year")[metric_col]
                        .agg(
                            p10=lambda x: x.quantile(0.90),
                            p50="median",
                            p90=lambda x: x.quantile(0.10),
                        )
                        .reset_index()
                        .dropna(subset=["p50"])
                    )
                    if stats.empty:
                        continue

                    color = AREA_PALETTE[i % len(AREA_PALETTE)]

                    # Banda P10-P90
                    fig_overlay.add_trace(go.Scatter(
                        x=pd.concat([stats["start_year"], stats["start_year"][::-1]]),
                        y=pd.concat([stats["p10"], stats["p90"][::-1]]),
                        fill="toself",
                        fillcolor=color,
                        opacity=0.10,
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"Banda {area}",
                    ))

                    # P50 por área
                    fig_overlay.add_trace(go.Scatter(
                        x=stats["start_year"],
                        y=stats["p50"],
                        mode="lines+markers",
                        name=area,
                        line=dict(color=color, width=2.5),
                        marker=dict(size=7),
                        hovertemplate=(
                            f"<b>{area}</b><br>"
                            "Campaña: %{x}<br>"
                            f"P50: %{{y:.0f}}<extra></extra>"
                        ),
                    ))
                    # Anotaciones P50
                    for _, row in stats.iterrows():
                        fig_overlay.add_annotation(
                            x=row["start_year"], y=row["p50"],
                            text=f"{row['p50']:.0f}",
                            showarrow=False, yshift=12,
                            font=dict(color=color, size=9),
                        )

                fig_overlay.update_layout(
                    title=f"{chart_name} — P50 por Área (superposición)",
                    xaxis_title=X_AXIS_LABEL,
                    yaxis_title=y_label,
                    template="plotly_white",
                    legend=LEGEND_BOTTOM,
                    height=480,
                )
                st.plotly_chart(fig_overlay, use_container_width=True)

            else:
                # ── Un gráfico por área ───────────────────────────────────────
                for area in sel_prod_areas:
                    df_area = df_vmut[
                        base_mask &
                        (df_vmut["areayacimiento"] == area)
                    ].dropna(subset=[metric_col])

                    if df_area.empty:
                        st.caption(f"Sin datos para {area} — {chart_name}")
                        continue

                    st.plotly_chart(
                        build_evolution_chart(
                            df_area,
                            metric_col,
                            f"{chart_name} — {area}",
                            y_label,
                            split_by_fluid=split_area,
                            invert_percentiles=True,
                        ),
                        use_container_width=True,
                    )

            st.divider()
