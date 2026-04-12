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

# Derived completion metrics — computed once, reused across all tabs
df_vmut["fracspacing"]        = df_vmut["longitud_rama_horizontal_m"] / df_vmut["cantidad_fracturas"]
df_vmut["prop_x_etapa"]       = df_vmut["arena_total_tn"] / df_vmut["cantidad_fracturas"]
df_vmut["proppant_intensity"]  = df_vmut["arena_total_tn"] / df_vmut["longitud_rama_horizontal_m"]
df_vmut["AS_x_vol"]           = df_vmut["arena_total_tn"] / (df_vmut["agua_inyectada_m3"] / 1000)
df_vmut["Qo_peak_x_etapa"]    = df_vmut["Qo_peak"] / df_vmut["cantidad_fracturas"]
df_vmut["Qg_peak_x_etapa"]    = df_vmut["Qg_peak"] / df_vmut["cantidad_fracturas"]
df_vmut = df_vmut.replace([np.inf, -np.inf], np.nan)

# One row per well for completion stats
df_vmut_dedup = df_vmut[df_vmut["longitud_rama_horizontal_m"] > 0].drop_duplicates(subset="sigla")

LEGEND_TOP  = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
FLUID_COLORS = {"Petrolífero": "green", "Gasífero": "red"}


# ── Shared chart helper — evolution lines ─────────────────────────────────────

def build_evolution_chart(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    group_col: str = "start_year",
    split_col: str = None,
    split_colors: dict = None,
) -> go.Figure:
    """
    P50 + Max evolution line chart grouped by start_year.
    Optionally splits by a categorical column (e.g. tipopozoNEW).
    Annotations show values at each data point.
    Layout is applied once after all traces are added.
    """
    fig = go.Figure()

    splits = [(None, df)] if split_col is None else [
        (val, df[df[split_col] == val]) for val in (split_colors or {}).keys()
    ]

    for label, sub in splits:
        color_p50 = (split_colors or {}).get(label, "#1f77b4")
        color_max = (split_colors or {}).get(label, "#aec7e8")

        stats = (
            sub.groupby(group_col)[metric_col]
            .agg(p50="median", max="max")
            .reset_index()
            .dropna(subset=["p50", "max"])
        )
        if stats.empty:
            continue

        suffix = f" - {label}" if label else ""

        fig.add_trace(go.Scatter(
            x=stats[group_col], y=stats["p50"],
            mode="lines+markers", name=f"P50{suffix}",
            line=dict(color=color_p50, width=2), marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=stats[group_col], y=stats["max"],
            mode="lines+markers", name=f"Max{suffix}",
            line=dict(color=color_max, width=2, dash="dash"), marker=dict(size=7),
        ))

        # Annotations added after traces — never inside the trace loop
        for _, row in stats.iterrows():
            fig.add_annotation(
                x=row[group_col], y=row["p50"],
                text=f"{row['p50']:.0f}", showarrow=False, yshift=12,
                font=dict(color=color_p50, size=9),
            )
            fig.add_annotation(
                x=row[group_col], y=row["max"],
                text=f"{row['max']:.0f}", showarrow=False, yshift=-14,
                font=dict(color=color_max, size=9),
            )

    # Layout applied once after all traces and annotations
    fig.update_layout(
        title=title,
        xaxis_title="Campana",
        yaxis_title=y_label,
        template="plotly_white",
        legend=LEGEND_TOP,
    )
    return fig


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Indicadores de Actividad",
    "Estrategia de Completacion",
    "Productividad",
    "Perfiles de Pozo",
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
        title="Pozos enganchados por campana (Fm. Vaca Muerta)",
        xaxis_title="Campana", yaxis_title="Cantidad de Pozos",
        template="plotly_white", legend=LEGEND_TOP,
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
        title="Total Arena Bombeada vs % Arena Importada por Ano",
        xaxis_title="Campana", yaxis_title="Arena Bombeada (tn)",
        yaxis2=dict(title="% Arena Importada", overlaying="y", side="right"),
        template="plotly_white", legend=LEGEND_TOP,
    )
    st.write("### Evolucion de Arena Bombeada")
    st.plotly_chart(fig_arena, use_container_width=True)


# ── Tab 2: Completion strategy ────────────────────────────────────────────────

with tab2:

    COMPLETION_CHARTS = {
        "Longitud de Rama Lateral": (
            df_vmut_dedup, "longitud_rama_horizontal_m",
            "Evolucion de la Rama Lateral (Fm. Vaca Muerta)", "Longitud de Rama (m)",
        ),
        "Cantidad de Etapas": (
            df_vmut_dedup, "cantidad_fracturas",
            "Evolucion de Cantidad de Etapas (Fm. Vaca Muerta)", "Cantidad de Etapas",
        ),
        "Arena Bombeada": (
            df_vmut_dedup[df_vmut_dedup["arena_total_tn"] > 0],
            "arena_total_tn",
            "Evolucion de Arena Bombeada (Fm. Vaca Muerta)", "Arena Total (tn)",
        ),
        "Fracspacing": (
            df_vmut_dedup[df_vmut_dedup["fracspacing"] > 0],
            "fracspacing",
            "Evolucion del Fracspacing (Fm. Vaca Muerta)", "Fracspacing (m)",
        ),
        "Agua Inyectada": (
            df_vmut[df_vmut["agua_inyectada_m3"].notna()],
            "agua_inyectada_m3",
            "Evolucion del P50 de Agua Inyectada (Fm. Vaca Muerta)", "Agua Inyectada (m3)",
        ),
        "Propante por Etapa": (
            df_vmut[df_vmut["prop_x_etapa"] > 0],
            "prop_x_etapa",
            "Evolucion de Propante por Etapa (Fm. Vaca Muerta)", "Prop x Etapa (tn/etapa)",
        ),
        "Concentracion AS por Vol. Inyectado": (
            df_vmut[df_vmut["AS_x_vol"] > 0],
            "AS_x_vol",
            "Evolucion de la Concentracion de Agente de Sosten (Fm. Vaca Muerta)",
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
                split_col="tipopozoNEW",
                split_colors={"Petrolífero": "green", "Gasífero": "red"},
            ),
            use_container_width=True,
        )
        st.divider()


# ── Tab 3: Productivity ───────────────────────────────────────────────────────

with tab3:

    PRODUCTIVITY_CHARTS = {
        "Qo Pico - Petrolífero": (
            df_vmut[df_vmut["tipopozoNEW"] == "Petrolífero"],
            "Qo_peak",
            "Tipo Petrolífero: Evolucion de Caudal Pico",
            "Caudal de Petroleo (m3/d)",
        ),
        "Qg Pico - Gasífero": (
            df_vmut[df_vmut["tipopozoNEW"] == "Gasífero"],
            "Qg_peak",
            "Tipo Gasífero: Evolucion de Caudal Pico",
            "Caudal de Gas (km3/d)",
        ),
        "Qo Pico x Etapa - Petrolífero": (
            df_vmut[(df_vmut["tipopozoNEW"] == "Petrolífero") & (df_vmut["start_year"] > 2012)],
            "Qo_peak_x_etapa",
            "Tipo Petrolífero: Evolucion de Caudal Pico por Etapa",
            "Caudal de Petroleo (m3/d/etapa)",
        ),
        "Qg Pico x Etapa - Gasífero": (
            df_vmut[(df_vmut["tipopozoNEW"] == "Gasífero") & (df_vmut["start_year"] > 2012)],
            "Qg_peak_x_etapa",
            "Tipo Gasífero: Evolucion de Caudal Pico por Etapa",
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
            build_evolution_chart(df_p, metric, title, y_label),
            use_container_width=True,
        )
        st.divider()


# ── Tab 4: Well profiles (company-filtered, multi-well diagnostic) ────────────

with tab4:

    st.subheader("Perfiles de Pozo por Empresa")

    selected_company = st.selectbox(
        "Seleccionar empresa:",
        options=sorted(data_filtered["empresaNEW"].dropna().unique()),
        key="tab4_company",
    )

    company_data = data_filtered[data_filtered["empresaNEW"] == selected_company].copy()

    available_fluids = sorted(company_data["tipopozoNEW"].dropna().unique())
    selected_fluid   = st.selectbox(
        "Seleccionar tipo de fluido:",
        options=available_fluids,
        key="tab4_fluid",
    )

    fluid_data      = company_data[company_data["tipopozoNEW"] == selected_fluid]
    available_wells = sorted(fluid_data["sigla"].dropna().unique())

    selected_wells = st.multiselect(
        "Seleccionar pozos a comparar:",
        options=available_wells,
        key="tab4_wells",
    )

    if not selected_wells:
        st.caption("Selecciona al menos un pozo para visualizar sus perfiles.")
    else:
        well_data = fluid_data[fluid_data["sigla"].isin(selected_wells)].copy()

        # Recompute clean monotonic cumulative per well from monthly volumes
        well_data = well_data.sort_values(["sigla", "date"])
        well_data["Gp_clean"] = well_data.groupby("sigla")["prod_gas"].cumsum()
        well_data["Np_clean"] = well_data.groupby("sigla")["prod_pet"].cumsum()
        well_data["Wp_clean"] = well_data.groupby("sigla")["prod_agua"].cumsum()
        well_data["GOR"] = (well_data["Gp_clean"] / well_data["Np_clean"] * 1000).replace([np.inf, -np.inf], np.nan)
        well_data["WOR"] = (well_data["Wp_clean"] / well_data["Np_clean"]).replace([np.inf, -np.inf], np.nan)
        well_data["WGR"] = (well_data["Wp_clean"] / well_data["Gp_clean"] * 1000).replace([np.inf, -np.inf], np.nan)

        if selected_fluid == "Gasífero":
            DIAG_PLOTS = {
                "Qg vs Gp":  ("Gp_clean", "gas_rate", "Gp (km3)",  "Qg (km3/d)"),
                "GOR vs Gp": ("Gp_clean", "GOR",      "Gp (km3)",  "GOR (m3/km3)"),
            }
        else:
            DIAG_PLOTS = {
                "Qo vs Np":  ("Np_clean", "oil_rate", "Np (m3)",   "Qo (m3/d)"),
                "WOR vs Np": ("Np_clean", "WOR",      "Np (m3)",   "WOR (m3/m3)"),
            }

        col_sel, col_log = st.columns([3, 1])
        with col_sel:
            selected_diag = st.multiselect(
                f"Seleccionar graficos diagnostico ({selected_fluid}):",
                options=list(DIAG_PLOTS.keys()),
                default=list(DIAG_PLOTS.keys()),
                key="tab4_diag",
            )
        with col_log:
            use_semilog = st.checkbox("Escala semilog (eje Y)", key="tab4_semilog")

        palette = px.colors.qualitative.Set2

        for plot_name in selected_diag:
            x_col, y_col, x_label, y_label = DIAG_PLOTS[plot_name]
            fig      = go.Figure()
            all_y    = []

            for i, well in enumerate(selected_wells):
                wd = well_data[well_data["sigla"] == well].dropna(subset=[x_col, y_col])
                if wd.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=wd[x_col], y=wd[y_col],
                    mode="lines+markers", name=well,
                    line=dict(color=palette[i % len(palette)]),
                    hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}",
                ))
                all_y.extend(wd[y_col].tolist())

            clean_y = pd.Series(all_y).replace([np.inf, -np.inf], np.nan).dropna()
            y_range = (
                None if use_semilog or clean_y.empty
                else [max(0, np.percentile(clean_y, 1)), np.percentile(clean_y, 99) * 1.10]
            )

            fig.update_layout(
                title=f"{selected_fluid} - {plot_name}",
                xaxis_title=x_label,
                yaxis_title=y_label,
                yaxis=dict(type="log" if use_semilog else "linear", range=y_range),
                hovermode="x unified",
                legend_title="Pozo",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)