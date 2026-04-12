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
    st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la Página Principal.")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.header(":blue[Reporte Extensivo de Completación y Producción en Vaca Muerta]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))
st.sidebar.image(Image.open("McCain.png"))
st.sidebar.caption(
    "Nota: Los pozos clasificados como 'Otro tipo' son reclasificados como "
    "'Gasíferos' o 'Petrolíferos' usando el criterio de GOR según McCain."
)

selected_company = st.sidebar.selectbox(
    "Seleccionar empresa:",
    options=sorted(data_sorted["empresaNEW"].unique()),
)

data_filtered = data_sorted[
    (data_sorted["tef"] > 0) &
    (data_sorted["empresaNEW"] == selected_company)
]


# ── Data preparation ──────────────────────────────────────────────────────────

df_frac    = load_frac_data()
summary_df = create_summary_dataframe(data_filtered)

df_merged = (
    pd.merge(df_frac, summary_df, on="sigla", how="outer")
    .drop_duplicates()
)

df_vmut = df_merged[
    (df_merged["formprod"]         == "VMUT") &
    (df_merged["sub_tipo_recurso"] == "SHALE")
].copy()

# Derived columns used across tabs
df_vmut["fracspacing"]       = df_vmut["longitud_rama_horizontal_m"] / df_vmut["cantidad_fracturas"]
df_vmut["prop_x_etapa"]      = df_vmut["arena_total_tn"] / df_vmut["cantidad_fracturas"]
df_vmut["proppant_intensity"] = df_vmut["arena_total_tn"] / df_vmut["longitud_rama_horizontal_m"]
df_vmut["AS_x_vol"]          = df_vmut["arena_total_tn"] / (df_vmut["agua_inyectada_m3"] / 1000)
df_vmut["Qo_peak_x_etapa"]   = df_vmut["Qo_peak"] / df_vmut["cantidad_fracturas"]
df_vmut["Qg_peak_x_etapa"]   = df_vmut["Qg_peak"] / df_vmut["cantidad_fracturas"]
df_vmut = df_vmut.replace([np.inf, -np.inf], np.nan)

df_vmut_dedup = df_vmut[df_vmut["longitud_rama_horizontal_m"] > 0].drop_duplicates(subset="sigla")


# ── Shared layout constants ───────────────────────────────────────────────────

LEGEND_TOP = dict(
    orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5
)


# ── Shared chart helper — evolution line chart ────────────────────────────────

def build_evolution_chart(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    extra_traces: list | None = None,
) -> go.Figure:
    """
    Builds a P50 + Max evolution line chart grouped by start_year.
    Optionally accepts extra_traces (list of dicts) for secondary axes.
    """
    stats = (
        df.groupby("start_year")[y_col]
        .agg(p50="median", max="max")
        .reset_index()
        .dropna()
    )
    stats["start_year"] = stats["start_year"].astype(int).astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stats["start_year"], y=stats["max"],
        mode="lines+markers", name="Max",
        line=dict(color="blue", dash="dash"), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=stats["start_year"], y=stats["p50"],
        mode="lines+markers", name="P50",
        line=dict(color="magenta"), marker=dict(size=8),
    ))
    for i, row in stats.iterrows():
        fig.add_annotation(x=row["start_year"], y=row["max"],
                           text=f"{row['max']:.0f}", showarrow=False,
                           yshift=12, font=dict(color="blue", size=10))
        fig.add_annotation(x=row["start_year"], y=row["p50"],
                           text=f"{row['p50']:.0f}", showarrow=False,
                           yshift=-15, font=dict(color="magenta", size=10))

    if extra_traces:
        for t in extra_traces:
            fig.add_trace(t)

    fig.update_layout(
        title=title,
        xaxis_title="Campaña",
        yaxis_title=y_label,
        template="plotly_white",
        legend=LEGEND_TOP,
    )
    return fig


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Indicadores de Actividad",
    "Estrategia de Completación",
    "Productividad",
    "Perfiles de Pozo",
])


# ── Tab 1: Activity indicators ────────────────────────────────────────────────

with tab1:

    # Wells drilled per vintage
    wells_by_year = (
        df_vmut.groupby(["start_year", "tipopozoNEW"])["sigla"]
        .nunique()
        .reset_index(name="count")
    )
    pivot_wells = wells_by_year.pivot_table(
        index="start_year", columns="tipopozoNEW", values="count", fill_value=0
    ).drop(columns=["Inyección de Agua", "Inyección de Gas"], errors="ignore")

    fig_activity = go.Figure()
    for fluid, color in [("Petrolífero", "green"), ("Gasífero", "red")]:
        if fluid not in pivot_wells.columns:
            continue
        fig_activity.add_trace(go.Scatter(
            x=pivot_wells.index, y=pivot_wells[fluid],
            mode="lines+markers", name=fluid,
            line=dict(color=color), marker=dict(size=8),
        ))
        for x, y in zip(pivot_wells.index, pivot_wells[fluid]):
            fig_activity.add_annotation(
                x=x, y=y, text=str(int(y)),
                showarrow=False, yshift=15,
                font=dict(size=10, color=color),
            )
    fig_activity.update_layout(
        title=f"Pozos enganchados por campaña — {selected_company} (Fm. Vaca Muerta)",
        xaxis_title="Año de Puesta en Marcha",
        yaxis_title="Cantidad de Pozos",
        legend_title="Tipo de Pozo",
        template="plotly_white",
    )
    st.plotly_chart(fig_activity, use_container_width=True)

    st.divider()

    # Arena evolution
    df_con_frac = df_vmut[df_vmut["id_base_fractura_adjiv"].notna()].copy()
    pivot_arena = (
        df_con_frac.groupby("start_year")
        .agg(
            arena_nacional =("arena_bombeada_nacional_tn",  "sum"),
            arena_importada=("arena_bombeada_importada_tn", "sum"),
            arena_total    =("arena_total_tn",              "sum"),
        )
        .reset_index()
    )
    pivot_arena["perc_importada"] = (
        pivot_arena["arena_importada"] / pivot_arena["arena_total"] * 100
    ).round(1)
    pivot_arena["start_year"] = pivot_arena["start_year"].astype(int).astype(str)

    fig_arena = go.Figure()
    fig_arena.add_trace(go.Scatter(
        x=pivot_arena["start_year"], y=pivot_arena["arena_total"],
        mode="lines+markers", name="Arena Total (tn)",
        line=dict(width=3),
    ))
    fig_arena.add_trace(go.Scatter(
        x=pivot_arena["start_year"], y=pivot_arena["perc_importada"],
        mode="lines+markers", name="% Arena Importada",
        line=dict(color="green", width=3),
        yaxis="y2",
    ))
    fig_arena.update_layout(
        title=f"Arena Bombeada vs % Importada — {selected_company}",
        xaxis_title="Campaña",
        yaxis_title="Arena Bombeada (tn)",
        yaxis2=dict(title="% Arena Importada", overlaying="y", side="right"),
        template="plotly_white",
        legend=LEGEND_TOP,
    )
    st.write("### Evolución de Arena Bombeada")
    st.plotly_chart(fig_arena, use_container_width=True)


# ── Tab 2: Completion strategy ────────────────────────────────────────────────

with tab2:

    COMPLETION_CHARTS = {
        "Longitud de Rama (m)":              ("longitud_rama_horizontal_m", "Longitud de Rama (m)"),
        "Cantidad de Etapas":                ("cantidad_fracturas",          "Etapas"),
        "Arena Bombeada (tn)":               ("arena_total_tn",              "Arena (tn)"),
        "Fracspacing (m)":                   ("fracspacing",                 "Fracspacing (m)"),
        "Agua Inyectada (km3)":              ("agua_inyectada_m3",           "Agua Inyectada (m3)"),
        "Propante por Etapa (tn/etapa)":     ("prop_x_etapa",               "Prop x Etapa (tn/etapa)"),
        "AS por Volumen Inyectado (tn/km3)": ("AS_x_vol",                   "AS x Vol (tn/km3)"),
    }

    selected_completion = st.multiselect(
        "Seleccionar gráficos de completación:",
        options=list(COMPLETION_CHARTS.keys()),
        default=["Longitud de Rama (m)", "Cantidad de Etapas", "Arena Bombeada (tn)"],
    )

    # Fracspacing: split by fluid type — handle separately
    FRACSPACING_KEY = "Fracspacing (m)"

    for chart_name in selected_completion:
        col, y_label = COMPLETION_CHARTS[chart_name]

        if chart_name == FRACSPACING_KEY:
            # Split by fluid type for fracspacing
            fig_fsp = go.Figure()
            for fluid, color_p50, color_min in [
                ("Gasífero",    "red",   "#F08080"),
                ("Petrolífero", "green", "#90EE90"),
            ]:
                df_fluid = df_vmut_dedup[df_vmut_dedup["tipopozoNEW"] == fluid]
                fsp_stats = (
                    df_fluid.groupby("start_year")[col]
                    .agg(p50="median", min="min")
                    .reset_index()
                    .dropna()
                )
                fsp_stats["start_year"] = fsp_stats["start_year"].astype(int).astype(str)
                fig_fsp.add_trace(go.Scatter(
                    x=fsp_stats["start_year"], y=fsp_stats["p50"],
                    mode="lines+markers", name=f"{fluid} P50",
                    line=dict(color=color_p50),
                ))
                fig_fsp.add_trace(go.Scatter(
                    x=fsp_stats["start_year"], y=fsp_stats["min"],
                    mode="lines+markers", name=f"{fluid} Min",
                    line=dict(color=color_p50, dash="dash"),
                ))
                for _, row in fsp_stats.iterrows():
                    fig_fsp.add_annotation(
                        x=row["start_year"], y=row["p50"],
                        text=f"{row['p50']:.0f}", showarrow=False,
                        yshift=12, font=dict(color=color_p50, size=10),
                    )
            fig_fsp.update_layout(
                title=f"Evolución del Fracspacing — {selected_company} (Fm. Vaca Muerta)",
                xaxis_title="Campaña", yaxis_title="Fracspacing (m)",
                template="plotly_white", legend=LEGEND_TOP,
            )
            st.plotly_chart(fig_fsp, use_container_width=True)
        else:
            source = df_vmut_dedup if col in ["longitud_rama_horizontal_m", "cantidad_fracturas"] else df_vmut
            st.plotly_chart(
                build_evolution_chart(
                    source, col, y_label,
                    f"Evolución de {chart_name} — {selected_company} (Fm. Vaca Muerta)",
                ),
                use_container_width=True,
            )


# ── Tab 3: Productivity ───────────────────────────────────────────────────────

with tab3:

    PRODUCTIVITY_CHARTS = {
        "Qo Pico — Petrolífero (m3/d)":          ("Petrolífero", "Qo_peak",        "Caudal de Petróleo (m3/d)"),
        "Qg Pico — Gasífero (km3/d)":            ("Gasífero",    "Qg_peak",        "Caudal de Gas (km3/d)"),
        "Qo Pico x Etapa — Petrolífero":         ("Petrolífero", "Qo_peak_x_etapa","Qo x Etapa (m3/d/etapa)"),
        "Qg Pico x Etapa — Gasífero":            ("Gasífero",    "Qg_peak_x_etapa","Qg x Etapa (km3/d/etapa)"),
    }

    selected_prod = st.multiselect(
        "Seleccionar gráficos de productividad:",
        options=list(PRODUCTIVITY_CHARTS.keys()),
        default=list(PRODUCTIVITY_CHARTS.keys())[:2],
    )

    for chart_name in selected_prod:
        fluid, metric_col, y_label = PRODUCTIVITY_CHARTS[chart_name]
        df_fluid = df_vmut[df_vmut["tipopozoNEW"] == fluid].dropna(subset=[metric_col])

        stats = (
            df_fluid.groupby("start_year")[metric_col]
            .agg(
                max   ="max",
                p50   =lambda x: np.percentile(x.dropna(), 50),
                p10   =lambda x: np.percentile(x.dropna(), 10),
                p90   =lambda x: np.percentile(x.dropna(), 90),
            )
            .reset_index()
        )
        stats["start_year"] = stats["start_year"].astype(int).astype(str)

        color = "green" if fluid == "Petrolífero" else "red"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stats["start_year"], y=stats["max"],
            mode="lines+markers", name="Max",
            line=dict(dash="dot", color=color), marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=stats["start_year"], y=stats["p50"],
            mode="lines+markers", name="P50",
            line=dict(color=color), marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=stats["start_year"], y=stats["p90"],
            mode="lines+markers", name="P90",
            line=dict(color="black", width=1), marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=stats["start_year"], y=stats["p10"],
            mode="lines+markers", name="P10",
            line=dict(color="black", width=1), marker=dict(size=8),
        ))
        for _, row in stats.iterrows():
            fig.add_annotation(x=row["start_year"], y=row["max"],
                               text=f"{row['max']:.0f}", showarrow=False,
                               yshift=12, font=dict(color=color, size=10))
            fig.add_annotation(x=row["start_year"], y=row["p50"],
                               text=f"{row['p50']:.0f}", showarrow=False,
                               yshift=-15, font=dict(color=color, size=10))
        fig.update_layout(
            title=f"{chart_name} — {selected_company} (Fm. Vaca Muerta)",
            xaxis_title="Campaña", yaxis_title=y_label,
            template="plotly_white", legend=LEGEND_TOP,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: Well profiles ──────────────────────────────────────────────────────

with tab4:

    st.subheader("Perfiles de Pozo — Gráficos Diagnóstico")

    available_siglas = sorted(
        df_vmut["sigla"].dropna().unique()
    )

    if not available_siglas:
        st.info("No hay pozos disponibles para la empresa seleccionada.")
    else:
        selected_wells = st.multiselect(
            "Seleccionar pozos:",
            options=available_siglas,
            default=[],
        )

        if not selected_wells:
            st.caption("Seleccioná al menos un pozo para ver los perfiles.")
        else:
            # Filter production data for selected wells
            well_prod = data_filtered[data_filtered["sigla"].isin(selected_wells)].copy()

            # Determine fluid type per well for plot selection
            fluid_map = (
                data_filtered[["sigla", "tipopozoNEW"]]
                .drop_duplicates(subset="sigla")
                .set_index("sigla")["tipopozoNEW"]
                .to_dict()
            )

            # Recompute clean monotonic cumulative per well
            well_prod = well_prod.sort_values(["sigla", "date"])
            well_prod["Gp_clean"] = well_prod.groupby("sigla")["prod_gas"].cumsum()
            well_prod["Np_clean"] = well_prod.groupby("sigla")["prod_pet"].cumsum()
            well_prod["Wp_clean"] = well_prod.groupby("sigla")["prod_agua"].cumsum()
            well_prod["GOR"] = (well_prod["Gp_clean"] / well_prod["Np_clean"] * 1000).replace([np.inf, -np.inf], np.nan)
            well_prod["WOR"] = (well_prod["Wp_clean"] / well_prod["Np_clean"]).replace([np.inf, -np.inf], np.nan)
            well_prod["WGR"] = (well_prod["Wp_clean"] / well_prod["Gp_clean"] * 1000).replace([np.inf, -np.inf], np.nan)

            # Determine available plots based on fluid types of selected wells
            fluids_selected = set(fluid_map.get(w) for w in selected_wells)
            GAS_PLOTS = {
                "Qg vs Gp":  ("Gp_clean", "gas_rate", "Gp (km3)",  "Qg (km3/d)"),
                "GOR vs Gp": ("Gp_clean", "GOR",      "Gp (km3)",  "GOR (m3/km3)"),
            }
            OIL_PLOTS = {
                "Qo vs Np":  ("Np_clean", "oil_rate", "Np (m3)",   "Qo (m3/d)"),
                "WOR vs Np": ("Np_clean", "WOR",      "Np (m3)",   "WOR (m3/m3)"),
            }
            available_diag: dict = {}
            if "Gasífero" in fluids_selected:
                available_diag.update(GAS_PLOTS)
            if "Petrolífero" in fluids_selected:
                available_diag.update(OIL_PLOTS)
            if not available_diag:
                available_diag = {**GAS_PLOTS, **OIL_PLOTS}

            col_sel, col_log = st.columns([3, 1])
            with col_sel:
                selected_diag = st.multiselect(
                    "Seleccionar gráficos diagnóstico:",
                    options=list(available_diag.keys()),
                    default=[],
                )
            with col_log:
                use_semilog = st.checkbox("Escala semilog (eje Y)", value=False)

            palette = px.colors.qualitative.Set2

            if selected_diag:
                for plot_name in selected_diag:
                    x_col, y_col, x_label, y_label = available_diag[plot_name]
                    fig = go.Figure()
                    all_y = []
                    for i, well in enumerate(selected_wells):
                        wd = well_prod[well_prod["sigla"] == well].dropna(subset=[x_col, y_col])
                        if wd.empty:
                            continue
                        fig.add_trace(go.Scatter(
                            x=wd[x_col], y=wd[y_col],
                            mode="lines+markers", name=well,
                            line=dict(color=palette[i % len(palette)]),
                            hovertemplate=f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}",
                        ))
                        all_y.extend(wd[y_col].tolist())

                    # Robust y-axis
                    clean_y = pd.Series(all_y).replace([np.inf, -np.inf], np.nan).dropna()
                    y_range = (
                        [max(0, np.percentile(clean_y, 1)), np.percentile(clean_y, 99) * 1.10]
                        if not clean_y.empty and not use_semilog else None
                    )
                    fig.update_layout(
                        title=f"{plot_name} — {selected_company}",
                        xaxis_title=x_label,
                        yaxis_title=y_label,
                        yaxis=dict(type="log" if use_semilog else "linear", range=y_range),
                        hovermode="x unified",
                        legend_title="Pozo",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Seleccioná al menos un gráfico diagnóstico para visualizarlo.")