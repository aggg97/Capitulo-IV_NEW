import numpy as np
import pandas as pd
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
    data_sorted["date"] = pd.to_datetime(
        data_sorted["anio"].astype(str) + "-" + data_sorted["mes"].astype(str) + "-1"
    )
    data_sorted["gas_rate"] = data_sorted["prod_gas"] / data_sorted["tef"]
    data_sorted["oil_rate"] = data_sorted["prod_pet"] / data_sorted["tef"]
    data_sorted["water_rate"] = data_sorted["prod_agua"] / data_sorted["tef"]
    data_sorted = data_sorted.sort_values(by=["sigla", "date"], ascending=True)
    data_sorted["empresaNEW"] = data_sorted["empresa"].replace(COMPANY_REPLACEMENTS)
    data_sorted = get_fluid_classification(data_sorted)
    st.info("Utilizando datos recuperados de la memoria.")
else:
    st.warning("⚠️ No se han cargado los datos. Por favor, vuelve a la Página Principal.")
    st.stop()


# ── Data preparation ──────────────────────────────────────────────────────────

st.header(":blue[Ranking y Records]")
st.sidebar.image(Image.open("Vaca Muerta rig.png"))
st.sidebar.image(Image.open("McCain.png"))
st.sidebar.caption(
    "Nota: Los pozos clasificados como 'Otro tipo' son reclasificados como "
    "'Gasíferos' o 'Petrolíferos' usando el criterio de GOR según McCain."
)

data_filtered = data_sorted[data_sorted["tef"] > 0]

df_frac = load_frac_data()
summary_df = create_summary_dataframe(data_filtered)

df_merged = (
    pd.merge(df_frac, summary_df, on="sigla", how="outer")
    .drop_duplicates()
)

df_vmut = df_merged[
    (df_merged["formprod"] == "VMUT") &
    (df_merged["sub_tipo_recurso"] == "SHALE")
].copy()

df_vmut["fracspacing"] = df_vmut["longitud_rama_horizontal_m"] / df_vmut["cantidad_fracturas"]
df_vmut["prop_x_etapa"] = df_vmut["arena_total_tn"] / df_vmut["cantidad_fracturas"]
df_vmut["proppant_intensity"] = df_vmut["arena_total_tn"] / df_vmut["longitud_rama_horizontal_m"]
df_vmut["AS_x_vol"] = df_vmut["arena_total_tn"] / (df_vmut["agua_inyectada_m3"] / 1000)
df_vmut["Qo_peak_x_etapa"] = df_vmut["Qo_peak"] / df_vmut["cantidad_fracturas"]
df_vmut["Qg_peak_x_etapa"] = df_vmut["Qg_peak"] / df_vmut["cantidad_fracturas"]
df_vmut["AS_x_volumen_inyectado"] = df_vmut["arena_total_tn"] / (
    df_vmut["agua_inyectada_m3"].replace(0, pd.NA) / 1000
)

df_vmut = df_vmut.replace([np.inf, -np.inf], np.nan)

df_vmut_dedup = df_vmut[df_vmut["longitud_rama_horizontal_m"] > 0].drop_duplicates(subset="sigla")

all_years = sorted(df_vmut["start_year"].dropna().unique().astype(int), reverse=True)
petro_data = df_vmut[df_vmut["tipopozoNEW"] == "Petrolífero"]
gas_data = df_vmut[df_vmut["tipopozoNEW"] == "Gasífero"]


# ── Shared helpers ────────────────────────────────────────────────────────────

def ranking_table(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    display_cols: dict,
    top_n: int,
    ascending: bool = False,
) -> pd.DataFrame:
    agg = {metric_col: "max" if not ascending else "min"}
    extra = {c: "first" for c in display_cols if c not in [group_col, metric_col]}
    agg.update(extra)
    result = (
        df.groupby(group_col)
        .agg(agg)
        .reset_index()
        .sort_values(metric_col, ascending=ascending)
        .head(top_n)
        .rename(columns=display_cols)
    )
    return result


def multi_year_dot_chart(
    df: pd.DataFrame,
    metric_col: str,
    label_col: str,
    title: str,
    y_label: str,
    fluid_filter: str | None = None,
    ascending: bool = False,
) -> go.Figure:
    if fluid_filter:
        df = df[df["tipopozoNEW"] == fluid_filter]

    df = df.dropna(subset=["start_year", metric_col, label_col]).copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    agg_func = "min" if ascending else "max"
    yearly = (
        df.groupby("start_year")
        .apply(lambda g: g.loc[g[metric_col].agg(agg_func) == g[metric_col], [metric_col, label_col]].iloc[0])
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["start_year"],
        y=yearly[metric_col],
        mode="markers+lines+text",
        text=yearly[label_col],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=10),
        hovertemplate="Año: %{x}<br>" + y_label + ": %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Campaña",
        yaxis_title=y_label,
        template="plotly_white",
    )
    return fig


def multi_year_company_line(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    fluid_filter: str | None = None,
    top_n_companies: int = 5,
    ascending: bool = False,
) -> go.Figure:
    import plotly.express as px

    if fluid_filter:
        df = df[df["tipopozoNEW"] == fluid_filter]

    df = df.dropna(subset=["start_year", "empresaNEW", metric_col]).copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    company_metric = df.groupby("empresaNEW")[metric_col].median()
    top_companies = (
        company_metric.nsmallest(top_n_companies).index
        if ascending else
        company_metric.nlargest(top_n_companies).index
    )

    df_top = df[df["empresaNEW"].isin(top_companies)]
    yearly = (
        df_top.groupby(["start_year", "empresaNEW"])[metric_col]
        .median()
        .reset_index()
    )

    fig = go.Figure()
    palette = px.colors.qualitative.Set2

    for i, company in enumerate(top_companies):
        yd = yearly[yearly["empresaNEW"] == company]
        fig.add_trace(go.Scatter(
            x=yd["start_year"],
            y=yd[metric_col],
            mode="lines+markers",
            name=company,
            line=dict(color=palette[i % len(palette)]),
            hovertemplate=f"{company}<br>Año: %{{x}}<br>{y_label}: %{{y:.0f}}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Campaña",
        yaxis_title=y_label,
        template="plotly_white",
        legend_title="Empresa",
    )
    return fig


def comparative_boxplot(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    fluid_filter: str | None = None,
    by: str = "start_year",
) -> go.Figure:
    import plotly.express as px

    plot_df = df.copy()

    if fluid_filter:
        plot_df = plot_df[plot_df["tipopozoNEW"] == fluid_filter]

    plot_df = plot_df.dropna(subset=[metric_col, by])

    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    fig = px.box(
        plot_df,
        x=by,
        y=metric_col,
        color="tipopozoNEW" if fluid_filter is None else None,
        points="outliers",
        template="plotly_white",
        title=title,
    )

    fig.update_layout(
        xaxis_title="Campaña" if by == "start_year" else "Empresa",
        yaxis_title=y_label,
        showlegend=True if fluid_filter is None else False,
    )

    return fig


# ── Year selector & top-N slider ──────────────────────────────────────────────

st.sidebar.divider()
selected_year = st.sidebar.selectbox("Seleccionar campaña:", all_years)
top_n = st.sidebar.slider("Top N a mostrar:", min_value=3, max_value=10, value=5)

df_year = df_vmut[df_vmut["start_year"] == selected_year]
df_year_dedup = df_vmut_dedup[df_vmut_dedup["start_year"] == selected_year]
petro_year = df_year[df_year["tipopozoNEW"] == "Petrolífero"]
gas_year = df_year[df_year["tipopozoNEW"] == "Gasífero"]


# ── Helper para tablas dinámicas ──────────────────────────────────────────────

def make_yearly_top_table(df, group_cols, agg_dict, sort_col, ascending, head_n, rename_map):
    grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
    sorted_ = grouped.sort_values(["start_year", sort_col], ascending=[True, ascending])
    top = sorted_.groupby("start_year").head(head_n)

    rows = []
    prev_year = None
    for _, row in top.iterrows():
        yr = int(row["start_year"]) if row["start_year"] != prev_year else " "
        entry = {"Campaña": yr}
        for src, dst in rename_map.items():
            val = row[src]
            if src != "start_year":
                try:
                    val = int(val) if pd.notna(val) and val > 0 else None
                except Exception:
                    val = val
            entry[dst] = val
        rows.append(entry)
        prev_year = row["start_year"]
    return pd.DataFrame(rows)


# ── TABS ──────────────────────────────────────────────────────────────────────

tab_graficos, tab_tablas, tab_comparativas = st.tabs(
    ["📈 Gráficos", "📊 Tablas Dinámicas", "📦 Visualizaciones Comparativas"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — GRÁFICOS
# ════════════════════════════════════════════════════════════════════════════

with tab_graficos:

    st.subheader("Ranking de Mayor Actividad por Empresa", divider="blue")

    activity = (
        df_year.groupby(["empresaNEW", "tipopozoNEW"])["sigla"]
        .nunique()
        .reset_index(name="Pozos")
    )

    for fluid in ["Petrolífero", "Gasífero"]:
        fluid_act = activity[activity["tipopozoNEW"] == fluid].sort_values("Pozos", ascending=False).head(top_n)
        if not fluid_act.empty:
            st.write(f"**{fluid} — Top {top_n} empresas por pozos enganchados ({selected_year})**")
            st.dataframe(
                fluid_act[["empresaNEW", "Pozos"]].rename(columns={"empresaNEW": "Empresa"}),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Ranking según Longitud de Rama", divider="blue")

    st.write(f"**Top {top_n} pozos con mayor longitud de rama ({selected_year})**")
    st.dataframe(
        ranking_table(
            df_year_dedup,
            "sigla",
            "longitud_rama_horizontal_m",
            {"sigla": "Pozo", "empresaNEW": "Empresa", "longitud_rama_horizontal_m": "Longitud Rama (m)"},
            top_n,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.write(f"**Top {top_n} empresas por P50 de longitud de rama ({selected_year})**")
    st.dataframe(
        df_year_dedup.groupby("empresaNEW")["longitud_rama_horizontal_m"]
        .median().reset_index()
        .rename(columns={"empresaNEW": "Empresa", "longitud_rama_horizontal_m": "P50 Longitud Rama (m)"})
        .sort_values("P50 Longitud Rama (m)", ascending=False).head(top_n),
        use_container_width=True,
        hide_index=True,
    )

    st.write("**Evolución multi-año — Record por campaña**")
    st.plotly_chart(
        multi_year_dot_chart(
            df_vmut_dedup,
            "longitud_rama_horizontal_m",
            "sigla",
            "Record de Longitud de Rama por Campaña",
            "Longitud de Rama (m)",
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — P50 por empresa**")
    st.plotly_chart(
        multi_year_company_line(
            df_vmut_dedup,
            "longitud_rama_horizontal_m",
            "P50 Longitud de Rama por Empresa y Campaña",
            "Longitud de Rama (m)",
            top_n_companies=top_n,
        ),
        use_container_width=True,
    )

    st.subheader("Ranking según Cantidad de Etapas", divider="blue")

    st.write(f"**Top {top_n} pozos con mayor cantidad de etapas ({selected_year})**")
    st.dataframe(
        ranking_table(
            df_year_dedup,
            "sigla",
            "cantidad_fracturas",
            {"sigla": "Pozo", "empresaNEW": "Empresa", "cantidad_fracturas": "Etapas"},
            top_n,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.write(f"**Top {top_n} empresas por P50 de etapas ({selected_year})**")
    st.dataframe(
        df_year_dedup.groupby("empresaNEW")["cantidad_fracturas"]
        .median().reset_index()
        .rename(columns={"empresaNEW": "Empresa", "cantidad_fracturas": "P50 Etapas"})
        .sort_values("P50 Etapas", ascending=False).head(top_n),
        use_container_width=True,
        hide_index=True,
    )

    st.write("**Evolución multi-año — Record de etapas por campaña**")
    st.plotly_chart(
        multi_year_dot_chart(
            df_vmut_dedup,
            "cantidad_fracturas",
            "sigla",
            "Record de Etapas por Campaña",
            "Cantidad de Etapas",
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — P50 etapas por empresa**")
    st.plotly_chart(
        multi_year_company_line(
            df_vmut_dedup,
            "cantidad_fracturas",
            "P50 Etapas por Empresa y Campaña",
            "Cantidad de Etapas",
            top_n_companies=top_n,
        ),
        use_container_width=True,
    )

    st.subheader("Ranking según Caudales Pico", divider="blue")

    st.write(f"**Petrolífero — Top {top_n} pozos por Qo pico ({selected_year})**")
    st.dataframe(
        petro_year.drop_duplicates("sigla")
        .nlargest(top_n, "Qo_peak")[["sigla", "empresaNEW", "Qo_peak", "cantidad_fracturas", "fracspacing"]]
        .rename(columns={
            "sigla": "Pozo",
            "empresaNEW": "Empresa",
            "Qo_peak": "Qo Pico (m3/d)",
            "cantidad_fracturas": "Etapas",
            "fracspacing": "Fracspacing (m)",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.write(f"**Petrolífero — Top {top_n} empresas por P50 Qo pico ({selected_year})**")
    st.dataframe(
        petro_year.groupby("empresaNEW")["Qo_peak"].median().reset_index()
        .rename(columns={"empresaNEW": "Empresa", "Qo_peak": "P50 Qo Pico (m3/d)"})
        .sort_values("P50 Qo Pico (m3/d)", ascending=False).head(top_n),
        use_container_width=True,
        hide_index=True,
    )

    st.write(f"**Gasífero — Top {top_n} pozos por Qg pico ({selected_year})**")
    st.dataframe(
        gas_year.drop_duplicates("sigla")
        .nlargest(top_n, "Qg_peak")[["sigla", "empresaNEW", "Qg_peak", "cantidad_fracturas", "fracspacing"]]
        .rename(columns={
            "sigla": "Pozo",
            "empresaNEW": "Empresa",
            "Qg_peak": "Qg Pico (km3/d)",
            "cantidad_fracturas": "Etapas",
            "fracspacing": "Fracspacing (m)",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.write(f"**Gasífero — Top {top_n} empresas por P50 Qg pico ({selected_year})**")
    st.dataframe(
        gas_year.groupby("empresaNEW")["Qg_peak"].median().reset_index()
        .rename(columns={"empresaNEW": "Empresa", "Qg_peak": "P50 Qg Pico (km3/d)"})
        .sort_values("P50 Qg Pico (km3/d)", ascending=False).head(top_n),
        use_container_width=True,
        hide_index=True,
    )

    st.write("**Evolución multi-año — Record Qo pico por campaña**")
    st.plotly_chart(
        multi_year_dot_chart(
            df_vmut,
            "Qo_peak",
            "sigla",
            "Record de Qo Pico por Campaña",
            "Qo Pico (m3/d)",
            "Petrolífero",
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — Record Qg pico por campaña**")
    st.plotly_chart(
        multi_year_dot_chart(
            df_vmut,
            "Qg_peak",
            "sigla",
            "Record de Qg Pico por Campaña",
            "Qg Pico (km3/d)",
            "Gasífero",
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — P50 Qo pico por empresa**")
    st.plotly_chart(
        multi_year_company_line(
            df_vmut,
            "Qo_peak",
            "P50 Qo Pico por Empresa y Campaña",
            "Qo Pico (m3/d)",
            "Petrolífero",
            top_n_companies=top_n,
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — P50 Qg pico por empresa**")
    st.plotly_chart(
        multi_year_company_line(
            df_vmut,
            "Qg_peak",
            "P50 Qg Pico por Empresa y Campaña",
            "Qg Pico (km3/d)",
            "Gasífero",
            top_n_companies=top_n,
        ),
        use_container_width=True,
    )

    st.subheader("Ranking según Arena Bombeada", divider="blue")

    df_arena = df_vmut_dedup[df_vmut_dedup["arena_total_tn"] > 0]
    df_arena_year = df_arena[df_arena["start_year"] == selected_year]

    st.write(f"**Top {top_n} pozos por arena total bombeada ({selected_year})**")
    st.dataframe(
        ranking_table(
            df_arena_year,
            "sigla",
            "arena_total_tn",
            {"sigla": "Pozo", "empresaNEW": "Empresa", "arena_total_tn": "Arena Total (tn)"},
            top_n,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.write(f"**Top {top_n} empresas por P50 arena bombeada ({selected_year})**")
    st.dataframe(
        df_arena_year.groupby("empresaNEW")["arena_total_tn"].median().reset_index()
        .rename(columns={"empresaNEW": "Empresa", "arena_total_tn": "P50 Arena Total (tn)"})
        .sort_values("P50 Arena Total (tn)", ascending=False).head(top_n),
        use_container_width=True,
        hide_index=True,
    )

    st.write("**Evolución multi-año — Record de arena por campaña**")
    st.plotly_chart(
        multi_year_dot_chart(
            df_arena,
            "arena_total_tn",
            "sigla",
            "Record de Arena Bombeada por Campaña",
            "Arena Total (tn)",
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — P50 arena por empresa**")
    st.plotly_chart(
        multi_year_company_line(
            df_arena,
            "arena_total_tn",
            "P50 Arena Bombeada por Empresa y Campaña",
            "Arena Total (tn)",
            top_n_companies=top_n,
        ),
        use_container_width=True,
    )

    st.subheader("Ranking según Fracspacing", divider="blue")
    st.caption("Fracspacing = longitud_rama / etapas. Menor fracspacing = completación más agresiva.")

    df_fsp = df_vmut_dedup[df_vmut_dedup["fracspacing"] > 0]
    df_fsp_year = df_fsp[df_fsp["start_year"] == selected_year]

    for fluid in ["Petrolífero", "Gasífero"]:
        fluid_fsp = df_fsp_year[df_fsp_year["tipopozoNEW"] == fluid]
        st.write(f"**{fluid} — Top {top_n} pozos con menor fracspacing ({selected_year})**")
        st.dataframe(
            ranking_table(
                fluid_fsp,
                "sigla",
                "fracspacing",
                {"sigla": "Pozo", "empresaNEW": "Empresa", "fracspacing": "Fracspacing (m)"},
                top_n,
                ascending=True,
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.write("**Evolución multi-año — Mínimo fracspacing por campaña (Petrolífero)**")
    st.plotly_chart(
        multi_year_dot_chart(
            df_fsp,
            "fracspacing",
            "sigla",
            "Mínimo Fracspacing por Campaña — Petrolífero",
            "Fracspacing (m)",
            "Petrolífero",
            ascending=True,
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — P50 fracspacing por empresa (Petrolífero)**")
    st.plotly_chart(
        multi_year_company_line(
            df_fsp,
            "fracspacing",
            "P50 Fracspacing por Empresa y Campaña — Petrolífero",
            "Fracspacing (m)",
            "Petrolífero",
            top_n_companies=top_n,
            ascending=True,
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — Mínimo fracspacing por campaña (Gasífero)**")
    st.plotly_chart(
        multi_year_dot_chart(
            df_fsp,
            "fracspacing",
            "sigla",
            "Mínimo Fracspacing por Campaña — Gasífero",
            "Fracspacing (m)",
            "Gasífero",
            ascending=True,
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — P50 fracspacing por empresa (Gasífero)**")
    st.plotly_chart(
        multi_year_company_line(
            df_fsp,
            "fracspacing",
            "P50 Fracspacing por Empresa y Campaña — Gasífero",
            "Fracspacing (m)",
            "Gasífero",
            top_n_companies=top_n,
            ascending=True,
        ),
        use_container_width=True,
    )

    st.subheader("Ranking según Propante por Etapa", divider="blue")
    st.caption("Prop x Etapa = arena_total_tn / etapas.")

    df_prop = df_vmut[df_vmut["prop_x_etapa"] > 0]
    df_prop_year = df_prop[df_prop["start_year"] == selected_year]

    for fluid in ["Petrolífero", "Gasífero"]:
        fluid_prop = df_prop_year[df_prop_year["tipopozoNEW"] == fluid]
        st.write(f"**{fluid} — Top {top_n} pozos con mayor prop x etapa ({selected_year})**")
        st.dataframe(
            ranking_table(
                fluid_prop,
                "sigla",
                "prop_x_etapa",
                {"sigla": "Pozo", "empresaNEW": "Empresa", "prop_x_etapa": "Prop x Etapa (tn/etapa)"},
                top_n,
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.write("**Evolución multi-año — P50 propante por etapa por empresa (Petrolífero)**")
    st.plotly_chart(
        multi_year_company_line(
            df_prop,
            "prop_x_etapa",
            "P50 Propante por Etapa — Petrolífero",
            "Prop x Etapa (tn/etapa)",
            "Petrolífero",
            top_n_companies=top_n,
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — P50 propante por etapa por empresa (Gasífero)**")
    st.plotly_chart(
        multi_year_company_line(
            df_prop,
            "prop_x_etapa",
            "P50 Propante por Etapa — Gasífero",
            "Prop x Etapa (tn/etapa)",
            "Gasífero",
            top_n_companies=top_n,
        ),
        use_container_width=True,
    )

    st.subheader("Ranking según Caudal Pico por Etapa", divider="blue")

    df_qetapa = df_vmut[df_vmut["start_year"] > 2012]
    df_qetapa_year = df_qetapa[df_qetapa["start_year"] == selected_year]

    for fluid, metric, label in [
        ("Petrolífero", "Qo_peak_x_etapa", "Qo Pico x Etapa (m3/d/etapa)"),
        ("Gasífero", "Qg_peak_x_etapa", "Qg Pico x Etapa (km3/d/etapa)"),
    ]:
        fluid_q = df_qetapa_year[df_qetapa_year["tipopozoNEW"] == fluid].dropna(subset=[metric])
        st.write(f"**{fluid} — Top {top_n} pozos por {label} ({selected_year})**")
        st.dataframe(
            ranking_table(
                fluid_q,
                "sigla",
                metric,
                {
                    "sigla": "Pozo",
                    "empresaNEW": "Empresa",
                    metric: label,
                    "cantidad_fracturas": "Etapas",
                    "fracspacing": "Fracspacing (m)",
                },
                top_n,
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.write("**Evolución multi-año — Record Qo pico x etapa por campaña**")
    st.plotly_chart(
        multi_year_dot_chart(
            df_qetapa.dropna(subset=["Qo_peak_x_etapa"]),
            "Qo_peak_x_etapa",
            "sigla",
            "Record Qo Pico x Etapa por Campaña",
            "Qo Pico x Etapa (m3/d/etapa)",
            "Petrolífero",
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — Record Qg pico x etapa por campaña**")
    st.plotly_chart(
        multi_year_dot_chart(
            df_qetapa.dropna(subset=["Qg_peak_x_etapa"]),
            "Qg_peak_x_etapa",
            "sigla",
            "Record Qg Pico x Etapa por Campaña",
            "Qg Pico x Etapa (km3/d/etapa)",
            "Gasífero",
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — P50 Qo pico x etapa por empresa**")
    st.plotly_chart(
        multi_year_company_line(
            df_qetapa.dropna(subset=["Qo_peak_x_etapa"]),
            "Qo_peak_x_etapa",
            "P50 Qo Pico x Etapa por Empresa y Campaña",
            "Qo Pico x Etapa (m3/d/etapa)",
            "Petrolífero",
            top_n_companies=top_n,
        ),
        use_container_width=True,
    )

    st.write("**Evolución multi-año — P50 Qg pico x etapa por empresa**")
    st.plotly_chart(
        multi_year_company_line(
            df_qetapa.dropna(subset=["Qg_peak_x_etapa"]),
            "Qg_peak_x_etapa",
            "P50 Qg Pico x Etapa por Empresa y Campaña",
            "Qg Pico x Etapa (km3/d/etapa)",
            "Gasífero",
            top_n_companies=top_n,
        ),
        use_container_width=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TABLAS DINÁMICAS
# ════════════════════════════════════════════════════════════════════════════

with tab_tablas:

    st.caption("Tablas con Top 3 histórico por campaña para todos los indicadores de completación y producción.")

    st.subheader("Actividad por Empresa", divider="blue")

    for fluid in ["Petrolífero", "Gasífero"]:
        act_fluid = (
            df_vmut[df_vmut["tipopozoNEW"] == fluid]
            .groupby(["start_year", "empresaNEW"])["sigla"]
            .nunique()
            .reset_index(name="Pozos")
        )
        top_act = (
            act_fluid.sort_values(["start_year", "Pozos"], ascending=[True, False])
            .groupby("start_year").head(3)
        )
        rows = []
        prev = None
        for _, row in top_act.iterrows():
            yr = int(row["start_year"]) if row["start_year"] != prev else " "
            rows.append({"Campaña": yr, "Empresa": row["empresaNEW"], "Pozos Enganchados": int(row["Pozos"])})
            prev = row["start_year"]
        st.write(f"**{fluid}: Top 3 Empresas con Mayor Actividad por Campaña**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Longitud de Rama", divider="blue")

    tbl = make_yearly_top_table(
        df_vmut_dedup,
        group_cols=["start_year", "sigla", "empresaNEW"],
        agg_dict={"longitud_rama_horizontal_m": "max"},
        sort_col="longitud_rama_horizontal_m",
        ascending=False,
        head_n=3,
        rename_map={
            "sigla": "Sigla",
            "empresaNEW": "Empresa",
            "longitud_rama_horizontal_m": "Máx. Longitud de Rama (m)"
        },
    )
    st.write("**Top 3 Pozos con Mayor Longitud de Rama por Campaña**")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    avg_lr = (
        df_vmut_dedup.groupby(["start_year", "empresaNEW"])["longitud_rama_horizontal_m"]
        .median().reset_index(name="p50_lr")
    )
    top_avg_lr = avg_lr.sort_values(["start_year", "p50_lr"], ascending=[True, False]).groupby("start_year").head(3)
    rows = []
    prev = None
    for _, row in top_avg_lr.iterrows():
        yr = int(row["start_year"]) if row["start_year"] != prev else " "
        rows.append({"Campaña": yr, "Empresa": row["empresaNEW"], "P50 Longitud de Rama (m)": int(row["p50_lr"])})
        prev = row["start_year"]
    st.write("**Top 3 Empresas por P50 Longitud de Rama por Campaña**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Cantidad de Etapas", divider="blue")

    tbl_etapas = make_yearly_top_table(
        df_vmut_dedup,
        group_cols=["start_year", "sigla", "empresaNEW"],
        agg_dict={"cantidad_fracturas": "max"},
        sort_col="cantidad_fracturas",
        ascending=False,
        head_n=3,
        rename_map={
            "sigla": "Sigla",
            "empresaNEW": "Empresa",
            "cantidad_fracturas": "Máx. Etapas"
        },
    )
    st.write("**Top 3 Pozos con Mayor Cantidad de Etapas por Campaña**")
    st.dataframe(tbl_etapas, use_container_width=True, hide_index=True)

    avg_etapas = (
        df_vmut_dedup.groupby(["start_year", "empresaNEW"])["cantidad_fracturas"]
        .median().reset_index(name="p50_etapas")
    )
    top_avg_etapas = avg_etapas.sort_values(["start_year", "p50_etapas"], ascending=[True, False]).groupby("start_year").head(3)
    rows = []
    prev = None
    for _, row in top_avg_etapas.iterrows():
        yr = int(row["start_year"]) if row["start_year"] != prev else " "
        rows.append({"Campaña": yr, "Empresa": row["empresaNEW"], "P50 Etapas": int(row["p50_etapas"])})
        prev = row["start_year"]
    st.write("**Top 3 Empresas por P50 Etapas por Campaña**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Caudales Pico", divider="blue")

    for fluid, metric, label in [
        ("Petrolífero", "Qo_peak", "Caudal Pico Petróleo (m3/d)"),
        ("Gasífero", "Qg_peak", "Caudal Pico Gas (km3/d)"),
    ]:
        base = df_vmut[df_vmut["tipopozoNEW"] == fluid].copy()

        grp = (
            base.groupby(["start_year", "sigla", "empresaNEW"])
            .agg({metric: "max", "cantidad_fracturas": "median", "fracspacing": "median"})
            .reset_index()
        )
        top3 = grp.sort_values(["start_year", metric], ascending=[True, False]).groupby("start_year").head(3)
        rows = []
        prev = None
        for _, row in top3.iterrows():
            yr = int(row["start_year"]) if row["start_year"] != prev else " "
            rows.append({
                "Campaña": yr,
                "Sigla": row["sigla"],
                "Empresa": row["empresaNEW"],
                label: int(row[metric]) if pd.notna(row[metric]) else None,
                "Etapas": int(row["cantidad_fracturas"]) if pd.notna(row["cantidad_fracturas"]) and row["cantidad_fracturas"] > 0 else None,
                "Fracspacing (m)": int(row["fracspacing"]) if pd.notna(row["fracspacing"]) and row["fracspacing"] > 0 else None,
            })
            prev = row["start_year"]
        st.write(f"**{fluid}: Top 3 Pozos con Mayor {label} por Campaña**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        emp = base.groupby(["start_year", "empresaNEW"])[metric].median().reset_index(name="p50")
        top3_emp = emp.sort_values(["start_year", "p50"], ascending=[True, False]).groupby("start_year").head(3)
        rows2 = []
        prev = None
        for _, row in top3_emp.iterrows():
            yr = int(row["start_year"]) if row["start_year"] != prev else " "
            rows2.append({"Campaña": yr, "Empresa": row["empresaNEW"], f"P50 {label}": round(row["p50"], 0)})
            prev = row["start_year"]
        st.write(f"**{fluid}: Top 3 Empresas por P50 {label} por Campaña**")
        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

    st.subheader("Arena Bombeada", divider="blue")

    df_arena_all = df_vmut_dedup[
        (df_vmut_dedup["start_year"] >= 2012) &
        (df_vmut_dedup["arena_total_tn"] > 0) &
        (df_vmut_dedup["arena_total_tn"].notna())
    ].copy()

    tbl_arena = make_yearly_top_table(
        df_arena_all,
        group_cols=["start_year", "sigla", "empresaNEW"],
        agg_dict={"arena_total_tn": "max"},
        sort_col="arena_total_tn",
        ascending=False,
        head_n=3,
        rename_map={"sigla": "Sigla", "empresaNEW": "Empresa", "arena_total_tn": "Máx. Arena Bombeada (tn)"},
    )
    st.write("**Top 3 Pozos con Mayor Arena Bombeada por Campaña**")
    st.dataframe(tbl_arena, use_container_width=True, hide_index=True)

    avg_arena = df_arena_all.groupby(["start_year", "empresaNEW"])["arena_total_tn"].median().reset_index(name="p50_arena")
    top_avg_arena = avg_arena.sort_values(["start_year", "p50_arena"], ascending=[True, False]).groupby("start_year").head(3)
    rows = []
    prev = None
    for _, row in top_avg_arena.iterrows():
        yr = int(row["start_year"]) if row["start_year"] != prev else " "
        rows.append({"Campaña": yr, "Empresa": row["empresaNEW"], "P50 Arena Bombeada (tn)": int(row["p50_arena"])})
        prev = row["start_year"]
    st.write("**Top 3 Empresas por P50 Arena Bombeada por Campaña**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Fracspacing", divider="blue")
    st.caption("Fracspacing = longitud_rama / etapas. Menor valor = completación más agresiva.")

    df_fsp_all = df_vmut_dedup[
        (df_vmut_dedup["fracspacing"].notna()) &
        (df_vmut_dedup["fracspacing"] > 0)
    ].copy()

    for fluid in ["Petrolífero", "Gasífero"]:
        base_fsp = df_fsp_all[df_fsp_all["tipopozoNEW"] == fluid]

        tbl_fsp = make_yearly_top_table(
            base_fsp,
            group_cols=["start_year", "sigla", "empresaNEW"],
            agg_dict={"fracspacing": "min"},
            sort_col="fracspacing",
            ascending=True,
            head_n=3,
            rename_map={"sigla": "Sigla", "empresaNEW": "Empresa", "fracspacing": "Mín. Fracspacing (m)"},
        )
        st.write(f"**{fluid}: Top 3 Pozos con Menor Fracspacing por Campaña**")
        st.dataframe(tbl_fsp, use_container_width=True, hide_index=True)

        p50_fsp = base_fsp.groupby(["start_year", "empresaNEW"])["fracspacing"].median().reset_index(name="p50_fsp")
        top_p50 = p50_fsp.sort_values(["start_year", "p50_fsp"], ascending=[True, True]).groupby("start_year").head(3)
        rows = []
        prev = None
        for _, row in top_p50.iterrows():
            yr = int(row["start_year"]) if row["start_year"] != prev else " "
            rows.append({"Campaña": yr, "Empresa": row["empresaNEW"], "P50 Fracspacing (m)": int(row["p50_fsp"])})
            prev = row["start_year"]
        st.write(f"**{fluid}: Top 3 Empresas por P50 Fracspacing por Campaña**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Propante por Etapa", divider="blue")
    st.caption("Prop x Etapa = arena_total_tn / cantidad_fracturas.")

    df_prop_all = df_vmut[
        (df_vmut["prop_x_etapa"].notna()) &
        (df_vmut["prop_x_etapa"] > 0)
    ].copy()

    for fluid in ["Petrolífero", "Gasífero"]:
        base_prop = df_prop_all[df_prop_all["tipopozoNEW"] == fluid]

        tbl_prop_max = make_yearly_top_table(
            base_prop,
            group_cols=["start_year", "sigla", "empresaNEW"],
            agg_dict={"prop_x_etapa": "max"},
            sort_col="prop_x_etapa",
            ascending=False,
            head_n=3,
            rename_map={"sigla": "Sigla", "empresaNEW": "Empresa", "prop_x_etapa": "Prop x Etapa (tn/etapa)"},
        )
        st.write(f"**{fluid}: Top 3 Pozos con Mayor Prop x Etapa por Campaña**")
        st.dataframe(tbl_prop_max, use_container_width=True, hide_index=True)

        p50_prop = base_prop.groupby(["start_year", "empresaNEW"])["prop_x_etapa"].median().reset_index(name="p50_prop")
        top_p50_prop = p50_prop.sort_values(["start_year", "p50_prop"], ascending=[True, False]).groupby("start_year").head(3)
        rows = []
        prev = None
        for _, row in top_p50_prop.iterrows():
            yr = int(row["start_year"]) if row["start_year"] != prev else " "
            rows.append({"Campaña": yr, "Empresa": row["empresaNEW"], "P50 Prop x Etapa (tn/etapa)": round(row["p50_prop"], 0)})
            prev = row["start_year"]
        st.write(f"**{fluid}: Top 3 Empresas por P50 Prop x Etapa por Campaña**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Agente de Sostén por Volumen Inyectado", divider="blue")

    df_as_all = df_vmut[
        (df_vmut["start_year"] >= 2012) &
        (df_vmut["AS_x_volumen_inyectado"] > 0) &
        (df_vmut["AS_x_volumen_inyectado"].notna())
    ].copy()

    tbl_as = make_yearly_top_table(
        df_as_all,
        group_cols=["start_year", "sigla", "empresaNEW"],
        agg_dict={"AS_x_volumen_inyectado": "max"},
        sort_col="AS_x_volumen_inyectado",
        ascending=False,
        head_n=3,
        rename_map={
            "sigla": "Sigla",
            "empresaNEW": "Empresa",
            "AS_x_volumen_inyectado": "AS x Vol. Inyectado (tn/1000m3)"
        },
    )
    st.write("**Top 3 Pozos con Mayor AS por Volumen Inyectado por Campaña**")
    st.dataframe(tbl_as, use_container_width=True, hide_index=True)

    p50_as = df_as_all.groupby(["start_year", "empresaNEW"])["AS_x_volumen_inyectado"].median().reset_index(name="p50_as")
    top_p50_as = p50_as.sort_values(["start_year", "p50_as"], ascending=[True, False]).groupby("start_year").head(3)
    rows = []
    prev = None
    for _, row in top_p50_as.iterrows():
        yr = int(row["start_year"]) if row["start_year"] != prev else " "
        rows.append({"Campaña": yr, "Empresa": row["empresaNEW"], "P50 AS x Vol. Inyectado (tn/1000m3)": int(row["p50_as"])})
        prev = row["start_year"]
    st.write("**Top 3 Empresas por P50 AS por Volumen Inyectado por Campaña**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Caudal Pico por Etapa", divider="blue")

    df_qe_all = df_vmut[df_vmut["start_year"] > 2012].copy()

    for fluid, metric, label in [
        ("Petrolífero", "Qo_peak_x_etapa", "Qo Pico x Etapa (m3/d/etapa)"),
        ("Gasífero", "Qg_peak_x_etapa", "Qg Pico x Etapa (km3/d/etapa)"),
    ]:
        base_qe = df_qe_all[
            (df_qe_all["tipopozoNEW"] == fluid) &
            (df_qe_all[metric].notna()) &
            (df_qe_all[metric] > 0)
        ]

        grp_qe = (
            base_qe.groupby(["start_year", "sigla", "empresaNEW"])
            .agg({metric: "max", "cantidad_fracturas": "median", "fracspacing": "median"})
            .reset_index()
        )
        top3_qe = grp_qe.sort_values(["start_year", metric], ascending=[True, False]).groupby("start_year").head(3)
        rows = []
        prev = None
        for _, row in top3_qe.iterrows():
            yr = int(row["start_year"]) if row["start_year"] != prev else " "
            rows.append({
                "Campaña": yr,
                "Sigla": row["sigla"],
                "Empresa": row["empresaNEW"],
                label: int(row[metric]) if pd.notna(row[metric]) else None,
                "Etapas": int(row["cantidad_fracturas"]) if pd.notna(row["cantidad_fracturas"]) and row["cantidad_fracturas"] > 0 else None,
                "Fracspacing (m)": int(row["fracspacing"]) if pd.notna(row["fracspacing"]) and row["fracspacing"] > 0 else None,
            })
            prev = row["start_year"]
        st.write(f"**{fluid}: Top 3 Pozos con Mayor {label} por Campaña**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        emp_qe = base_qe.groupby(["start_year", "empresaNEW"])[metric].median().reset_index(name="p50")
        top3_emp_qe = emp_qe.sort_values(["start_year", "p50"], ascending=[True, False]).groupby("start_year").head(3)
        rows2 = []
        prev = None
        for _, row in top3_emp_qe.iterrows():
            yr = int(row["start_year"]) if row["start_year"] != prev else " "
            rows2.append({"Campaña": yr, "Empresa": row["empresaNEW"], f"P50 {label}": round(row["p50"], 0)})
            prev = row["start_year"]
        st.write(f"**{fluid}: Top 3 Empresas por P50 {label} por Campaña**")
        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — VISUALIZACIONES COMPARATIVAS
# ════════════════════════════════════════════════════════════════════════════

with tab_comparativas:

    st.subheader("Visualizaciones Comparativas", divider="blue")
    st.caption(
        "Boxplots para comparar distribución, dispersión y outliers por campaña o empresa. "
        "Sirven para distinguir récords aislados de cambios estructurales."
    )

    comp_metric = st.selectbox(
        "Seleccionar métrica comparativa:",
        [
            "Longitud de Rama",
            "Cantidad de Etapas",
            "Arena Total",
            "Fracspacing",
            "Prop x Etapa",
            "Qo Pico",
            "Qg Pico",
            "Qo Pico x Etapa",
            "Qg Pico x Etapa",
        ],
        key="comp_metric"
    )

    compare_mode = st.radio(
        "Comparar por:",
        ["Campaña", "Empresa"],
        horizontal=True,
        key="compare_mode"
    )

    fluid_mode = st.radio(
        "Filtrar fluido:",
        ["Todos", "Petrolífero", "Gasífero"],
        horizontal=True,
        key="fluid_mode"
    )

    by_col = "start_year" if compare_mode == "Campaña" else "empresaNEW"
    fluid_filter = None if fluid_mode == "Todos" else fluid_mode

    metric_map = {
        "Longitud de Rama": (
            df_vmut_dedup[df_vmut_dedup["longitud_rama_horizontal_m"] > 0],
            "longitud_rama_horizontal_m",
            "Longitud de Rama (m)",
        ),
        "Cantidad de Etapas": (
            df_vmut_dedup[df_vmut_dedup["cantidad_fracturas"] > 0],
            "cantidad_fracturas",
            "Cantidad de Etapas",
        ),
        "Arena Total": (
            df_vmut_dedup[df_vmut_dedup["arena_total_tn"] > 0],
            "arena_total_tn",
            "Arena Total (tn)",
        ),
        "Fracspacing": (
            df_vmut_dedup[df_vmut_dedup["fracspacing"] > 0],
            "fracspacing",
            "Fracspacing (m)",
        ),
        "Prop x Etapa": (
            df_vmut[df_vmut["prop_x_etapa"] > 0],
            "prop_x_etapa",
            "Prop x Etapa (tn/etapa)",
        ),
        "Qo Pico": (
            df_vmut[df_vmut["Qo_peak"] > 0],
            "Qo_peak",
            "Qo Pico (m3/d)",
        ),
        "Qg Pico": (
            df_vmut[df_vmut["Qg_peak"] > 0],
            "Qg_peak",
            "Qg Pico (km3/d)",
        ),
        "Qo Pico x Etapa": (
            df_vmut[df_vmut["Qo_peak_x_etapa"] > 0],
            "Qo_peak_x_etapa",
            "Qo Pico x Etapa (m3/d/etapa)",
        ),
        "Qg Pico x Etapa": (
            df_vmut[df_vmut["Qg_peak_x_etapa"] > 0],
            "Qg_peak_x_etapa",
            "Qg Pico x Etapa (km3/d/etapa)",
        ),
    }

    plot_df, metric_col, y_label = metric_map[comp_metric]

    if by_col == "empresaNEW":
        if fluid_filter:
            plot_df = plot_df[plot_df["tipopozoNEW"] == fluid_filter]

        company_metric = plot_df.groupby("empresaNEW")[metric_col].median()

        if metric_col == "fracspacing":
            top_emp = company_metric.nsmallest(top_n).index
        else:
            top_emp = company_metric.nlargest(top_n).index

        plot_df = plot_df[plot_df["empresaNEW"].isin(top_emp)]

    fig = comparative_boxplot(
        plot_df,
        metric_col=metric_col,
        title=f"{comp_metric} — distribución por {'campaña' if by_col == 'start_year' else 'empresa'}",
        y_label=y_label,
        fluid_filter=fluid_filter,
        by=by_col,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("**Cómo leer el boxplot**")
    st.markdown(
        "- La línea dentro de la caja es la mediana.\n"
        "- La caja representa el rango intercuartílico (P25–P75).\n"
        "- Los bigotes muestran el rango típico.\n"
        "- Los puntos aislados representan outliers.\n"
        "- Si toda la caja sube o baja entre campañas, hay un cambio estructural; "
        "si solo cambian algunos puntos, puede ser efecto de pocos pozos."
    )