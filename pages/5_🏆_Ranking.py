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
    data_sorted["date"]        = pd.to_datetime(data_sorted["anio"].astype(str) + "-" + data_sorted["mes"].astype(str) + "-1")
    data_sorted["gas_rate"]    = data_sorted["prod_gas"]  / data_sorted["tef"]
    data_sorted["oil_rate"]    = data_sorted["prod_pet"]  / data_sorted["tef"]
    data_sorted["water_rate"]  = data_sorted["prod_agua"] / data_sorted["tef"]
    data_sorted                = data_sorted.sort_values(by=["sigla", "date"], ascending=True)
    data_sorted["empresaNEW"]  = data_sorted["empresa"].replace(COMPANY_REPLACEMENTS)
    data_sorted                = get_fluid_classification(data_sorted)
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

df_frac     = load_frac_data()
summary_df  = create_summary_dataframe(data_filtered)

# Merge fracture + summary
df_merged = (
    pd.merge(df_frac, summary_df, on="sigla", how="outer")
    .drop_duplicates()
)

# Keep only Vaca Muerta Shale
df_vmut = df_merged[
    (df_merged["formprod"]        == "VMUT") &
    (df_merged["sub_tipo_recurso"] == "SHALE")
].copy()

# Derived columns used across multiple sections
df_vmut["fracspacing"]      = df_vmut["longitud_rama_horizontal_m"] / df_vmut["cantidad_fracturas"]
df_vmut["prop_x_etapa"]     = df_vmut["arena_total_tn"] / df_vmut["cantidad_fracturas"]
df_vmut["proppant_intensity"]= df_vmut["arena_total_tn"] / df_vmut["longitud_rama_horizontal_m"]
df_vmut["AS_x_vol"]         = df_vmut["arena_total_tn"] / (df_vmut["agua_inyectada_m3"] / 1000)
df_vmut["Qo_peak_x_etapa"]  = df_vmut["Qo_peak"] / df_vmut["cantidad_fracturas"]
df_vmut["Qg_peak_x_etapa"]  = df_vmut["Qg_peak"] / df_vmut["cantidad_fracturas"]

# Replace inf from divisions
df_vmut = df_vmut.replace([np.inf, -np.inf], np.nan)

# Deduplicated version (one row per well) for completion stats
df_vmut_dedup = df_vmut[df_vmut["longitud_rama_horizontal_m"] > 0].drop_duplicates(subset="sigla")

all_years    = sorted(df_vmut["start_year"].dropna().unique().astype(int), reverse=True)
petro_data   = df_vmut[df_vmut["tipopozoNEW"] == "Petrolífero"]
gas_data     = df_vmut[df_vmut["tipopozoNEW"] == "Gasífero"]


# ── Shared helpers ────────────────────────────────────────────────────────────

def ranking_table(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    display_cols: dict,
    top_n: int,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Groups df by group_col, aggregates metric_col, returns top_n rows
    sorted by metric. display_cols maps original → display column names.
    """
    agg   = {metric_col: "max" if not ascending else "min"}
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
    """
    Dot chart showing the record (max or min) value per year, labeled
    with the well or company name that holds the record.
    """
    if fluid_filter:
        df = df[df["tipopozoNEW"] == fluid_filter]

    agg_func = "min" if ascending else "max"
    yearly   = (
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
) -> go.Figure:
    """
    Line chart showing median metric per year for the top N companies
    (ranked by overall median). One line per company.
    """
    import plotly.express as px
    if fluid_filter:
        df = df[df["tipopozoNEW"] == fluid_filter]

    top_companies = (
        df.groupby("empresaNEW")[metric_col]
        .median()
        .nlargest(top_n_companies)
        .index
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


# ── Year selector & top-N slider (shared across all sections) ─────────────────

st.sidebar.divider()
selected_year = st.sidebar.selectbox("Seleccionar campaña:", all_years)
top_n         = st.sidebar.slider("Top N a mostrar:", min_value=3, max_value=10, value=5)

df_year       = df_vmut[df_vmut["start_year"] == selected_year]
df_year_dedup = df_vmut_dedup[df_vmut_dedup["start_year"] == selected_year]
petro_year    = df_year[df_year["tipopozoNEW"] == "Petrolífero"]
gas_year      = df_year[df_year["tipopozoNEW"] == "Gasífero"]


# ── Section 1: Activity ranking ───────────────────────────────────────────────

st.subheader("Ranking de Mayor Actividad por Empresa", divider="blue")

activity = (
    df_year.groupby(["empresaNEW", "tipopozoNEW"])["sigla"]
    .nunique()
    .reset_index(name="Pozos")
)

for fluid, color in [("Petrolífero", "green"), ("Gasífero", "red")]:
    fluid_act = activity[activity["tipopozoNEW"] == fluid].sort_values("Pozos", ascending=False).head(top_n)
    if not fluid_act.empty:
        st.write(f"**{fluid} — Top {top_n} empresas por pozos enganchados ({selected_year})**")
        st.dataframe(
            fluid_act[["empresaNEW", "Pozos"]].rename(columns={"empresaNEW": "Empresa"}),
            use_container_width=True, hide_index=True,
        )


# ── Section 2: Arm length ─────────────────────────────────────────────────────

st.subheader("Ranking según Longitud de Rama", divider="blue")

# Table — top wells
st.write(f"**Top {top_n} pozos con mayor longitud de rama ({selected_year})**")
st.dataframe(
    ranking_table(
        df_year_dedup, "sigla", "longitud_rama_horizontal_m",
        {"sigla": "Pozo", "empresaNEW": "Empresa", "longitud_rama_horizontal_m": "Longitud Rama (m)"},
        top_n,
    ),
    use_container_width=True, hide_index=True,
)

# Table — top companies (median)
st.write(f"**Top {top_n} empresas por P50 de longitud de rama ({selected_year})**")
st.dataframe(
    df_year_dedup.groupby("empresaNEW")["longitud_rama_horizontal_m"]
    .median().reset_index()
    .rename(columns={"empresaNEW": "Empresa", "longitud_rama_horizontal_m": "P50 Longitud Rama (m)"})
    .sort_values("P50 Longitud Rama (m)", ascending=False).head(top_n),
    use_container_width=True, hide_index=True,
)

# Multi-year charts
st.write("**Evolución multi-año — Record por campaña**")
st.plotly_chart(multi_year_dot_chart(
    df_vmut_dedup, "longitud_rama_horizontal_m", "sigla",
    "Record de Longitud de Rama por Campaña", "Longitud de Rama (m)",
), use_container_width=True)

st.write("**Evolución multi-año — P50 por empresa**")
st.plotly_chart(multi_year_company_line(
    df_vmut_dedup, "longitud_rama_horizontal_m",
    "P50 Longitud de Rama por Empresa y Campaña", "Longitud de Rama (m)",
), use_container_width=True)


# ── Section 3: Fracture stages ────────────────────────────────────────────────

st.subheader("Ranking según Cantidad de Etapas", divider="blue")

st.write(f"**Top {top_n} pozos con mayor cantidad de etapas ({selected_year})**")
st.dataframe(
    ranking_table(
        df_year_dedup, "sigla", "cantidad_fracturas",
        {"sigla": "Pozo", "empresaNEW": "Empresa", "cantidad_fracturas": "Etapas"},
        top_n,
    ),
    use_container_width=True, hide_index=True,
)

st.write(f"**Top {top_n} empresas por P50 de etapas ({selected_year})**")
st.dataframe(
    df_year_dedup.groupby("empresaNEW")["cantidad_fracturas"]
    .median().reset_index()
    .rename(columns={"empresaNEW": "Empresa", "cantidad_fracturas": "P50 Etapas"})
    .sort_values("P50 Etapas", ascending=False).head(top_n),
    use_container_width=True, hide_index=True,
)

st.write("**Evolución multi-año — Record de etapas por campaña**")
st.plotly_chart(multi_year_dot_chart(
    df_vmut_dedup, "cantidad_fracturas", "sigla",
    "Record de Etapas por Campaña", "Cantidad de Etapas",
), use_container_width=True)

st.write("**Evolución multi-año — P50 etapas por empresa**")
st.plotly_chart(multi_year_company_line(
    df_vmut_dedup, "cantidad_fracturas",
    "P50 Etapas por Empresa y Campaña", "Cantidad de Etapas",
), use_container_width=True)


# ── Section 4: Peak rates ─────────────────────────────────────────────────────

st.subheader("Ranking según Caudales Pico", divider="blue")

# Petrolífero
st.write(f"**Petrolífero — Top {top_n} pozos por Qo pico ({selected_year})**")
st.dataframe(
    petro_year.drop_duplicates("sigla")
    .nlargest(top_n, "Qo_peak")[["sigla", "empresaNEW", "Qo_peak", "cantidad_fracturas", "fracspacing"]]
    .rename(columns={
        "sigla": "Pozo", "empresaNEW": "Empresa",
        "Qo_peak": "Qo Pico (m3/d)", "cantidad_fracturas": "Etapas",
        "fracspacing": "Fracspacing (m)",
    }),
    use_container_width=True, hide_index=True,
)

st.write(f"**Petrolífero — Top {top_n} empresas por P50 Qo pico ({selected_year})**")
st.dataframe(
    petro_year.groupby("empresaNEW")["Qo_peak"].median().reset_index()
    .rename(columns={"empresaNEW": "Empresa", "Qo_peak": "P50 Qo Pico (m3/d)"})
    .sort_values("P50 Qo Pico (m3/d)", ascending=False).head(top_n),
    use_container_width=True, hide_index=True,
)

# Gasífero
st.write(f"**Gasífero — Top {top_n} pozos por Qg pico ({selected_year})**")
st.dataframe(
    gas_year.drop_duplicates("sigla")
    .nlargest(top_n, "Qg_peak")[["sigla", "empresaNEW", "Qg_peak", "cantidad_fracturas", "fracspacing"]]
    .rename(columns={
        "sigla": "Pozo", "empresaNEW": "Empresa",
        "Qg_peak": "Qg Pico (km3/d)", "cantidad_fracturas": "Etapas",
        "fracspacing": "Fracspacing (m)",
    }),
    use_container_width=True, hide_index=True,
)

st.write(f"**Gasífero — Top {top_n} empresas por P50 Qg pico ({selected_year})**")
st.dataframe(
    gas_year.groupby("empresaNEW")["Qg_peak"].median().reset_index()
    .rename(columns={"empresaNEW": "Empresa", "Qg_peak": "P50 Qg Pico (km3/d)"})
    .sort_values("P50 Qg Pico (km3/d)", ascending=False).head(top_n),
    use_container_width=True, hide_index=True,
)

# Multi-year
st.write("**Evolución multi-año — Record Qo pico por campaña**")
st.plotly_chart(multi_year_dot_chart(
    df_vmut, "Qo_peak", "sigla",
    "Record de Qo Pico por Campaña", "Qo Pico (m3/d)", "Petrolífero",
), use_container_width=True)

st.write("**Evolución multi-año — Record Qg pico por campaña**")
st.plotly_chart(multi_year_dot_chart(
    df_vmut, "Qg_peak", "sigla",
    "Record de Qg Pico por Campaña", "Qg Pico (km3/d)", "Gasífero",
), use_container_width=True)

st.write("**Evolución multi-año — P50 Qo pico por empresa**")
st.plotly_chart(multi_year_company_line(
    df_vmut, "Qo_peak",
    "P50 Qo Pico por Empresa y Campaña", "Qo Pico (m3/d)", "Petrolífero",
), use_container_width=True)

st.write("**Evolución multi-año — P50 Qg pico por empresa**")
st.plotly_chart(multi_year_company_line(
    df_vmut, "Qg_peak",
    "P50 Qg Pico por Empresa y Campaña", "Qg Pico (km3/d)", "Gasífero",
), use_container_width=True)


# ── Section 5: Proppant ───────────────────────────────────────────────────────

st.subheader("Ranking según Arena Bombeada", divider="blue")

df_arena = df_vmut_dedup[df_vmut_dedup["arena_total_tn"] > 0]
df_arena_year = df_arena[df_arena["start_year"] == selected_year]

st.write(f"**Top {top_n} pozos por arena total bombeada ({selected_year})**")
st.dataframe(
    ranking_table(
        df_arena_year, "sigla", "arena_total_tn",
        {"sigla": "Pozo", "empresaNEW": "Empresa", "arena_total_tn": "Arena Total (tn)"},
        top_n,
    ),
    use_container_width=True, hide_index=True,
)

st.write(f"**Top {top_n} empresas por P50 arena bombeada ({selected_year})**")
st.dataframe(
    df_arena_year.groupby("empresaNEW")["arena_total_tn"].median().reset_index()
    .rename(columns={"empresaNEW": "Empresa", "arena_total_tn": "P50 Arena Total (tn)"})
    .sort_values("P50 Arena Total (tn)", ascending=False).head(top_n),
    use_container_width=True, hide_index=True,
)

st.write("**Evolución multi-año — Record de arena por campaña**")
st.plotly_chart(multi_year_dot_chart(
    df_arena, "arena_total_tn", "sigla",
    "Record de Arena Bombeada por Campaña", "Arena Total (tn)",
), use_container_width=True)

st.write("**Evolución multi-año — P50 arena por empresa**")
st.plotly_chart(multi_year_company_line(
    df_arena, "arena_total_tn",
    "P50 Arena Bombeada por Empresa y Campaña", "Arena Total (tn)",
), use_container_width=True)


# ── Section 6: Fracspacing ────────────────────────────────────────────────────

st.subheader("Ranking según Fracspacing", divider="blue")
st.caption("Fracspacing = longitud_rama / etapas. Menor fracspacing = completación más agresiva.")

df_fsp      = df_vmut_dedup[df_vmut_dedup["fracspacing"] > 0]
df_fsp_year = df_fsp[df_fsp["start_year"] == selected_year]

for fluid in ["Petrolífero", "Gasífero"]:
    fluid_fsp = df_fsp_year[df_fsp_year["tipopozoNEW"] == fluid]
    st.write(f"**{fluid} — Top {top_n} pozos con menor fracspacing ({selected_year})**")
    st.dataframe(
        ranking_table(
            fluid_fsp, "sigla", "fracspacing",
            {"sigla": "Pozo", "empresaNEW": "Empresa", "fracspacing": "Fracspacing (m)"},
            top_n, ascending=True,
        ),
        use_container_width=True, hide_index=True,
    )

st.write("**Evolución multi-año — Mínimo fracspacing por campaña (Petrolífero)**")
st.plotly_chart(multi_year_dot_chart(
    df_fsp, "fracspacing", "sigla",
    "Mínimo Fracspacing por Campaña — Petrolífero", "Fracspacing (m)", "Petrolífero", ascending=True,
), use_container_width=True)

st.write("**Evolución multi-año — Mínimo fracspacing por campaña (Gasífero)**")
st.plotly_chart(multi_year_dot_chart(
    df_fsp, "fracspacing", "sigla",
    "Mínimo Fracspacing por Campaña — Gasífero", "Fracspacing (m)", "Gasífero", ascending=True,
), use_container_width=True)


# ── Section 7: Proppant per stage ─────────────────────────────────────────────

st.subheader("Ranking según Propante por Etapa", divider="blue")
st.caption("Prop x Etapa = arena_total_tn / etapas.")

df_prop      = df_vmut[df_vmut["prop_x_etapa"] > 0]
df_prop_year = df_prop[df_prop["start_year"] == selected_year]

for fluid in ["Petrolífero", "Gasífero"]:
    fluid_prop = df_prop_year[df_prop_year["tipopozoNEW"] == fluid]
    st.write(f"**{fluid} — Top {top_n} pozos con mayor prop x etapa ({selected_year})**")
    st.dataframe(
        ranking_table(
            fluid_prop, "sigla", "prop_x_etapa",
            {"sigla": "Pozo", "empresaNEW": "Empresa", "prop_x_etapa": "Prop x Etapa (tn/etapa)"},
            top_n,
        ),
        use_container_width=True, hide_index=True,
    )

st.write("**Evolución multi-año — P50 propante por etapa por empresa (Petrolífero)**")
st.plotly_chart(multi_year_company_line(
    df_prop, "prop_x_etapa",
    "P50 Propante por Etapa — Petrolífero", "Prop x Etapa (tn/etapa)", "Petrolífero",
), use_container_width=True)

st.write("**Evolución multi-año — P50 propante por etapa por empresa (Gasífero)**")
st.plotly_chart(multi_year_company_line(
    df_prop, "prop_x_etapa",
    "P50 Propante por Etapa — Gasífero", "Prop x Etapa (tn/etapa)", "Gasífero",
), use_container_width=True)


# ── Section 8: Peak rate per stage ───────────────────────────────────────────

st.subheader("Ranking según Caudal Pico por Etapa", divider="blue")

df_qetapa      = df_vmut[df_vmut["start_year"] > 2012]
df_qetapa_year = df_qetapa[df_qetapa["start_year"] == selected_year]

for fluid, metric, label in [
    ("Petrolífero", "Qo_peak_x_etapa", "Qo Pico x Etapa (m3/d/etapa)"),
    ("Gasífero",    "Qg_peak_x_etapa", "Qg Pico x Etapa (km3/d/etapa)"),
]:
    fluid_q = df_qetapa_year[df_qetapa_year["tipopozoNEW"] == fluid].dropna(subset=[metric])
    st.write(f"**{fluid} — Top {top_n} pozos por {label} ({selected_year})**")
    st.dataframe(
        ranking_table(
            fluid_q, "sigla", metric,
            {"sigla": "Pozo", "empresaNEW": "Empresa", metric: label,
             "cantidad_fracturas": "Etapas", "fracspacing": "Fracspacing (m)"},
            top_n,
        ),
        use_container_width=True, hide_index=True,
    )

st.write("**Evolución multi-año — Record Qo pico x etapa por campaña**")
st.plotly_chart(multi_year_dot_chart(
    df_qetapa.dropna(subset=["Qo_peak_x_etapa"]),
    "Qo_peak_x_etapa", "sigla",
    "Record Qo Pico x Etapa por Campaña", "Qo Pico x Etapa (m3/d/etapa)", "Petrolífero",
), use_container_width=True)

st.write("**Evolución multi-año — Record Qg pico x etapa por campaña**")
st.plotly_chart(multi_year_dot_chart(
    df_qetapa.dropna(subset=["Qg_peak_x_etapa"]),
    "Qg_peak_x_etapa", "sigla",
    "Record Qg Pico x Etapa por Campaña", "Qg Pico x Etapa (km3/d/etapa)", "Gasífero",
), use_container_width=True)