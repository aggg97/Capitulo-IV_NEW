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

st.header(f":blue[Reporte Extensivo de Completación y Producción en Vaca Muerta]")
image = Image.open("Vaca Muerta rig.png")
st.sidebar.image(image)

st.sidebar.caption("")
st.sidebar.caption(
    "Nota: Para excluir los pozos clasificados como 'Otro tipo', "
    "se crea una nueva columna que utiliza la definición de fluido basada "
    "en el criterio de GOR según McCain. Esto permite reclasificar estos pozos como "
    "'Gasíferos' o 'Petrolíferos' de manera más precisa"
)
image_mccain = Image.open("McCain.png")
st.sidebar.image(image_mccain)


# ── Data preparation ──────────────────────────────────────────────────────────

data_filtered = data_sorted[data_sorted["tef"] > 0]

# Load fracture data via shared utility (cached)
df_frac = load_frac_data()

# Per-well summary via shared utility (requires tipopozoNEW — already added above)
summary_df = create_summary_dataframe(data_filtered)

# Merge frac + summary
df_merged = (
    pd.merge(df_frac, summary_df, on="sigla", how="outer")
    .drop_duplicates()
)


# ── Derived columns used throughout the page ──────────────────────────────────

df_merged["prod_total"]     = df_merged["Np"].fillna(0) + df_merged["Gp"].fillna(0)
df_merged["sin_datos_frac"] = df_merged["id_base_fractura_adjiv"].isna()
df_merged["anio_inicio"]    = pd.to_datetime(df_merged["date"]).dt.year

campos_criticos = ["longitud_rama_horizontal_m", "cantidad_fracturas", "arena_total_tn"]
df_merged["score_calidad"] = (
    df_merged[campos_criticos].notna().sum(axis=1) / len(campos_criticos) * 100
).round(1)


# ── Global KPIs ───────────────────────────────────────────────────────────────

st.subheader("Diagnóstico de Calidad de Datos por Empresa", divider="blue")

st.info("""
Esta sección muestra dónde faltan datos de fractura en los pozos más relevantes 
de Vaca Muerta.

El foco está en producción, no en cantidad de pozos, para evidenciar qué empresas 
tienen información crítica incompleta que puede afectar análisis y rankings.
""")

_df_dedup        = df_merged.drop_duplicates("sigla")
total_pozos_g    = _df_dedup["sigla"].nunique()
pozos_sin_frac_g = int(_df_dedup["sin_datos_frac"].sum())
prod_total_g     = df_merged["prod_total"].sum()
prod_sin_frac_g  = df_merged[df_merged["sin_datos_frac"]]["prod_total"].sum()
pct_prod_g       = prod_sin_frac_g / prod_total_g * 100 if prod_total_g else 0
score_medio_g    = _df_dedup["score_calidad"].mean()

st.subheader("Resumen Global", divider="grey")
_c1, _c2, _c3, _c4, _c5 = st.columns(5)
_c1.metric("Total Pozos",         f"{total_pozos_g:,}")
_c2.metric("Sin Datos Fractura",  f"{pozos_sin_frac_g:,}",
           delta=f"-{pozos_sin_frac_g/total_pozos_g*100:.1f}% del total", delta_color="inverse")
_c3.metric("Producción Total",    f"{prod_total_g:,.0f}")
_c4.metric("Prod. sin Fractura",  f"{prod_sin_frac_g:,.0f}",
           delta=f"-{pct_prod_g:.1f}% del total", delta_color="inverse")
_c5.metric("Score Calidad Medio", f"{score_medio_g:.1f} / 100")

st.divider()


# ── Ranking by missing-data impact ────────────────────────────────────────────

st.subheader("Ranking Data Management: Impacto por Producción sin Datos de Fractura")

_sin_frac_stats = (
    df_merged[df_merged["sin_datos_frac"]]
    .groupby("empresaNEW")
    .agg(prod_sin_frac=("prod_total", "sum"), pozos_sin_frac=("sigla", "nunique"))
    .reset_index()
)
ranking_dm = (
    df_merged.groupby("empresaNEW")
    .agg(prod_total=("prod_total", "sum"), pozos_total=("sigla", "nunique"))
    .reset_index()
    .merge(_sin_frac_stats, on="empresaNEW", how="left")
)
ranking_dm[["prod_sin_frac", "pozos_sin_frac"]] = ranking_dm[["prod_sin_frac", "pozos_sin_frac"]].fillna(0)
ranking_dm["pct_incompleto"] = (ranking_dm["prod_sin_frac"] / ranking_dm["prod_total"]) * 100
ranking_dm = ranking_dm.sort_values("prod_sin_frac", ascending=False)

ranking_dm["prod_total_fmt"]     = ranking_dm["prod_total"].map("{:,.0f}".format)
ranking_dm["prod_sin_frac_fmt"]  = ranking_dm["prod_sin_frac"].map("{:,.0f}".format)
ranking_dm["pct_incompleto_fmt"] = ranking_dm["pct_incompleto"].map("{:.1f}%".format)
ranking_dm["pozos_sin_frac_fmt"] = ranking_dm["pozos_sin_frac"].astype(int).map("{:,}".format)

st.dataframe(
    ranking_dm[["empresaNEW", "prod_total_fmt", "prod_sin_frac_fmt",
                "pct_incompleto_fmt", "pozos_sin_frac_fmt"]].rename(columns={
        "empresaNEW":          "Empresa",
        "prod_total_fmt":      "Prod. Total",
        "prod_sin_frac_fmt":   "Prod. sin Fractura",
        "pct_incompleto_fmt":  "% Incompleto",
        "pozos_sin_frac_fmt":  "Pozos sin Fractura",
    }),
    use_container_width=True,
    hide_index=True,
)

# Bubble chart
fig_dm = px.scatter(
    ranking_dm,
    x="pct_incompleto",
    y="prod_total",
    size="prod_sin_frac",
    size_max=60,
    color="pct_incompleto",
    color_continuous_scale="RdYlGn_r",
    range_color=[0, 100],
    hover_name="empresaNEW",
    hover_data={
        "prod_total":     ":,.0f",
        "prod_sin_frac":  ":,.0f",
        "pct_incompleto": ":.1f",
        "pozos_total":    True,
        "pozos_sin_frac": True,
    },
    text="empresaNEW",
    title="Mapa de Riesgo: Producción Total vs % Datos Incompletos",
    labels={
        "pct_incompleto": "% Producción sin datos de fractura",
        "prod_total":     "Producción Total",
        "prod_sin_frac":  "Prod. sin datos de fractura",
    },
)
fig_dm.update_traces(
    textposition="top center",
    textfont=dict(size=10),
    marker=dict(line=dict(width=1, color="white")),
)
fig_dm.update_layout(
    template="plotly_white",
    xaxis_title="% Producción sin datos de fractura",
    yaxis_title="Producción Total",
    yaxis_tickformat=",",
    coloraxis_colorbar=dict(title="% Incompleto", ticksuffix="%"),
)
fig_dm.add_vline(x=50, line_dash="dash", line_color="orange",
                 annotation_text="50% umbral", annotation_position="top right")
fig_dm.add_vline(x=80, line_dash="dash", line_color="red",
                 annotation_text="80% crítico", annotation_position="top right")
st.plotly_chart(fig_dm, use_container_width=True)


# ── Heatmap: empresa × año ────────────────────────────────────────────────────

st.subheader("Evolución Temporal de Datos Incompletos por Empresa", divider="grey")
st.caption("Porcentaje de pozos sin datos de fractura por empresa y año. Verde = completo. Rojo = crítico.")

pivot_temporal = (
    df_merged.groupby(["empresaNEW", "anio_inicio"])["sin_datos_frac"]
    .mean()
    .mul(100)
    .round(1)
    .unstack(fill_value=None)
)
pivot_temporal = pivot_temporal.loc[
    pivot_temporal.mean(axis=1).sort_values(ascending=False).index
]

fig_heat = go.Figure(data=go.Heatmap(
    z=pivot_temporal.values,
    x=pivot_temporal.columns.astype(str).tolist(),
    y=pivot_temporal.index.tolist(),
    colorscale="RdYlGn_r",
    zmin=0,
    zmax=100,
    text=pivot_temporal.map(lambda v: f"{v:.0f}%" if pd.notna(v) else "N/D").values,
    texttemplate="%{text}",
    textfont=dict(size=10),
    hoverongaps=False,
    colorbar=dict(title="% Incompleto", ticksuffix="%"),
))
fig_heat.update_layout(
    template="plotly_white",
    title="% Pozos sin Datos de Fractura — Empresa × Año",
    xaxis_title="Año",
    yaxis_title="Empresa",
    height=max(350, 30 * len(pivot_temporal)),
)
st.plotly_chart(fig_heat, use_container_width=True)


# ── Score de calidad por formación ────────────────────────────────────────────

st.subheader("Score de Calidad de Datos por Formación", divider="grey")
st.caption("Score promedio (0–100) según completitud de: longitud de rama, cantidad de fracturas y arena total.")

score_form = (
    df_merged.groupby("formprod")
    .agg(score_medio=("score_calidad", "mean"), pozos=("sigla", "nunique"))
    .reset_index()
    .sort_values("score_medio", ascending=True)
)
score_form["score_medio"] = score_form["score_medio"].round(1)
score_form["color"] = score_form["score_medio"].apply(
    lambda s: "#1E8449" if s >= 70 else ("#F39C12" if s >= 40 else "#C0392B")
)

fig_score = go.Figure(go.Bar(
    x=score_form["score_medio"],
    y=score_form["formprod"],
    orientation="h",
    text=score_form["score_medio"].astype(str) + " pts",
    textposition="outside",
    marker_color=score_form["color"],
    customdata=score_form["pozos"],
    hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<br>Pozos: %{customdata}<extra></extra>",
))
fig_score.update_layout(
    template="plotly_white",
    title="Score de Calidad Promedio por Formación",
    xaxis_title="Score de Calidad (0–100)",
    yaxis_title="Formación",
    xaxis_range=[0, 115],
    height=max(300, 35 * len(score_form)),
)
fig_score.add_vline(x=70, line_dash="dot", line_color="#1E8449",
                    annotation_text="Umbral aceptable (70)", annotation_position="top right")
fig_score.add_vline(x=40, line_dash="dot", line_color="#C0392B",
                    annotation_text="Umbral crítico (40)", annotation_position="bottom right")
st.plotly_chart(fig_score, use_container_width=True)


# ── Per-company drilldown ─────────────────────────────────────────────────────

empresa_objetivo = st.selectbox(
    "Seleccionar Empresa",
    sorted(df_merged["empresaNEW"].dropna().unique()),
)

df_emp = df_merged.drop_duplicates("sigla").copy()
df_emp = df_emp[df_emp["empresaNEW"] == empresa_objetivo]

total_pozos    = df_emp["sigla"].nunique()
pozos_sin_frac = df_emp["sin_datos_frac"].sum()
pct            = (pozos_sin_frac / total_pozos) * 100 if total_pozos > 0 else 0
prod_emp       = df_emp["prod_total"].sum()
prod_sin_emp   = df_emp[df_emp["sin_datos_frac"]]["prod_total"].sum()
pct_prod_emp   = (prod_sin_emp / prod_emp * 100) if prod_emp > 0 else 0
score_emp      = df_emp["score_calidad"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Pozos",        total_pozos)
col2.metric("Sin Datos Fractura", int(pozos_sin_frac),
            delta=f"-{pct:.1f}%", delta_color="inverse")
col3.metric("% Incompleto",       f"{pct:.1f}%")
col4.metric("Prod. sin Fractura", f"{prod_sin_emp:,.0f}",
            delta=f"-{pct_prod_emp:.1f}% del total", delta_color="inverse")
col5.metric("Score Calidad",      f"{score_emp:.1f} / 100")

# Breakdown by well type
resumen_tipo = (
    df_emp.groupby("tipopozoNEW")
    .agg(total=("sigla", "count"), sin_frac=("sin_datos_frac", "sum"))
    .reset_index()
)
resumen_tipo["pct"] = (resumen_tipo["sin_frac"] / resumen_tipo["total"]) * 100
resumen_tipo["color"] = resumen_tipo["pct"].apply(
    lambda p: "#1E8449" if p < 40 else ("#F39C12" if p < 70 else "#C0392B")
)

fig_tipo = go.Figure(go.Bar(
    x=resumen_tipo["tipopozoNEW"],
    y=resumen_tipo["pct"],
    text=resumen_tipo["pct"].round(1).astype(str) + "%",
    textposition="outside",
    marker_color=resumen_tipo["color"],
    customdata=resumen_tipo[["total", "sin_frac"]].values,
    hovertemplate=(
        "<b>%{x}</b><br>% incompleto: %{y:.1f}%<br>"
        "Total: %{customdata[0]}<br>Sin fractura: %{customdata[1]}<extra></extra>"
    ),
))
fig_tipo.update_layout(
    template="plotly_white",
    title="Datos Incompletos por Tipo de Pozo",
    yaxis_title="% Incompleto",
    xaxis_title="Tipo de Pozo",
    yaxis_range=[0, 115],
)
st.plotly_chart(fig_tipo, use_container_width=True)

# Temporal evolution for selected company
st.markdown("#### Evolución Temporal de Completitud")

df_emp_full = df_merged[df_merged["empresaNEW"] == empresa_objetivo].copy()
evol_anio = (
    df_emp_full.groupby("anio_inicio")["sin_datos_frac"]
    .agg(total="count", sin_frac="sum")
    .reset_index()
)
evol_anio["pct_incompleto"] = (evol_anio["sin_frac"] / evol_anio["total"] * 100).round(1)
evol_anio["pct_completo"]   = 100 - evol_anio["pct_incompleto"]

fig_evol = go.Figure()
fig_evol.add_trace(go.Bar(
    x=evol_anio["anio_inicio"], y=evol_anio["pct_completo"],
    name="Con datos", marker_color="#1E8449",
    hovertemplate="%{x}: %{y:.1f}% completo<extra></extra>",
))
fig_evol.add_trace(go.Bar(
    x=evol_anio["anio_inicio"], y=evol_anio["pct_incompleto"],
    name="Sin datos", marker_color="#C0392B",
    hovertemplate="%{x}: %{y:.1f}% incompleto<extra></extra>",
))
fig_evol.update_layout(
    template="plotly_white",
    title=f"Completitud de Datos de Fractura — {empresa_objetivo}",
    barmode="stack",
    yaxis_title="% Pozos",
    xaxis_title="Año de Inicio",
    yaxis_range=[0, 110],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_evol, use_container_width=True)

# Wells with missing frac data
with st.expander("Ver pozos sin datos de fractura"):
    st.dataframe(
        df_emp[df_emp["sin_datos_frac"]][["sigla", "tipopozoNEW", "formprod", "score_calidad"]]
        .sort_values("score_calidad")
        .head(20),
        use_container_width=True,
        hide_index=True,
    )