"""
utils.py — Shared constants and helpers for the Vaca Muerta dashboard.

Import from any page with:
    from utils import COMPANY_REPLACEMENTS, DATASET_URL, DATASET_FRAC_URL
    from utils import get_fluid_classification, load_frac_data, create_summary_dataframe

Dependency order for callers
─────────────────────────────
1. Load production data         → load_and_sort_data()  (defined in main page)
2. Classify fluid type          → get_fluid_classification(df)
3. Load fracture data           → load_frac_data()
4. Build per-well summary       → create_summary_dataframe(df)
   └─ requires tipopozoNEW column (step 2 must run first)
"""

import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta


# ── Dataset URLs ──────────────────────────────────────────────────────────────

DATASET_URL = (
    "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c"
    "/resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725/download/"
    "produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv"
)

DATASET_FRAC_URL = (
    "http://datos.energia.gob.ar/dataset/71fa2e84-0316-4a1b-af68-7f35e41f58d7"
    "/resource/2280ad92-6ed3-403e-a095-50139863ab0d/download/"
    "datos-de-fractura-de-pozos-de-hidrocarburos-adjunto-iv-actualizacin-diaria.csv"
)


# ── Company name standardisation ──────────────────────────────────────────────

COMPANY_REPLACEMENTS = {
    "PAN AMERICAN ENERGY (SUCURSAL ARGENTINA) LLC": "PAN AMERICAN ENERGY",
    "PAN AMERICAN ENERGY SL":                       "PAN AMERICAN ENERGY",
    "VISTA ENERGY ARGENTINA SAU":                   "VISTA",
    "Vista Oil & Gas Argentina SA":                 "VISTA",
    "VISTA OIL & GAS ARGENTINA SAU":                "VISTA",
    "WINTERSHALL DE ARGENTINA S.A.":                "WINTERSHALL",
    "WINTERSHALL ENERGÍA S.A.":                     "WINTERSHALL",
    "PLUSPETROL S.A.":                              "PLUSPETROL",
    "PLUSPETROL CUENCA NEUQUINA S.R.L.":            "PLUSPETROL",
}


# ── Conversion factors ────────────────────────────────────────────────────────

BARRELS_PER_M3 = 6.28981


# ── Fluid classification (McCain GOR criterion) ───────────────────────────────

def get_fluid_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies each well (sigla) as 'Gasífero' or 'Petrolífero' using the
    McCain GOR criterion (GOR > 3000 → Gasífero), and reclassifies wells
    originally tagged as 'Otro tipo' using this criterion.

    Adds 'tipopozoNEW' column to the returned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: sigla, Np, Gp, Wp, tipopozo.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 'tipopozoNEW' merged in.
    """
    cum = (
        df.groupby("sigla")[["Np", "Gp", "Wp"]]
        .max()
        .reset_index()
    )

    cum["GOR"] = (cum["Gp"] / cum["Np"] * 1000).fillna(100_000)
    cum["WOR"] = (cum["Wp"] / cum["Np"]).fillna(100_000)
    cum["WGR"] = (cum["Wp"] / cum["Gp"] * 1000).fillna(100_000)

    cum["Fluido McCain"] = cum.apply(
        lambda r: "Gasífero" if r["Np"] == 0 or r["GOR"] > 3_000 else "Petrolífero",
        axis=1,
    )

    tipopozo_unique = df[["sigla", "tipopozo"]].drop_duplicates(subset=["sigla"])
    cum = cum.merge(tipopozo_unique, on="sigla", how="left")

    cum["tipopozoNEW"] = cum.apply(
        lambda r: r["Fluido McCain"] if r["tipopozo"] == "Otro tipo" else r["tipopozo"],
        axis=1,
    )

    return df.merge(cum[["sigla", "tipopozoNEW"]], on="sigla", how="left")


# ── Fracture data loader ──────────────────────────────────────────────────────

@st.cache_data
def load_frac_data() -> pd.DataFrame:
    """
    Loads, computes total proppant, and applies quality cut-offs to the
    fracture dataset. Returns one row per fracture job.

    Cut-offs applied
    ────────────────
    - longitud_rama_horizontal_m > 100
    - cantidad_fracturas         > 6
    - arena_total_tn             > 100

    These remove pilot/appraisal wells and data-entry errors that would
    distort completion statistics.
    """
    df = pd.read_csv(DATASET_FRAC_URL)
    df["arena_total_tn"] = df["arena_bombeada_nacional_tn"] + df["arena_bombeada_importada_tn"]
    df = df[
        (df["longitud_rama_horizontal_m"] > 100) &
        (df["cantidad_fracturas"]         > 6)   &
        (df["arena_total_tn"]             > 100)
    ]
    return df


# ── Per-well summary dataframe ────────────────────────────────────────────────

def create_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a per-well summary with peak rates, cumulative production,
    start year, and EUR at 30 / 90 / 180 days.
    Fully vectorized — no groupby+apply to avoid pandas 2.x index promotion bug.
    """
    if "tipopozoNEW" not in df.columns:
        raise ValueError(
            "Column 'tipopozoNEW' not found. "
            "Call get_fluid_classification(df) before create_summary_dataframe(df)."
        )

    df = df.copy().sort_values(["sigla", "date"])

    df["Qo_peak"]    = df.groupby("sigla")["oil_rate"].transform("max")
    df["Qg_peak"]    = df.groupby("sigla")["gas_rate"].transform("max")
    df["start_year"] = df.groupby("sigla")["anio"].transform("min")

    # Fecha de inicio por pozo
    df["_start_date"] = df.groupby("sigla")["date"].transform("first")

    # Tipo de fluido por pozo
    df["_fluid"] = df.groupby("sigla")["tipopozoNEW"].transform("first")

    # EUR vectorizado: para cada intervalo, marcar filas dentro de la ventana
    # y tomar el máximo de Np o Gp según tipo
    for col, days in [("EUR_30", 30), ("EUR_90", 90), ("EUR_180", 180)]:
        within = df["date"] <= (df["_start_date"] + pd.to_timedelta(days, unit="D"))
        # Valor acumulado: Np si petrolífero, Gp si gasífero
        val = df["Np"].where(df["_fluid"] == "Petrolífero", df["Gp"])
        df[col] = (
            val.where(within)
               .groupby(df["sigla"])
               .transform("max")
        )

    df = df.drop(columns=["_start_date", "_fluid"], errors="ignore")

    agg = dict(
        date       =("date",       "first"),
        start_year =("start_year", "first"),
        Np         =("Np",         "max"),
        Gp         =("Gp",         "max"),
        Wp         =("Wp",         "max"),
        Qo_peak    =("Qo_peak",    "max"),
        Qg_peak    =("Qg_peak",    "max"),
        EUR_30     =("EUR_30",     "max"),
        EUR_90     =("EUR_90",     "max"),
        EUR_180    =("EUR_180",    "max"),
    )
    for col in ["empresaNEW", "formprod", "sub_tipo_recurso", "tipopozoNEW"]:
        if col in df.columns:
            agg[col] = (col, "first")

    return df.groupby("sigla").agg(**agg).reset_index()