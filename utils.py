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

    Parameters
    ----------
    df : pd.DataFrame
        Filtered production data. Must already contain 'tipopozoNEW'
        (call get_fluid_classification first) and 'empresaNEW'.

    Returns
    -------
    pd.DataFrame
        One row per sigla with aggregated metrics.
    """
    if "tipopozoNEW" not in df.columns:
        raise ValueError(
            "Column 'tipopozoNEW' not found. "
            "Call get_fluid_classification(df) before create_summary_dataframe(df)."
        )

    df = df.copy()

    # Guard: sigla must be a regular column, not the index
    if "sigla" not in df.columns:
        if df.index.name == "sigla":
            df = df.reset_index()
        else:
            raise KeyError(
                "La columna 'sigla' no está presente en el DataFrame de entrada. "
                "Verificar que el DataFrame viene de load_and_sort_data() sin modificaciones al índice."
            )

    df["Qo_peak"]    = df.groupby("sigla")["oil_rate"].transform("max")
    df["Qg_peak"]    = df.groupby("sigla")["gas_rate"].transform("max")
    df["start_year"] = df.groupby("sigla")["anio"].transform("min")

    def calculate_eur(group: pd.DataFrame) -> pd.DataFrame:
        group      = group.sort_values("date")
        start_date = group["date"].iloc[0]
        cum_col    = "Np" if group["tipopozoNEW"].iloc[0] == "Petrolífero" else "Gp"
        for col, days in [("EUR_30", 30), ("EUR_90", 90), ("EUR_180", 180)]:
            cutoff     = start_date + relativedelta(days=days)
            group[col] = group.loc[group["date"] <= cutoff, cum_col].max()
        return group

    # ── Pandas 2.x robust groupby+apply ──────────────────────────────────────
    # groupby+apply in pandas ≥ 2.0 may promote the group key ('sigla') into
    # the index instead of keeping it as a column, depending on the pandas
    # minor version and whether include_groups was set.  The block below
    # normalises the result so 'sigla' is always a plain column afterward.
    try:
        # pandas ≥ 2.2: pass include_groups=False to silence FutureWarning,
        # then restore sigla from the group key via reset_index.
        result = df.groupby("sigla", group_keys=True).apply(
            calculate_eur, include_groups=False
        )
    except TypeError:
        # pandas < 2.2: include_groups not supported, fall back silently
        result = df.groupby("sigla", group_keys=False).apply(calculate_eur)

    # Normalise index → always make sigla a regular column
    if isinstance(result.index, pd.MultiIndex):
        # MultiIndex → first level is sigla, second is original row index
        result = result.reset_index(level=0).reset_index(drop=True)
    elif result.index.name == "sigla":
        result = result.reset_index()
    else:
        result = result.reset_index(drop=True)

    # If include_groups=False dropped sigla from columns, restore from index op above
    if "sigla" not in result.columns:
        raise KeyError(
            "No se pudo restaurar la columna 'sigla' tras groupby+apply. "
            f"Columnas disponibles: {list(result.columns)}. "
            "Verificar versión de pandas con: import pandas; print(pandas.__version__)"
        )

    df = result

    # ── Aggregation ───────────────────────────────────────────────────────────
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