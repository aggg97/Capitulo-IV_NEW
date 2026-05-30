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


# ── Unit conversion helpers ───────────────────────────────────────────────────

def to_km3_per_day(m3_per_day: float) -> float:
    """Convert m³/d to km³/d."""
    return m3_per_day / 1_000

def to_mmbbl_per_day(m3_per_day: float) -> float:
    """Convert m³/d to MMBBL/d (million barrels per day)."""
    return (m3_per_day * BARRELS_PER_M3) / 1_000_000

def to_kbbl_per_day(m3_per_day: float) -> float:
    """Convert m³/d to KBBL/d (thousand barrels per day)."""
    return (m3_per_day * BARRELS_PER_M3) / 1_000

def to_bbl_per_day(m3_per_day: float) -> float:
    """Convert m³/d to bbl/d."""
    return m3_per_day * BARRELS_PER_M3

def from_bbl_to_m3(bbl: float) -> float:
    """Convert barrels to m³."""
    return bbl / BARRELS_PER_M3


# ── Safe production rate calculations (division by zero protection) ────────────

def safe_divide(numerator, denominator, fillna_value=0):
    """Safely divide with fallback for zero/null denominator."""
    import numpy as np
    return np.where(denominator > 0, numerator / denominator, fillna_value)

def calculate_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate normalized production rates (oil, gas, water) protecting against
    division by zero. Returns a copy with new columns: gas_rate, oil_rate, water_rate.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: prod_pet, prod_gas, prod_agua, tef.

    Returns
    -------
    pd.DataFrame
        Copy with safe rates added.
    """
    import numpy as np
    df = df.copy()
    df["gas_rate"]   = safe_divide(df["prod_gas"],   df["tef"], fillna_value=np.nan)
    df["oil_rate"]   = safe_divide(df["prod_pet"],   df["tef"], fillna_value=np.nan)
    df["water_rate"] = safe_divide(df["prod_agua"],  df["tef"], fillna_value=np.nan)
    return df

def calculate_gor_wc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate GOR (gas-oil ratio) and water cut with protection against division by zero.
    Returns a copy with new columns: GOR, water_cut_pct.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: prod_gas, prod_pet, prod_agua.

    Returns
    -------
    pd.DataFrame
        Copy with GOR and water_cut_pct added.
    """
    import numpy as np
    df = df.copy()
    df["GOR"] = safe_divide(df["prod_gas"], df["prod_pet"], fillna_value=np.nan)
    total_liquid = df["prod_pet"] + df["prod_agua"]
    df["water_cut_pct"] = safe_divide(df["prod_agua"], total_liquid, fillna_value=0) * 100
    return df


# ── Incremental production & new wells analysis ───────────────────────────────

def calculate_monthly_incremental(monthly_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate month-over-month changes in production rates.
    Useful for detecting organic growth, plateau conditions, and decline acceleration.

    Parameters
    ----------
    monthly_totals : pd.DataFrame
        Must have columns: date, oil_rate, gas_rate (sorted by date).

    Returns
    -------
    pd.DataFrame
        Copy with new columns: oil_rate_change, oil_rate_change_pct,
        gas_rate_change, gas_rate_change_pct.
    """
    import numpy as np
    df = monthly_totals.copy().sort_values("date").reset_index(drop=True)
    df["oil_rate_change"]     = df["oil_rate"].diff()
    df["oil_rate_change_pct"] = (df["oil_rate"].pct_change() * 100).round(2)
    df["gas_rate_change"]     = df["gas_rate"].diff()
    df["gas_rate_change_pct"] = (df["gas_rate"].pct_change() * 100).round(2)
    return df

def calculate_new_wells_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count new well starts by month. Requires well start dates already in data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: sigla, date, anio. Assumes df is already sorted.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, new_wells_count.
    """
    well_first_appearance = (
        df.groupby("sigla")["date"].min()
        .reset_index()
        .rename(columns={"date": "well_start_date"})
    )
    new_wells_monthly = (
        well_first_appearance.groupby(
            well_first_appearance["well_start_date"].dt.to_period("M")
        ).size()
        .reset_index(name="new_wells_count")
    )
    new_wells_monthly["date"] = new_wells_monthly["well_start_date"].dt.to_timestamp()
    return new_wells_monthly[["date", "new_wells_count"]]

def calculate_base_decline_contribution(df: pd.DataFrame, latest_date: str) -> dict:
    """
    Estimate contribution from existing wells (base decline) vs. new wells.
    Compares production from wells existing in previous period vs. current.

    Returns dict with:
    - base_production: rate from wells that existed in previous month
    - total_production: current total rate
    - new_well_contribution: estimate of new well contribution
    - active_wells_current: number of producing wells now
    - active_wells_previous: number of producing wells last period
    - new_wells_added: difference
    """
    from dateutil.relativedelta import relativedelta
    
    latest_date = pd.Timestamp(latest_date)
    previous_date = latest_date - relativedelta(months=1)
    
    # Wells active in current period
    current_wells = set(df[df["date"] == latest_date]["sigla"].unique())
    
    # Wells active in previous period
    previous_wells = set(df[df["date"] == previous_date]["sigla"].unique())
    
    # Wells that existed in previous month (base)
    base_well_set = current_wells & previous_wells
    
    # Production from base wells (current month)
    base_prod = df[
        (df["date"] == latest_date) & 
        (df["sigla"].isin(base_well_set))
    ]["oil_rate"].sum()
    
    # Production from all wells (current month)
    total_prod = df[df["date"] == latest_date]["oil_rate"].sum()
    
    # New well contribution (rough estimate)
    new_contrib = total_prod - base_prod
    
    return {
        "base_production": base_prod,
        "total_production": total_prod,
        "new_well_contribution": new_contrib,
        "active_wells_current": len(current_wells),
        "active_wells_previous": len(previous_wells),
        "new_wells_added": len(current_wells - previous_wells),
    }


# ── Robust YoY calculation ────────────────────────────────────────────────────

def calculate_yoy_metrics(monthly_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate year-over-year growth rates robustly using DateOffset instead of
    brittle integer column comparisons.

    Parameters
    ----------
    monthly_totals : pd.DataFrame
        Must have columns: date, oil_rate, gas_rate (sorted by date).

    Returns
    -------
    pd.DataFrame
        With new columns: oil_rate_prev_year, oil_yoy_pct, gas_rate_prev_year, gas_yoy_pct.
    """
    from dateutil.relativedelta import relativedelta
    import numpy as np
    
    df = monthly_totals.copy().sort_values("date").reset_index(drop=True)
    df["date_prev_year"] = df["date"] + pd.DateOffset(years=-1)
    
    # Merge to get previous year data
    df_prev = df[["date", "oil_rate", "gas_rate"]].rename(
        columns={"date": "date_prev_year", "oil_rate": "oil_rate_prev_year",
                 "gas_rate": "gas_rate_prev_year"}
    )
    
    df = df.merge(df_prev, on="date_prev_year", how="left")
    
    # Calculate YoY percentages safely
    df["oil_yoy_pct"] = (
        np.where(
            df["oil_rate_prev_year"] > 0,
            (df["oil_rate"] - df["oil_rate_prev_year"]) / df["oil_rate_prev_year"] * 100,
            np.nan
        )
    ).round(2)
    
    df["gas_yoy_pct"] = (
        np.where(
            df["gas_rate_prev_year"] > 0,
            (df["gas_rate"] - df["gas_rate_prev_year"]) / df["gas_rate_prev_year"] * 100,
            np.nan
        )
    ).round(2)
    
    return df


# ── Vintage/Cohort productivity analysis ──────────────────────────────────────

def calculate_productivity_by_vintage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate productivity metrics aggregated by well start year (vintage/cohort).
    Includes: count of wells, avg oil/gas rate, EUR estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: sigla, start_year, oil_rate, gas_rate, Np, Gp.

    Returns
    -------
    pd.DataFrame
        Aggregated by start_year with columns: wells_count, avg_oil_rate,
        avg_gas_rate, median_oil_rate, median_gas_rate, total_np, total_gp.
    """
    cohort = (
        df.groupby("start_year").agg(
            wells_count=("sigla", "nunique"),
            avg_oil_rate=("oil_rate", "mean"),
            avg_gas_rate=("gas_rate", "mean"),
            median_oil_rate=("oil_rate", "median"),
            median_gas_rate=("gas_rate", "median"),
            total_np=("Np", "sum"),
            total_gp=("Gp", "sum"),
        )
        .reset_index()
    )
    return cohort


# ── Operator benchmarking ─────────────────────────────────────────────────────

def calculate_operator_metrics(df: pd.DataFrame, latest_date, company_col="empresaNEW") -> pd.DataFrame:
    """
    Calculate per-operator KPIs: active wells, productivity, market share.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered data with columns: empresaNEW (or custom), sigla, oil_rate, gas_rate, date.
    latest_date : Timestamp
        Reference date for filtering.
    company_col : str
        Column name for operator/company.

    Returns
    -------
    pd.DataFrame
        Aggregated by company with: active_wells, total_oil_rate, total_gas_rate,
        avg_oil_per_well, market_share_oil_pct.
    """
    latest = df[df["date"] == latest_date].copy()
    
    operators = (
        latest.groupby(company_col).agg(
            active_wells=("sigla", "nunique"),
            total_oil_rate=("oil_rate", "sum"),
            total_gas_rate=("gas_rate", "sum"),
        )
        .reset_index()
    )
    
    operators["avg_oil_per_well"] = (
        operators["total_oil_rate"] / operators["active_wells"]
    ).round(2)
    
    total_oil = operators["total_oil_rate"].sum()
    operators["market_share_oil_pct"] = (
        (operators["total_oil_rate"] / total_oil * 100).round(2)
    )
    
    return operators.sort_values("total_oil_rate", ascending=False)


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