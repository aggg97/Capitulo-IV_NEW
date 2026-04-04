"""
utils.py — Shared constants and helpers for the Vaca Muerta dashboard.
Import from any page with: from utils import COMPANY_REPLACEMENTS, DATASET_URL, DATASET_FRAC_URL
"""

import pandas as pd


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
        Must contain columns: sigla, Np, Gp, Wp, tipopozo.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 'tipopozoNEW' merged in.
    """
    # Cumulative maxima per well
    cum = (
        df.groupby("sigla")[["Np", "Gp", "Wp"]]
        .max()
        .reset_index()
    )

    # Ratios — fill NaN (division by zero) with large sentinel
    cum["GOR"] = (cum["Gp"] / cum["Np"] * 1000).fillna(100_000)
    cum["WOR"] = (cum["Wp"] / cum["Np"]).fillna(100_000)
    cum["WGR"] = (cum["Wp"] / cum["Gp"] * 1000).fillna(100_000)

    # McCain fluid classification
    cum["Fluido McCain"] = cum.apply(
        lambda r: "Gasífero" if r["Np"] == 0 or r["GOR"] > 3_000 else "Petrolífero",
        axis=1,
    )

    # Merge original tipopozo (one unique row per sigla)
    tipopozo_unique = df[["sigla", "tipopozo"]].drop_duplicates(subset=["sigla"])
    cum = cum.merge(tipopozo_unique, on="sigla", how="left")

    # Reclassify 'Otro tipo' wells using McCain; keep original label otherwise
    cum["tipopozoNEW"] = cum.apply(
        lambda r: r["Fluido McCain"] if r["tipopozo"] == "Otro tipo" else r["tipopozo"],
        axis=1,
    )

    return df.merge(cum[["sigla", "tipopozoNEW"]], on="sigla", how="left")