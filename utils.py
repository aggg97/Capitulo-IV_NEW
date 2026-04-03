"""
utils.py — Shared constants and helpers for the Vaca Muerta dashboard.
Import from any page with: from utils import COMPANY_REPLACEMENTS, DATASET_URL, DATASET_FRAC_URL
"""

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
