"""
hospital_selector.py
=====================
Restricts the PMC dataset to hospitals that are actually usable for the
Digital Twin: Public facilities with complete staffing/resource fields and
total_beds > 0. Private facilities are excluded because they have
systematically missing staffing data in this dataset -- they are not
usable, not because private care is out of scope conceptually.

The project brief fixes the default (and only) selectable facility to
Kamla Nehru Hospital, so this module also validates that it is indeed a
valid selectable facility.
"""

import pandas as pd
import config


def get_selectable_hospitals(pmc_df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of PMC facilities that satisfy the twin's data requirements."""
    df = pmc_df.copy()

    is_public = df["Class : (Public / Private)"].str.strip().str.lower() == "public"
    df = df[is_public]

    for field in config.REQUIRED_FIELDS:
        df = df[df[field].notna()]

    df = df[pd.to_numeric(df["Number of Beds in facility type"], errors="coerce") > 0]

    return df.reset_index(drop=True)


def get_facility(pmc_df: pd.DataFrame, name: str = config.DEFAULT_FACILITY) -> pd.Series:
    """
    Return the row for the requested facility, after validating it is a
    selectable (public, complete-data) facility. Raises if not found/valid.
    """
    selectable = get_selectable_hospitals(pmc_df)
    match = selectable[selectable["Facility Name"].str.strip() == name]
    if match.empty:
        raise ValueError(
            f"'{name}' is not a selectable facility (not public, missing "
            f"required fields, or zero beds)."
        )
    return match.iloc[0]
