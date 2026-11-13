"""
Focused time-series inspection of Middlesex Hospital in the Connecticut
HHS hospital-capacity extract.

Reads the original CSV and writes plots, tables, and a report into this
folder. It does not edit the source file or any other project file.
Missing and suppressed values are left blank. Nothing is imputed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE = Path(r"C:\Users\SHREENIDHI\Downloads\export.csv")
OUT = Path(__file__).resolve().parent
PLOTS = OUT / "plots"
TABLES = OUT / "tables"

FIELDS = {
    "beds": "inpatient_beds_7_day_avg",
    "used": "inpatient_beds_used_7_day_avg",
    "covid_adm": "previous_day_admission_adult_covid_confirmed_7_day_sum",
    "icu_beds": "total_icu_beds_7_day_avg",
    "icu_used": "icu_beds_used_7_day_avg",
}
COVERAGE = {
    "beds": "inpatient_beds_7_day_coverage",
    "used": "inpatient_beds_used_7_day_coverage",
    "covid_adm": "previous_day_admission_adult_covid_confirmed_7_day_coverage",
    "icu_beds": "total_icu_beds_7_day_coverage",
    "icu_used": "icu_beds_used_7_day_coverage",
}

NAVY = "#1f4e79"
TEAL = "#0f6f6a"
AMBER = "#b45309"
GRAY = "#6b7280"
RED = "#b91c1c"


def parse_measure(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (numeric values, suppressed mask). Null stays null. No fill."""
    text = series.astype("string").str.strip()
    suppressed = text.eq("-999,999") | text.str.replace(",", "", regex=False).eq("-999999")
    cleaned = text.str.replace(",", "", regex=False)
    values = pd.to_numeric(cleaned, errors="coerce")
    values = values.mask(suppressed | values.eq(-999999))
    return values, suppressed.fillna(False)


def yearly_or_quarterly(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    grouped = frame.groupby(frame["week"].dt.to_period(freq), observed=True)
    rows = []
    for period, g in grouped:
        row = {
            "period": str(period),
            "weeks_in_file": int(len(g)),
            "corrected_weeks": int(g["is_corrected"].sum()),
        }
        for key in ("beds", "used", "icu_beds", "icu_used", "covid_adm"):
            observed = g[key].dropna()
            row[f"{key}_n"] = int(observed.shape[0])
            row[f"{key}_mean"] = float(observed.mean()) if len(observed) else np.nan
            row[f"{key}_suppressed"] = int(g[f"{key}_suppressed"].sum())
            row[f"{key}_missing"] = int(g[key].isna().sum() - g[f"{key}_suppressed"].sum()) if False else int((g[key].isna() & ~g[f"{key}_suppressed"]).sum())
        both = g["occupancy_pct"].dropna()
        row["occupancy_n"] = int(both.shape[0])
        row["occupancy_mean"] = float(both.mean()) if len(both) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def mark_corrected(ax, weeks, corrected):
    flagged = weeks[corrected.to_numpy()]
    if len(flagged) == 0:
        return
    ymin, ymax = ax.get_ylim()
    ax.plot(
        flagged,
        np.full(len(flagged), ymin),
        linestyle="None",
        marker="|",
        color=RED,
        markersize=10,
        label="Corrected week",
    )


def style_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(SOURCE, low_memory=False)
    name_hits = sorted(raw.loc[raw["hospital_name"].astype(str).str.contains("MIDDLESEX", case=False), "hospital_name"].unique())
    facility = raw.loc[raw["hospital_name"].eq("MIDDLESEX HOSPITAL")].copy()
    if facility.empty:
        raise SystemExit(f"MIDDLESEX HOSPITAL not found. Name hits: {name_hits}")

    identity = {
        "hospital_name": sorted(facility["hospital_name"].astype(str).unique()),
        "hospital_pk": sorted(int(x) for x in facility["hospital_pk"].unique()),
        "ccn": sorted(int(x) for x in facility["ccn"].unique()),
        "address": sorted(facility["address"].astype(str).unique()),
        "city": sorted(facility["city"].astype(str).unique()),
        "zip": sorted(int(x) for x in facility["zip"].unique()),
        "state": sorted(facility["state"].astype(str).unique()),
        "subtype": sorted(facility["hospital_subtype"].astype(str).unique()),
        "fips": sorted(int(x) for x in facility["fips_code"].unique()),
        "rows": int(len(facility)),
        "pk_equals_ccn": bool((facility["hospital_pk"] == facility["ccn"]).all()),
    }

    facility["week"] = pd.to_datetime(facility["collection_week"], format="%m/%d/%Y")
    facility = facility.sort_values("week").reset_index(drop=True)
    for key, column in FIELDS.items():
        values, suppressed = parse_measure(facility[column])
        facility[key] = values
        facility[f"{key}_suppressed"] = suppressed
        facility[f"{key}_missing"] = values.isna() & ~suppressed
    for key, column in COVERAGE.items():
        facility[f"{key}_coverage"] = pd.to_numeric(facility[column], errors="coerce")

    facility["occupancy_pct"] = np.where(
        facility["beds"].notna() & facility["used"].notna() & (facility["beds"] > 0),
        100.0 * facility["used"] / facility["beds"],
        np.nan,
    )
    facility["icu_occupancy_pct"] = np.where(
        facility["icu_beds"].notna() & facility["icu_used"].notna() & (facility["icu_beds"] > 0),
        100.0 * facility["icu_used"] / facility["icu_beds"],
        np.nan,
    )
    facility["is_corrected"] = facility["is_corrected"].astype(bool)

    first, last = facility["week"].min(), facility["week"].max()
    full_index = pd.date_range(first, last, freq="W-SUN")
    observed_weeks = pd.DatetimeIndex(facility["week"])
    missing_weeks = full_index.difference(observed_weeks)

    indexed = facility.set_index("week").reindex(full_index)
    indexed.index.name = "week"

    def series_for_plot(column: str) -> pd.Series:
        return indexed[column]

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })

    def finish(fig, ax, path: Path, ylabel: str, note: str):
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Collection week (Sunday starting the reporting week)")
        style_time_axis(ax)
        fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=8, color="#4b5563")
        fig.savefig(path, dpi=140)
        plt.close(fig)

    note = (
        "Middlesex Hospital, hospital_pk 70020. Source: export.csv. "
        "Gaps are weeks with no usable value. Nothing is interpolated. "
        "Red ticks mark weeks HHS flagged is_corrected."
    )

    # 1. Capacity
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(indexed.index, series_for_plot("beds"), color=NAVY, linewidth=1.6, label="Staffed inpatient beds")
    mark_corrected(ax, indexed.index, indexed["is_corrected"].fillna(False))
    ax.set_title("Inpatient bed capacity, Middlesex Hospital")
    ax.legend(frameon=False, loc="upper right")
    finish(fig, ax, PLOTS / "01_inpatient_capacity.png", "Staffed inpatient beds (7-day average)", note)

    # 2. Occupied
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(indexed.index, series_for_plot("used"), color=TEAL, linewidth=1.6, label="Occupied inpatient beds")
    mark_corrected(ax, indexed.index, indexed["is_corrected"].fillna(False))
    ax.set_title("Occupied inpatient beds, Middlesex Hospital")
    ax.legend(frameon=False, loc="upper right")
    finish(fig, ax, PLOTS / "02_occupied_inpatient_beds.png", "Occupied inpatient beds (7-day average)", note)

    # 3. Occupancy percentage
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(indexed.index, series_for_plot("occupancy_pct"), color=NAVY, linewidth=1.6, label="Occupancy")
    ax.axhline(90, color="#d1d5db", linewidth=0.8, linestyle="--")
    mark_corrected(ax, indexed.index, indexed["is_corrected"].fillna(False))
    ax.set_title("Inpatient occupancy, Middlesex Hospital")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, loc="upper right")
    finish(
        fig, ax, PLOTS / "03_inpatient_occupancy_pct.png",
        "Occupancy (occupied / staffed beds, %)",
        note + " Occupancy is blank when either bed field is missing or suppressed.",
    )

    # 4. COVID admissions, suppressed vs missing
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.08},
    )
    observed = indexed["covid_adm"]
    axes[0].plot(indexed.index, observed, color=NAVY, linewidth=1.5, label="Reported admissions")
    mark_corrected(axes[0], indexed.index, indexed["is_corrected"].fillna(False))
    axes[0].set_title("Adult confirmed COVID admissions, Middlesex Hospital")
    axes[0].set_ylabel("Admissions in the week (7-day sum)")
    axes[0].legend(frameon=False, loc="upper right")
    style_time_axis(axes[0])
    axes[0].spines["bottom"].set_visible(False)
    axes[0].tick_params(axis="x", length=0)

    status = np.full(len(indexed), np.nan)
    # 0 missing field, 1 suppressed, 2 reported. Absent facility-weeks stay nan.
    present = indexed["hospital_pk"].notna()
    status = pd.Series(np.nan, index=indexed.index)
    status[present & indexed["covid_adm_missing"].fillna(False)] = 0
    status[present & indexed["covid_adm_suppressed"].fillna(False)] = 1
    status[present & indexed["covid_adm"].notna()] = 2
    colors = {0: GRAY, 1: AMBER, 2: NAVY}
    labels = {0: "Missing", 1: "Suppressed (-999,999)", 2: "Reported number"}
    for code in (2, 1, 0):
        mask = status.eq(code)
        axes[1].scatter(
            indexed.index[mask.to_numpy()],
            np.full(int(mask.sum()), code),
            s=12,
            color=colors[code],
            label=labels[code],
            linewidths=0,
        )
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["Missing", "Suppressed", "Reported"])
    axes[1].set_ylim(-0.6, 2.6)
    style_time_axis(axes[1])
    axes[1].grid(False)
    axes[1].set_xlabel("Collection week (Sunday starting the reporting week)")
    fig.text(0.01, -0.01, note + " Suppressed means HHS replaced a count below 4. It is not zero.", ha="left", va="top", fontsize=8, color="#4b5563")
    fig.savefig(PLOTS / "04_covid_admissions.png", dpi=140)
    plt.close(fig)

    # 5. ICU capacity and use
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(indexed.index, series_for_plot("icu_beds"), color=NAVY, linewidth=1.6, label="Staffed ICU beds")
    ax.plot(indexed.index, series_for_plot("icu_used"), color=TEAL, linewidth=1.6, label="Occupied ICU beds")
    mark_corrected(ax, indexed.index, indexed["is_corrected"].fillna(False))
    ax.set_title("ICU capacity and occupied ICU beds, Middlesex Hospital")
    ax.legend(frameon=False, loc="upper right")
    finish(fig, ax, PLOTS / "05_icu_capacity_and_utilization.png", "ICU beds (7-day average)", note)

    # Supporting figure for the occupancy decomposition
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(indexed.index, series_for_plot("beds"), color=NAVY, linewidth=1.6, label="Staffed inpatient beds")
    ax.plot(indexed.index, series_for_plot("used"), color=TEAL, linewidth=1.6, label="Occupied inpatient beds")
    mark_corrected(ax, indexed.index, indexed["is_corrected"].fillna(False))
    ax.set_title("Capacity and occupancy on the same scale, Middlesex Hospital")
    ax.legend(frameon=False, loc="upper right")
    finish(
        fig, ax, PLOTS / "06_capacity_and_occupied.png",
        "Beds (7-day average)",
        note + " Occupancy falls when the gap between these two lines widens.",
    )

    yearly = yearly_or_quarterly(facility, "Y")
    quarterly = yearly_or_quarterly(facility, "Q")
    yearly.to_csv(TABLES / "yearly_averages.csv", index=False)
    quarterly.to_csv(TABLES / "quarterly_averages.csv", index=False)

    # Quality table
    quality_rows = []
    n = len(facility)
    for key, column in FIELDS.items():
        suppressed_n = int(facility[f"{key}_suppressed"].sum())
        missing_n = int(facility[f"{key}_missing"].sum())
        usable_n = int(facility[key].notna().sum())
        coverage = facility[f"{key}_coverage"]
        quality_rows.append({
            "column": column,
            "rows": n,
            "usable": usable_n,
            "usable_pct": round(100 * usable_n / n, 2),
            "missing": missing_n,
            "missing_pct": round(100 * missing_n / n, 2),
            "suppressed": suppressed_n,
            "suppressed_pct": round(100 * suppressed_n / n, 2),
            "coverage_mean": round(float(coverage.mean()), 2),
            "coverage_eq_7": int((coverage == 7).sum()),
            "coverage_eq_7_pct": round(100 * float((coverage == 7).mean()), 2),
            "coverage_eq_0": int((coverage == 0).sum()),
            "coverage_min": int(coverage.min()),
            "coverage_max": int(coverage.max()),
        })
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(TABLES / "data_quality.csv", index=False)

    # Capacity regimes: sustained level of staffed beds
    beds_obs = facility.loc[facility["beds"].notna(), ["week", "beds", "used"]].copy()
    beds_obs["level"] = (beds_obs["beds"].round(0) / 10).round().mul(10)
    # Change point when the 3-week median shifts by at least 20 beds
    roll = beds_obs["beds"].rolling(4, min_periods=3).median()
    shift = roll.diff().abs() >= 20
    change_weeks = beds_obs.loc[shift.fillna(False), "week"]

    def year_means(year: int) -> dict:
        g = facility.loc[facility["week"].dt.year.eq(year)]
        both = g["occupancy_pct"].dropna()
        return {
            "weeks": int(len(g)),
            "beds_mean": float(g["beds"].mean()),
            "used_mean": float(g["used"].mean()),
            "occ_mean": float(both.mean()) if len(both) else np.nan,
            "beds_n": int(g["beds"].notna().sum()),
            "used_n": int(g["used"].notna().sum()),
            "occ_n": int(len(both)),
        }

    y2020, y2023, y2024 = year_means(2020), year_means(2023), year_means(2024)
    # Counterfactuals use the mean of weeks where both fields exist, so the
    # ratio is the occupancy that those means imply.
    def both_means(year: int) -> tuple[float, float, float]:
        g = facility.loc[facility["week"].dt.year.eq(year) & facility["occupancy_pct"].notna()]
        return float(g["used"].mean()), float(g["beds"].mean()), float(g["occupancy_pct"].mean())

    u20, b20, o20 = both_means(2020)
    u23, b23, o23 = both_means(2023)
    u24, b24, o24 = both_means(2024)

    def period_means(period: str) -> tuple[float, float, float, int]:
        g = quarterly.loc[quarterly["period"].eq(period)].iloc[0]
        return float(g["used_mean"]), float(g["beds_mean"]), float(g["occupancy_mean"]), int(g["occupancy_n"])

    u_q3, b_q3, o_q3, n_q3 = period_means("2023Q3")
    u_q4, b_q4, o_q4, n_q4 = period_means("2023Q4")
    # The step the eye reads as 92% to 66% is 2023 Q3 to 2023 Q4.
    occ_if_only_occupied_changed = 100.0 * u_q4 / b_q3
    occ_if_only_capacity_changed = 100.0 * u_q3 / b_q4
    pp_total = o_q4 - o_q3
    pp_occupied = occ_if_only_occupied_changed - o_q3
    pp_capacity = occ_if_only_capacity_changed - o_q3
    pp_interaction = pp_total - pp_occupied - pp_capacity

    corrected_weeks = facility.loc[facility["is_corrected"], "week"]
    corrected_list = ", ".join(d.strftime("%Y-%m-%d") for d in corrected_weeks) or "none"

    # Largest week-to-week capacity jumps between successive observed weeks
    jump = beds_obs.copy()
    jump["prev_beds"] = jump["beds"].shift(1)
    jump["prev_week"] = jump["week"].shift(1)
    jump["delta"] = jump["beds"] - jump["prev_beds"]
    big_jumps = jump.loc[jump["delta"].abs() >= 20, ["prev_week", "week", "prev_beds", "beds", "delta"]]

    # Coverage value counts
    coverage_counts = {}
    for key, column in COVERAGE.items():
        vc = facility[f"{key}_coverage"].value_counts(dropna=False).sort_index()
        coverage_counts[column] = {str(int(k)) if pd.notna(k) else "null": int(v) for k, v in vc.items()}

    icu_occ = facility["icu_occupancy_pct"].dropna()
    usable_occ = facility["occupancy_pct"].dropna()

    def fmt(x: float, digits: int = 1) -> str:
        if pd.isna(x):
            return "n/a"
        return f"{x:.{digits}f}"

    def fmt_table(frame: pd.DataFrame) -> str:
        show = frame.copy()
        for col in show.columns:
            if pd.api.types.is_float_dtype(show[col]):
                show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
        header = "| " + " | ".join(show.columns) + " |"
        rule = "| " + " | ".join("---" for _ in show.columns) + " |"
        body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in show.itertuples(index=False))
        return "\n".join([header, rule, body])

    yearly_view = yearly.rename(columns={
        "period": "Year",
        "weeks_in_file": "Weeks",
        "beds_mean": "Capacity",
        "beds_n": "Capacity n",
        "used_mean": "Occupied",
        "used_n": "Occupied n",
        "occupancy_mean": "Occupancy %",
        "occupancy_n": "Occupancy n",
        "covid_adm_mean": "COVID admissions",
        "covid_adm_n": "Admissions n",
        "covid_adm_suppressed": "Admissions suppressed",
        "icu_beds_mean": "ICU capacity",
        "icu_used_mean": "ICU occupied",
        "corrected_weeks": "Corrected weeks",
    })[
        ["Year", "Weeks", "Capacity", "Capacity n", "Occupied", "Occupied n",
         "Occupancy %", "Occupancy n", "COVID admissions", "Admissions n",
         "Admissions suppressed", "ICU capacity", "ICU occupied", "Corrected weeks"]
    ]
    quarterly_view = quarterly.rename(columns={
        "period": "Quarter",
        "weeks_in_file": "Weeks",
        "beds_mean": "Capacity",
        "used_mean": "Occupied",
        "occupancy_mean": "Occupancy %",
        "occupancy_n": "n",
        "corrected_weeks": "Corrected",
    })[["Quarter", "Weeks", "Capacity", "Occupied", "Occupancy %", "n", "Corrected"]]

    jump_lines = []
    for rec in big_jumps.itertuples(index=False):
        jump_lines.append(
            f"- {rec.prev_week.date()} to {rec.week.date()}: staffed beds {rec.prev_beds:.1f} to {rec.beds:.1f} ({rec.delta:+.1f})"
        )
    jump_text = "\n".join(jump_lines) if jump_lines else "- No successive observed week changed by 20 or more staffed beds."

    gap_text = ", ".join(d.strftime("%Y-%m-%d") for d in missing_weeks) or "none"

    pk = identity["hospital_pk"][0]
    ccn_stored = identity["ccn"][0]
    report = f"""# Middlesex Hospital time-series inspection

Source file: `{SOURCE}` ({SOURCE.stat().st_size:,} bytes). This folder is a read-only analysis. The original CSV was not edited, and no missing or suppressed value was imputed or interpolated.

The hospital in the file is **MIDDLESEX HOSPITAL**, not the system name “Middlesex Health”. Middlesex Health is the organization; the reporting facility is the Middletown hospital. No other row in this extract has “Middlesex” in the hospital name ({", ".join(name_hits)}).

## Facility identity

| Field | Value |
|---|---|
| Hospital name | {identity["hospital_name"][0]} |
| City | {identity["city"][0]} |
| Address | {identity["address"][0]} |
| State | {identity["state"][0]} |
| Hospital subtype | {identity["subtype"][0]} |
| `hospital_pk` as stored | {pk} |
| `ccn` as stored | {ccn_stored} |
| `hospital_pk` equals `ccn` on every row | {identity["pk_equals_ccn"]} |
| ZIP as stored | {identity["zip"][0]} |
| County FIPS as stored | {identity["fips"][0]} |
| Rows | {identity["rows"]} |
| Distinct names, keys, addresses, cities, subtypes | 1 each |

Every selected row is the same facility: one `hospital_pk`, one CCN, one name, one address, one subtype. `hospital_pk` and `ccn` are the same integer on every row. They are stored as 5-digit integers. A Connecticut CMS certification number is normally 6 digits beginning with 0, so the value on a CMS file would be **070020**. ZIP and county FIPS have the same leading-zero truncation ({identity["zip"][0]} and {identity["fips"][0]} in this file).

## Period

`collection_week` runs from **{first.date()}** to **{last.date()}**. All {identity["rows"]} dates are Sundays. The reporting week starts on that Sunday. There are **{len(missing_weeks)} internal gaps** between the first and last week. Missing Sundays: {gap_text}.

`is_corrected` is true on **{int(facility["is_corrected"].sum())}** of {identity["rows"]} weeks. Those weeks stay in every average and plot. Corrected dates: {corrected_list}.

## A. Time series

Plots are in `plots/`. Lines break where a week is absent, missing, or suppressed. They are not connected across those weeks.

- `plots/01_inpatient_capacity.png` — `inpatient_beds_7_day_avg`
- `plots/02_occupied_inpatient_beds.png` — `inpatient_beds_used_7_day_avg`
- `plots/03_inpatient_occupancy_pct.png` — occupied divided by staffed beds, in percent, only where both values are usable and staffed beds are above 0
- `plots/04_covid_admissions.png` — `previous_day_admission_adult_covid_confirmed_7_day_sum`, with reported, suppressed, and missing weeks marked separately
- `plots/05_icu_capacity_and_utilization.png` — `total_icu_beds_7_day_avg` and `icu_beds_used_7_day_avg`
- `plots/06_capacity_and_occupied.png` — capacity and occupied beds together, for the occupancy explanation

Observed inpatient occupancy ranges from **{fmt(usable_occ.min())}%** to **{fmt(usable_occ.max())}%** (median {fmt(usable_occ.median())}%, n = {len(usable_occ)}). Observed ICU occupancy ranges from **{fmt(icu_occ.min())}%** to **{fmt(icu_occ.max())}%** (median {fmt(icu_occ.median())}%, n = {len(icu_occ)}).

## B. Why occupancy moves

Averages below use observed weeks only. Suppressed and missing weeks are excluded from the mean, not filled in. `n` is the number of weeks that entered the occupancy mean.

### Yearly means

{fmt_table(yearly_view)}

### Quarterly means

{fmt_table(quarterly_view)}

Full numeric tables: `tables/yearly_averages.csv` and `tables/quarterly_averages.csv`.

### The drop from about 92% to about 66%

That pair of figures is real, and it is **not** the 2020 annual mean falling smoothly into 2024. The 2020 annual mean occupancy is **{fmt(o20)}%**, because 2020 Q2 was about 70% while 2020 Q3 and Q4 were about 94% and 92%. The same high band returns in 2022 Q4 (**92.1%**) and 2023 Q1 (**93.2%**). The low band is **2023 Q4 (64.7%)**, **2024 Q1 (66.9%)**, and **2024 Q2 (64.5%)**.

The break is the week of **1 October 2023**. Reported staffed beds move from 168.0 on 24 September to 198.9 on 1 October and 222.0 on 8 October, then stay near 220 through 21 April 2024. The week of 1 October is one of the nine `is_corrected` weeks, but the new level continues for every week after it, so this is not a one-week edit.

| | 2023 Q3 | 2023 Q4 |
|---|---:|---:|
| Weeks with occupancy | {n_q3} | {n_q4} |
| Mean staffed beds | {fmt(b_q3)} | {fmt(b_q4)} |
| Mean occupied beds | {fmt(u_q3)} | {fmt(u_q4)} |
| Mean occupancy | {fmt(o_q3)}% | {fmt(o_q4)}% |

Holding 2023 Q3 capacity fixed, the Q4 occupied-bed level would be **{fmt(occ_if_only_occupied_changed)}%**. That accounts for {fmt(pp_occupied)} percentage points of the {fmt(pp_total)} point change. Holding 2023 Q3 occupied beds fixed, the Q4 capacity would be **{fmt(occ_if_only_capacity_changed)}%**, which is {fmt(pp_capacity)} percentage points. The interaction of the two shifts is {fmt(pp_interaction)} points.

**The decline is primarily a change in reported capacity.** Occupied beds fall by about {fmt(u_q3 - u_q4)} beds. Reported staffed beds rise by about {fmt(b_q4 - b_q3)} beds. The same pattern holds into 2024: mean occupied beds are {fmt(u24)} and mean staffed beds are {fmt(b24)}, against a 2020 mean occupancy of {fmt(o20)}% on {fmt(b20)} staffed beds and {fmt(u20)} occupied beds.

### Did the denominator change substantially?

Yes. Successive observed weeks where staffed inpatient beds jumped by 20 or more:

{jump_text}

Read as levels rather than as every small jump, reported staffed beds sit near 220 for most of 2021, near 168 from January 2022 through September 2023, and near 220 again from October 2023 through April 2024. Spring 2020 is the other unstable stretch: one observed week at 132, then about 230, then a step down toward 170. Occupied beds do not follow those steps. A twin that treats `inpatient_beds_7_day_avg` as a physical bed count will read the steps as beds opening and closing. The file does not say whether the hospital changed its staffed-bed definition.

## C. Data quality

Missing means the row exists and the cell is empty. Suppressed means the cell is the text `-999,999`, HHS’s code for a sum or average below 4. Those are counted separately. Coverage is the number of days in the week that the hospital reported the element (0–7). A coverage of 0 is an explicit non-report; the matching average is then null, and that null is counted as missing below, not as suppression.

{fmt_table(quality.rename(columns={
    "column": "Column",
    "usable": "Usable",
    "usable_pct": "Usable %",
    "missing": "Missing",
    "missing_pct": "Missing %",
    "suppressed": "Suppressed",
    "suppressed_pct": "Suppressed %",
    "coverage_mean": "Mean coverage",
    "coverage_eq_7": "Coverage = 7",
    "coverage_eq_7_pct": "Coverage = 7, %",
    "coverage_eq_0": "Coverage = 0",
})[[
    "Column", "Usable", "Usable %", "Missing", "Missing %", "Suppressed", "Suppressed %",
    "Mean coverage", "Coverage = 7", "Coverage = 7, %", "Coverage = 0",
]])}

Coverage value counts (days reported → number of weeks):

{chr(10).join(f"- `{col}`: {counts}" for col, counts in coverage_counts.items())}

Corrected observations were kept. {int(facility["is_corrected"].sum())} weeks have `is_corrected` true. They are included in the means above and marked with red ticks on the plots.

Internal calendar gaps between {first.date()} and {last.date()}: **{len(missing_weeks)}** ({gap_text}).

No value was imputed, interpolated, or replaced. Suppressed COVID admissions are not plotted as zero or as −999,999.

## D. Assessment for a single-hospital weekly twin

Middlesex Hospital is a usable single-facility weekly series for a historical twin, with the limits below. This inspection does not replace your selection decision.

**Capacity monitoring.** Suitable for watching the reported staffed-bed series. It is usable on {quality.loc[quality.column.eq(FIELDS["beds"]), "usable_pct"].iloc[0]}% of weeks, and coverage is 7 on {quality.loc[quality.column.eq(FIELDS["beds"]), "coverage_eq_7_pct"].iloc[0]}% of weeks. The series has three sustained levels: about 220 beds through 2021, about 168 beds from January 2022 through September 2023, and about 220 beds from October 2023 onward. A monitor has to treat a step between those levels as a change in the reported number, and check it before describing it as beds opened or closed.

**Occupancy forecasting.** Weekly occupancy is observable on {len(usable_occ)} weeks, and it moves within a band for long stretches before a cliff in October 2023. A model of the percentage will treat that cliff as a demand change unless reported capacity is an input, because occupied beds stayed near 140–150. Forecasting occupied beds is the steadier target: the 2022, 2023, and 2024 yearly means are {fmt(float(yearly.loc[yearly.period.eq("2022"), "used_mean"].iloc[0]))}, {fmt(u23)}, and {fmt(u24)}. The low-occupancy regime is the 30 weeks from 2023 Q4 through 2024 Q2. There is no all-cause admission or discharge series to explain why occupied beds move.

**COVID-related demand.** Adult confirmed COVID admissions are usable on {quality.loc[quality.column.eq(FIELDS["covid_adm"]), "usable_pct"].iloc[0]}% of weeks, with {int(quality.loc[quality.column.eq(FIELDS["covid_adm"]), "suppressed"].iloc[0])} suppressed weeks and {int(quality.loc[quality.column.eq(FIELDS["covid_adm"]), "missing"].iloc[0])} missing weeks. The reported series shows the waves, including a peak well above the usual week. Suppressed weeks are counts of 1, 2, or 3, so they are not quiet zeros. Leaving them out understates how often small admission weeks occurred. This field is COVID-confirmed adult admissions only.

**ICU utilization.** ICU capacity is present on every week and is almost flat: 24 staffed beds, except a stretch at 26 during part of 2021. Occupied ICU beds are usable on {quality.loc[quality.column.eq(FIELDS["icu_used"]), "usable_pct"].iloc[0]}% of weeks, with {int(quality.loc[quality.column.eq(FIELDS["icu_used"]), "missing"].iloc[0])} missing weeks and {int(quality.loc[quality.column.eq(FIELDS["icu_used"]), "suppressed"].iloc[0])} suppressed weeks. The missing weeks are concentrated at the start of the series, including all of 2020 Q2. From late 2020 onward the occupied series is continuous and moves between roughly 15 and 24 beds, so the percentage moves between {fmt(icu_occ.min())}% and {fmt(icu_occ.max())}% because the unit is small. That supports a weekly ICU utilization review. It does not support a story of ICU capacity expanding and contracting.

**Capacity what-if simulations.** You can replay a historical week and ask what occupancy would have been at another staffed-bed count, using that week’s occupied beds as the patient load. The October 2023 step is itself that arithmetic: occupied beds stayed near 140–150 while reported capacity went from 168 to 220, and occupancy fell by about 25 points. A scenario should name its baseline, either the long 168-bed plateau or the later 220-bed plateau. This is not a simulation of admissions, length of stay, staffing, or discharge.

### Limits that apply to all five uses

- One Connecticut short-term hospital, weekly, from {first.date()} to {last.date()}.
- 2024 contains {y2024["weeks"]} weeks, ending 21 April 2024, when this reporting requirement was ending.
- Staffed-bed steps are large enough to dominate occupancy even when the patient count is steady.
- `-999,999` censors adult COVID admissions below 4 (12 weeks). Occupied ICU beds are not suppressed at this hospital; 17 of those weeks are missing instead, mostly in 2020.
- There is no all-cause admission, discharge, or length-of-stay series, and no doctor or nurse count.
- {int(facility["is_corrected"].sum())} weeks were adjusted by HHS. They remain in the series and are flagged.
- CCN, ZIP, and FIPS in this file are missing a leading zero.
"""

    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    # Machine-readable identity and decomposition for review
    summary = {
        "hospital_name": identity["hospital_name"][0],
        "hospital_pk_stored": pk,
        "ccn_stored": ccn_stored,
        "ccn_with_leading_zero": f"0{ccn_stored}",
        "rows": identity["rows"],
        "first_week": str(first.date()),
        "last_week": str(last.date()),
        "internal_gaps": int(len(missing_weeks)),
        "corrected_weeks": int(facility["is_corrected"].sum()),
        "occ_2020": o20,
        "occ_2023": o23,
        "occ_2024": o24,
        "beds_2020": b20,
        "beds_2023": b23,
        "beds_2024": b24,
        "used_2020": u20,
        "used_2023": u23,
        "used_2024": u24,
        "occ_2024_if_only_occupied_changed_from_2023": occ_if_only_occupied_changed,
        "occ_2024_if_only_capacity_changed_from_2023": occ_if_only_capacity_changed,
    }
    pd.Series(summary).to_csv(TABLES / "identity_and_decomposition.csv", header=["value"])
    print("rows", identity["rows"], "gaps", len(missing_weeks), "corrected", int(facility["is_corrected"].sum()))
    print("occ", round(o20, 2), round(o23, 2), round(o24, 2))
    print("beds", round(b20, 2), round(b23, 2), round(b24, 2))
    print("used", round(u20, 2), round(u23, 2), round(u24, 2))
    print("jumps", len(big_jumps))


if __name__ == "__main__":
    main()
