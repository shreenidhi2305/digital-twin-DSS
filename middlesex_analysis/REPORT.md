# Middlesex Hospital time-series inspection

Source file: `C:\Users\SHREENIDHI\Downloads\export.csv` (4,020,461 bytes). This folder is a read-only analysis. The original CSV was not edited, and no missing or suppressed value was imputed or interpolated.

The hospital in the file is **MIDDLESEX HOSPITAL**, not the system name “Middlesex Health”. Middlesex Health is the organization; the reporting facility is the Middletown hospital. No other row in this extract has “Middlesex” in the hospital name (MIDDLESEX HOSPITAL).

## Facility identity

| Field | Value |
|---|---|
| Hospital name | MIDDLESEX HOSPITAL |
| City | MIDDLETOWN |
| Address | 28 CRESCENT ST |
| State | CT |
| Hospital subtype | Short Term |
| `hospital_pk` as stored | 70020 |
| `ccn` as stored | 70020 |
| `hospital_pk` equals `ccn` on every row | True |
| ZIP as stored | 6457 |
| County FIPS as stored | 9007 |
| Rows | 215 |
| Distinct names, keys, addresses, cities, subtypes | 1 each |

Every selected row is the same facility: one `hospital_pk`, one CCN, one name, one address, one subtype. `hospital_pk` and `ccn` are the same integer on every row. They are stored as 5-digit integers. A Connecticut CMS certification number is normally 6 digits beginning with 0, so the value on a CMS file would be **070020**. ZIP and county FIPS have the same leading-zero truncation (6457 and 9007 in this file).

## Period

`collection_week` runs from **2020-03-15** to **2024-04-21**. All 215 dates are Sundays. The reporting week starts on that Sunday. There are **0 internal gaps** between the first and last week. Missing Sundays: none.

`is_corrected` is true on **9** of 215 weeks. Those weeks stay in every average and plot. Corrected dates: 2022-05-08, 2022-09-25, 2022-10-30, 2023-01-08, 2023-02-05, 2023-03-12, 2023-04-09, 2023-05-07, 2023-10-01.

## A. Time series

Plots are in `plots/`. Lines break where a week is absent, missing, or suppressed. They are not connected across those weeks.

- `plots/01_inpatient_capacity.png` — `inpatient_beds_7_day_avg`
- `plots/02_occupied_inpatient_beds.png` — `inpatient_beds_used_7_day_avg`
- `plots/03_inpatient_occupancy_pct.png` — occupied divided by staffed beds, in percent, only where both values are usable and staffed beds are above 0
- `plots/04_covid_admissions.png` — `previous_day_admission_adult_covid_confirmed_7_day_sum`, with reported, suppressed, and missing weeks marked separately
- `plots/05_icu_capacity_and_utilization.png` — `total_icu_beds_7_day_avg` and `icu_beds_used_7_day_avg`
- `plots/06_capacity_and_occupied.png` — capacity and occupied beds together, for the occupancy explanation

Observed inpatient occupancy ranges from **58.5%** to **100.0%** (median 83.5%, n = 212). Observed ICU occupancy ranges from **56.5%** to **98.8%** (median 85.5%, n = 198).

## B. Why occupancy moves

Averages below use observed weeks only. Suppressed and missing weeks are excluded from the mean, not filled in. `n` is the number of weeks that entered the occupancy mean.

### Yearly means

| Year | Weeks | Capacity | Capacity n | Occupied | Occupied n | Occupancy % | Occupancy n | COVID admissions | Admissions n | Admissions suppressed | ICU capacity | ICU occupied | Corrected weeks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 42 | 192.6 | 39 | 160.3 | 39 | 85.1 | 39 | 15.1 | 19 | 6 | 24.0 | 20.1 | 0 |
| 2021 | 52 | 220.3 | 52 | 173.6 | 52 | 78.8 | 52 | 18.9 | 51 | 1 | 25.1 | 21.2 | 0 |
| 2022 | 52 | 168.8 | 52 | 148.0 | 52 | 87.6 | 52 | 37.6 | 52 | 0 | 24.0 | 20.3 | 3 |
| 2023 | 53 | 181.6 | 53 | 148.6 | 53 | 83.2 | 53 | 33.5 | 53 | 0 | 24.0 | 20.7 | 6 |
| 2024 | 16 | 220.1 | 16 | 146.3 | 16 | 66.5 | 16 | 7.3 | 11 | 5 | 24.0 | 20.2 | 0 |

### Quarterly means

| Quarter | Weeks | Capacity | Occupied | Occupancy % | n | Corrected |
| --- | --- | --- | --- | --- | --- | --- |
| 2020Q1 | 3 |  |  |  | 0 | 0 |
| 2020Q2 | 13 | 222.1 | 152.3 | 69.6 | 13 | 0 |
| 2020Q3 | 13 | 168.4 | 157.5 | 94.0 | 13 | 0 |
| 2020Q4 | 13 | 187.3 | 171.1 | 91.7 | 13 | 0 |
| 2021Q1 | 13 | 221.0 | 169.2 | 76.6 | 13 | 0 |
| 2021Q2 | 13 | 221.0 | 167.8 | 75.9 | 13 | 0 |
| 2021Q3 | 13 | 221.0 | 174.9 | 79.1 | 13 | 0 |
| 2021Q4 | 13 | 218.2 | 182.3 | 83.6 | 13 | 0 |
| 2022Q1 | 13 | 170.7 | 151.7 | 88.8 | 13 | 0 |
| 2022Q2 | 13 | 168.0 | 139.9 | 83.3 | 13 | 1 |
| 2022Q3 | 13 | 168.3 | 145.3 | 86.4 | 13 | 1 |
| 2022Q4 | 13 | 168.3 | 155.0 | 92.1 | 13 | 1 |
| 2023Q1 | 13 | 168.7 | 157.2 | 93.2 | 13 | 3 |
| 2023Q2 | 13 | 168.0 | 145.3 | 86.5 | 13 | 2 |
| 2023Q3 | 13 | 168.0 | 151.0 | 89.9 | 13 | 0 |
| 2023Q4 | 14 | 218.8 | 141.3 | 64.7 | 14 | 1 |
| 2024Q1 | 13 | 220.1 | 147.3 | 66.9 | 13 | 0 |
| 2024Q2 | 3 | 220.0 | 142.0 | 64.5 | 3 | 0 |

Full numeric tables: `tables/yearly_averages.csv` and `tables/quarterly_averages.csv`.

### The drop from about 92% to about 66%

That pair of figures is real, and it is **not** the 2020 annual mean falling smoothly into 2024. The 2020 annual mean occupancy is **85.1%**, because 2020 Q2 was about 70% while 2020 Q3 and Q4 were about 94% and 92%. The same high band returns in 2022 Q4 (**92.1%**) and 2023 Q1 (**93.2%**). The low band is **2023 Q4 (64.7%)**, **2024 Q1 (66.9%)**, and **2024 Q2 (64.5%)**.

The break is the week of **1 October 2023**. Reported staffed beds move from 168.0 on 24 September to 198.9 on 1 October and 222.0 on 8 October, then stay near 220 through 21 April 2024. The week of 1 October is one of the nine `is_corrected` weeks, but the new level continues for every week after it, so this is not a one-week edit.

| | 2023 Q3 | 2023 Q4 |
|---|---:|---:|
| Weeks with occupancy | 13 | 14 |
| Mean staffed beds | 168.0 | 218.8 |
| Mean occupied beds | 151.0 | 141.3 |
| Mean occupancy | 89.9% | 64.7% |

Holding 2023 Q3 capacity fixed, the Q4 occupied-bed level would be **84.1%**. That accounts for -5.7 percentage points of the -25.2 point change. Holding 2023 Q3 occupied beds fixed, the Q4 capacity would be **69.0%**, which is -20.9 percentage points. The interaction of the two shifts is 1.4 points.

**The decline is primarily a change in reported capacity.** Occupied beds fall by about 9.6 beds. Reported staffed beds rise by about 50.8 beds. The same pattern holds into 2024: mean occupied beds are 146.3 and mean staffed beds are 220.1, against a 2020 mean occupancy of 85.1% on 192.6 staffed beds and 160.3 occupied beds.

### Did the denominator change substantially?

Yes. Successive observed weeks where staffed inpatient beds jumped by 20 or more:

- 2020-04-05 to 2020-04-12: staffed beds 132.0 to 212.0 (+80.0)
- 2020-06-28 to 2020-07-05: staffed beds 232.0 to 211.4 (-20.6)
- 2020-07-05 to 2020-07-12: staffed beds 211.4 to 170.8 (-40.6)
- 2021-12-19 to 2021-12-26: staffed beds 221.0 to 184.9 (-36.1)
- 2023-09-24 to 2023-10-01: staffed beds 168.0 to 198.9 (+30.9)
- 2023-10-01 to 2023-10-08: staffed beds 198.9 to 222.0 (+23.1)

Read as levels rather than as every small jump, reported staffed beds sit near 220 for most of 2021, near 168 from January 2022 through September 2023, and near 220 again from October 2023 through April 2024. Spring 2020 is the other unstable stretch: one observed week at 132, then about 230, then a step down toward 170. Occupied beds do not follow those steps. A twin that treats `inpatient_beds_7_day_avg` as a physical bed count will read the steps as beds opening and closing. The file does not say whether the hospital changed its staffed-bed definition.

## C. Data quality

Missing means the row exists and the cell is empty. Suppressed means the cell is the text `-999,999`, HHS’s code for a sum or average below 4. Those are counted separately. Coverage is the number of days in the week that the hospital reported the element (0–7). A coverage of 0 is an explicit non-report; the matching average is then null, and that null is counted as missing below, not as suppression.

| Column | Usable | Usable % | Missing | Missing % | Suppressed | Suppressed % | Mean coverage | Coverage = 7 | Coverage = 7, % | Coverage = 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inpatient_beds_7_day_avg | 212 | 98.6 | 3 | 1.4 | 0 | 0.0 | 6.8 | 204 | 94.9 | 3 |
| inpatient_beds_used_7_day_avg | 212 | 98.6 | 3 | 1.4 | 0 | 0.0 | 6.8 | 204 | 94.9 | 3 |
| previous_day_admission_adult_covid_confirmed_7_day_sum | 186 | 86.5 | 17 | 7.9 | 12 | 5.6 | 6.4 | 192 | 89.3 | 17 |
| total_icu_beds_7_day_avg | 215 | 100.0 | 0 | 0.0 | 0 | 0.0 | 6.9 | 207 | 96.3 | 0 |
| icu_beds_used_7_day_avg | 198 | 92.1 | 17 | 7.9 | 0 | 0.0 | 6.3 | 191 | 88.8 | 17 |

Coverage value counts (days reported → number of weeks):

- `inpatient_beds_7_day_coverage`: {'0': 3, '1': 1, '4': 1, '5': 4, '6': 2, '7': 204}
- `inpatient_beds_used_7_day_coverage`: {'0': 3, '1': 1, '4': 1, '5': 4, '6': 2, '7': 204}
- `previous_day_admission_adult_covid_confirmed_7_day_coverage`: {'0': 17, '3': 2, '5': 3, '6': 1, '7': 192}
- `total_icu_beds_7_day_coverage`: {'3': 1, '5': 4, '6': 3, '7': 207}
- `icu_beds_used_7_day_coverage`: {'0': 17, '1': 2, '5': 3, '6': 2, '7': 191}

Corrected observations were kept. 9 weeks have `is_corrected` true. They are included in the means above and marked with red ticks on the plots.

Internal calendar gaps between 2020-03-15 and 2024-04-21: **0** (none).

No value was imputed, interpolated, or replaced. Suppressed COVID admissions are not plotted as zero or as −999,999.

## D. Assessment for a single-hospital weekly twin

Middlesex Hospital is a usable single-facility weekly series for a historical twin, with the limits below. This inspection does not replace your selection decision.

**Capacity monitoring.** Suitable for watching the reported staffed-bed series. It is usable on 98.6% of weeks, and coverage is 7 on 94.88% of weeks. The series has three sustained levels: about 220 beds through 2021, about 168 beds from January 2022 through September 2023, and about 220 beds from October 2023 onward. A monitor has to treat a step between those levels as a change in the reported number, and check it before describing it as beds opened or closed.

**Occupancy forecasting.** Weekly occupancy is observable on 212 weeks, and it moves within a band for long stretches before a cliff in October 2023. A model of the percentage will treat that cliff as a demand change unless reported capacity is an input, because occupied beds stayed near 140–150. Forecasting occupied beds is the steadier target: the 2022, 2023, and 2024 yearly means are 148.0, 148.6, and 146.3. The low-occupancy regime is the 30 weeks from 2023 Q4 through 2024 Q2. There is no all-cause admission or discharge series to explain why occupied beds move.

**COVID-related demand.** Adult confirmed COVID admissions are usable on 86.51% of weeks, with 12 suppressed weeks and 17 missing weeks. The reported series shows the waves, including a peak well above the usual week. Suppressed weeks are counts of 1, 2, or 3, so they are not quiet zeros. Leaving them out understates how often small admission weeks occurred. This field is COVID-confirmed adult admissions only.

**ICU utilization.** ICU capacity is present on every week and is almost flat: 24 staffed beds, except a stretch at 26 during part of 2021. Occupied ICU beds are usable on 92.09% of weeks, with 17 missing weeks and 0 suppressed weeks. The missing weeks are concentrated at the start of the series, including all of 2020 Q2. From late 2020 onward the occupied series is continuous and moves between roughly 15 and 24 beds, so the percentage moves between 56.5% and 98.8% because the unit is small. That supports a weekly ICU utilization review. It does not support a story of ICU capacity expanding and contracting.

**Capacity what-if simulations.** You can replay a historical week and ask what occupancy would have been at another staffed-bed count, using that week’s occupied beds as the patient load. The October 2023 step is itself that arithmetic: occupied beds stayed near 140–150 while reported capacity went from 168 to 220, and occupancy fell by about 25 points. A scenario should name its baseline, either the long 168-bed plateau or the later 220-bed plateau. This is not a simulation of admissions, length of stay, staffing, or discharge.

### Limits that apply to all five uses

- One Connecticut short-term hospital, weekly, from 2020-03-15 to 2024-04-21.
- 2024 contains 16 weeks, ending 21 April 2024, when this reporting requirement was ending.
- Staffed-bed steps are large enough to dominate occupancy even when the patient count is steady.
- `-999,999` censors adult COVID admissions below 4 (12 weeks). Occupied ICU beds are not suppressed at this hospital; 17 of those weeks are missing instead, mostly in 2020.
- There is no all-cause admission, discharge, or length-of-stay series, and no doctor or nurse count.
- 9 weeks were adjusted by HHS. They remain in the series and are flagged.
- CCN, ZIP, and FIPS in this file are missing a leading zero.
