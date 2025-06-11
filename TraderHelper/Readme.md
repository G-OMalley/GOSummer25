# Natural Gas Market Intelligence Repository

This repository contains the full suite of data and model inputs used to analyze, interpret, and forecast U.S. natural gas fundamentals, prices, and weekly EIA storage changes. It is organized by data source, with final output files concentrated in the `/INFO` directory for centralized access.

---

## 📂 Folder Structure

- **Platts/** – Data from S&P Global Commodity Insights (Platts), including price history, power market fundamentals, and modeled gas balances.
- **EIAApi/** – Direct API-pulled data from the U.S. Energy Information Administration (EIA), such as regional storage totals and changes.
- **Weather/** – Historical and forecasted weather data for key U.S. cities, including temperature anomalies and degree day calculations.
- **Power/** – LMP-based daily power prices by ISO and location, used to infer demand-side stress and generation stack positioning.
- **INFO/** – Cleaned, final-use datasets for modeling and analysis, including historical prices, fundamentals, storage levels, and forecasts.

---

## 📊 Core Datasets (in /INFO)

| Filename                      | Description |
|------------------------------|-------------|
| `PRICES.csv`                 | Daily fixed prices by market component since 2015. Used for basis calculations and historical trend analysis. |
| `HistoricalFOM.csv`          | Monthly First-of-Month fixed prices and basis values for each market component. Used to understand forward price relationships. |
| `CriterionStorageChange.csv` | Daily changes in gas storage for various facilities. Summable and normalizable to compare fill/withdrawal speed over time. |
| `Fundy.csv` & `FundyForecast.csv` | Modeled daily fundamentals (Power, ResCom, Ind, etc.) by region. Primary inputs for EIA change prediction. |
| `CriterionExtra.csv` & `CriterionExtraForecast.csv` | Modeled demand-side data with additional breakdowns. Useful for regional usage and forecast performance. |
| `PlattsPowerFundy.csv`       | ISO-level power fundamentals including fuel mix and peak load. Allows inference of gas usage stack position. |
| `PlattsCONUSFundamentalsHIST.csv` | National gas supply/demand balance. ImpliedStorageChange is helpful for directional storage estimates. |
| `WEATHER.csv` & `WEATHERforecast.csv` | Historical and forecast temperatures by city. Min/Max deltas from 10yr norms are key demand indicators. |
| `PowerPrices.csv`            | Daily Max LMPs by ISO/location. Used to signal real-time stress and price-based demand effects. |
| `EIAchanges.csv`             | Weekly EIA storage change (target for modeling). Contains national and regional values. |
| `EIAtotals.csv`              | Total gas in storage weekly by region. Provides inventory context. |

---

## 🧠 Analytical Approach

This repository supports the construction of predictive models for U.S. gas storage (EIA) changes, as well as broader insights into:
- Natural gas basis valuation
- Seasonal trading patterns
- Weather-normalized demand shifts
- Power sector stress and fuel substitution
- Storage inventory risk assessments

---

## 🤖 AI & Automation Notes

- Column headers and formats have been standardized across files for easy ingestion by machine learning models.
- Weather and power data are regionally mapped to align with EIA storage regions.
- Final targets for modeling reside in `EIAchanges.csv`, with the `Period` column as the anchor.
- All data is intended to be used *as-is* unless otherwise noted; some modeled data may be revised retroactively (e.g., Platts CONUS fundamentals).


