## 🎯 Project Goals: Advanced Market Framing Analysis

This project aims to develop a "Trader Helper" capable of performing advanced market framing by analyzing relationships across natural gas storage, weather, power markets, and price behavior. The key analytical goals are broken down as follows:

### I. Storage Positioning and Historical Reaction 🧊

1.  **Current Storage Assessment:**
    * Determine current working gas in storage by EIA region and total Lower 48 using the latest data from `EIAtotals.csv`.
    * Compare current levels to the 5-year and 10-year averages for the same week, expressing deviations in Bcf and percentage.
2.  **Historical Price Context:**
    * Analyze price behavior in the 2-4 weeks following historical periods when storage levels were similar to the current state in early June.
    * Determine if historical price responses were muted or directional.
    * Investigate correlations or non-linear patterns between inventory tightness and future price movements.
3.  **Regional Imbalances:**
    * Flag EIA regions as "tight" (significantly below historical norms with recent draws/low injections) or "loose" (significantly above average and still injecting heavily).
4.  **Behavioral & Physical Insights:**
    * Assess if current storage positioning suggests physical constraints or behavioral shifts in injection patterns (e.g., slowed injections despite ample space, or aggressive injections into already-full storage).

### II. Storage Change Momentum and Comparative Patterns 🔄

5.  **Injection/Withdrawal Pace:**
    * Compute the rolling 4-week injection/withdrawal total for each EIA region using `EIAchanges.csv`.
    * Compare this pace to prior years that had similar starting storage levels.
6.  **Identify Inflection Points:**
    * Detect if a region has switched from net draw to net inject (or vice-versa).
    * Identify sharp slowdowns or accelerations in the pace of storage changes.
7.  **Historical Pattern Comparison:**
    * Compare current storage behavior against historical periods, such as tight end-of-season years (e.g., 2016, 2022).
    * Determine if large injections coincide with expected low inventories or with abnormally full storage (signaling behavioral shifts).
8.  **Price Responsiveness to Momentum:**
    * Assess if local hub prices reflect recent storage injection/withdrawal momentum in their respective regions.
    * Analyze if basis spreads (e.g., Henry Hub vs. regional hubs) are widening or narrowing in response to regional storage tightness or looseness.

### III. Weather Setup and Demand Risk Outlook 🌡️

9.  **Seasonal Context & Forecast Overview:**
    * Classify the current date by season (summer/winter).
    * Summarize the 15-day temperature and demand (CDD/HDD) forecast across major demand centers using `WEATHERforecast.csv`.
10. **Focus on Demand-Driving Temperatures:**
    * Prioritize above-normal Minimum Temperatures in summer (overnight cooling demand).
    * Prioritize below-normal Maximum Temperatures in winter (all-day heating demand).
11. **Regional Demand Anomalies:**
    * Aggregate CDD/HDD anomalies (forecast vs. normal) by region from `WEATHERforecast.csv` to assess upcoming demand surges or lulls.
12. **Historical Weather Impact on Storage:**
    * Correlate similar historical temperature setups in past Junes (from `WEATHER.csv`) with corresponding bullish or bearish storage surprises (`EIAchanges.csv`).

### IV. Power Market Alignment ⚡

13. **Power & Gas Price Synchronization:**
    * Compare recent trends in Maximum LMPs (from `PowerPrices.csv`) with Henry Hub spot price movements (from `NGPrices.csv` or `PRICES.csv`).
14. **Identify Correlation Breakdowns:**
    * Determine where and when the correlation between power and gas prices is strong versus breaking down.
    * Investigate potential reasons for breakdowns (e.g., renewables shortfall, regional congestion), though direct data for these causes may not be in the primary files.
15. **Historical Divergences & Corrections:**
    * Analyze historical instances of divergences between LMP and natural gas prices and whether they preceded price corrections.

### V. Forward Market Insight (ICE Daily) 📈

16. **Forward Basis Analysis:**
    * Utilize "Mark0" basis levels from `ICE_daily.xlsx - Daily_pricingLIVE.csv`.
    * Compare current basis marks to:
        * 1-year average for the same delivery month (from `HistoricalFOM.csv`).
        * 5-year average for the same delivery month (from `HistoricalFOM.csv`, if applicable).
        * Past month’s trailing spot basis average (calculated from `NGPrices.csv` or `PRICES.csv`).
17. **Fundamental Disconnects in Basis:**
    * Identify forward market components where basis levels appear disconnected from the underlying fundamentals suggested by storage levels, weather forecasts, and other market conditions.