# Natural Gas Trading & Forecasting Project – Summary Notes

## 📊 Project Goal
Develop a forecasting model to predict **natural gas prices or spreads** on a **regional basis**, using a combination of:
- Market components (e.g., prices, flows)
- External drivers (e.g., weather, LNG, nuclear generation)
- ICE futures and basis prices
- Anchored primarily around **Henry Hub**

---

## 🔧 Core Components

### 1. Market Component (Pick One)
You are deciding between two primary modeling bases:
- `Prices` (uncertain which regional or hub price to use)
- `Flows` (pipeline, LNG, or regional demand/supply)

These will be tested for explanatory power using defined **criteria**.

---

### 2. Flows (Per Region)
Track gas flows across:
- Pipeline flows (interstate/intrastate)
- LNG terminals (imports/exports)
- Power burn by region

Used to generate **regional signals** rather than national averages.

---

### 3. External Drivers (Forward Month Forecast)
Forecast drivers include:
- **Weather**: HDD, CDD, anomalies
- **Prices**: Henry Hub, regional hubs, TTF, NBP
- **LNG**: Import/export volumes (US/EU/Asia)
- **Nuclear**: Generation outages or capacity

These are expected to influence forward market signals.

---

### 4. `Mark 0` – ICE Daily Baseline
First version of the model:
- Uses **ICE daily forward prices**
- Tests simple linear or regression-based forecasting logic
- Could be used to trade spreads or outright prices

---

## 📉 Forecast Design Logic

### 5. Historical Price Anchors
You're experimenting with different pricing reference points:
- `EOM` (End of Month)
- `FOM` (First of Month)

Used to test which timing best correlates with next-month outcomes.

---

### 6. Basis of Henry Hub
- Regional price basis = (Regional Price - Henry Hub)
- Looking at **March Basis** as a potential seasonal anchor
- This sets the stage for **spread trading** or **relative forecasting**

---

### 7. Emphasis on: `PER MARKET COMP` (Highlighted)
Modeling should occur **per market component**:
- Per hub (e.g., SoCal, Algonquin, Chicago)
- Per driver (weather, LNG, nuclear)
- Per flow region

Forecasts will not be one-size-fits-all.

---

### 8. Henry Hub as Anchor
Henry Hub appears to be your **benchmark**, and:
- Regional forecasts might be basis to HH
- Spread trades and hedge decisions will revolve around HH vs. other hubs

---

## ✅ External Context & Suggestions

### Recommended Inputs
- **Weather**: NOAA/ECMWF forecasts, HDD/CDD per EIA region
- **Flows**: Genscape, pipeline bulletin boards, LNG schedules
- **Prices**: ICE Futures (daily), S&P Platts, EIA datasets
- **Storage**: EIA weekly storage levels (national + regional)
- **Power Mix**: Nuclear outages, coal vs gas burn from ISO data

---

## 🛠️ Suggested Next Steps

1. **Select Your Core Market Component** (prices or flows)
2. **Align Features to Forecast Month** (e.g., forward weather, flow projections)
3. **Run Correlation / SHAP Analysis** to test explanatory power
4. **Build `Mark 0` Baseline Model**:
   - Predict next month's ICE daily close using linear regression
   - Inputs: weather forecast, LNG volume, recent price trends

---

## 📁 Optional Starter Script
Would you like a Python script that:
- Pulls ICE prices
- Gathers NOAA weather
- Calculates Henry Hub basis
- Forecasts next-month regional prices or spreads?

