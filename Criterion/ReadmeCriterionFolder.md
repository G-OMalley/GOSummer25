# Criterion Folder – TraderHelper

> **Mission**: Deliver clean, analysis-ready fundamentals for U.S. natural-gas trading—from CONUS down to individual pipeline segments—so that humans *and* automated agents can spot market-moving signals fast, retrain models, and stay ahead of the curve.

---

## 1 Why Criterion Exists
Criterion is the data-harvesting nerve-center of the private **TraderHelper** repo.  
It taps a Postgres database (or local CSV backups) and produces regional balances, LNG feed-gas, storage deltas, pipeline flows vs operationally available capacity (OA), and every other series listed in `all_database_tickers_and_descriptions.csv`.

These outputs power:

* Exploratory notebooks & dashboards  
* Feature pipes for ML models (weekly EIA, monthly FoM, RL agents)  
* Real-time anomaly alerts (coming soon)  

---

## 2 Folder Layout (⌂ = this folder)
~~~text
Criterion/⌂
├─ CriterionOrchestrator.py         # One-click updater
├─ UpdateCriterionStorage.py        # Storage deltas (net injections / withdrawals)
├─ UpdateCriterionLNG.py            # Historical + forward LNG feed-gas
├─ UpdateCriterionHenryFlows.py     # Henry Hub pipeline flows & OA
├─ UpdateCriterionLocs.py           # Location metadata refresh
├─ UpdateAndForecastFundy.py        # Big Fundy engine (actuals + forecasts)
├─ UpdateCriterionNuclear.py        # Utility for nuclear-gen tickers
├─ database_tables_list.csv         # Master mapping (core fundamentals)
├─ CriterionExtra_tables_list.csv   # Extra fundamentals mapping
├─ all_database_tickers_and_descriptions.csv  # “Full menu” of available series
├─ NuclearPairs.csv                 # Plant ↔ ticker helper
└─ INFO/                            # All CSV outputs drop here
~~~

---

## 3 Quick-Start (Ad-hoc runs)

1. **Clone & env**

   ~~~bash
   git clone <private-repo-url>
   cd TraderHelper/Criterion
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r ../requirements.txt                  # repo-root reqs
   ~~~

2. **Create `.env`** (in *this* folder or next to each script)

   ~~~env
   DB_USER=xxx              # Optional – leave blank to run in offline/CSV mode
   DB_PASSWORD=xxx
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=criterion
   ~~~

3. **Run everything**

   ~~~bash
   python CriterionOrchestrator.py
   ~~~

   Fresh CSVs will appear under `INFO/`.

> *Tip*: Each script can also be called directly—handy while prototyping.

---

## 4 Script Matrix

| Script | Purpose | Key Inputs | Main Outputs (→ INFO/) |
|--------|---------|-----------|-------------------------|
| `UpdateCriterionStorage.py` | Net storage change per facility & region (daily) | Postgres view `flows_schedule` | `CriterionStorageChange.csv` |
| `UpdateCriterionLNG.py` | LNG feed-gas actuals & 60-day forecast | LNG ticker list (hard-coded) | `CriterionLNGHist.csv`, `CriterionLNGForecast.csv` |
| `UpdateCriterionHenryFlows.py` | Henry Hub pipelines – scheduled vs OA | `locs_list.csv`, DB flows | `CriterionHenryFlows.csv` |
| `UpdateAndForecastFundy.py` | **Big engine** – pulls every ticker in mapping files, pivots to wide table, computes regional balances, and builds ARIMA-style forecasts | `database_tables_list.csv`, `CriterionExtra_tables_list.csv`, Postgres function (see below) | `Fundy.csv`, `FundyForecast.csv`, `CriterionExtra.csv`, `CriterionExtraForecast.csv` |
| `UpdateCriterionLocs.py` | Refresh location metadata, flag orphans | DB metadata tables | `locs_list.csv` (in-place) |
| `UpdateNuclear.py` | Quick nuclear-generation sanity test | Nuclear ticker | Stdout only |

---

## 5 Postgres Helper Function

All series pull through one stored procedure:

~~~sql
SELECT *
FROM data_series.fin_json_to_excel_tickers(
    ticker_array   => ARRAY['TICKER1','TICKER2'],
    start_date     => '2010-01-01',
    end_date       => NOW(),
    frequency      => 'daily'      -- accepts daily, weekly, monthly
);
~~~

Expected columns:

| date | ticker | value | unit |
|------|--------|-------|------|
| 2025-06-10 | LNG.GOM.FEEDGAS | 10.25 | Bcf |

If the DB is offline, scripts fall back to the latest CSVs in `INFO/`.

---

## 6 Data-Flow Snapshot

~~~mermaid
flowchart TD
    subgraph Update_Run
        Orchestrator -->|1| Storage[UpdateCriterionStorage]
        Orchestrator -->|2| LNG[UpdateCriterionLNG]
        Orchestrator -->|3| Fundy[UpdateAndForecastFundy]
    end
    Storage --> INFO["INFO/ CSV vault"]
    LNG --> INFO
    Fundy --> INFO
~~~

*Change scope freely—the mermaid above simply mirrors the current call order.*

---

## 7 Agents & Automation

* **Refresh** – An AI agent can schedule the orchestrator (e.g., daily at 05:00 ET) and watch the `INFO/` folder for missing or zero-row files.  
* **Anomaly detection** – Flag spikes > ±3 σ vs 30-day mean, or storage numbers that break physical limits.  
* **Trend surfacing** – Auto-rank top movers (flow-minus-OA %, regional balance swings) and push to Slack.  
* **Model retraining hooks** – When LNG feed-gas forecast error > x %, trigger a notebook that refits the ARIMA modules inside `UpdateAndForecastFundy.py`.

---

## 8 Testing & Validation (roadmap)

| Check | Why it matters | Proposed tool |
|-------|----------------|---------------|
| **Row count drift** | Catch missing days | `pytest` + `pandas.testing` |
| **Schema assert** | Columns haven’t disappeared | `pytest-schema` |
| **Value sanity** | No negative LNG feed-gas | Custom assert helpers |

Run `pytest` before every PR merge.

---

## 9 Contribution & Licensing

This repo is **private**.  
Feel free to branch and PR; commit messages should follow Conventional Commits (`feat:`, `fix:`, `refactor:`…).  
No external redistribution without approval.