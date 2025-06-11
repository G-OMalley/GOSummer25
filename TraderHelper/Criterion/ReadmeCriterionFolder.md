# Criterion Folder

## Overview

This folder contains scripts and supporting data files related to interacting with the Criterion database. The primary purpose is to extract, process, and update fundamental energy data, including storage levels, location information, nuclear power output, and broader market fundamentals, which are then used to generate key analytical files like `Fundy.csv`, `FundyForecast.csv`, `CriterionExtra.csv`, `CriterionExtraForecast.csv`, `Nuclear.csv`, and `NuclearForecast.csv` (stored in the main `INFO` directory).  

## Key Data Files (Supporting Files within Criterion Folder)

These CSV files provide metadata or input for the scripts in this folder:

* **`all_database_tickers_and_descriptions.csv`**:
    * **Purpose**: A comprehensive list of all available data tables and tickers within the Criterion database that we have access to. Serves as a reference.
* **`database_tables_list.csv`**:
    * **Purpose**: Specifies the subset of Criterion database tables that are actively used by the `UpdateAndForecastFundy.py` script to create the `Fundy.csv` and `FundyForecast.csv` files.
* **`CriterionExtra_tables_list.csv`**:
    * **Purpose**: Lists other specific tables from the Criterion database that have historical and forecast data. These are used by relevant scripts to generate the `CriterionExtra.csv` and `CriterionExtraForecast.csv` files.
* **`CriterionLOCS.csv`**:
    * **Purpose**: Contains important location identifiers relevant to the natural gas industry. These typically represent points of contact between pipelines and end-users (e.g., delivery points, interconnects). 
* **`NuclearPairs.csv`**:
    * **Purpose**: This file is used as an input for the `UpdateNuclear.py` script, containing mappings or parameters needed to process nuclear power plant data correctly.

## Scripts

The Python scripts in this folder are responsible for interacting with the Criterion database and processing the data:

* **`UpdateCriterionStorage.py`**:
    * **Purpose**: Connects to the Criterion database to fetch and update data on daily changes in natural gas storage levels across various regions in the country.
    * **Inputs**: Uses connection details for the Criterion database. References specific tickers or tables related to storage.
    * **Outputs**: Produces or updates a data file (e.g., `CriterionStorageChange.csv` in the `INFO` directory) containing the latest storage change information.
    * **Key Logic**: Queries storage-related tables, processes date/value information.

* **`UpdateCriterionLocs.py`**:
    * **Purpose**: Checks the Criterion database to identify and update the list of available physical locations (points) relevant to our analysis.
    * **Inputs**: Criterion database connection. May use `CriterionLOCS.csv` as a base or for comparison.
    * **Outputs**: Updates `CriterionLOCS.csv` or another master list of locations if new ones are found or existing ones change status.

* **`UpdateAndForecastFundy.py`**:
    * **Purpose**: This script is central to generating core fundamental data. It connects to the Criterion database, uses the tables listed in `database_tables_list.csv`, processes the data, and creates/updates the `Fundy.csv` (historical fundamentals) and `FundyForecast.csv` (forecasted fundamentals) files.
    * **Inputs**:
        * Criterion database connection.
        * `database_tables_list.csv` (to know which tables to query).
    * **Outputs**:
        * `Fundy.csv` (likely in `INFO/`)
        * `FundyForecast.csv` (likely in `INFO/`)
    * **Key Logic**: Iterates through specified tables, extracts historical and forecast data, aggregates/transforms it as needed, and saves it in a structured format.

* **`UpdateNuclear.py`**:
    * **Purpose**: Fetches and processes data related to nuclear power generation. This data is then used to create/update `Nuclear.csv` (historical nuclear output) and `NuclearForecast.csv` (forecasted nuclear output).
    * **Inputs**:
        * Criterion database connection.
        * `NuclearPairs.csv` (for specific processing parameters or mappings).
    * **Outputs**:
        * `Nuclear.csv` (likely in `INFO/`)
        * `NuclearForecast.csv` (likely in `INFO/`)
    * **Key Logic**: Queries nuclear-related data from Criterion, applies logic based on `NuclearPairs.csv`, and structures the output for historical and forecast datasets.

## Workflow & Dependencies

1.  **Metadata Setup**: Files like `all_database_tickers_and_descriptions.csv`, `database_tables_list.csv`, `CriterionExtra_tables_list.csv`, `CriterionLOCS.csv`, and `NuclearPairs.csv` should be accurate and present as they guide the behavior of the scripts.
2.  **Data Fetching & Processing**:
    * `UpdateCriterionStorage.py` can be run to get the latest storage figures.
    * `UpdateCriterionLocs.py` can be run to ensure location data is current.
    * `UpdateNuclear.py` can be run to get nuclear generation data.
    * `UpdateAndForecastFundy.py` (and similar scripts for `CriterionExtra` if any) can then be run to generate the core analytical CSV files.
3.  **Output Usage**: The CSV files generated by these scripts (presumably in the `INFO` folder) are then consumed by downstream analytical processes or models for the "trader helper."