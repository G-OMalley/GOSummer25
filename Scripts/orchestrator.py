# orchestrator.py
# Main script to drive the market analysis and generate a consolidated HTML report.

import os
from pathlib import Path
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import market_analyzer

# --- Configuration ---
# PLEASE UPDATE THESE PATHS AND SETTINGS AS NEEDED
BASE_DIR = Path(__file__).resolve().parent
INFO_DIR = BASE_DIR.parent / "INFO"
OUTPUT_DIR = BASE_DIR / "MarketAnalysis_Report_Output"
OUTPUT_CHARTS_SUBDIR = "charts"

PRICES_FILE = INFO_DIR / "PRICES.csv"
ICE_FILE = INFO_DIR / "ICE_daily.xlsx"
HISTORICAL_FOM_FILE = INFO_DIR / "HistoricalFOM.csv"
PRICE_ADMIN_FILE = INFO_DIR / "PriceAdmin.csv"
WEATHER_FILE = INFO_DIR / "WEATHER.csv"
WEATHER_FORECAST_FILE = INFO_DIR / "WEATHERforecast.csv"

# This list controls the order of regions in the final report.
SPECIFIC_REGION_ORDER = [
    "SouthCentral - Henry", "SouthCentral - Perryville", "Southeast",
    "Northeast - Demand", "Northeast - Appalachia", "SouthCentral - Midcon",
    "Midwest - Chicago", "Midwest - Demand", "SouthCentral - TexCoast",
    "SouthCentral - WesTex", "Rockies", "Gulf Coast"
]

TODAY = date.today()

# Configuration for the regional charts
REGIONAL_CHART_START_DATE = (TODAY.replace(day=1) - relativedelta(months=1))
REGIONAL_CHART_END_DATE = (TODAY.replace(day=1) + relativedelta(months=2) - relativedelta(days=1))
REGIONAL_FORWARD_MARK_DATE = REGIONAL_CHART_END_DATE
REGIONAL_PRIOR_YEARS_FOR_AVG_BASIS = [TODAY.year - 1, TODAY.year - 2, TODAY.year - 3]
# Ensure month list is sorted and unique
REGIONAL_MONTH_INDICES_FOR_AVG_BASIS = sorted(list(set([
    (TODAY - relativedelta(months=1)).month, TODAY.month,
    (TODAY + relativedelta(months=1)).month, (TODAY + relativedelta(months=2)).month
])))

# Configuration for the spread grid heatmaps
SPREAD_GRID_NUM_PERIODS = 3
SPREAD_GRID_PERIOD_DAYS = 15

# Optional mapping if names in ICE file differ from PRICES.csv
MARKET_COMPONENT_NAME_MAP_ICE_TO_PRICES = {
    "Consumers": "Michcon", "Eastern Gas-North": "Eastern Gas-South",
    "ENT-STX Map": "NGPL-STX", "EP-Keystone": "Waha", "FGT-CG": "FGT-Z3",
}

# --- Main Orchestration Logic ---
def main():
    print("Orchestrator script starting...")
    print(f"Orchestrator main() function started. Today's date: {TODAY.strftime('%Y-%m-%d')}")
    print(f"Using INFO directory: {INFO_DIR.resolve()}")
    print(f"Output will be saved in: {OUTPUT_DIR.resolve()}")

    output_charts_path = OUTPUT_DIR / OUTPUT_CHARTS_SUBDIR
    output_charts_path.mkdir(parents=True, exist_ok=True)

    # --- Load All Data ---
    prices_df, active_market_components = market_analyzer.load_and_filter_prices(PRICES_FILE, TODAY)
    if prices_df.empty or not active_market_components:
        print("CRITICAL: No active market components found or error loading prices. Exiting.")
        return

    regional_groups, unregioned_details = market_analyzer.load_ice_data_and_group_by_region(
        ICE_FILE, active_market_components, component_name_map=MARKET_COMPONENT_NAME_MAP_ICE_TO_PRICES
    )

    df_fom_historical = market_analyzer.load_historical_fom(HISTORICAL_FOM_FILE)
    component_to_city_symbol_map, city_symbol_to_title_map, df_weather = market_analyzer.load_key_and_weather_data(PRICE_ADMIN_FILE, WEATHER_FILE)

    # --- Validation Step ---
    print("\n--- Validating Component Mapping ---")
    price_admin_components = set(component_to_city_symbol_map.keys())
    missing_components = [comp for comp in active_market_components if comp not in price_admin_components]
    if missing_components:
        print("WARNING: The following active components from PRICES.csv are NOT found in PriceAdmin.csv and will be skipped:")
        for comp in missing_components:
            print(f"  - {comp}")
    else:
        print("Validation successful: All active components are mapped in PriceAdmin.csv.")
    print("-" * 35 + "\n")

    # --- Build HTML Report ---
    html_parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>Consolidated Market Analysis Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; background-color: #f4f4f4; color: #333; }",
        "h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }",
        "h2 { color: #2980b9; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 40px; }",
        "h3 { color: #16a085; margin-top: 30px; text-align: center; }",
        "h4 { color: #8e44ad; margin-top: 25px; border-bottom: 1px dashed #9b59b6; padding-bottom: 5px; }",
        ".region-section, .component-section, .unregioned-section { background-color: #fff; padding: 20px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }",
        ".chart-container, .table-container { margin-bottom: 20px; text-align: center; }",
        "img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top:10px; }",
        "table { width: auto; margin-left: auto; margin-right: auto; border-collapse: collapse; font-size: 0.9em; }",
        "th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }",
        "th { background-color: #ecf0f1; }",
        "caption { font-size: 1.1em; font-weight: bold; margin-bottom: 10px; color: #333; }",
        ".flex-row { display: flex; flex-wrap: wrap; justify-content: space-around; align-items: flex-start; gap: 20px; margin-top: 15px; margin-bottom:15px; }",
        ".flex-item { flex: 1 1 45%; min-width: 300px; box-sizing: border-box; padding: 5px; }",
        ".spread-grid-layout, .monthly-charts-layout { display: flex; flex-wrap: wrap; justify-content: space-around; gap: 15px; margin-top:15px; }",
        ".spread-grid-item, .monthly-chart-item { flex: 1 1 30%; min-width: 300px; }",
        "hr { border: 0; height: 1px; background: #bdc3c7; margin: 40px 0; }",
        "</style></head><body>",
        f"<h1>Consolidated Market Analysis Report - {TODAY.strftime('%B %d, %Y')}</h1>"
    ]
    
    # Consolidate regional and unregioned components into a single list to process
    all_sections = []
    processed_regions = set()
    for region_name in SPECIFIC_REGION_ORDER:
        if region_name in regional_groups and region_name not in processed_regions:
            all_sections.append((region_name, regional_groups[region_name]))
            processed_regions.add(region_name)

    for region_name, components in regional_groups.items():
        if region_name not in processed_regions:
            all_sections.append((region_name, components))
            processed_regions.add(region_name)
    
    if unregioned_details:
        all_sections.append(("Unregioned", unregioned_details))

    # --- Main Loop to Generate All Report Sections ---
    for section_name, components_in_section in all_sections:
        is_unregioned = section_name == "Unregioned"
        html_parts.append(f"<div class='region-section'><h2>{'Unregioned Active Components' if is_unregioned else f'Region: {section_name}'}</h2>")

        if not is_unregioned:
            html_parts.append("<h3>Regional Overview</h3>")
            daily_overlay_img = market_analyzer.generate_regional_daily_overlay_chart(section_name, components_in_section, prices_df, REGIONAL_CHART_START_DATE, REGIONAL_CHART_END_DATE, REGIONAL_FORWARD_MARK_DATE, output_charts_path)
            if daily_overlay_img: html_parts.append(f"<div class='chart-container'><img src='{OUTPUT_CHARTS_SUBDIR}/{daily_overlay_img}' alt='Daily Overlay for {section_name}'></div>")
            
            spread_grids = market_analyzer.generate_regional_avg_spread_grid_heatmaps(section_name, components_in_section, prices_df, TODAY, output_charts_path, SPREAD_GRID_NUM_PERIODS, SPREAD_GRID_PERIOD_DAYS)
            if spread_grids:
                html_parts.append("<div class='spread-grid-layout'>")
                html_parts.extend([f"<div class='spread-grid-item'><img src='{OUTPUT_CHARTS_SUBDIR}/{img}' alt='Spread Grid'></div>" for img in spread_grids])
                html_parts.append("</div>")

            monthly_charts = market_analyzer.generate_regional_historical_monthly_basis(section_name, components_in_section, prices_df, REGIONAL_PRIOR_YEARS_FOR_AVG_BASIS, REGIONAL_MONTH_INDICES_FOR_AVG_BASIS, output_charts_path)
            if monthly_charts:
                html_parts.append("<div class='monthly-charts-layout'>")
                html_parts.extend([f"<div class='monthly-chart-item'><img src='{OUTPUT_CHARTS_SUBDIR}/{img}' alt='Monthly Chart'></div>" for img in monthly_charts])
                html_parts.append("</div>")

        html_parts.append("<h3>Individual Component Details</h3>")
        for comp_detail in components_in_section:
            comp_name = comp_detail['prices_csv_name']
            fom_mark0 = comp_detail['forward_value']
            html_parts.append(f"<div class='component-section'><h4>Component: {comp_name}</h4>")
            
            cash_basis_df = market_analyzer.calculate_individual_monthly_cash_basis_table(prices_df, comp_name)
            html_parts.append("<div class='flex-row'>")
            if cash_basis_df is not None and not cash_basis_df.empty:
                html_parts.append(f"<div class='flex-item table-container'>{market_analyzer.style_individual_heatmap_table(cash_basis_df, 'Heat Map by Month (Yearly Seasonality)', 0)}</div>")
                html_parts.append(f"<div class='flex-item table-container'>{market_analyzer.style_individual_heatmap_table(cash_basis_df, 'Heat Map by Year (Historical Monthly Strength)', 1)}</div>")
            else:
                html_parts.append("<div class='flex-item table-container'><p>No data for cash basis heatmaps.</p></div><div class='flex-item table-container'></div>")
            html_parts.append("</div>")
            
            if not df_fom_historical.empty:
                fom_vs_cash_df = market_analyzer.generate_individual_fom_vs_cash_table(cash_basis_df, df_fom_historical, comp_name, TODAY)
                html_parts.append(f"<div class='table-container'>{market_analyzer.style_individual_fom_vs_cash_table(fom_vs_cash_df, 'First of Month (FoM) vs. Average Cash Basis')}</div>")
            
            html_parts.append("<div class='flex-row'>")
            html_parts.append(f"<div class='flex-item chart-container'>{market_analyzer.generate_individual_basis_history_chart(prices_df, comp_name, fom_mark0, output_charts_path, TODAY)}</div>")
            
            city_symbol = component_to_city_symbol_map.get(comp_name)
            if city_symbol:
                city_title = city_symbol_to_title_map.get(city_symbol, city_symbol)
                scatter_html = market_analyzer.generate_individual_temp_scatter_plot(prices_df, df_weather, comp_name, city_title, city_symbol, output_charts_path, TODAY, weather_forecast_file_path=WEATHER_FORECAST_FILE)
                html_parts.append(f"<div class='flex-item chart-container'>{scatter_html}</div>")
            else:
                html_parts.append(f"<div class='flex-item chart-container'><p>Temperature scatter plot not available for {comp_name} (no city mapping in PriceAdmin.csv).</p></div>")
            
            html_parts.append("</div>") 
            html_parts.append("</div>") 
        html_parts.append("</div><hr>") 

    # --- Finalize and Write Report ---
    html_parts.append("</body></html>")
    final_html_content = "\n".join(html_parts)

    report_file_path = OUTPUT_DIR / "Consolidated_Market_Report.html"
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(final_html_content)
        print(f"\nSuccessfully generated consolidated HTML report: {report_file_path.resolve()}")
        print(f"Supporting charts saved in: {output_charts_path.resolve()}")
    except Exception as e:
        print(f"Error writing HTML output to '{report_file_path}': {e}")


if __name__ == '__main__':
    main()
