# orchestrator.py
# Main script to drive the market analysis and generate a consolidated HTML report.

print("Orchestrator script starting...") # Diagnostic print

import os
from pathlib import Path
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

print("Orchestrator: Basic imports successful.") # Diagnostic print
print("Orchestrator: Attempting to import market_analyzer...") # Diagnostic print
try:
    import market_analyzer # Your new module
    print("Orchestrator: Successfully imported market_analyzer.") # Diagnostic print
except ImportError as e:
    print(f"Orchestrator: FAILED to import market_analyzer. Error: {e}")
    exit() # Stop if import fails
except Exception as e:
    print(f"Orchestrator: An unexpected error occurred during market_analyzer import: {e}")
    exit()


# --- Configuration ---
# PLEASE UPDATE THESE PATHS AND SETTINGS AS NEEDED
# Assumes orchestrator.py is in a 'Scripts' directory, and 'INFO' is a sibling to 'Scripts'
BASE_DIR = Path(__file__).resolve().parent 
INFO_DIR = BASE_DIR.parent / "INFO" 
OUTPUT_DIR = BASE_DIR / "MarketAnalysis_Report_Output" 
OUTPUT_CHARTS_SUBDIR = "charts" 

PRICES_FILE = INFO_DIR / "PRICES.csv"
ICE_FILE = INFO_DIR / "ICE_daily.xlsx"
HISTORICAL_FOM_FILE = INFO_DIR / "HistoricalFOM.csv"
KEY_FILE = INFO_DIR / "KEY.xlsx" 
WEATHER_FILE = INFO_DIR / "WEATHER.csv" 

SPECIFIC_REGION_ORDER = [
    "SouthCentral - Henry",
    "SouthCentral - Perryville",
    "SouthCentral - Midcon",
    "Midwest - Chicago",
    "Midwest - Demand",
    "Northeast - Appalachia",
    "Northeast - Demand", 
    "Southeast",
    "SouthCentral - TexCoast",
    "SouthCentral - WesTex"
]

TODAY = date.today() 

REGIONAL_CHART_START_DATE = (TODAY.replace(day=1) - relativedelta(months=1))
REGIONAL_CHART_END_DATE = (TODAY.replace(day=1) + relativedelta(months=2) - relativedelta(days=1)) 
REGIONAL_FORWARD_MARK_DATE = REGIONAL_CHART_END_DATE 

REGIONAL_PRIOR_YEARS_FOR_AVG_BASIS = [TODAY.year - 1, TODAY.year - 2, TODAY.year - 3]
REGIONAL_MONTH_INDICES_FOR_AVG_BASIS = [
    (TODAY - relativedelta(months=1)).month, TODAY.month,
    (TODAY + relativedelta(months=1)).month, (TODAY + relativedelta(months=2)).month
] 

SPREAD_GRID_NUM_PERIODS = 3
SPREAD_GRID_PERIOD_DAYS = 15

MARKET_COMPONENT_NAME_MAP_ICE_TO_PRICES = {
    "Consumers": "Michcon", "Eastern Gas-North": "Eastern Gas-South",
    "ENT-STX Map": "NGPL-STX", "EP-Keystone": "Waha", "FGT-CG": "FGT-Z3",
}
COMPONENT_NAME_MAP_FOR_ICE_LOOKUP = MARKET_COMPONENT_NAME_MAP_ICE_TO_PRICES


# --- Main Orchestration Logic ---
def main():
    print(f"Orchestrator main() function started. Today's date: {TODAY.strftime('%Y-%m-%d')}") # Changed from "Orchestrator started"
    print(f"Using INFO directory: {INFO_DIR.resolve()}")
    print(f"Output will be saved in: {OUTPUT_DIR.resolve()}")

    output_charts_path = OUTPUT_DIR / OUTPUT_CHARTS_SUBDIR
    output_charts_path.mkdir(parents=True, exist_ok=True)

    prices_df, active_market_components = market_analyzer.load_and_filter_prices(
        PRICES_FILE,
        TODAY
    )
    if prices_df.empty or not active_market_components:
        print("No active market components found or error loading prices. Exiting.")
        return

    regional_groups, unregioned_components_details = market_analyzer.load_ice_data_and_group_by_region(
        ICE_FILE,
        active_market_components,
        component_name_map=COMPONENT_NAME_MAP_FOR_ICE_LOOKUP
    )

    df_fom_historical = market_analyzer.load_historical_fom(HISTORICAL_FOM_FILE)
    component_to_city_map, df_weather = market_analyzer.load_key_and_weather_data(KEY_FILE, WEATHER_FILE)

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

    all_regions_to_process = []
    processed_region_names = set()

    for region_name in SPECIFIC_REGION_ORDER:
        if region_name in regional_groups and region_name not in processed_region_names:
            all_regions_to_process.append((region_name, regional_groups[region_name]))
            processed_region_names.add(region_name)

    for region_name, components in regional_groups.items():
        if region_name not in processed_region_names:
            all_regions_to_process.append((region_name, components))
            processed_region_names.add(region_name)

    for region_name, components_in_this_region in all_regions_to_process:
        html_parts.append(f"<div class='region-section'><h2>Region: {region_name}</h2>")

        html_parts.append("<h3>Regional Overview</h3>")
        daily_overlay_img_name = market_analyzer.generate_regional_daily_overlay_chart(
            region_name, components_in_this_region, prices_df,
            REGIONAL_CHART_START_DATE, REGIONAL_CHART_END_DATE, REGIONAL_FORWARD_MARK_DATE,
            output_charts_path
        )
        if daily_overlay_img_name:
            html_parts.append(f"<div class='chart-container'><img src='{OUTPUT_CHARTS_SUBDIR}/{daily_overlay_img_name}' alt='Daily Overlay for {region_name}'></div>")

        spread_grid_img_names = market_analyzer.generate_regional_avg_spread_grid_heatmaps(
            region_name, components_in_this_region, prices_df, TODAY, output_charts_path,
            SPREAD_GRID_NUM_PERIODS, SPREAD_GRID_PERIOD_DAYS
        )
        if spread_grid_img_names:
            html_parts.append("<div class='spread-grid-layout'>")
            for img_name in spread_grid_img_names:
                html_parts.append(f"<div class='spread-grid-item'><img src='{OUTPUT_CHARTS_SUBDIR}/{img_name}' alt='Spread Grid for {region_name}'></div>")
            html_parts.append("</div>")

        monthly_hist_img_names = market_analyzer.generate_regional_historical_monthly_basis(
            region_name, components_in_this_region, prices_df,
            REGIONAL_PRIOR_YEARS_FOR_AVG_BASIS, REGIONAL_MONTH_INDICES_FOR_AVG_BASIS,
            output_charts_path
        )
        if monthly_hist_img_names:
            html_parts.append("<div class='monthly-charts-layout'>")
            for img_name in monthly_hist_img_names:
                html_parts.append(f"<div class='monthly-chart-item'><img src='{OUTPUT_CHARTS_SUBDIR}/{img_name}' alt='Monthly Historical Basis for {region_name}'></div>")
            html_parts.append("</div>")

        html_parts.append("<h3>Individual Component Details</h3>")
        for comp_detail in components_in_this_region:
            comp_name = comp_detail['prices_csv_name']
            fom_mark0 = comp_detail['forward_value']

            html_parts.append(f"<div class='component-section'><h4>Component: {comp_name}</h4>")

            cash_basis_heatmap_df = market_analyzer.calculate_individual_monthly_cash_basis_table(prices_df, comp_name)
            html_parts.append("<div class='flex-row'>") 
            if cash_basis_heatmap_df is not None:
                heatmap_by_month_html = market_analyzer.style_individual_heatmap_table(cash_basis_heatmap_df, "Heat Map by Month (Yearly Seasonality)", 0)
                heatmap_by_year_html = market_analyzer.style_individual_heatmap_table(cash_basis_heatmap_df, "Heat Map by Year (Historical Monthly Strength)", 1)
                html_parts.append(f"<div class='flex-item table-container'>{heatmap_by_month_html}</div>")
                html_parts.append(f"<div class='flex-item table-container'>{heatmap_by_year_html}</div>")
            else:
                html_parts.append("<div class='flex-item table-container'><p>No data for cash basis heatmaps.</p></div>")
                html_parts.append("<div class='flex-item table-container'></div>") 
            html_parts.append("</div>") 


            if not df_fom_historical.empty:
                fom_vs_cash_df = market_analyzer.generate_individual_fom_vs_cash_table(cash_basis_heatmap_df, df_fom_historical, comp_name, TODAY)
                fom_vs_cash_html = market_analyzer.style_individual_fom_vs_cash_table(fom_vs_cash_df, "First of Month (FoM) vs. Average Cash Basis")
                html_parts.append(f"<div class='table-container'>{fom_vs_cash_html}</div>")
            else:
                html_parts.append("<p>Historical FoM data not available for FoM vs. Cash table.</p>")

            html_parts.append("<div class='flex-row'>") 
            basis_history_html = market_analyzer.generate_individual_basis_history_chart(prices_df, comp_name, fom_mark0, output_charts_path, TODAY)
            html_parts.append(f"<div class='flex-item chart-container'>{basis_history_html}</div>")
            
            city_for_comp = component_to_city_map.get(comp_name)
            temp_scatter_html_content = ""
            if city_for_comp and not df_weather.empty:
                temp_scatter_html_content = market_analyzer.generate_individual_temp_scatter_plot(prices_df, df_weather, comp_name, city_for_comp, output_charts_path, TODAY)
            else:
                temp_scatter_html_content = f"<p>Temperature scatter plot not available for {comp_name} (no city mapping or weather data).</p>"
            html_parts.append(f"<div class='flex-item chart-container'>{temp_scatter_html_content}</div>")
            html_parts.append("</div>") 

            html_parts.append("</div>") 
        html_parts.append("</div><hr>") 


    if unregioned_components_details:
        html_parts.append("<div class='unregioned-section'><h2>Unregioned Active Components</h2>")
        for comp_detail in unregioned_components_details:
            comp_name = comp_detail['prices_csv_name']
            fom_mark0 = comp_detail['forward_value']

            html_parts.append(f"<div class='component-section'><h4>Component: {comp_name} (Unregioned)</h4>")
            
            cash_basis_heatmap_df = market_analyzer.calculate_individual_monthly_cash_basis_table(prices_df, comp_name)
            html_parts.append("<div class='flex-row'>") 
            if cash_basis_heatmap_df is not None:
                heatmap_by_month_html = market_analyzer.style_individual_heatmap_table(cash_basis_heatmap_df, "Heat Map by Month (Yearly Seasonality)", 0)
                heatmap_by_year_html = market_analyzer.style_individual_heatmap_table(cash_basis_heatmap_df, "Heat Map by Year (Historical Monthly Strength)", 1)
                html_parts.append(f"<div class='flex-item table-container'>{heatmap_by_month_html}</div>")
                html_parts.append(f"<div class='flex-item table-container'>{heatmap_by_year_html}</div>")
            else:
                html_parts.append("<div class='flex-item table-container'><p>No data for cash basis heatmaps.</p></div>")
                html_parts.append("<div class='flex-item table-container'></div>") 
            html_parts.append("</div>") 


            if not df_fom_historical.empty:
                fom_vs_cash_df = market_analyzer.generate_individual_fom_vs_cash_table(cash_basis_heatmap_df, df_fom_historical, comp_name, TODAY)
                fom_vs_cash_html = market_analyzer.style_individual_fom_vs_cash_table(fom_vs_cash_df, "First of Month (FoM) vs. Average Cash Basis")
                html_parts.append(f"<div class='table-container'>{fom_vs_cash_html}</div>")
            else:
                html_parts.append("<p>Historical FoM data not available for FoM vs. Cash table.</p>")

            html_parts.append("<div class='flex-row'>") 
            basis_history_html = market_analyzer.generate_individual_basis_history_chart(prices_df, comp_name, fom_mark0, output_charts_path, TODAY)
            html_parts.append(f"<div class='flex-item chart-container'>{basis_history_html}</div>")

            city_for_comp = component_to_city_map.get(comp_name)
            temp_scatter_html_content = ""
            if city_for_comp and not df_weather.empty:
                temp_scatter_html_content = market_analyzer.generate_individual_temp_scatter_plot(prices_df, df_weather, comp_name, city_for_comp, output_charts_path, TODAY)
            else:
                temp_scatter_html_content = f"<p>Temperature scatter plot not available for {comp_name} (no city mapping or weather data).</p>"
            html_parts.append(f"<div class='flex-item chart-container'>{temp_scatter_html_content}</div>")
            html_parts.append("</div>") 
            
            html_parts.append("</div>") 
        html_parts.append("</div>")


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
