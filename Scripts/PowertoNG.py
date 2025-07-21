import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_generation_table_html(df, iso_name, output_folder, current_date):
    """
    Creates a styled, non-interactive HTML report with a data table
    showing historical generation data.
    """
    print(f"  - Generating static HTML table for {iso_name}...")
    
    # --- 1. Prepare Data for the Table ---
    all_year_data = []

    # Loop through the current year and the 5 prior years
    for year in range(current_date.year, current_date.year - 6, -1):
        
        # Define the date range for the specific year we are processing
        center_date_for_year = datetime(year, current_date.month, current_date.day)
        start_date = center_date_for_year - timedelta(days=45)
        end_date = center_date_for_year + timedelta(days=44) # 90 days total
        
        # Filter the main dataframe for the specific ISO, item, and date range
        item_name = f"{iso_name} - Total Generation"
        mask = (
            (df['Item'] == item_name) &
            (df['Date'] >= start_date) &
            (df['Date'] <= end_date)
        )
        year_data = df.loc[mask].copy()
        
        # Create a 'Day' column (e.g., 'Jul-15') to use for alignment
        year_data['Day'] = year_data['Date'].dt.strftime('%b-%d')
        
        # Rename the 'Value' column to the current year
        year_data.rename(columns={'Value': str(year)}, inplace=True)
        
        # Keep only the 'Day' and the year's data column, and set 'Day' as the index
        all_year_data.append(year_data[['Day', str(year)]].set_index('Day'))

    # --- 2. Combine all years into a single DataFrame ---
    if not all_year_data:
        print(f"    - No data found for {iso_name}. Skipping report.")
        return

    # Concatenate all the yearly dataframes. They will align on the 'Day' index.
    report_df = pd.concat(all_year_data, axis=1)
    
    # Ensure the rows are in chronological order
    date_labels = [(current_date - timedelta(days=45) + timedelta(days=i)).strftime('%b-%d') for i in range(91)]
    report_df = report_df.reindex(date_labels).round(2)

    # Convert the final DataFrame to a styled HTML table
    html_table = report_df.to_html(classes='styled-table', border=0, na_rep='-')

    # --- 3. Create the HTML file ---
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Generation Report - {iso_name}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; margin: 0; padding: 20px; color: #333; }}
            .container {{ width: 90%; max-width: 1400px; margin: 30px auto; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 5px; }}
            h3 {{ color: #7f8c8d; text-align: center; margin-top: 0; font-weight: normal;}}
            .styled-table {{
                border-collapse: collapse;
                margin: 25px auto;
                font-size: 0.9em;
                width: 100%;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                border-radius: 5px 5px 0 0;
                overflow: hidden;
            }}
            .styled-table thead tr {{
                background-color: #009879;
                color: #ffffff;
                text-align: right;
            }}
            .styled-table th,
            .styled-table td {{
                padding: 12px 15px;
                text-align: right;
            }}
            .styled-table th:first-child,
            .styled-table td:first-child {{
                text-align: left;
                font-weight: bold;
            }}
            .styled-table tbody tr {{
                border-bottom: 1px solid #dddddd;
            }}
            .styled-table tbody tr:nth-of-type(even) {{
                background-color: #f3f3f3;
            }}
            .styled-table tbody tr:last-of-type {{
                border-bottom: 2px solid #009879;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{iso_name} Daily Total Generation</h1>
            <h3>Historical Comparison (MWa)</h3>
            {html_table}
        </div>
    </body>
    </html>
    """
    
    # Save the HTML content to a file
    report_path = os.path.join(output_folder, f"{iso_name}_Generation_Report.html")
    with open(report_path, 'w') as f:
        f.write(html_template)
    print(f"    - Successfully created HTML report: {report_path}")


def generate_generation_reports():
    """
    Main function to generate the historical generation chart reports for each ISO.
    """
    # --- Configuration ---
    base_path = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper"
    info_path = os.path.join(base_path, "INFO")
    output_folder = os.path.join(base_path, "Scripts", "MarketAnalysis_Report_Output")
    
    fundy_file = os.path.join(info_path, "PlattsPowerFundy.csv")
    
    # Define the current date for the report. Using a fixed date for reproducibility.
    # In a live environment, you would use datetime.now()
    current_date = datetime(2023, 5, 15) # Example Date
    
    # Define the ISOs to report on
    isos_to_report = ['ERCOT', 'ISONE', 'MISO', 'NYISO', 'PJM', 'SPP']

    print("--- Starting ISO Generation Report Generation ---")

    try:
        # --- Load Data ---
        print("Loading fundamental data...")
        fundy_df = pd.read_csv(fundy_file)
        
        # --- Prepare Data ---
        print("Preparing data...")
        fundy_df['Date'] = pd.to_datetime(fundy_df['Date'], errors='coerce')
        fundy_df.dropna(subset=['Date'], inplace=True)
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # --- Generate a report for each ISO ---
        print("\nGenerating individual HTML reports...")
        for iso in isos_to_report:
            create_generation_table_html(fundy_df, iso, output_folder, current_date)
        
        print("\nAll reports generated successfully.")

    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found. Details: {e}")
        print("Please ensure the file path is correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    generate_generation_reports()
