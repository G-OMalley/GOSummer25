# explore_daily_peak_report.py

import os
import pandas as pd
from dotenv import load_dotenv
from gridstatusio import GridStatusClient
from datetime import datetime, timedelta
import traceback

# --- Global Variables for Client and API Key Status ---
API_KEY_LOADED = False
GS_CLIENT = None

# --- Load API Key and Initialize Client ---
try:
    load_dotenv() # Assumes .env is in the current (Power/) or parent (TraderHelper/) directory
    GRIDSTATUS_API_KEY = os.environ.get("GRIDSTATUS_API_KEY")

    if GRIDSTATUS_API_KEY:
        print("API Key retrieved from environment.")
        API_KEY_LOADED = True
        try:
            GS_CLIENT = GridStatusClient(api_key=GRIDSTATUS_API_KEY)
            print("GridStatusClient initialized successfully.")
        except Exception as e_client:
            print(f"Error initializing GridStatusClient: {e_client}")
            GS_CLIENT = None
            # traceback.print_exc()
    else:
        print("ERROR: GRIDSTATUS_API_KEY not found in environment variables.")
        print("Ensure your .env file is correctly placed and configured.")

except Exception as e_dotenv:
    print(f"Error during .env loading or initial setup: {e_dotenv}")
    # traceback.print_exc()

# --- Main Function to Explore Daily Peak Report ---
def explore_peak_report():
    if not (API_KEY_LOADED and GS_CLIENT):
        print("Cannot proceed: API Key or GridStatusClient not initialized.")
        return

    print("\n--- Exploring Daily Peak Report Endpoint ---")

    # Get ISO input from user
    # Common ISOs: CAISO, ERCOT, MISO, PJM, NYISO, ISONE, SPP
    iso_input = input("Enter ISO (e.g., caiso, ercot, miso, pjm, nyiso, isone, spp): ").strip().lower()
    if not iso_input:
        print("No ISO entered. Exiting.")
        return

    # Get Market Date input from user
    default_market_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d') # Yesterday
    market_date_input = input(f"Enter Market Date (YYYY-MM-DD) [default: {default_market_date}]: ").strip()
    if not market_date_input:
        market_date_input = default_market_date

    try:
        # Validate date format
        datetime.strptime(market_date_input, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD. Exiting.")
        return

    print(f"\nFetching daily peak report for ISO: '{iso_input}' on Market Date: '{market_date_input}'...")

    try:
        report_data = GS_CLIENT.get_daily_peak_report(iso=iso_input, market_date=market_date_input)

        # Replace the existing 'if report_data is not None:' block in your
# ColGridStatAPI.py script with this new version:

        if report_data is not None:
            print("\n--- Report Data Received ---")
            if isinstance(report_data, dict):
                print("Report is a dictionary. Processing sections...")
                
                # Print top-level info
                print(f"\nOverall Report Info:")
                print(f"  ISO: {report_data.get('ISO')}")
                print(f"  Market Date: {report_data.get('market_date')}")
                print(f"  Timezone: {report_data.get('timezone')}")

                dataframes_to_save = {} # Store dataframes to optionally save

                # Process 'peak_dam_lmp'
                peak_dam_lmp_data = report_data.get('peak_dam_lmp')
                if isinstance(peak_dam_lmp_data, list) and peak_dam_lmp_data:
                    print("\nPeak DAM LMP Details (at various locations):")
                    df_peak_dam_lmp = pd.DataFrame(peak_dam_lmp_data)
                    print(df_peak_dam_lmp.to_string())
                    dataframes_to_save['peak_dam_lmp'] = df_peak_dam_lmp
                elif peak_dam_lmp_data is not None:
                    print(f"\nPeak DAM LMP Details: {peak_dam_lmp_data}") # If it's not a list or is empty

                # Process 'peak_load'
                peak_load_data = report_data.get('peak_load')
                if isinstance(peak_load_data, dict):
                    print("\nPeak Load Details (including fuel mix at peak load):")
                    # Transpose the dictionary to make keys into an 'Attribute' column and values into a 'Value' column
                    # Or create a single-row DataFrame
                    df_peak_load = pd.DataFrame([peak_load_data]) # Makes a single-row DataFrame
                    print(df_peak_load.to_string())
                    dataframes_to_save['peak_load_with_fuelmix'] = df_peak_load
                elif peak_load_data is not None:
                    print(f"\nPeak Load Details: {peak_load_data}")

                # Process 'peak_net_load'
                peak_net_load_data = report_data.get('peak_net_load')
                if isinstance(peak_net_load_data, dict):
                    print("\nPeak Net Load Details (including fuel mix at peak net load):")
                    df_peak_net_load = pd.DataFrame([peak_net_load_data]) # Single-row DataFrame
                    print(df_peak_net_load.to_string())
                    dataframes_to_save['peak_net_load_with_fuelmix'] = df_peak_net_load
                elif peak_net_load_data is not None:
                    print(f"\nPeak Net Load Details: {peak_net_load_data}")

                # Ask user if they want to save the extracted DataFrames
                if dataframes_to_save:
                    save_csv = input("\nDo you want to save the extracted report sections to CSV files? (y/n): ").strip().lower()
                    if save_csv == 'y':
                        output_dir = os.path.dirname(__file__) # Saves in the same directory as the script
                        for name, df_to_save in dataframes_to_save.items():
                            csv_filename = f"{iso_input}_daily_report_{market_date_input.replace('-', '')}_{name}.csv"
                            full_csv_path = os.path.join(output_dir, csv_filename)
                            try:
                                df_to_save.to_csv(full_csv_path, index=False)
                                print(f"Section '{name}' saved successfully to: {full_csv_path}")
                            except Exception as e_csv:
                                print(f"Error saving section '{name}' to CSV: {e_csv}")
                else:
                    print("\nNo structured data sections found to save as CSVs.")


            elif isinstance(report_data, pd.DataFrame): # Original handling if it was a DataFrame
                print(f"Report is a Pandas DataFrame with shape: {report_data.shape}")
                print("First 5 rows of the report:")
                print(report_data.head().to_string())
                # ... (keep your existing DataFrame saving logic here if needed) ...

            else:
                print("Report data is not a dictionary or DataFrame. Printing as is:")
                print(report_data)
        else:
            print("No data returned from the daily peak report endpoint (response is None).")

    except AttributeError as ae:
        if "get_daily_peak_report" in str(ae):
            print(f"\nERROR: The method 'get_daily_peak_report' might not be available.")
            print(f"Please ensure your 'gridstatusio' library is version 0.6.3 or newer.")
            print(f"You can upgrade using: pip install --upgrade gridstatusio")
        else:
            print(f"\nAn AttributeError occurred: {ae}")
            # traceback.print_exc()
    except Exception as e:
        print(f"\nAn error occurred while fetching or processing the report: {e}")
        # traceback.print_exc()

# --- Main Execution Block ---
if __name__ == "__main__":
    explore_peak_report()
    print("\n--- Exploration Script Finished ---")