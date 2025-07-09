import icepython as ice
import pandas as pd
import csv
import re
import os

def discover_all_codes(description, type_filter):
    """
    Performs a broad search to discover all unique codes and their names for a product type.
    Returns a dictionary mapping root codes to their descriptive names.
    """
    print(f"--- Discovering all codes for: '{description}' ---")
    discovered_codes = {}
    try:
        search_results = ice.get_search(description, rows=500, filters=type_filter, symbolsOnly=False)
        if not search_results:
            print("--> No contracts found.")
            return discovered_codes

        print(f"--> Found {len(search_results)} contracts. Processing...")
        for symbol, desc, _, _ in search_results:
            if "ICE LOTS" in desc:
                continue
            
            # Match 3-character alphanumeric codes
            root_match = re.match(r'^([A-Z0-9]{3})', symbol)
            if not root_match:
                continue
            
            root_code = root_match.group(1)
            if root_code not in discovered_codes:
                name_match = re.search(r' - (.*?)( - |$)', desc)
                point_name = name_match.group(1).strip() if name_match else root_code
                discovered_codes[root_code] = point_name
                
    except Exception as e:
        print(f"--> An error occurred during discovery: {e}")
        
    print(f"--> Discovery complete. Found {len(discovered_codes)} unique codes.")
    return discovered_codes

def build_enriched_market_table(input_filename, output_filename="enriched_market_data.csv"):
    """
    Creates an enriched table by matching codes from PriceAdmin.csv with their
    official ICE names and appends any undiscovered codes.
    """
    # --- STAGE 1 (NEW): Discover all codes from the API first for efficiency ---
    print("\n--- STAGE 1: Discovering all available market codes from API ---")
    all_financial_codes = discover_all_codes("NG Basis LD1", 'F.TYP.0')
    all_physical_codes = discover_all_codes("NG Firm Phys", 'F.TYP.2')
    all_physical_codes.update(discover_all_codes("NG Firm Phys, BS", 'F.TYP.3'))

    # --- STAGE 2 (NEW): Process the input CSV and enrich using the discovered data ---
    print(f"\n--- STAGE 2: Processing and enriching '{input_filename}' ---")
    output_rows = []
    used_codes = set()

    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"--> ERROR: Input file '{input_filename}' not found.")
        return

    for _, row in df.iterrows():
        ice_market = row.get('ICE Market', '')
        
        phys_code = str(row.get('Monthly Basis', '')).strip()
        fin_code = str(row.get('Fin Basis', '')).strip()

        # Add non-empty codes to the set of used codes
        if phys_code and phys_code != 'nan':
            used_codes.add(phys_code)
        if fin_code and fin_code != 'nan':
            used_codes.add(fin_code)

        # --- EDIT: Get names from the pre-fetched dictionaries instead of new API calls ---
        phys_name = all_physical_codes.get(phys_code, "")
        fin_name = all_financial_codes.get(fin_code, "")

        output_rows.append([ice_market, phys_code, phys_name, fin_code, fin_name])
    
    print(f"--> Finished processing {len(output_rows)} rows from input file.")

    # --- STAGE 3: Append unused codes to the bottom ---
    print("\n--- STAGE 3: Appending unused codes ---")
    
    appended_phys_count = 0
    for code, name in sorted(all_physical_codes.items()):
        if code not in used_codes:
            output_rows.append(['', code, name, '', ''])
            appended_phys_count += 1
    print(f"--> Appended {appended_phys_count} unused Physical codes.")

    appended_fin_count = 0
    for code, name in sorted(all_financial_codes.items()):
        if code not in used_codes:
            output_rows.append(['', '', '', code, name])
            appended_fin_count += 1
    print(f"--> Appended {appended_fin_count} unused Financial codes.")

    # --- STAGE 4: Write the final CSV ---
    print(f"\n--- STAGE 4: Writing {len(output_rows)} total rows to '{output_filename}' ---")
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ICE Market', 'Physical Code', 'Physical Name (from API)', 'Financial Code', 'Financial Name (from API)'])
            writer.writerows(output_rows)
        print(f"--> SUCCESS: File '{output_filename}' created.")
    except Exception as e:
        print(f"--> ERROR writing CSV: {e}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(base_dir, 'INFO', 'PriceAdmin.csv')
    
    build_enriched_market_table(input_filename=csv_path)
