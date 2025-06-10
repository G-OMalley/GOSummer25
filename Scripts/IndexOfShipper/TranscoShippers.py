import pandas as pd
import os

# Load only the alternate format file
file_path = "Scripts/IndexOfShipper/Index_of_Customers_Filing_04_2025_Excel_Format.xls"
df = pd.read_excel(file_path, header=None, engine="xlrd")

# Corrected: pipeline name is in cell B12 (row 11, column 1)
pipeline_name = df.iloc[11, 1] if len(df.columns) > 1 and len(df) > 11 else "Unknown"

contracts = []
i = 0
while i < len(df):
    row = df.iloc[i]
    if row[0] == "D":
        block_start = i
        block_end = i + 1
        while block_end < len(df) and df.iloc[block_end][0] != "D":
            block_end += 1
        block = df.iloc[block_start:block_end]

        # Check for 'A' presence in block
        if any(r[0] == "A" for _, r in block.iterrows()):
            shipper = row[1]
            expiration = row[7]  # Column H
            max_qty = row[10]     # Column K

            # Skip entries with no max quantity
            if pd.isna(max_qty):
                i = block_end
                continue

            receipt_points = []
            delivery_points = []

            for _, point_row in block.iterrows():
                if point_row[1] == "S8":
                    receipt_points.append((point_row[2], point_row[6]))
                elif point_row[1] == "S9":
                    delivery_points.append((point_row[2], point_row[6]))

            contracts.append({
                "Pipeline Name": pipeline_name,
                "Shipper Name": shipper,
                "Expiration Date": expiration,
                "Max Quantity": max_qty,
                "Receipt Points": receipt_points,
                "Delivery Points": delivery_points
            })
        i = block_end
    else:
        i += 1

# Save final result
contracts_df = pd.DataFrame(contracts)
output_path = r"C:\\Users\\patri\\OneDrive\\Desktop\\Coding\\TraderHelper\\Scripts\\IndexOfShipper\\Customers\\AMAcounterparties.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
contracts_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
