import pandas as pd
import os

# Define the files to process
standard_format_files = [
    "Scripts/IndexOfShipper/IC0003062504.xls",
    "Scripts/IndexOfShipper/IC0003072504.xls",
    "Scripts/IndexOfShipper/IC0003092504.xls",
    "Scripts/IndexOfShipper/IC0003112504.xls",
    "Scripts/IndexOfShipper/IC0006262504.xls",
    "Scripts/IndexOfShipper/IC0006272504.xls",
    "Scripts/IndexOfShipper/IC0012172504.xls"
]

alt_format_files = [
    "Scripts/IndexOfShipper/Index_of_Customers_Filing_04_2025_Excel_Format.xls"
]

all_contracts = []

# Process standard format files
for file_path in standard_format_files:
    df = pd.read_excel(file_path, header=None, engine="xlrd")
    pipeline_name = df.iloc[6, 10]

    SHIPPER_COL = 4
    EXPIRATION_COL = 18
    MAX_QTY_COL = 24
    TYPE_COL = 11
    POINT_NAME_COL = 13
    POINT_QTY_COL = 24
    RECEIPT_TYPE = "M2"
    DELIVERY_TYPE = "MQ"

    i = 0
    while i < len(df):
        row = df.iloc[i]
        if row[0] == "D" and any(df.iloc[j][0] == "A" for j in range(i + 1, min(i + 10, len(df)))):
            shipper = row[SHIPPER_COL]
            expiration = row[EXPIRATION_COL]
            max_qty = row[MAX_QTY_COL]
            receipt_points = []
            delivery_points = []
            i += 1
            while i < len(df) and df.iloc[i][0] != "D":
                current = df.iloc[i]
                if current[TYPE_COL] == RECEIPT_TYPE:
                    receipt_points.append((current[POINT_NAME_COL], current[POINT_QTY_COL]))
                elif current[TYPE_COL] == DELIVERY_TYPE:
                    delivery_points.append((current[POINT_NAME_COL], current[POINT_QTY_COL]))
                i += 1
            all_contracts.append({
                "Pipeline Name": pipeline_name,
                "Shipper Name": shipper,
                "Expiration Date": expiration,
                "Max Quantity": max_qty,
                "Receipt Points": receipt_points,
                "Delivery Points": delivery_points
            })
        else:
            i += 1

# Process alternate format files
for file_path in alt_format_files:
    df = pd.read_excel(file_path, header=None, engine="xlrd")
    pipeline_name = df.iloc[6, 10] if len(df.columns) > 10 and len(df) > 6 else "Unknown"

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
                max_qty = row[8]     # Column I
                receipt_points = []
                delivery_points = []

                for _, point_row in block.iterrows():
                    if point_row[1] == "S8":
                        receipt_points.append((point_row[2], point_row[6]))
                    elif point_row[1] == "S9":
                        delivery_points.append((point_row[2], point_row[6]))

                all_contracts.append({
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
contracts_df = pd.DataFrame(all_contracts)
output_path = r"C:\\Users\\patri\\OneDrive\\Desktop\\Coding\\TraderHelper\\Scripts\\IndexOfShipper\\Customers\\AMAcounterparties.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
contracts_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
