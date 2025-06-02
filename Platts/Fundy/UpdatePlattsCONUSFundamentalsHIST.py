# update_gdmf_fundamentals.py

import os
import requests
import pandas as pd
import datetime
from io import BytesIO
from dotenv import load_dotenv

# --- Config ---
PLATTS_AUTH_API_URL = "https://api.ci.spglobal.com/auth/api"
PLATTS_NEWS_API_URL = "https://api.ci.spglobal.com/news-insights"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, '..', '.env')
HIST_CSV_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'INFO', 'PlattsCONUSFundamentalsHIST.csv')

# --- Auth ---
def get_access_token(username, password):
    payload = {"username": username, "password": password}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(PLATTS_AUTH_API_URL, data=payload, headers=headers)
    response.raise_for_status()
    return response.json().get("access_token")

# --- Find latest content ID ---
def find_latest_package(token):
    url = f"{PLATTS_NEWS_API_URL}/v1/search/packages"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {
    "field": "publication",
    "filter": 'publication:"Gas Daily Market Fundamentals Data"',
    "PageSize": 1
}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    results = response.json().get("results", [])
    if results:
        return results[0]["id"]
    return None

# --- Download Excel content (in memory) ---
def download_excel_content(token, content_id):
    url = f"{PLATTS_NEWS_API_URL}/v1/content/{content_id}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return BytesIO(response.content)

# --- Update Historical CSV ---
def update_historical_csv(excel_io, hist_csv_path):
    print("Reading 'US SupplyDemand' tab from downloaded Excel content...")
    new_df = pd.read_excel(excel_io, sheet_name="US SupplyDemand")
    new_df.columns = new_df.columns.map(str)
    new_df.rename(columns={new_df.columns[0]: "GasDate"}, inplace=True)
    new_df["GasDate"] = pd.to_datetime(new_df["GasDate"])

    print(f"Reading historical CSV: {hist_csv_path}")
    hist_df = pd.read_csv(hist_csv_path)
    hist_df["GasDate"] = pd.to_datetime(hist_df["GasDate"])

    combined = pd.concat([hist_df, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["GasDate"], keep="last", inplace=True)
    combined.sort_values("GasDate", inplace=True)

    combined.to_csv(hist_csv_path, index=False)
    print(f"Updated CSV saved to: {hist_csv_path}")

# --- Main ---
if __name__ == "__main__":
    print(f"Loading .env from: {ENV_PATH}")
    load_dotenv(ENV_PATH)

    username = os.getenv("PLATTS_USERNAME")
    password = os.getenv("PLATTS_PASSWORD")
    if not username or not password:
        raise ValueError("Missing PLATTS_USERNAME or PLATTS_PASSWORD in .env")

    print("Authenticating to Platts...")
    token = get_access_token(username, password)

    print("Searching for latest GDMF package...")
    content_id = find_latest_package(token)
    if not content_id:
        raise RuntimeError("Could not find latest GDMF content package.")

    print(f"Downloading Excel content for content ID: {content_id}")
    excel_io = download_excel_content(token, content_id)

    print("Merging with historical data...")
    update_historical_csv(excel_io, HIST_CSV_PATH)

    print("✅ Done.")
