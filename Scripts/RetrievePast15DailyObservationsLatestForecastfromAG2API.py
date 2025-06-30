"""
DISCLAIMER AND LIMITATION OF LIABILITY

This script is provided "as is" without warranty of any kind, either expressed or implied,
including but not limited to the implied warranties of merchantability and fitness for a
particular purpose. The user assumes full responsibility for the use of this script and any
results or consequences that may arise.

Under no circumstances shall the author or distributor of this script be liable for any
direct, indirect, incidental, special, exemplary, or consequential damages (including, but not
limited to, procurement of substitute goods or services; loss of use, data, or profits; or
business interruption) however caused and on any theory of liability, whether in contract,
strict liability, or tort (including negligence or otherwise) arising in any way out of the use
of this script, even if advised of the possibility of such damage.

By using this script, you agree to the terms of this disclaimer.
"""

import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta

# --- Configuration ---
city_id = "KJFK"
account = "colonialenergy"
profile = "ppatten@colonialenergy.com"
password = "ewq474"

# --- Dynamic date range ---
start_date = (datetime.today() - timedelta(days=15)).strftime("%m/%d/%Y")
end_date = (datetime.today() - timedelta(days=1)).strftime("%m/%d/%Y")

# --- Final column order ---
final_columns = [
    "CityId", "Date", "MinTemp", "MaxTemp", "Max Dewpoint", "Avg Cloud Cover",
    "Max Surface Wind", "Min Feels Like", "Max Feels Like", "Daily Precip Amount", "Source"
]

# --- Get Observed Daily Data ---
daily_url = (
    f"https://www.wsitrader.com/Services/CSVDownloadService.svc/GetHistoricalObservations"
    f"?Account={account}&profile={profile}&password={password}"
    f"&TempUnits=F&HistoricalProductID=HISTORICAL_DAILY_OBSERVED"
    f"&StartDate={start_date}&EndDate={end_date}&IsDisplayDates=false&IsTemp=true"
    f"&CityIds[]={city_id}"
)
daily_response = requests.get(daily_url)
daily_df = pd.read_csv(StringIO(daily_response.text), skiprows=3, header=None)
daily_df.columns = ["Date", "MinTemp", "MaxTemp", "AvgTemp", "Precip"]
daily_df["Date"] = pd.to_datetime(daily_df["Date"]).dt.strftime("%Y-%m-%d")

# --- Get Hourly Observed Data ---
hourly_url = (
    f"https://www.wsitrader.com/Services/CSVDownloadService.svc/GetHistoricalObservations"
    f"?Account={account}&Profile={profile}&Password={password}"
    f"&HistoricalProductID=HISTORICAL_HOURLY_OBSERVED"
    f"&DataTypes[]=dewpoint&DataTypes[]=temperature&DataTypes[]=cloudCover"
    f"&DataTypes[]=windSpeed&DataTypes[]=precipitation&DataTypes[]=heatIndex&DataTypes[]=windChill"
    f"&TempUnits=F&StartDate={start_date}&EndDate={end_date}&CityIds[]={city_id}"
)
hourly_response = requests.get(hourly_url)
columns = ["Date", "Hour", "Temperature", "Dewpoint", "WindChill", "HeatIndex", "WindSpeed", "CloudCover", "Precip"]
hourly_df = pd.read_csv(StringIO(hourly_response.text), skiprows=2, header=None, names=columns)
hourly_df["Date"] = pd.to_datetime(hourly_df["Date"]).dt.strftime("%Y-%m-%d")
hourly_df["CloudCover"] = hourly_df["CloudCover"].astype(str).str.replace("%", "", regex=False).astype(float)

def calculate_feels_like(row):
    if row["Temperature"] > 80:
        return row["HeatIndex"]
    elif row["Temperature"] < 50 and row["WindSpeed"] >= 3:
        return row["WindChill"]
    else:
        return row["Temperature"]

hourly_df["FeelsLike"] = hourly_df.apply(calculate_feels_like, axis=1)

hourly_summary = hourly_df.groupby("Date").agg({
    "Dewpoint": "max",
    "CloudCover": "mean",
    "WindSpeed": "max",
    "FeelsLike": ["min", "max"]
}).reset_index()
hourly_summary.columns = [
    "Date", "Max Dewpoint", "Avg Cloud Cover", "Max Surface Wind",
    "Min Feels Like", "Max Feels Like"
]
hourly_summary["Avg Cloud Cover"] = hourly_summary["Avg Cloud Cover"].round().astype(int)

combined_df = pd.merge(daily_df, hourly_summary, on="Date", how="inner")
combined_df.insert(0, "CityId", city_id)
combined_df["Daily Precip Amount"] = combined_df.pop("Precip")
combined_df["Source"] = "Observed"
combined_df = combined_df[final_columns]

# --- Get Forecast Data ---
forecast_url = (
    f"https://www.wsitrader.com/Services/CSVDownloadService.svc/GetHourlyForecast"
    f"?Account={account}&Profile={profile}&Password={password}"
    f"&region=NA&SiteIds[]={city_id}&TempUnits=F"
)
forecast_response = requests.get(forecast_url)
forecast_columns = [
    "DateHour", "Temperature", "TempDiff", "TempNormal", "DewPoint", "CloudCover",
    "FeelsLikeTemp", "FeelsLikeTempDiff", "Precip", "WindDirection", "WindSpeed", "GHI"
]
forecast_df = pd.read_csv(StringIO(forecast_response.text), skiprows=2, header=None, names=forecast_columns)
forecast_df["Date"] = pd.to_datetime(forecast_df["DateHour"], format="mixed").dt.strftime("%Y-%m-%d")

for col in ["Temperature", "DewPoint", "CloudCover", "FeelsLikeTemp", "Precip", "WindSpeed"]:
    forecast_df[col] = pd.to_numeric(forecast_df[col], errors="coerce")

forecast_summary = forecast_df.groupby("Date").agg({
    "Temperature": ["min", "max"],
    "DewPoint": "max",
    "FeelsLikeTemp": ["min", "max"],
    "WindSpeed": "max",
    "Precip": "sum",
    "CloudCover": "mean"
}).reset_index()
forecast_summary.columns = [
    "Date", "Min Temperature", "Max Temperature", "Max Dewpoint",
    "Min Feels Like Temp", "Max Feels Like Temp",
    "Max Wind Speed", "Total Precip", "Avg Cloud Cover"
]

forecast_summary["Min Temperature"] = forecast_summary["Min Temperature"].round(0).astype(int)
forecast_summary["Max Temperature"] = forecast_summary["Max Temperature"].round(0).astype(int)

forecast_summary["Max Temperature"] = forecast_summary.apply(
    lambda row: row["Max Temperature"] + (2 * (row["Max Temperature"] - row["Min Temperature"]) / 18)
    if (row["Max Temperature"] - row["Min Temperature"]) <= 18 else row["Max Temperature"] + 2,
    axis=1
)
forecast_summary["Min Temperature"] -= 1

forecast_summary["MinTemp"] = forecast_summary["Min Temperature"]
forecast_summary["MaxTemp"] = forecast_summary["Max Temperature"].round(0).astype(int)
forecast_summary["Max Dewpoint"] = forecast_summary["Max Dewpoint"].round(0).astype(int)
forecast_summary["Avg Cloud Cover"] = forecast_summary["Avg Cloud Cover"].round(0).astype(int)
forecast_summary["Max Surface Wind"] = forecast_summary["Max Wind Speed"].round(0).astype(int)
forecast_summary["Min Feels Like"] = forecast_summary["Min Feels Like Temp"].round(0).astype(int)
forecast_summary["Max Feels Like"] = forecast_summary["Max Feels Like Temp"].round(0).astype(int)
forecast_summary["Daily Precip Amount"] = forecast_summary["Total Precip"]
forecast_summary["CityId"] = city_id
forecast_summary["Source"] = "Forecast"

forecast_output_df = forecast_summary[final_columns]

# Combine and save
final_df = pd.concat([combined_df, forecast_output_df], ignore_index=True)
output_file = f"Past15DayObservationsLatest15DayForecast_{city_id}.csv"
final_df.to_csv(output_file, index=False)

print(f"✅ Combined data saved to: {output_file}")
