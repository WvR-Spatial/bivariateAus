import xarray as xr
import pandas as pd
import urllib.request
import json
import os

print("Starting SILO data processing pipeline...")

# 1. Download the SILO NetCDF data for a specific year (e.g., 2023)
urls = {
    'daily_rain.nc': "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/daily_rain/2023.daily_rain.nc",
    'max_temp.nc': "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/max_temp/2023.max_temp.nc",
    'min_temp.nc': "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/min_temp/2023.min_temp.nc"
}

for filename, url in urls.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename} from AWS...")
        urllib.request.urlretrieve(url, filename)

print("Calculating annual aggregates...")

# 2. Open datasets with xarray
ds_rain = xr.open_dataset('daily_rain.nc')
ds_tmax = xr.open_dataset('max_temp.nc')
ds_tmin = xr.open_dataset('min_temp.nc')

# 3. Perform temporal aggregation
# Sum daily rainfall to get the annual total
annual_rain = ds_rain['daily_rain'].sum(dim='time')
# Average the maximum and minimum temperatures
annual_temp = (ds_tmax['max_temp'].mean(dim='time') + ds_tmin['min_temp'].mean(dim='time')) / 2

# Create a combined dataset
ds_combined = xr.Dataset({
    'ppt': annual_rain,
    'temp': annual_temp
})

# 4. Convert to a Pandas DataFrame
df = ds_combined.to_dataframe().reset_index()

# Drop ocean pixels (NaN values) and unnecessary coordinates
df = df.dropna()

# Round values to reduce file size significantly
df['lon'] = df['lon'].round(3)
df['lat'] = df['lat'].round(3)
df['ppt'] = df['ppt'].round(1)
df['temp'] = df['temp'].round(2)

print("Exporting to JSON...")

# 5. Export to a flat JSON array
records = df[['lon', 'lat', 'ppt', 'temp']].to_dict(orient='records')

# Use separators to remove whitespace and minify the JSON
with open('climate_data.json', 'w') as f:
    json.dump(records, f, separators=(',', ':'))

print("Success! climate_data.json is ready for your web map.")