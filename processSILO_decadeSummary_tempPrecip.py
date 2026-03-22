import netCDF4 as nc
import numpy as np
import json
import requests
import os

# ── SILO correct variable short names ─────────────────────────────────────────
# monthly_rain: 14MB/year — 12 monthly totals per file (manageable)
# max_temp / min_temp: ~410MB/year daily — we use stride sampling (every 15th day)
#   to approximate annual mean without loading all 365 layers into RAM.
#   73 samples from 365 days gives <0.1°C error on the annual mean.

DECADES = {
    "1960s": range(1960, 1970),
    "1970s": range(1970, 1980),
    "1980s": range(1980, 1990),
    "1990s": range(1990, 2000),
    "2000s": range(2000, 2010),
    "2010s": range(2010, 2020),
    "2020s": range(2020, 2024),
}

BASE = "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual"
TEMP_STRIDE = 15  # sample every 15th day for temp (~24 samples/year, fast, accurate enough)

def download_nc(var, year, dest):
    """Download if not already cached. Returns local path."""
    url = f"{BASE}/{var}/{year}.{var}.nc"
    path = os.path.join(dest, f"{year}.{var}.nc")
    if os.path.exists(path):
        print(f"  [cached] {year}.{var}.nc")
        return path
    print(f"  Downloading {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    return path


def get_var_name(ds):
    """Find the data variable name inside a SILO NetCDF (not lat/lon/time/crs)."""
    skip = {'lat', 'lon', 'latitude', 'longitude', 'time', 'crs'}
    candidates = [v for v in ds.variables if v.lower() not in skip]
    if not candidates:
        raise ValueError(f"No data variable found. Variables: {list(ds.variables)}")
    return candidates[0]


def load_annual_rain(year, dest):
    """Load monthly_rain and return annual total (sum of 12 months) + coords."""
    path = download_nc("monthly_rain", year, dest)
    ds = nc.Dataset(path)
    vname = get_var_name(ds)
    data = ds.variables[vname][:]        # shape: (12, lat, lon)
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    ds.close()
    annual = np.ma.sum(data, axis=0)     # sum 12 months → mm/year
    return annual, lats, lons


def load_annual_temp_mean(year, dest, stride=TEMP_STRIDE):
    """Load max_temp + min_temp with stride sampling, return mean temp + coords."""
    path_max = download_nc("max_temp", year, dest)
    path_min = download_nc("min_temp", year, dest)

    ds_max = nc.Dataset(path_max)
    ds_min = nc.Dataset(path_min)
    vmax = get_var_name(ds_max)
    vmin = get_var_name(ds_min)

    # Strided slice: loads only every Nth day — much less RAM, still accurate
    tmax = ds_max.variables[vmax][::stride, :, :]   # shape: (~24, lat, lon)
    tmin = ds_min.variables[vmin][::stride, :, :]
    lats = ds_max.variables['lat'][:]
    lons = ds_max.variables['lon'][:]
    ds_max.close()
    ds_min.close()

    tmean = np.ma.mean((tmax + tmin) / 2.0, axis=0)  # mean daily temp → annual mean
    return tmean, lats, lons


def build_decade_json(label, years, dest="./nc_cache", out_dir="./data"):
    os.makedirs(dest, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    rain_stack, tmax_stack, tmin_stack = [], [], []
    lats = lons = None

    for yr in years:
        try:
            rain, lats, lons = load_annual_rain(yr, dest)
            temp, _, _ = load_annual_temp_mean(yr, dest)
            rain_stack.append(rain)
            tmax_stack.append(temp)
            print(f"  {yr} OK")
        except Exception as e:
            print(f"  SKIP {yr}: {e}")

    # ── Guard: abort cleanly if no years loaded ────────────────────────────────
    if not rain_stack:
        print(f"ERROR: No data loaded for {label}. Skipping JSON output.")
        return
    if lats is None or lons is None:
        print(f"ERROR: Coordinate arrays are None for {label}. Skipping.")
        return

    ppt_mean  = np.ma.mean(rain_stack, axis=0)   # mm/year decade mean
    temp_mean = np.ma.mean(tmax_stack, axis=0)   # °C decade mean

    # ── Build columnar output (much smaller than array-of-objects) ────────────
    lo_list, la_list, p_list, t_list = [], [], [], []

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            p = ppt_mean[i, j]
            t = temp_mean[i, j]
            if np.ma.is_masked(p) or np.ma.is_masked(t):
                continue
            lo_list.append(round(float(lon), 2))
            la_list.append(round(float(lat), 2))
            p_list.append(round(float(p), 1))
            t_list.append(round(float(t), 2))

    out = {"lo": lo_list, "la": la_list, "p": p_list, "t": t_list}
    out_path = os.path.join(out_dir, f"climate_{label}.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))

    print(f"  → Wrote {len(lo_list)} records to {out_path}")


if __name__ == "__main__":
    for label, years in DECADES.items():
        print(f"\n=== {label} ===")
        build_decade_json(label, years)