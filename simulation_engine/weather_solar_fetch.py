import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


def fetch_nasa_power_irradiance(
    latitude,
    longitude,
    start_date,
    end_date,
    community="RE",
    time_standard="UTC",
    output_csv=None,
    timeout=20,
):
    """Fetch hourly irradiance from NASA POWER API.

    Returns a DataFrame with columns:
    - timestamp
    - shortwave_radiation (W/m2)
    - ghi_wm2 (alias of shortwave_radiation)
    - temp_2m_c (deg C)
    - wind_10m_ms (m/s)
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start": start_dt.strftime("%Y%m%d"),
        "end": end_dt.strftime("%Y%m%d"),
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS10M",
        "community": community,
        "format": "JSON",
        "time-standard": time_standard,
    }

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    irradiance_param = (
        payload.get("properties", {})
        .get("parameter", {})
        .get("ALLSKY_SFC_SW_DWN", {})
    )
    temp_param = (
        payload.get("properties", {})
        .get("parameter", {})
        .get("T2M", {})
    )
    wind_param = (
        payload.get("properties", {})
        .get("parameter", {})
        .get("WS10M", {})
    )

    if not irradiance_param:
        raise ValueError("NASA POWER response does not include ALLSKY_SFC_SW_DWN.")

    timestamps = []
    radiation = []
    temps = []
    winds = []

    for ts_key, value in sorted(irradiance_param.items()):
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue

        if value_float == -999.0 or pd.isna(value_float):
            continue

        # NASA hourly keys are formatted as YYYYMMDDHH.
        timestamps.append(datetime.strptime(str(ts_key), "%Y%m%d%H"))

        # NASA provides ALLSKY_SFC_SW_DWN in W/m2 (no conversion needed).
        radiation.append(value_float)
        temp_val = temp_param.get(ts_key, None)
        wind_val = wind_param.get(ts_key, None)

        try:
            temp_float = float(temp_val)
            temps.append(None if temp_float == -999.0 or pd.isna(temp_float) else temp_float)
        except (TypeError, ValueError):
            temps.append(None)

        try:
            wind_float = float(wind_val)
            winds.append(None if wind_float == -999.0 or pd.isna(wind_float) else wind_float)
        except (TypeError, ValueError):
            winds.append(None)

    if not timestamps or not radiation:
        raise ValueError("No valid hourly irradiance values returned by NASA POWER.")

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "shortwave_radiation": radiation,
            "temp_2m_c": temps,
            "wind_10m_ms": winds,
        }
    )
    df["ghi_wm2"] = df["shortwave_radiation"]

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


def fetch_weather_irradiance(*args, **kwargs):
    """Convenience wrapper for NASA POWER irradiance fetch."""
    return fetch_nasa_power_irradiance(*args, **kwargs)


def fetch_open_meteo_forecast(
    latitude,
    longitude,
    days_ahead=7,
    output_csv=None,
    timeout=20,
):
    """Fetch forecast hourly irradiance from Open-Meteo API (free, no key).

    Returns a DataFrame with columns:
    - timestamp
    - shortwave_radiation (W/m2)
    - ghi_wm2 (alias)
    - temp_2m_c (deg C)
    - wind_10m_ms (m/s)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "shortwave_radiation,temperature_2m,wind_speed_10m",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "forecast_days": min(16, max(1, days_ahead)),
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    radiation = hourly.get("shortwave_radiation", [])
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("wind_speed_10m", [])

    if not times or not radiation:
        raise ValueError("Open-Meteo forecast missing hourly data.")

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(times),
            "shortwave_radiation": radiation,
            "temp_2m_c": temps,
            "wind_10m_ms": winds,
        }
    )
    df["ghi_wm2"] = df["shortwave_radiation"]

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


def fetch_latest_nasa_weather(
    latitude,
    longitude,
    days=7,
    nasa_latency_days=2,
    max_lookback_days=1825,
    step_days=7,
    output_csv=None,
    timeout=20,
):
    """Fetch most recent available hourly NASA POWER weather data.

    NASA POWER is near-real-time, usually with a short publication delay.
    """
    last_error = None
    max_attempts = max(1, int(max_lookback_days / max(1, step_days)))

    for extra_latency in range(0, max_attempts * step_days, step_days):
        effective_latency = nasa_latency_days + extra_latency
        latest_available = datetime.now(timezone.utc).date() - timedelta(days=effective_latency)
        start_date = latest_available - timedelta(days=max(1, days) - 1)

        try:
            df = fetch_nasa_power_irradiance(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=latest_available.strftime("%Y-%m-%d"),
                output_csv=output_csv,
                timeout=timeout,
            )

            # Found a valid window; keep the latest one possible.
            if df is not None and not df.empty:
                return df
        except ValueError as err:
            last_error = err

    raise ValueError(
        "Unable to fetch valid hourly NASA POWER data in lookback horizon. "
        f"Last error: {last_error}"
    )


if __name__ == "__main__":
    import sys

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = os.path.join(project_root, "outputs", "weather_irradiance.csv")

    # Use forecast mode if requested, otherwise historical
    mode = sys.argv[1] if len(sys.argv) > 1 else "historical"

    if mode == "forecast":
        df = fetch_open_meteo_forecast(
            latitude=28.367,
            longitude=79.430,
            days_ahead=7,
            output_csv=out_path,
        )
        print("✅ Fetched Open-Meteo 7-day forecast")
    else:
        df = fetch_latest_nasa_weather(
            latitude=28.367,
            longitude=79.430,
            days=7,
            nasa_latency_days=2,
            max_lookback_days=1825,
            step_days=7,
            output_csv=out_path,
        )
        print("✅ Fetched latest NASA historical data")

    print(f"Weather rows: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Output: {out_path}")
