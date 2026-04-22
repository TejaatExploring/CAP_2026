# Weather Service (Separate Module)

This module fetches hourly solar irradiance + weather context data from NASA POWER for use in simulation.

## Data fetched

- ALLSKY_SFC_SW_DWN -> GHI (kWh/m^2/hour)
- ALLSKY_SFC_SW_DNI -> DNI (kWh/m^2/hour)
- ALLSKY_SFC_SW_DIFF -> DHI (kWh/m^2/hour)
- CLRSKY_SFC_SW_DWN -> clear-sky GHI (kWh/m^2/hour)
- WS10M -> wind speed at 10m (m/s)
- PRECTOTCORR -> precipitation (mm/hour)

## Run

From project root:

```bash
python3 weather_service/fetch_weather.py \
	--lat 28.3670 --lon 79.4304 \
	--start 2021-01-01 --days 30 \
	--out outputs/weather_hourly_30d.csv
```

If you omit `--end`, the script uses `--days` (default `30`) starting from `--start`.

## Output

CSV columns:

- timestamp_utc
- ghi_kwh_m2
- dni_kwh_m2
- dhi_kwh_m2
- clear_sky_ghi_kwh_m2
- wind_speed_10m_mps
- precipitation_mm_h
- cloud_factor
- weather_label

## Notes

- Uses NASA POWER hourly point endpoint.
- Handles retry/backoff and filters invalid placeholder values.
- Keep this module separate from synthetic_load code.
