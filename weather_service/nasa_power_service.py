"""NASA POWER weather service for hourly solar-weather context.

Fetches:
- ALLSKY_SFC_SW_DWN (GHI, kWh/m^2/hour)
- ALLSKY_SFC_SW_DNI (DNI, kWh/m^2/hour)
- ALLSKY_SFC_SW_DIFF (DHI, kWh/m^2/hour)
- CLRSKY_SFC_SW_DWN (clear-sky GHI, kWh/m^2/hour)
- WS10M (10m wind speed, m/s)
- PRECTOTCORR (precipitation, mm/hour)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


@dataclass(frozen=True)
class WeatherPoint:
    """One hourly irradiance sample."""

    timestamp_utc: datetime
    ghi_kwh_m2: float
    dni_kwh_m2: float
    dhi_kwh_m2: float
    clear_sky_ghi_kwh_m2: float
    wind_speed_10m_mps: float
    precipitation_mm_h: float
    cloud_factor: float
    weather_label: str


class NasaPowerWeatherService:
    """Client for NASA POWER hourly point weather data."""

    def __init__(self, retries: int = 3, timeout_seconds: int = 30, backoff_seconds: float = 1.5) -> None:
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self.backoff_seconds = backoff_seconds

    def fetch_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        time_standard: str = "UTC",
    ) -> list[WeatherPoint]:
        """Fetch hourly irradiance values between start and end dates (inclusive)."""
        self._validate_inputs(latitude, longitude, start_date, end_date, time_standard)

        params = {
            "parameters": (
                "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,"
                "CLRSKY_SFC_SW_DWN,WS10M,PRECTOTCORR"
            ),
            "community": "RE",
            "longitude": f"{longitude:.6f}",
            "latitude": f"{latitude:.6f}",
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": time_standard,
        }

        url = f"{NASA_POWER_BASE_URL}?{urlencode(params)}"
        payload = self._get_json_with_retry(url)
        return self._parse_payload(payload)

    def save_to_csv(self, points: list[WeatherPoint], output_path: Path) -> None:
        """Save weather points to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(
                "timestamp_utc,ghi_kwh_m2,dni_kwh_m2,dhi_kwh_m2,"
                "clear_sky_ghi_kwh_m2,wind_speed_10m_mps,precipitation_mm_h,"
                "cloud_factor,weather_label\n"
            )
            for p in points:
                f.write(
                    f"{p.timestamp_utc.strftime('%Y-%m-%d %H:%M:%S')},"
                    f"{p.ghi_kwh_m2:.6f},{p.dni_kwh_m2:.6f},{p.dhi_kwh_m2:.6f},"
                    f"{p.clear_sky_ghi_kwh_m2:.6f},{p.wind_speed_10m_mps:.6f},"
                    f"{p.precipitation_mm_h:.6f},{p.cloud_factor:.6f},{p.weather_label}\n"
                )

    def to_records(self, points: list[WeatherPoint]) -> list[dict[str, Any]]:
        """Convert points to JSON-friendly dictionaries."""
        rows: list[dict[str, Any]] = []
        for p in points:
            rec = asdict(p)
            rec["timestamp_utc"] = p.timestamp_utc.isoformat(sep=" ")
            rows.append(rec)
        return rows

    def _validate_inputs(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        time_standard: str,
    ) -> None:
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError("latitude must be between -90 and 90")
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError("longitude must be between -180 and 180")
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        if time_standard not in {"UTC", "LST"}:
            raise ValueError("time_standard must be UTC or LST")

    def _get_json_with_retry(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urlopen(url, timeout=self.timeout_seconds) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"NASA POWER returned status {resp.status}")
                    content = resp.read().decode("utf-8")
                    return json.loads(content)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as err:
                last_error = err
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * attempt)
                else:
                    break

        raise RuntimeError(f"Failed to fetch NASA POWER data after {self.retries} attempts: {last_error}")

    def _parse_payload(self, payload: dict[str, Any]) -> list[WeatherPoint]:
        try:
            params = payload["properties"]["parameter"]
            ghi = params["ALLSKY_SFC_SW_DWN"]
            dni = params["ALLSKY_SFC_SW_DNI"]
            dhi = params["ALLSKY_SFC_SW_DIFF"]
            clear_ghi = params["CLRSKY_SFC_SW_DWN"]
            wind = params["WS10M"]
            rain = params["PRECTOTCORR"]
        except KeyError as err:
            raise RuntimeError(f"Unexpected NASA POWER payload shape: missing {err}") from err

        keys = sorted(
            set(ghi.keys())
            & set(dni.keys())
            & set(dhi.keys())
            & set(clear_ghi.keys())
            & set(wind.keys())
            & set(rain.keys())
        )
        points: list[WeatherPoint] = []
        for key in keys:
            # NASA key format: YYYYMMDDHH
            ts = datetime.strptime(key, "%Y%m%d%H")
            g = float(ghi[key])
            dn = float(dni[key])
            df = float(dhi[key])
            cg = float(clear_ghi[key])
            ws = float(wind[key])
            pr = float(rain[key])
            # NASA missing values are often <= -999
            if g <= -900 or dn <= -900 or df <= -900 or cg <= -900 or ws <= -900 or pr <= -900:
                continue

            if cg > 1e-6:
                cloud_factor = max(0.0, min(1.0, g / cg))
            else:
                cloud_factor = 0.0

            label = self._weather_label(ghi=g, clear_ghi=cg, wind=ws, rain=pr, cloud_factor=cloud_factor)

            points.append(
                WeatherPoint(
                    timestamp_utc=ts,
                    ghi_kwh_m2=g,
                    dni_kwh_m2=dn,
                    dhi_kwh_m2=df,
                    clear_sky_ghi_kwh_m2=cg,
                    wind_speed_10m_mps=ws,
                    precipitation_mm_h=pr,
                    cloud_factor=cloud_factor,
                    weather_label=label,
                )
            )

        if not points:
            raise RuntimeError("No valid weather points were parsed from NASA POWER response")

        return points

    def _weather_label(self, ghi: float, clear_ghi: float, wind: float, rain: float, cloud_factor: float) -> str:
        """Derive a practical weather tag for PV evaluation."""
        is_night = clear_ghi < 0.02 and ghi < 0.02
        if is_night:
            return "night"

        if rain >= 0.2:
            return "rainy"

        if wind >= 8.0:
            return "windy"

        if cloud_factor < 0.55:
            return "cloudy"

        if cloud_factor >= 0.8:
            return "sunny"

        return "partly_cloudy"
