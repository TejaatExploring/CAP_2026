"""CLI to fetch hourly irradiance + weather context data from NASA POWER.

Example:
python3 weather_service/fetch_weather.py \
  --lat 28.3670 --lon 79.4304 \
  --start 2021-01-01 --end 2021-01-30 \
  --out outputs/weather_hourly.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from nasa_power_service import NasaPowerWeatherService


def parse_date(raw: str) -> datetime.date:
    """Parse YYYY-MM-DD date string."""
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Invalid date '{raw}'. Use YYYY-MM-DD") from err


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch hourly weather context from NASA POWER")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--start", type=parse_date, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_date, default=None, help="End date (YYYY-MM-DD). If omitted, --days is used")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch when --end is omitted")
    parser.add_argument("--time-standard", choices=["UTC", "LST"], default="UTC", help="NASA POWER time standard")
    parser.add_argument("--out", type=Path, default=Path("outputs/weather_hourly_30d.csv"), help="Output CSV path")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    args = parser.parse_args()

    if args.days <= 0:
        raise ValueError("--days must be a positive integer")

    end_date = args.end if args.end else (args.start + timedelta(days=args.days - 1))
    if end_date < args.start:
        raise ValueError("--end must be on or after --start")

    service = NasaPowerWeatherService(retries=args.retries, timeout_seconds=args.timeout)
    points = service.fetch_hourly(
        latitude=args.lat,
        longitude=args.lon,
        start_date=args.start,
        end_date=end_date,
        time_standard=args.time_standard,
    )
    service.save_to_csv(points, args.out)

    print("Weather fetch completed")
    print(f"Rows: {len(points)}")
    print(f"Output: {args.out}")
    print(f"Range: {points[0].timestamp_utc} to {points[-1].timestamp_utc}")


if __name__ == "__main__":
    main()
