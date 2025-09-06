from pathlib import Path
from dotenv import load_dotenv
import os
import re
from datetime import date
import requests

_loaded = False

def ensure_env_loaded() -> None:
    global _loaded
    if _loaded:
        return
    cwd = Path(__file__).resolve().parents[1]
    for p in [cwd / ".env", cwd / ".env.local"]:
        if p.exists():
            load_dotenv(dotenv_path=p)
    _loaded = True


# Weather helpers
OPEN_METEO_GEOCODE_URL = os.getenv("OPEN_METEO_GEOCODE_URL", "https://geocoding-api.open-meteo.com/v1/search")
OPEN_METEO_FORECAST_URL = os.getenv("OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")
HTTP_TIMEOUT = float(os.getenv("WEATHER_HTTP_TIMEOUT", "20"))
DEFAULT_PAST_DAYS = int(os.getenv("WEATHER_PAST_DAYS", "7"))
DEFAULT_FORECAST_DAYS = int(os.getenv("WEATHER_FORECAST_DAYS", "7"))
DEFAULT_CAPITAL_NAME = os.getenv("WEATHER_DEFAULT_CAPITAL_NAME", "Dehradun, Uttarakhand, India")
DEFAULT_CAPITAL_LAT = float(os.getenv("WEATHER_DEFAULT_CAPITAL_LAT", "30.3165"))
DEFAULT_CAPITAL_LON = float(os.getenv("WEATHER_DEFAULT_CAPITAL_LON", "78.0322"))


def geocode_place(name: str) -> tuple[float, float, str] | None:
    params = {"name": name, "count": 1, "language": "en", "format": "json"}
    r = requests.get(OPEN_METEO_GEOCODE_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json() or {}
    results = data.get("results") or []
    if not results:
        norm = re.sub(r"[^a-z]", "", name.lower())
        if norm in {"uttarakhand", "uttrakhand", "uttarkhand"}:
            return DEFAULT_CAPITAL_LAT, DEFAULT_CAPITAL_LON, DEFAULT_CAPITAL_NAME
        return None
    hit = results[0]
    lat = float(hit.get("latitude"))
    lon = float(hit.get("longitude"))
    display = ", ".join([p for p in [hit.get("name"), hit.get("admin1"), hit.get("country")] if p])
    return lat, lon, display


def get_forecast(lat: float, lon: float, days: int = DEFAULT_FORECAST_DAYS, past_days: int = DEFAULT_PAST_DAYS) -> dict[str, object]:
    days = max(1, min(days, 14))
    past_days = max(0, min(past_days, 14))
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "forecast_days": days,
        "past_days": past_days,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "windspeed_10m_max",
            "sunrise",
            "sunset",
        ]),
        "current_weather": True,
    }
    r = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_weather_for_place(place: str, days: int = DEFAULT_FORECAST_DAYS) -> str:
    g = geocode_place(place)
    if not g:
        return f"Couldn't find a location for '{place}'."
    lat, lon, display = g
    data = get_forecast(lat, lon, days)
    return format_weather_response(display, data)


def get_weather_data_for_place(place: str, days: int = DEFAULT_FORECAST_DAYS) -> tuple[str, dict[str, object]]:
    g = geocode_place(place)
    if not g:
        raise ValueError(f"Couldn't find a location for '{place}'.")
    lat, lon, display = g
    data = get_forecast(lat, lon, days)
    return display, data


def _safe_list(d: dict[str, object], key: str) -> list[object]:
    daily = d.get("daily")
    v = daily.get(key) if isinstance(daily, dict) else None
    return v if isinstance(v, list) else []


def format_weather_response(display: str, data: dict[str, object]) -> str:
    cw = data.get("current_weather") or {}
    current = None
    if cw:
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        desc = f"Current: {temp}°C, wind {wind} km/h" if temp is not None else None
        current = desc

    dates = _safe_list(data, "time")
    tmax = _safe_list(data, "temperature_2m_max")
    tmin = _safe_list(data, "temperature_2m_min")
    prcp = _safe_list(data, "precipitation_sum")
    rain = _safe_list(data, "rain_sum")
    windmax = _safe_list(data, "windspeed_10m_max")

    lines: list[str] = [f"Weather for {display}:"]
    if current:
        lines.append(current)

    today = date.today().isoformat()
    past_lines: list[str] = []
    future_lines: list[str] = []
    for i, d in enumerate(dates):
        try:
            line = (
                f"{d}: max {tmax[i]}°C / min {tmin[i]}°C, precip {prcp[i]} mm (rain {rain[i]} mm), wind max {windmax[i]} km/h"
            )
        except Exception:
            break
        if isinstance(d, str) and d < today:
            past_lines.append(line)
        else:
            future_lines.append(line)

    if past_lines:
        lines.append("Past week:")
        lines.extend(past_lines[-DEFAULT_PAST_DAYS:])
    if future_lines:
        lines.append("Forecast:")
        lines.extend(future_lines)

    return "\n".join(lines)
