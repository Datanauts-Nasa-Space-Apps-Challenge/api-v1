"""
lunar_site_api.py
=================

API única para evaluar un sitio concreto de la Luna:
  1. Comunicación directa con la Tierra (Deep Space Network), evitando el
     relieve que rodea al módulo.
  2. Energía solar disponible: iluminación con el relieve, eclipses por la
     Tierra, irradiancia y producción de paneles.

Ambos análisis comparten el relieve lunar y las efemérides.

Fuentes de datos (NASA):
- Efemérides del Sol, la Tierra y la Luna: NASA/JPL Horizons API (DE441)
- Irradiancia solar total (TSI): NASA TSIS-1 (LASP LISIRD)
- Relieve lunar: modelos de elevación LRO LOLA
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import threading
import time as time_module
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from pyproj import CRS as ProjCRS, Transformer
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

HORIZONS_API_URL = os.environ.get(
    "HORIZONS_API_URL",
    "https://ssd.jpl.nasa.gov/api/horizons.api",
)
HORIZONS_TIMEOUT_S = 60
HORIZONS_MAX_RETRIES = 3
HORIZONS_MAX_ROWS_PER_REQUEST = 80_000
HORIZONS_MAX_PARALLEL = 3

MAX_SAMPLES = 100_000

HORIZONS_SUN = "10"
HORIZONS_EARTH = "399"
HORIZONS_MOON = "301"

AU_KM = 149_597_870.700
SUN_RADIUS_KM = 695_700.0
EARTH_RADIUS_KM = 6_371.0

MOON_RADIUS_KM = 1737.4
MOON_RADIUS_M = MOON_RADIUS_KM * 1000.0

LISIRD_TSI_URL = os.environ.get(
    "LISIRD_TSI_URL",
    "https://lasp.colorado.edu/lisird/latis/dap/tsis_tsi_24hr.csv",
)
LISIRD_TIMEOUT_S = 30
DEFAULT_TSI_W_M2 = 1361.0
TSI_FUTURE_WINDOW_DAYS = 27

MIN_GROUND_STATION_ELEVATION_DEG = 10.0

GROUND_STATIONS = {
    "Goldstone": {"lon_deg": -116.89, "lat_deg": 35.4267, "height_m": 1000.0},
    "Madrid": {"lon_deg": -4.2483, "lat_deg": 40.4272, "height_m": 741.0},
    "Canberra": {"lon_deg": 148.9819, "lat_deg": -35.4022, "height_m": 693.0},
}

MAX_SUN_DEPRESSION_FOR_MAST_DEG = -3.0
MAX_EARTH_DEPRESSION_FOR_ANTENNA_DEG = -3.0

MAST_HEIGHT_OPTIONS_M = [2, 5, 10, 20, 50, 100, 200, 500]

_PGDA = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/"
LUNAR_DEM_CATALOG = [
    {"id": "LOLA_80S_20MPP", "url": _PGDA + "LDEM_80S_20MPP_ADJ.TIF",
     "south_limit_deg": -80.0, "resolution_m": 20.0},
    {"id": "LOLA_75S_30MPP", "url": _PGDA + "LDEM_75S_30MPP_ADJ.TIF",
     "south_limit_deg": -75.0, "resolution_m": 30.0},
    {"id": "LOLA_60S_60MPP", "url": _PGDA + "LDEM_60S_60MPP_ADJ.TIF",
     "south_limit_deg": -60.0, "resolution_m": 60.0},
]
LUNAR_DEM_GLOBAL = {
    "id": "LOLA_GLOBAL_118M",
    "url": "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif",
    "south_limit_deg": None,
    "resolution_m": 118.45,
}

GLOBAL_DEM_MAX_DISTANCE_M = 50_000.0
NEAR_FIELD_RADIUS_M = 15_000.0
FAR_FIELD_MAX_PIXELS = 2048

CACHE_DIR = Path(
    os.environ.get("LUNAR_SITE_CACHE_DIR", Path.home() / ".cache" / "lunar_site_api")
)

app = FastAPI(
    title="Lunar Landing Site Assessment API",
    description=(
        "Evalúa un sitio lunar: comunicación directa con la Tierra (DSN) y "
        "energía solar disponible, teniendo en cuenta el relieve LRO LOLA."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Modelos de entrada
# ---------------------------------------------------------------------------

PanelMode = Literal["two_axis", "vertical_tracking", "horizontal", "fixed"]


class HorizonPoint(BaseModel):
    azimuth_deg: float
    elevation_deg: float


class SiteRequest(BaseModel):
    lat: float
    lon: float
    elevation_m: float = 0.0
    start_utc: str
    duration_days: float = 7.0
    step_minutes: int = 10
    min_sun_elevation_deg: float = 0.0
    min_earth_elevation_deg: float = 0.0
    min_ground_station_elevation_deg: float = MIN_GROUND_STATION_ELEVATION_DEG
    lunar_terrain: bool = True
    dem_path: Optional[str] = None
    dem_url: Optional[str] = None
    horizon_mask: Optional[list[HorizonPoint]] = None
    antenna_height_m: float = 2.0
    horizon_azimuth_step_deg: float = 1.0
    horizon_search_distance_m: float = 100_000.0


class SolarRequest(BaseModel):
    lat: float
    lon: float
    elevation_m: float = 0.0
    start_utc: str
    duration_days: float = 29.5
    step_minutes: int = 30
    min_sun_visible_fraction: float = 0.5
    lunar_terrain: bool = True
    dem_path: Optional[str] = None
    dem_url: Optional[str] = None
    horizon_mask: Optional[list[HorizonPoint]] = None
    panel_height_m: float = 2.0
    panel_area_m2: float = 1.0
    panel_efficiency: float = 0.30
    system_losses: float = 0.10
    panel_mode: PanelMode = "vertical_tracking"
    panel_tilt_deg: float = 90.0
    panel_azimuth_deg: float = 0.0
    load_w: Optional[float] = None
    use_measured_tsi: bool = True
    horizon_azimuth_step_deg: float = 1.0
    horizon_search_distance_m: float = 100_000.0
    include_timeseries: bool = False


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def parse_utc(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Fecha UTC inválida: {value}",
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def validate_coordinates(lat: float, lon: float):
    if not -90 <= lat <= 90:
        raise HTTPException(status_code=400, detail="La latitud debe estar entre -90 y 90 grados.")
    if not -180 <= lon <= 180:
        raise HTTPException(status_code=400, detail="La longitud debe estar entre -180 y 180 grados.")


def angular_distance_to_south_pole(lat_deg: float) -> float:
    if lat_deg <= 0:
        return 90.0 + lat_deg
    return 180.0 + lat_deg


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Relieve lunar
# ---------------------------------------------------------------------------

class TerrainError(Exception):
    pass


def select_lunar_dem(lat_deg: float, max_distance_m: float) -> tuple[dict, float]:
    margin_deg = math.degrees(max_distance_m / MOON_RADIUS_M) + 0.2
    for product in LUNAR_DEM_CATALOG:
        if lat_deg <= product["south_limit_deg"] - margin_deg:
            return product, max_distance_m

    if abs(lat_deg) > 85.0:
        raise TerrainError("No hay DEM polar norte automático; pasa dem_path o dem_url.")
    return LUNAR_DEM_GLOBAL, min(max_distance_m, GLOBAL_DEM_MAX_DISTANCE_M)


def _destination_on_moon(lat_deg, lon_deg, azimuth_deg, distance_m):
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    bearing = np.radians(azimuth_deg)
    delta = np.asarray(distance_m) / MOON_RADIUS_M

    lat2 = np.arcsin(
        math.sin(lat1) * np.cos(delta)
        + math.cos(lat1) * np.sin(delta) * np.cos(bearing)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(delta) * math.cos(lat1),
        np.cos(delta) - math.sin(lat1) * np.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2 * math.pi) - math.pi
    return np.degrees(lat2), np.degrees(lon2)


def _elevation_angle_deg(ground_radius_m, observer_radius_m, target_radius_m, distance_m):
    theta = distance_m / ground_radius_m
    return np.degrees(np.arctan2(
        target_radius_m * np.cos(theta) - observer_radius_m,
        target_radius_m * np.sin(theta),
    ))


class LunarTerrainHorizon:
    PROFILE_SAMPLES = 400

    def __init__(
        self,
        lat_deg: float,
        lon_deg: float,
        observer_height_m: float = 2.0,
        fallback_elevation_m: float = 0.0,
        azimuth_step_deg: float = 1.0,
        max_distance_m: float = 100_000.0,
        dem_source: Optional[str] = None,
        dem_id: str = "custom",
    ):
        if not RASTERIO_AVAILABLE:
            raise TerrainError("rasterio y pyproj son necesarios: pip install rasterio pyproj")

        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.observer_height_m = observer_height_m
        self.azimuth_step_deg = azimuth_step_deg

        if dem_source is None:
            product, max_distance_m = select_lunar_dem(lat_deg, max_distance_m)
            dem_source = product["url"]
            dem_id = product["id"]

        self.dem_source = dem_source
        self.dem_id = dem_id
        self.max_distance_m = max_distance_m
        self._load_or_compute(fallback_elevation_m)

    def _cache_file(self) -> Path:
        source_key = self.dem_source
        if not source_key.startswith("http"):
            path = Path(source_key)
            source_key = f"{path.resolve()}:{path.stat().st_mtime_ns}"
        key = "|".join(map(str, [
            source_key, round(self.lat_deg, 6), round(self.lon_deg, 6),
            self.azimuth_step_deg, int(self.max_distance_m), self.PROFILE_SAMPLES,
        ]))
        return CACHE_DIR / f"profile_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"

    def _load_or_compute(self, fallback_elevation_m: float):
        cache = self._cache_file()
        if cache.exists():
            data = np.load(cache)
        else:
            data = self._compute_profiles()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **data)
            tmp.replace(cache)

        self.azimuths = data["azimuths"]
        self.distances = data["distances"]
        self.heights = data["heights"]
        self.sample_lat = data["sample_lat"]
        self.sample_lon = data["sample_lon"]

        self.dem_resolution_m = float(data["dem_resolution_m"])
        self.min_obstacle_distance_m = 2.0 * self.dem_resolution_m
        self.heights = np.where(
            self.distances[None, :] < self.min_obstacle_distance_m, np.nan, self.heights,
        )

        site_ground = float(data["site_ground_m"])
        self.ground_elevation_m = (
            site_ground if np.isfinite(site_ground) else fallback_elevation_m
        )
        self.ground_from_dem = bool(np.isfinite(site_ground))
        self.observer_radius_m = (
            MOON_RADIUS_M + self.ground_elevation_m + self.observer_height_m
        )

        target_r = MOON_RADIUS_M + np.where(np.isfinite(self.heights), self.heights, -1e7)
        self.angles = _elevation_angle_deg(
            MOON_RADIUS_M, self.observer_radius_m, target_r, self.distances[None, :],
        )
        self.worst_index = np.argmax(self.angles, axis=1)
        rows = np.arange(len(self.azimuths))
        self.horizon_deg = self.angles[rows, self.worst_index]

    def _compute_profiles(self) -> dict:
        azimuths = np.arange(0.0, 360.0, self.azimuth_step_deg)
        distances = np.geomspace(10.0, self.max_distance_m, self.PROFILE_SAMPLES)

        az_grid, d_grid = np.meshgrid(azimuths, distances, indexing="ij")
        lats, lons = _destination_on_moon(self.lat_deg, self.lon_deg, az_grid, d_grid)

        source = self.dem_source
        path = f"/vsicurl/{source}" if source.startswith("http") else source

        env = dict(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MAX_RETRY="8",
            GDAL_HTTP_RETRY_DELAY="0.5",
            GDAL_HTTP_RETRY_CODES="403,429,500,502,503,504",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
        )
        try:
            with rasterio.Env(**env), rasterio.open(path) as ds:
                crs = ProjCRS.from_wkt(ds.crs.to_wkt())
                to_dem = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)

                site_x, site_y = to_dem.transform(self.lon_deg, self.lat_deg)
                xs, ys = to_dem.transform(lons, lats)

                ds_res = ds.res
                scale = ds.scales[0] or 1.0
                offset = ds.offsets[0] or 0.0

                near = d_grid <= NEAR_FIELD_RADIUS_M
                heights = np.full(d_grid.shape, np.nan)

                grid_near = self._read_grid(ds, xs[near], ys[near], site_x, site_y, None)
                heights[near] = grid_near.sample(xs[near], ys[near])
                site_ground = float(grid_near.sample(np.array([site_x]), np.array([site_y]))[0])

                if np.any(~near):
                    grid_far = self._read_grid(
                        ds, xs[~near], ys[~near], site_x, site_y, FAR_FIELD_MAX_PIXELS,
                    )
                    heights[~near] = grid_far.sample(xs[~near], ys[~near])

                heights = heights * scale + offset
                site_ground = site_ground * scale + offset
        except rasterio.errors.RasterioIOError as exc:
            raise TerrainError(f"No se pudo leer el DEM {source}: {exc}") from exc

        return {
            "azimuths": azimuths,
            "distances": distances,
            "heights": heights.astype(np.float32),
            "sample_lat": lats.astype(np.float32),
            "sample_lon": lons.astype(np.float32),
            "site_ground_m": np.float64(site_ground),
            "dem_resolution_m": np.float64(max(abs(ds_res[0]), abs(ds_res[1]))),
        }

    @staticmethod
    def _read_grid(ds, xs, ys, site_x, site_y, max_pixels):
        pad = 2 * max(abs(ds.res[0]), abs(ds.res[1]))
        left = min(xs.min(), site_x) - pad
        right = max(xs.max(), site_x) + pad
        bottom = min(ys.min(), site_y) - pad
        top = max(ys.max(), site_y) + pad

        window = rasterio.windows.from_bounds(left, bottom, right, top, ds.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, ds.width, ds.height))

        out_h, out_w = int(window.height), int(window.width)
        if max_pixels and max(out_h, out_w) > max_pixels:
            factor = max(out_h, out_w) / max_pixels
            out_h = max(1, int(out_h / factor))
            out_w = max(1, int(out_w / factor))

        data = ds.read(
            1, window=window, out_shape=(out_h, out_w),
            resampling=Resampling.bilinear, masked=True,
        ).astype(np.float64).filled(np.nan)

        transform = ds.window_transform(window) * rasterio.Affine.scale(
            window.width / out_w, window.height / out_h,
        )
        return _Grid(data, transform)

    def elevation_at_azimuth(self, azimuth_deg) -> np.ndarray:
        return np.interp(azimuth_deg, self.azimuths, self.horizon_deg, period=360.0)

    def obstacle_at_azimuth(self, azimuth_deg: float) -> dict:
        k = int(np.rint(azimuth_deg / self.azimuth_step_deg) % len(self.azimuths))
        j = int(self.worst_index[k])
        height = float(self.heights[k, j])
        return {
            "azimuth_deg": float(self.azimuths[k]),
            "horizon_elevation_deg": round(float(self.horizon_deg[k]), 3),
            "distance_km": round(float(self.distances[j]) / 1000.0, 3),
            "terrain_height_m": round(height, 1) if np.isfinite(height) else None,
            "height_above_site_m": (
                round(height - self.ground_elevation_m, 1) if np.isfinite(height) else None
            ),
            "lat_deg": round(float(self.sample_lat[k, j]), 5),
            "lon_deg": round(float(self.sample_lon[k, j]), 5),
            "type": "terrain",
        }

    def required_mast_height_m(self, azimuth_deg, target_elevation_deg) -> np.ndarray:
        k = np.rint(np.asarray(azimuth_deg) / self.azimuth_step_deg).astype(int) % len(self.azimuths)
        e = np.radians(np.clip(np.asarray(target_elevation_deg, dtype=float), -89.0, 89.0))

        theta = self.distances[None, :] / MOON_RADIUS_M
        r_t = MOON_RADIUS_M + np.nan_to_num(self.heights[k], nan=-1e7)
        r_needed = r_t * (np.cos(theta) - np.sin(theta) * np.tan(e)[:, None])
        h = np.max(r_needed, axis=1) - (MOON_RADIUS_M + self.ground_elevation_m)
        return np.maximum(h, 0.0)

    def to_mask(self) -> list[dict]:
        return [
            {"azimuth_deg": float(a), "elevation_deg": round(float(e), 3)}
            for a, e in zip(self.azimuths, self.horizon_deg)
        ]

    def info(self) -> dict:
        return {
            "dem": self.dem_id,
            "dem_source": self.dem_source,
            "search_distance_km": round(self.max_distance_m / 1000.0, 1),
            "azimuth_step_deg": self.azimuth_step_deg,
            "site_ground_elevation_m": round(self.ground_elevation_m, 1),
            "observer_height_m": self.observer_height_m,
        }


class _Grid:
    def __init__(self, data: np.ndarray, transform):
        self.data = data
        self.inv = ~transform

    def sample(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols, rows = self.inv * (np.asarray(xs), np.asarray(ys))
        cols = np.asarray(cols) - 0.5
        rows = np.asarray(rows) - 0.5
        h, w = self.data.shape

        c0 = np.floor(cols).astype(int)
        r0 = np.floor(rows).astype(int)
        fc = cols - c0
        fr = rows - r0
        c0 = np.clip(c0, 0, w - 1); c1 = np.clip(c0 + 1, 0, w - 1)
        r0 = np.clip(r0, 0, h - 1); r1 = np.clip(r0 + 1, 0, h - 1)

        d = self.data
        return (
            d[r0, c0] * (1 - fc) * (1 - fr) + d[r0, c1] * fc * (1 - fr)
            + d[r1, c0] * (1 - fc) * fr + d[r1, c1] * fc * fr
        )


def _smooth_horizon_deg(ground_radius_m: float, observer_radius_m: float) -> float:
    ratio = min(1.0, ground_radius_m / observer_radius_m)
    return -math.degrees(math.acos(ratio))


_terrain_cache: dict = {}
_terrain_lock = threading.Lock()


def _observer_height_m(request) -> float:
    if hasattr(request, "antenna_height_m"):
        return request.antenna_height_m
    return request.panel_height_m


def get_lunar_terrain(request) -> Optional[LunarTerrainHorizon]:
    if request.dem_path:
        if not Path(request.dem_path).exists():
            raise HTTPException(status_code=400, detail=f"No se encontró el DEM: {request.dem_path}")
        source, dem_id = request.dem_path, "local"
    elif request.dem_url:
        source, dem_id = request.dem_url, "url"
    elif request.horizon_mask:
        return None
    elif request.lunar_terrain:
        source, dem_id = None, None
    else:
        return None

    key = (
        source, round(request.lat, 6), round(request.lon, 6),
        _observer_height_m(request), request.elevation_m,
        request.horizon_azimuth_step_deg, request.horizon_search_distance_m,
    )
    with _terrain_lock:
        if key not in _terrain_cache:
            try:
                kwargs = dict(
                    lat_deg=request.lat,
                    lon_deg=request.lon,
                    observer_height_m=_observer_height_m(request),
                    fallback_elevation_m=request.elevation_m,
                    azimuth_step_deg=request.horizon_azimuth_step_deg,
                    max_distance_m=request.horizon_search_distance_m,
                )
                if source:
                    kwargs.update(dem_source=source, dem_id=dem_id)
                _terrain_cache[key] = LunarTerrainHorizon(**kwargs)
            except TerrainError as exc:
                raise HTTPException(status_code=502, detail=f"Relieve lunar: {exc}") from exc
        return _terrain_cache[key]


def build_horizon_function(request):
    terrain = get_lunar_terrain(request)
    if terrain is not None:
        return terrain.elevation_at_azimuth, terrain.to_mask(), terrain

    if request.horizon_mask:
        mask = sorted(request.horizon_mask, key=lambda item: item.azimuth_deg)
        azimuths = np.array([item.azimuth_deg for item in mask])
        elevations = np.array([item.elevation_deg for item in mask])

        def manual_horizon(azimuth_deg):
            return np.interp(azimuth_deg, azimuths, elevations, period=360.0)

        return manual_horizon, [
            {"azimuth_deg": float(a), "elevation_deg": float(e)}
            for a, e in zip(azimuths, elevations)
        ], None

    return (lambda azimuth_deg: np.zeros_like(np.asarray(azimuth_deg, dtype=float))), [], None


# ---------------------------------------------------------------------------
# Cliente Horizons y TSI
# ---------------------------------------------------------------------------

class HorizonsError(Exception):
    pass


_horizons_session = requests.Session()
_horizons_session.headers.update({"User-Agent": "lunar-site-api/2.1"})
_horizons_lock = threading.Semaphore(HORIZONS_MAX_PARALLEL)


def _q(value: str) -> str:
    return f"'{value}'"


def _horizons_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _call_horizons(params: dict) -> str:
    query = {"format": "json"}
    query.update({key: _q(str(value)) for key, value in params.items()})

    last_error = None
    for attempt in range(HORIZONS_MAX_RETRIES):
        try:
            with _horizons_lock:
                response = _horizons_session.get(
                    HORIZONS_API_URL, params=query, timeout=HORIZONS_TIMEOUT_S,
                )
            if response.status_code == 503:
                raise HorizonsError("Horizons está ocupado (HTTP 503).")
            if response.status_code != 200:
                raise HorizonsError(f"Horizons HTTP {response.status_code}: {response.text[:300]}")

            payload = response.json()
            if "error" in payload:
                raise ValueError(payload["error"])
            return payload["result"]
        except ValueError as exc:
            raise HorizonsError(f"Petición rechazada: {exc}") from exc
        except (requests.RequestException, HorizonsError) as exc:
            last_error = exc
            time_module.sleep(2 ** attempt)

    raise HorizonsError(f"No se pudo contactar con Horizons: {last_error}")


def _parse_observer_table(result: str):
    if "$$SOE" not in result or "$$EOE" not in result:
        raise HorizonsError("Respuesta de Horizons sin tabla válida.")

    header_block, body = result.split("$$SOE", 1)
    body = body.split("$$EOE", 1)[0]

    header_line = next(
        line for line in reversed(header_block.splitlines()) if "Date__(UT)" in line
    )
    columns = [col.strip() for col in header_line.split(",")]

    i_az = columns.index(next(c for c in columns if c.startswith("Azi")))
    i_el = columns.index(next(c for c in columns if c.startswith("Elev")))
    i_delta = columns.index(next(c for c in columns if c.startswith("delta")))

    times, azimuths, elevations, distances = [], [], [], []
    for line in body.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) <= i_delta:
            continue
        stamp = fields[0]
        fmt = "%Y-%b-%d %H:%M:%S.%f" if "." in stamp else (
            "%Y-%b-%d %H:%M:%S" if stamp.count(":") == 2 else "%Y-%b-%d %H:%M"
        )
        times.append(datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc))
        azimuths.append(float(fields[i_az]))
        elevations.append(float(fields[i_el]))
        distances.append(float(fields[i_delta]))

    return times, np.array(azimuths), np.array(elevations), np.array(distances)


@lru_cache(maxsize=256)
def horizons_observer_altaz(
    target: str,
    center_body: str,
    lon_deg: float,
    lat_deg: float,
    height_km: float,
    start_iso: str,
    stop_iso: str,
    step_minutes: int,
    refracted: bool = False,
):
    start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    stop_dt = datetime.fromisoformat(stop_iso.replace("Z", "+00:00"))

    chunk = timedelta(minutes=step_minutes * HORIZONS_MAX_ROWS_PER_REQUEST)
    all_times, all_az, all_el, all_dist = [], [], [], []

    chunk_start = start_dt
    while chunk_start <= stop_dt:
        chunk_stop = min(chunk_start + chunk, stop_dt)
        request_stop = chunk_stop if chunk_stop > chunk_start else (
            chunk_start + timedelta(minutes=step_minutes)
        )

        result = _call_horizons({
            "COMMAND": target,
            "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "OBSERVER",
            "CENTER": f"coord@{center_body}",
            "COORD_TYPE": "GEODETIC",
            "SITE_COORD": f"{lon_deg:.6f},{lat_deg:.6f},{height_km:.6f}",
            "START_TIME": _horizons_time(chunk_start),
            "STOP_TIME": _horizons_time(request_stop),
            "STEP_SIZE": f"{step_minutes}m",
            "QUANTITIES": "4,20",
            "CSV_FORMAT": "YES",
            "ANG_FORMAT": "DEG",
            "CAL_FORMAT": "CAL",
            "TIME_DIGITS": "SECONDS",
            "RANGE_UNITS": "KM",
            "APPARENT": "REFRACTED" if refracted else "AIRLESS",
            "EXTRA_PREC": "YES",
        })

        times, az, el, dist = _parse_observer_table(result)
        for t, a, e, d in zip(times, az, el, dist):
            if t > chunk_stop:
                break
            if all_times and t <= all_times[-1]:
                continue
            all_times.append(t)
            all_az.append(a)
            all_el.append(e)
            all_dist.append(d)

        chunk_start = chunk_stop + timedelta(minutes=step_minutes)

    distances = np.array(all_dist)
    if distances.size and np.nanmax(distances) < 100.0:
        distances = distances * AU_KM

    return tuple(all_times), np.array(all_az), np.array(all_el), distances


def topocentric_altaz(
    target: str,
    site_lat_deg: float,
    site_lon_deg: float,
    site_height_m: float,
    start_dt: datetime,
    end_dt: datetime,
    step_minutes: int,
):
    codes = {"sun": HORIZONS_SUN, "earth": HORIZONS_EARTH}
    times, az, el, dist = horizons_observer_altaz(
        codes[target], HORIZONS_MOON,
        round(site_lon_deg, 6), round(site_lat_deg, 6), round(site_height_m / 1000.0, 6),
        start_dt.isoformat(), end_dt.isoformat(), step_minutes,
    )
    return list(times), el, az, dist


def moon_elevation_from_ground_stations(start_dt: datetime, end_dt: datetime, step_minutes: int):
    def fetch(item):
        name, station = item
        times, _, el, _ = horizons_observer_altaz(
            HORIZONS_MOON, HORIZONS_EARTH,
            station["lon_deg"], station["lat_deg"], station["height_m"] / 1000.0,
            start_dt.isoformat(), end_dt.isoformat(), step_minutes, True,
        )
        return name, list(times), el

    with ThreadPoolExecutor(max_workers=HORIZONS_MAX_PARALLEL) as pool:
        results = list(pool.map(fetch, GROUND_STATIONS.items()))

    station_names = [name for name, _, _ in results]
    reference_times = results[0][1]
    elevations = np.vstack([el for _, _, el in results])
    best_indices = np.argmax(elevations, axis=0)
    best_elevations = np.max(elevations, axis=0)
    best_stations = [station_names[i] for i in best_indices]
    station_series = {name: el for name, _, el in results}

    return reference_times, best_elevations, best_stations, station_series


@lru_cache(maxsize=64)
def _fetch_tsi_daily(first_day: str, last_day: str) -> tuple[tuple[str, float], ...]:
    query = f"time,tsi_1au&time>={first_day}&time<={last_day}&formatTime(yyyy-MM-dd)"
    url = f"{LISIRD_TSI_URL}?{requests.utils.quote(query, safe='&=(),-')}"
    response = requests.get(url, timeout=LISIRD_TIMEOUT_S)
    response.raise_for_status()

    rows = []
    for line in response.text.splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if math.isfinite(value) and 1200.0 < value < 1500.0:
            rows.append((parts[0], value))
    return tuple(rows)


def get_tsi_series(times: list[datetime], use_measured: bool = True) -> tuple[np.ndarray, dict]:
    n = len(times)
    if not use_measured or n == 0:
        return np.full(n, DEFAULT_TSI_W_M2), {"source": "constante", "value_w_m2": DEFAULT_TSI_W_M2}

    days = [t.strftime("%Y-%m-%d") for t in times]
    first = (times[0] - timedelta(days=TSI_FUTURE_WINDOW_DAYS + 30)).strftime("%Y-%m-%d")
    last = (times[-1] + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        rows = _fetch_tsi_daily(first, last)
        if not rows:
            today = datetime.now(timezone.utc)
            rows = _fetch_tsi_daily((today - timedelta(days=30)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
    except requests.RequestException:
        return np.full(n, DEFAULT_TSI_W_M2), {"source": "constante (fallback)", "value_w_m2": DEFAULT_TSI_W_M2}

    measured = dict(rows)
    if not measured:
        return np.full(n, DEFAULT_TSI_W_M2), {"source": "constante", "value_w_m2": DEFAULT_TSI_W_M2}

    ordered = sorted(measured.items())
    fill_value = float(np.mean([v for _, v in ordered[-TSI_FUTURE_WINDOW_DAYS:]]))
    tsi = np.array([measured.get(day, fill_value) for day in days])
    return tsi, {"source": "NASA TSIS-1", "mean_tsi_1au_w_m2": round(float(np.mean(tsi)), 4)}


# ---------------------------------------------------------------------------
# Geometría solar y de paneles
# ---------------------------------------------------------------------------

def disk_fraction_above_line(center_offset_deg, radius_deg) -> np.ndarray:
    d = np.clip(np.asarray(center_offset_deg, dtype=float) / radius_deg, -1.0, 1.0)
    hidden = (np.arccos(d) - d * np.sqrt(1.0 - d * d)) / math.pi
    return 1.0 - hidden


def disk_overlap_fraction(separation_deg, sun_radius_deg, occulter_radius_deg) -> np.ndarray:
    c = np.asarray(separation_deg, dtype=float)
    rs = np.asarray(sun_radius_deg, dtype=float)
    ro = np.asarray(occulter_radius_deg, dtype=float)
    c, rs, ro = np.broadcast_arrays(c, rs, ro)
    out = np.zeros(c.shape)

    full = c <= np.abs(ro - rs)
    out[full] = np.minimum(1.0, (ro[full] / rs[full]) ** 2)

    part = (c < rs + ro) & ~full
    if np.any(part):
        cc, a, b = c[part], rs[part], ro[part]
        alpha = np.arccos(np.clip((cc**2 + a**2 - b**2) / (2 * cc * a), -1, 1))
        beta = np.arccos(np.clip((cc**2 + b**2 - a**2) / (2 * cc * b), -1, 1))
        area = a**2 * (alpha - np.sin(2 * alpha) / 2) + b**2 * (beta - np.sin(2 * beta) / 2)
        out[part] = area / (math.pi * a**2)
    return np.clip(out, 0.0, 1.0)


def angular_separation_deg(el1, az1, el2, az2) -> np.ndarray:
    e1, a1, e2, a2 = map(np.radians, (el1, az1, el2, az2))
    cos_c = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(cos_c, -1.0, 1.0)))


def panel_incidence_factor(mode: str, sun_el_deg, sun_az_deg, tilt_deg=90.0, panel_az_deg=0.0):
    e = np.radians(np.asarray(sun_el_deg, dtype=float))
    if mode == "two_axis":
        factor = np.ones_like(e)
    elif mode == "vertical_tracking":
        factor = np.cos(e)
    elif mode == "horizontal":
        factor = np.sin(e)
    elif mode == "fixed":
        t = math.radians(tilt_deg)
        da = np.radians(np.asarray(sun_az_deg, dtype=float) - panel_az_deg)
        factor = np.sin(e) * math.cos(t) + np.cos(e) * math.sin(t) * np.cos(da)
    else:
        raise ValueError(f"panel_mode desconocido: {mode}")
    return np.clip(factor, 0.0, 1.0)


def find_next_event(times: list[datetime], values: np.ndarray, threshold: float, condition: str):
    for i in range(len(values) - 1):
        current_ok = values[i] >= threshold if condition == "below" else values[i] < threshold
        next_ok = values[i + 1] >= threshold if condition == "below" else values[i + 1] < threshold

        if current_ok and not next_ok:
            denominator = values[i + 1] - values[i]
            fraction = 0.5 if denominator == 0 else (threshold - values[i]) / denominator
            return times[i] + fraction * (times[i + 1] - times[i])
    return None


def _runs(mask: np.ndarray):
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        yield i, j
        i = j + 1


def _r(value, digits=3):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


# ---------------------------------------------------------------------------
# Funciones principales de análisis
# ---------------------------------------------------------------------------

def summarize_earth_terrain(
    times, earth_alt, earth_az, earth_horizon, blocked, ground_available,
    deficit, antenna_needed, terrain, min_earth_deg,
) -> dict:
    step_h = (times[1] - times[0]).total_seconds() / 3600.0 if len(times) > 1 else 0.0
    lost = blocked & ground_available
    return {
        "terrain": terrain.info() if terrain else None,
        "hours_earth_blocked_by_terrain": round(float(blocked.sum() * step_h), 2),
        "hours_comms_lost_to_terrain": round(float(lost.sum() * step_h), 2),
        "fraction_comms_lost_to_terrain": round(float(np.mean(lost)), 4),
    }


def analyze_communications(request: SiteRequest) -> dict:
    validate_coordinates(request.lat, request.lon)
    start_dt = parse_utc(request.start_utc).replace(microsecond=0)
    end_dt = start_dt + timedelta(days=request.duration_days)

    with ThreadPoolExecutor(max_workers=4) as pool:
        terrain_future = pool.submit(build_horizon_function, request)
        sun_future = pool.submit(topocentric_altaz, "sun", request.lat, request.lon, request.elevation_m, start_dt, end_dt, request.step_minutes)
        earth_future = pool.submit(topocentric_altaz, "earth", request.lat, request.lon, request.elevation_m, start_dt, end_dt, request.step_minutes)
        ground_future = pool.submit(moon_elevation_from_ground_stations, start_dt, end_dt, request.step_minutes)
        
        times, sun_alt, sun_az, sun_distance = sun_future.result()
        earth_times, earth_alt, earth_az, earth_distance = earth_future.result()
        station_times, ground_elevation, best_station, station_series = ground_future.result()
        horizon_function, horizon_mask, terrain = terrain_future.result()

    sun_horizon = np.asarray(horizon_function(sun_az), dtype=float)
    earth_horizon = np.asarray(horizon_function(earth_az), dtype=float)
    sun_effective_elevation = sun_alt - sun_horizon
    earth_effective_elevation = earth_alt - earth_horizon

    sun_visible = sun_effective_elevation >= request.min_sun_elevation_deg
    earth_visible = earth_effective_elevation >= request.min_earth_elevation_deg
    ground_available = ground_elevation >= request.min_ground_station_elevation_deg
    direct_comms_available = earth_visible & ground_available

    return {
        "site": {"lat": request.lat, "lon": request.lon},
        "summary": {
            "sunlit_fraction": round(float(np.mean(sun_visible)), 4),
            "earth_visible_fraction": round(float(np.mean(earth_visible)), 4),
            "direct_comms_fraction": round(float(np.mean(direct_comms_available)), 4),
        },
        "horizon_mask": horizon_mask,
    }


def analyze_solar_energy(request: SolarRequest) -> dict:
    validate_coordinates(request.lat, request.lon)
    start_dt = parse_utc(request.start_utc).replace(microsecond=0)
    end_dt = start_dt + timedelta(days=request.duration_days)

    with ThreadPoolExecutor(max_workers=3) as pool:
        terrain_future = pool.submit(build_horizon_function, request)
        sun_future = pool.submit(topocentric_altaz, "sun", request.lat, request.lon, request.elevation_m, start_dt, end_dt, request.step_minutes)
        earth_future = pool.submit(topocentric_altaz, "earth", request.lat, request.lon, request.elevation_m, start_dt, end_dt, request.step_minutes)
        
        times, sun_alt, sun_az, sun_dist_km = sun_future.result()
        earth_times, earth_alt, earth_az, earth_dist_km = earth_future.result()
        horizon_function, horizon_mask, terrain = terrain_future.result()

    n = len(times)
    step_h = (times[1] - times[0]).total_seconds() / 3600.0 if n > 1 else request.step_minutes / 60.0

    sun_radius_deg = np.degrees(np.arcsin(SUN_RADIUS_KM / sun_dist_km))
    sun_horizon = np.asarray(horizon_function(sun_az), dtype=float)
    sun_effective_elevation = sun_alt - sun_horizon
    terrain_fraction = disk_fraction_above_line(sun_effective_elevation, sun_radius_deg)

    earth_radius_deg = np.degrees(np.arcsin(EARTH_RADIUS_KM / earth_dist_km))
    separation = angular_separation_deg(sun_alt, sun_az, earth_alt, earth_az)
    eclipse_fraction = disk_overlap_fraction(separation, sun_radius_deg, earth_radius_deg)

    visible_fraction = terrain_fraction * (1.0 - eclipse_fraction)
    tsi_1au, tsi_info = get_tsi_series(times, request.use_measured_tsi)
    irradiance = tsi_1au * (AU_KM / sun_dist_km) ** 2 * visible_fraction

    conversion = request.panel_area_m2 * request.panel_efficiency * (1.0 - request.system_losses)
    factor = panel_incidence_factor(request.panel_mode, sun_alt, sun_az, request.panel_tilt_deg, request.panel_azimuth_deg)
    p = irradiance * factor * conversion
    energy_wh = float(np.sum(p) * step_h)

    sunlit = visible_fraction >= request.min_sun_visible_fraction

    return {
        "site": {"lat": request.lat, "lon": request.lon},
        "illumination": {
            "hours_sunlit": round(float(sunlit.sum() * step_h), 2),
            "fraction_sunlit": round(float(np.mean(sunlit)), 4),
        },
        "panel": {
            "selected_mode": request.panel_mode,
            "total_energy_wh": round(energy_wh, 1),
        },
        "horizon_mask": horizon_mask,
    }


# ---------------------------------------------------------------------------
# Endpoints FastAPI
# ---------------------------------------------------------------------------

@app.get("/api/analyze")
def analyze_get(
    lat: float = Query(...), lon: float = Query(...), elevation_m: float = Query(0.0),
    start_utc: str = Query(...), duration_days: float = Query(7.0), step_minutes: int = Query(10),
    min_sun_elevation_deg: float = Query(0.0), min_earth_elevation_deg: float = Query(0.0),
    min_ground_station_elevation_deg: float = Query(MIN_GROUND_STATION_ELEVATION_DEG),
    lunar_terrain: bool = Query(True), dem_path: Optional[str] = Query(None), dem_url: Optional[str] = Query(None),
    antenna_height_m: float = Query(2.0), horizon_azimuth_step_deg: float = Query(1.0),
    horizon_search_distance_m: float = Query(100_000.0),
):
    return analyze_communications(SiteRequest(
        lat=lat, lon=lon, elevation_m=elevation_m, start_utc=start_utc,
        duration_days=duration_days, step_minutes=step_minutes,
        min_sun_elevation_deg=min_sun_elevation_deg, min_earth_elevation_deg=min_earth_elevation_deg,
        min_ground_station_elevation_deg=min_ground_station_elevation_deg,
        lunar_terrain=lunar_terrain, dem_path=dem_path, dem_url=dem_url,
        antenna_height_m=antenna_height_m, horizon_azimuth_step_deg=horizon_azimuth_step_deg,
        horizon_search_distance_m=horizon_search_distance_m,
    ))


@app.post("/api/analyze")
def analyze_post(request: SiteRequest):
    return analyze_communications(request)


@app.get("/api/solar-energy")
def solar_energy_get(
    lat: float = Query(...), lon: float = Query(...), elevation_m: float = Query(0.0),
    start_utc: str = Query(...), duration_days: float = Query(29.5), step_minutes: int = Query(30),
    min_sun_visible_fraction: float = Query(0.5), lunar_terrain: bool = Query(True),
    dem_path: Optional[str] = Query(None), dem_url: Optional[str] = Query(None),
    panel_height_m: float = Query(2.0), panel_area_m2: float = Query(1.0),
    panel_efficiency: float = Query(0.30), system_losses: float = Query(0.10),
    panel_mode: PanelMode = Query("vertical_tracking"), panel_tilt_deg: float = Query(90.0),
    panel_azimuth_deg: float = Query(0.0), load_w: Optional[float] = Query(None),
    use_measured_tsi: bool = Query(True), horizon_azimuth_step_deg: float = Query(1.0),
    horizon_search_distance_m: float = Query(100_000.0), include_timeseries: bool = Query(False),
):
    return analyze_solar_energy(SolarRequest(
        lat=lat, lon=lon, elevation_m=elevation_m, start_utc=start_utc,
        duration_days=duration_days, step_minutes=step_minutes,
        min_sun_visible_fraction=min_sun_visible_fraction, lunar_terrain=lunar_terrain,
        dem_path=dem_path, dem_url=dem_url, panel_height_m=panel_height_m,
        panel_area_m2=panel_area_m2, panel_efficiency=panel_efficiency,
        system_losses=system_losses, panel_mode=panel_mode, panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_deg=panel_azimuth_deg, load_w=load_w, use_measured_tsi=use_measured_tsi,
        horizon_azimuth_step_deg=horizon_azimuth_step_deg,
        horizon_search_distance_m=horizon_search_distance_m, include_timeseries=include_timeseries,
    ))


@app.post("/api/solar-energy")
def solar_energy_post(request: SolarRequest):
    return analyze_solar_energy(request)


@app.get("/")
def root():
    return {"message": "Lunar Landing Site Assessment API", "status": "active"}
