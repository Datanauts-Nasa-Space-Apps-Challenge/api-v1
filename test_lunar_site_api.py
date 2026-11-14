"""
test_lunar_site_api.py
======================
Pruebas unitarias para verificar el funcionamiento de la API.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
import numpy as np
from datetime import datetime, timezone

from lunar_site_api import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "active"


@patch("lunar_site_api.topocentric_altaz")
@patch("lunar_site_api.moon_elevation_from_ground_stations")
@patch("lunar_site_api.get_lunar_terrain")
def test_analyze_endpoint(mock_get_terrain, mock_moon_elev, mock_topo):
    mock_topo.return_value = (
        [datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)],
        np.array([45.0]),
        np.array([180.0]),
        np.array([150_000_000.0]),
    )
    mock_moon_elev.return_value = (
        [datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)],
        np.array([30.0]),
        ["Madrid"],
        {"Madrid": np.array([30.0])},
    )
    mock_get_terrain.return_value = None

    response = client.get(
        "/api/analyze",
        params={
            "lat": -89.0,
            "lon": 0.0,
            "start_utc": "2026-06-01T00:00:00Z",
            "duration_days": 1.0,
            "step_minutes": 60,
            "lunar_terrain": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "direct_comms_fraction" in data["summary"]


@patch("lunar_site_api.topocentric_altaz")
@patch("lunar_site_api.get_tsi_series")
@patch("lunar_site_api.get_lunar_terrain")
def test_solar_energy_endpoint(mock_get_terrain, mock_tsi, mock_topo):
    mock_topo.return_value = (
        [datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)],
        np.array([45.0]),
        np.array([180.0]),
        np.array([150_000_000.0]),
    )
    mock_tsi.return_value = (np.array([1361.0]), {"source": "mock"})
    mock_get_terrain.return_value = None

    response = client.get(
        "/api/solar-energy",
        params={
            "lat": -89.0,
            "lon": 0.0,
            "start_utc": "2026-06-01T00:00:00Z",
            "duration_days": 1.0,
            "step_minutes": 60,
            "lunar_terrain": False,
            "panel_mode": "vertical_tracking",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "panel" in data
    assert data["panel"]["selected_mode"] == "vertical_tracking"
