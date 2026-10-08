"""Thin wrapper around OpenDrift / OpenOil.

We never implement ocean physics ourselves. This module only:
  * builds an OpenOil (or OceanDrift fallback) model with the configured readers,
  * disables weathering for backward runs (evaporation, emulsification,
    dispersion, biodegradation are NOT reversible processes),
  * seeds elements and runs with a negative (backward) or positive time step,
  * returns trajectories as plain numpy arrays.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from app.core.errors import DriftError

log = logging.getLogger(__name__)

WEATHERING_KEYS = ("processes:evaporation", "processes:emulsification", "processes:dispersion",
                   "processes:biodegradation", "processes:update_oilfilm_thickness")


@dataclass
class DriftRun:
    times: list[datetime]          # output times (chronological order of the simulation)
    lon: np.ndarray                # (n_particles, n_times), NaN when inactive
    lat: np.ndarray
    status: np.ndarray             # final status string per particle
    model: str
    config: dict


def _set(o, key, value, applied: dict):
    try:
        o.set_config(key, value)
        applied[key] = value
    except Exception:  # key absent in this model/version
        pass


def build_model(model_name: str, readers: list, backward: bool, cfg: dict):
    """Return (model, applied_config, model_label)."""
    applied: dict = {}
    label = model_name
    if model_name == "OpenOil":
        try:
            from opendrift.models.openoil import OpenOil
            o = OpenOil(loglevel=50, weathering_model="noaa")
        except Exception as exc:
            log.warning("OpenOil unavailable (%s); falling back to OceanDrift", exc)
            from opendrift.models.oceandrift import OceanDrift
            o, label = OceanDrift(loglevel=50), "OceanDrift (fallback: OpenOil unavailable)"
    else:
        from opendrift.models.oceandrift import OceanDrift
        o = OceanDrift(loglevel=50)
    o.add_reader(readers)
    if backward:
        for k in WEATHERING_KEYS:
            _set(o, k, False, applied)
    _set(o, "drift:vertical_mixing", False, applied)          # surface slick transport
    _set(o, "environment:constant:horizontal_diffusivity", cfg.get("horizontal_diffusivity_m2s", 10.0), applied)
    _set(o, "drift:stokes_drift", False, applied)   # no wave forcing available -> not modelled (stated in report)
    _set(o, "drift:current_uncertainty", cfg.get("current_uncertainty_ms", 0.0), applied)
    _set(o, "drift:wind_uncertainty", cfg.get("wind_uncertainty_ms", 0.0), applied)
    _set(o, "general:coastline_action", "stranding", applied)
    # No wave reader in demo: OpenOil's fallback values are used for wave-dependent terms.
    for var, val in (("sea_surface_wave_significant_height", 0), ("sea_surface_wave_stokes_drift_x_velocity", 0),
                     ("sea_surface_wave_stokes_drift_y_velocity", 0)):
        _set(o, f"environment:fallback:{var}", val, applied)
    return o, applied, label


def run(model_name: str, readers: list, lon: np.ndarray, lat: np.ndarray, start: datetime,
        hours: float, time_step_s: int, output_step_s: int, wind_drift_factor: np.ndarray | float,
        oil_type: str, cfg: dict, weathering: bool = False) -> DriftRun:
    backward = time_step_s < 0
    o, applied, label = build_model(model_name, readers, backward and not weathering, cfg)
    seed_kw = dict(lon=lon, lat=lat, time=start.replace(tzinfo=None), z=0.0, wind_drift_factor=wind_drift_factor)
    if "OpenOil" in label:
        seed_kw["oil_type"] = oil_type
    try:
        o.seed_elements(**seed_kw)
        o.run(duration=timedelta(hours=hours), time_step=timedelta(seconds=time_step_s),
              time_step_output=timedelta(seconds=abs(output_step_s) * (1 if not backward else -1)))
    except Exception as exc:
        if label.startswith("OpenOil"):
            # Some OpenDrift versions refuse backward OpenOil; fall back to pure transport.
            log.warning("OpenOil run failed (%s); retrying with OceanDrift", exc)
            return run("OceanDrift", readers, lon, lat, start, hours, time_step_s, output_step_s,
                       wind_drift_factor, oil_type, cfg, weathering)
        raise DriftError(f"OpenDrift simulation failed: {exc}",
                         "Check forcing coverage/time range and the OpenDrift log.") from exc
    res = o.result
    lons = res["lon"].values.astype(float)
    lats = res["lat"].values.astype(float)
    t = [datetime.fromtimestamp(v.astype("datetime64[s]").astype(int), tz=start.tzinfo)
         for v in res["time"].values]
    lons[np.abs(lons) > 360] = np.nan     # masked values
    lats[np.abs(lats) > 90] = np.nan
    status = res["status"].values
    last = np.array([s[~np.isnan(s.astype(float))][-1] if np.any(~np.isnan(s.astype(float))) else -1
                     for s in status.astype(float)])
    meanings = dict(enumerate(res["status"].attrs.get("flag_meanings", "active").split()))
    return DriftRun(times=t, lon=lons, lat=lats, status=np.array([meanings.get(int(s), "unknown") for s in last]),
                    model=label, config=applied)
