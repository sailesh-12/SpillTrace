"""SYNTHETIC AIS for Indian waters — used when real AIS is unavailable.

SIH26143 explicitly allows this: "Real AIS if available may be used else synthetic data can be prepared for the
region of oil spill to demonstrate the functioning of the algorithm." Everything produced here is labelled
SYNTHETIC_DEMO and must never be presented as real vessel data:
  * provenance = SYNTHETIC_DEMO on the provider, every track and every candidate,
  * vessel names prefixed "SYN-", no IMO numbers,
  * MMSIs use India's MID 419 in the band 419900000-419999999 (realistic format; a collision with a real
    Indian MMSI in that band is possible, which is why the SYN- name and the SYNTHETIC label are mandatory).

How traffic is generated (deterministic per analysis, seeded by the analysis id):
  1. Region preset from the search bbox: Arabian Sea / west coast, Bay of Bengal / east coast, or the
     Sri Lanka east-west route; each preset fixes typical lane bearings and the vessel-type mix.
  2. Lane traffic: lanes at the preset bearings, offset across the search area (some pass near the slick);
     straight legs joined by smooth course alterations plus a gentle meander; vessels enter/leave at random
     times; speed per type with AR(1) variation.
  3. Fishing vessels: trawling legs with a slowly varying turn rate, occasional loitering, kept on a fishing
     ground; sparser class-B-style reporting.
  At most MAX_VESSELS (10) vessels in total (scenario + background).
  4. Reporting: class A every 60-180 s, class B / fishing every 180-360 s; AIS gaps as Poisson episodes
     (15-90 min, more frequent far offshore, mimicking satellite-only coverage); 10 m position jitter,
     SOG noise 0.3 kn, COG noise 2 deg; a few duplicates, invalid coordinates and position spikes so the real
     cleaning code is exercised.
  5. Optional scenario (ais.synthetic.scenario): a planted "release" tanker on the model's backtracked oil
     path at a random time in the release window (with a slowdown and a later AIS gap), a vessel crossing the
     observed slick at image time (nearest-vessel trap), and a vessel crossing the source area outside the
     window. Their roles are written to synthetic_truth.json; the scoring never reads it. Because the planted
     vessel is placed with the model's own backtracked path, the scenario demonstrates the matching/scoring
     logic — it is not evidence of detection skill.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.ais.provider import CANONICAL, AISProvider
from app.core.provenance import Provenance

KM_PER_DEG = 111.32
MAX_VESSELS = 10          # synthetic traffic is capped at 10 vessels in total (scenario + background)

# type -> (display type, speed range kn, report interval s, share)
TYPES = {
    "tanker": ("Tanker", (11, 14), (60, 180)),
    "container": ("Container", (14, 20), (60, 150)),
    "bulk": ("Bulk carrier", (10, 13), (60, 180)),
    "cargo": ("Cargo", (10, 15), (60, 180)),
    "offshore": ("Offshore supply", (8, 12), (60, 180)),
    "tug": ("Tug", (6, 10), (90, 240)),
    "passenger": ("Passenger", (14, 18), (60, 120)),
    "fishing": ("Fishing", (2, 8), (180, 360)),
}

# Indian-waters presets: lane bearings (deg, both directions used) and vessel-type mix
PRESETS = {
    "arabian_sea_west_coast": {
        "bearings": [160, 175, 95],          # coastal Gulf/Karachi <-> Mumbai/Kochi, approaches to JNPT/Mumbai
        "mix": {"tanker": .22, "container": .16, "bulk": .12, "cargo": .12, "offshore": .08, "tug": .04,
                "passenger": .03, "fishing": .23}},
    "bay_of_bengal_east_coast": {
        "bearings": [20, 35, 100],           # coastal Chennai/Vizag/Paradip/Haldia, route to Malacca
        "mix": {"tanker": .18, "container": .18, "bulk": .18, "cargo": .12, "offshore": .03, "tug": .04,
                "passenger": .04, "fishing": .23}},
    "sri_lanka_east_west_route": {
        "bearings": [90, 80, 120],           # Suez/Gulf <-> Malacca main route south of Sri Lanka
        "mix": {"tanker": .25, "container": .28, "bulk": .17, "cargo": .12, "offshore": .01, "tug": .02,
                "passenger": .03, "fishing": .12}},
    "generic_indian_ocean": {
        "bearings": [0, 60, 120],
        "mix": {"tanker": .2, "container": .2, "bulk": .15, "cargo": .15, "offshore": .03, "tug": .04,
                "passenger": .03, "fishing": .2}},
}


def region_preset(bbox) -> str:
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    if lat < 9.5 and 74 <= lon <= 90:
        return "sri_lanka_east_west_route"
    if 64 <= lon < 78 and 6 <= lat <= 25:
        return "arabian_sea_west_coast"
    if 78 <= lon <= 95 and 6 <= lat <= 23:
        return "bay_of_bengal_east_coast"
    return "generic_indian_ocean"


def _offset(lat, lon, dx_km, dy_km):
    return lat + dy_km / KM_PER_DEG, lon + dx_km / (KM_PER_DEG * math.cos(math.radians(lat)))


class SyntheticAISProvider(AISProvider):
    name = "Synthetic AIS generator (Indian waters)"
    provenance = Provenance.DEMO_SYNTHETIC.value

    def __init__(self, cfg_syn: dict | None = None, context: dict | None = None, seed_key: str = "sih26143"):
        c = dict(cfg_syn or {})
        self.n_vessels = min(int(c.get("vessels", MAX_VESSELS)), MAX_VESSELS)   # incl. scenario vessels
        self.scenario = bool(c.get("scenario", True))
        self.gap_rate_per_day = float(c.get("gap_rate_per_day", 2.0))
        self.context = context or {}
        self.seed = int(hashlib.sha256(f"{seed_key}:{c.get('seed', 0)}".encode()).hexdigest()[:8], 16)
        self.truth: dict = {}
        self._meta: dict = {}
        self._mmsis: set = set()

    # ---------------------------------------------------------------------------------------------
    def describe(self) -> dict:
        return {"provider": self.name, "provenance": self.provenance, "synthetic": True,
                "label": "SYNTHETIC AIS — generated for demonstration; NOT real vessel data",
                "preset": self.truth.get("preset"), "n_vessels": self.truth.get("n_vessels"),
                "scenario": self.scenario,
                "basis": "SIH26143 permits synthetic AIS for the spill region when real AIS is unavailable."}

    def get_vessel_metadata(self, mmsi: str) -> dict:
        return {"mmsi": mmsi, **self._meta.get(str(mmsi), {})}

    def _new_mmsi(self, rng) -> str:
        while True:
            m = f"419{rng.integers(900000, 1000000):06d}"
            if m not in self._mmsis:
                self._mmsis.add(m)
                return m

    # ---------------------------------------------------------------------------------------------
    def _emit(self, rng, rows, mmsi, name, vtype, t0, t1, pos_fn, speed_fn, interval, offshore=True):
        """Sample a trajectory given pos_fn(t)->(lat, lon, cog) and speed_fn(t)->kn, with AIS gaps + noise."""
        span = (t1 - t0).total_seconds()
        if span <= 0:
            return
        # gap episodes (coverage) — more offshore
        rate = self.gap_rate_per_day * (1.5 if offshore else 0.5) * span / 86400
        gaps = []
        for _ in range(rng.poisson(rate)):
            g0 = rng.uniform(0, span)
            gaps.append((g0, g0 + rng.uniform(15, 90) * 60))
        for g in self.truth.get("_forced_gaps", {}).get(mmsi, []):
            gaps.append(g)
        s = rng.uniform(0, interval[1])
        while s < span:
            if not any(a <= s <= b for a, b in gaps):
                t = t0 + timedelta(seconds=float(s))
                lat, lon, cog = pos_fn(t)
                rows.append((mmsi, t, lat + rng.normal(0, 1e-4), lon + rng.normal(0, 1e-4),
                             max(speed_fn(t) + rng.normal(0, 0.3), 0.0), (cog + rng.normal(0, 2)) % 360,
                             name, vtype))
            s += rng.uniform(*interval)

    @staticmethod
    def _smoothstep(x):
        x = min(max(x + 0.5, 0.0), 1.0)
        return x * x * (3 - 2 * x)

    def _lane_vessel(self, rng, rows, key, bbox, t0, t1, through=None, t_through=None, speed=None,
                     slowdown=None, name_tag=None, bearing=None):
        """Transiting vessel: straight legs along the lane joined by smooth course alterations (8-25 deg,
        spread over 15-35 min, alternating so the vessel keeps to its lane) plus a gentle +-1-2.5 deg meander
        (steering/sea state). Scenario vessels keep a steady course for +-1.5 h around the time they must pass
        their point, so the planted geometry (e.g. a discharge trail behind the ship) stays consistent."""
        vtype, (smin, smax), interval = TYPES[key]
        w, s_, e, n = bbox
        lat_c, lon_c = (s_ + n) / 2, (w + e) / 2
        if bearing is None:
            bearing = rng.choice(PRESETS[self.truth["preset"]]["bearings"]) + rng.normal(0, 6)
            bearing += 180 * rng.integers(0, 2)
        kn = speed or rng.uniform(smin, smax)
        span_km = 2.2 * max((e - w) * KM_PER_DEG * math.cos(math.radians(lat_c)), (n - s_) * KM_PER_DEG)
        protect = 0.0
        if through is None:
            # lane offset perpendicular to bearing, random across the area (some near the centre)
            off = rng.normal(0, span_km / 5)
            b = math.radians(bearing + 90)
            through = _offset(lat_c, lon_c, off * math.sin(b), off * math.cos(b))
            t_through = t0 + (t1 - t0) * rng.uniform(0, 1)
        else:
            protect = 1.5
        half_h = span_km / (kn * 1.852) / 2

        # heading programme (deg) as a function of hours from t_through
        amp, per = rng.uniform(1.0, 2.5), rng.uniform(1.5, 4.0)
        ph = math.pi / 2 if protect else rng.uniform(0, 2 * math.pi)
        turns = []
        for sgn in (1, -1):
            tau, sign = protect + rng.uniform(0.3, 2.0), rng.choice([-1, 1])
            while tau < half_h:
                turns.append((sgn * tau, sign * rng.uniform(8, 25), rng.uniform(0.25, 0.6)))
                sign, tau = -sign, tau + rng.uniform(1.5, 4.0)

        def heading(dt_h):
            hd = bearing + amp * (math.sin(2 * math.pi * dt_h / per + ph) - math.sin(ph))
            for tau, delta, width in turns:
                f = self._smoothstep((dt_h - tau) / width)
                hd += delta * f if tau > 0 else -delta * (1 - f)
            return hd

        def dist(dt_h):             # km along track from the through point (+ slowdown)
            d = kn * 1.852 * dt_h
            if slowdown:
                a0, a1, slow = slowdown
                if dt_h > a1:
                    d -= (kn - slow) * 1.852 * (a1 - a0)
                elif dt_h > a0:
                    d -= (kn - slow) * 1.852 * (dt_h - a0)
            return d

        # integrate the path outwards from the through point at 2-min steps
        step = 1 / 30
        n_half = int(math.ceil(half_h / step)) + 1
        hs = np.arange(-n_half, n_half + 1) * step
        la, lo = np.empty(len(hs)), np.empty(len(hs))
        la[n_half], lo[n_half] = through
        for i in range(n_half + 1, len(hs)):
            br = math.radians(heading((hs[i] + hs[i - 1]) / 2))
            dd = dist(hs[i]) - dist(hs[i - 1])
            la[i], lo[i] = _offset(la[i - 1], lo[i - 1], dd * math.sin(br), dd * math.cos(br))
        for i in range(n_half - 1, -1, -1):
            br = math.radians(heading((hs[i] + hs[i + 1]) / 2))
            dd = dist(hs[i + 1]) - dist(hs[i])
            la[i], lo[i] = _offset(la[i + 1], lo[i + 1], -dd * math.sin(br), -dd * math.cos(br))
        speed_state = {"v": kn}

        def pos(t):
            dt_h = (t - t_through).total_seconds() / 3600
            return float(np.interp(dt_h, hs, la)), float(np.interp(dt_h, hs, lo)), heading(dt_h) % 360

        def spd(t):
            dt_h = (t - t_through).total_seconds() / 3600
            if slowdown and slowdown[0] < dt_h < slowdown[1]:
                return slowdown[2]
            speed_state["v"] = 0.9 * speed_state["v"] + 0.1 * (kn + rng.normal(0, 0.8))
            return speed_state["v"]
        # active while inside an enlarged bbox
        a = max(t0, t_through - timedelta(hours=half_h))
        b = min(t1, t_through + timedelta(hours=half_h))
        mmsi = self._new_mmsi(rng)
        name = f"SYN-{name_tag or key.upper()}-{len(self._meta) + 1:02d}"
        self._meta[mmsi] = {"name": name, "type_raw": vtype, "imo": None}
        self._emit(rng, rows, mmsi, name, vtype, a, b, pos, spd, interval)
        return mmsi, name

    def _fishing_vessel(self, rng, rows, bbox, t0, t1):
        """Fishing vessel (trawler pattern): tows of 1.5-4 h at 2.5-4.5 kn along a tow axis (+-15 deg), hauling
        at 0.5-1.5 kn for 20-60 min, then a smooth ~180 deg turn onto the next, laterally offset tow; heading
        changes at most 3 deg/min, with a small meander. The tow axis turns back towards the fishing ground
        when the vessel drifts off it."""
        vtype, (smin, smax), interval = TYPES["fishing"]
        w, s_, e, n = bbox
        gx = rng.uniform(w + 0.25 * (e - w), e - 0.25 * (e - w))          # fishing-ground centre and radii
        gy = rng.uniform(s_ + 0.25 * (n - s_), n - 0.25 * (n - s_))
        rx, ry = 0.2 * (e - w), 0.2 * (n - s_)
        lat, lon = gy + rng.uniform(-ry, ry) / 2, gx + rng.uniform(-rx, rx) / 2
        axis = rng.uniform(0, 180)
        target = axis + 180 * rng.integers(0, 2)
        heading, wob = target, 0.0
        track = []                       # path at 1-min resolution
        # one fishing session of 12-30 h inside the window (AIS off / outside the area otherwise)
        sess = min(timedelta(hours=float(rng.uniform(12, 30))), t1 - t0)
        t0 = t0 + (t1 - t0 - sess) * float(rng.uniform(0, 1))
        t1 = t0 + sess
        t = t0
        phase, left = "tow", rng.uniform(2.5, 5) * 60           # minutes left in phase
        v = rng.uniform(2.5, 4.5)
        side = rng.choice([-1, 1])
        while t <= t1:
            left -= 1
            if left <= 0:
                if phase == "tow":
                    phase, left, v = "haul", rng.uniform(20, 60), rng.uniform(0.5, 1.5)
                else:
                    dx, dy = (lon - gx) / rx, (lat - gy) / ry
                    if dx * dx + dy * dy > 1:          # off the ground: next tow heads back towards its centre
                        target = math.degrees(math.atan2(-dx, -dy)) + rng.normal(0, 10)
                    else:
                        target = target + 180 + side * rng.uniform(5, 20)
                    side = -side
                    phase, left, v = "turn", 1e9, rng.uniform(1.0, 1.8)      # slow turn onto the next tow
            if phase == "turn" and abs((target - heading + 540) % 360 - 180) < 4:
                phase, left, v = "tow", rng.uniform(2.5, 5) * 60, rng.uniform(2.5, 4.5)
            wob = float(np.clip(0.95 * wob + rng.normal(0, 0.15), -2, 2))
            diff = (target - heading + 540) % 360 - 180
            heading = (heading + float(np.clip(diff, -2.5, 2.5)) + (wob if phase == "tow" else 0.3 * wob)) % 360
            vv = v + rng.normal(0, 0.05)
            track.append((t, lat, lon, max(vv, 0.2), heading))
            d = vv * 1.852 / 60
            lat, lon = _offset(lat, lon, d * math.sin(math.radians(heading)), d * math.cos(math.radians(heading)))
            t += timedelta(minutes=1)
        ts = np.array([x[0].timestamp() for x in track])

        def pos(tt):
            i = int(np.clip(np.searchsorted(ts, tt.timestamp()), 1, len(track) - 1))
            a, b = track[i - 1], track[i]
            f = (tt.timestamp() - ts[i - 1]) / max(ts[i] - ts[i - 1], 1)
            return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[4]

        def spd(tt):
            i = int(np.clip(np.searchsorted(ts, tt.timestamp()), 0, len(track) - 1))
            return track[i][3]
        mmsi = self._new_mmsi(rng)
        name = f"SYN-FISHING-{len(self._meta) + 1:02d}"
        self._meta[mmsi] = {"name": name, "type_raw": vtype, "imo": None}
        self._emit(rng, rows, mmsi, name, vtype, t0, t1, pos, spd, interval)

    # ---------------------------------------------------------------------------------------------
    def get_tracks(self, bbox, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        preset = region_preset(bbox)
        self.truth = {"warning": "SYNTHETIC scenario roles. The scoring never reads this file.", "preset": preset,
                      "bbox": list(bbox), "start": start_time.isoformat(), "end": end_time.isoformat(),
                      "vessels": [], "_forced_gaps": {}}
        rows: list = []
        ctx = self.context
        # ---- scenario vessels (need the analysis context) -----------------------------------------
        syn_scene = ctx.get("synthetic_scene")
        if self.scenario and syn_scene and syn_scene.get("discharging_vessel"):
            # SYNTHETIC SAR scene: its discharging ship (at the head of the trail at image time) gets a matching
            # AIS track on the trail's course, so SAR, AIS and evidence stay consistent in demonstration mode
            dv = syn_scene["discharging_vessel"]
            t_img = datetime.fromisoformat(dv["time"])
            mm, nm = self._lane_vessel(rng, rows, "tanker", bbox, start_time, end_time, through=(dv["lat"], dv["lon"]),
                                       t_through=t_img, speed=dv["speed_kn"], bearing=dv["course_deg"],
                                       name_tag="TANKER")
            gap0 = (t_img - start_time).total_seconds() + rng.uniform(1.5, 3) * 3600      # gap AFTER the image
            rows[:] = [r for r in rows if not (r[0] == mm and gap0 <= (r[1] - start_time).total_seconds()
                                                <= gap0 + rng.uniform(40, 80) * 60)]
            self.truth["vessels"].append({"mmsi": mm, "name": nm, "role": "discharging_vessel_in_synthetic_scene",
                                          "time_at_trail_head": t_img.isoformat(),
                                          "position": {"lat": dv["lat"], "lon": dv["lon"]}})
        elif self.scenario and ctx.get("corridor_point") and ctx.get("release_window"):
            r0, r1 = ctx["release_window"]
            tr = r0 + (r1 - r0) * rng.uniform(0.2, 0.8)
            p = ctx["corridor_point"](tr)
            if p is not None:
                gap0 = (tr - start_time).total_seconds() + rng.uniform(1, 3) * 3600
                mm, nm = self._lane_vessel(rng, rows, "tanker", bbox, start_time, end_time, through=p, t_through=tr,
                                           speed=12.5, slowdown=(-0.6, 0.6, 4.0), name_tag="TANKER")
                # forced AIS gap after the release (applied by re-emitting with the gap)
                rows[:] = [r for r in rows if not (r[0] == mm and gap0 <= (r[1] - start_time).total_seconds()
                                                    <= gap0 + rng.uniform(30, 60) * 60)]
                self.truth["vessels"].append({"mmsi": mm, "name": nm, "role": "planted_release",
                                              "release_time": tr.isoformat(), "position": {"lat": p[0], "lon": p[1]}})
                tw = r0 - timedelta(hours=float(rng.uniform(4, 6)))
                if tw > start_time:
                    mm, nm = self._lane_vessel(rng, rows, "cargo", bbox, start_time, end_time, through=p, t_through=tw,
                                               name_tag="CARGO")
                    self.truth["vessels"].append({"mmsi": mm, "name": nm, "role": "outside_release_window",
                                                  "time_at_source": tw.isoformat()})
        if self.scenario and ctx.get("spill_centroid") and ctx.get("observation_time"):
            c, obs = ctx["spill_centroid"], ctx["observation_time"]
            mm, nm = self._lane_vessel(rng, rows, "container", bbox, start_time, end_time,
                                       through=(c["lat"], c["lon"]), t_through=obs, name_tag="CONTAINER")
            self.truth["vessels"].append({"mmsi": mm, "name": nm, "role": "nearest_vessel_trap",
                                          "time_at_slick": obs.isoformat()})
        # ---- background traffic ------------------------------------------------------------------
        mix = PRESETS[preset]["mix"]
        keys, probs = list(mix), np.array(list(mix.values()))
        n_bg = max(self.n_vessels - len(self.truth["vessels"]), 0)
        kinds = list(rng.choice(keys, size=n_bg, p=probs / probs.sum()))
        while kinds.count("fishing") > 2:                   # keep at most 2 fishing vessels in a 10-vessel scene
            kinds[kinds.index("fishing")] = "cargo"
        for k in kinds:
            if k == "fishing":
                self._fishing_vessel(rng, rows, bbox, start_time, end_time)
            else:
                self._lane_vessel(rng, rows, k, bbox, start_time, end_time)
        df = pd.DataFrame(rows, columns=["mmsi", "timestamp", "lat", "lon", "sog", "cog", "vessel_name", "vessel_type"])
        df["imo"] = None
        w, s_, e, n = bbox
        df = df[df["lon"].between(w, e) & df["lat"].between(s_, n)]
        # realistic data defects for the cleaning stage
        if len(df) > 100:
            dup = df.sample(max(len(df) // 200, 1), random_state=int(self.seed % 2**31))
            bad = df.sample(2, random_state=1).assign(lat=91.0)
            spike = df.sample(2, random_state=2)
            spike = spike.assign(lon=spike["lon"] + 0.8,                       # ~85 km jump in 30 s -> impossible
                                 timestamp=spike["timestamp"] + pd.Timedelta(seconds=30))
            df = pd.concat([df, dup, bad, spike])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        self.truth["n_vessels"] = int(df["mmsi"].nunique())
        self.truth.pop("_forced_gaps", None)
        return df.sort_values(["mmsi", "timestamp"])[CANONICAL].reset_index(drop=True)
