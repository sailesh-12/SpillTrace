"""Plain-language, epistemically-labelled evidence summaries per candidate.

Each statement is tagged OBSERVED_FACT / MODEL_DERIVED / ASSUMPTION /
CANDIDATE_INFERENCE so an investigator can see what kind of claim it is.
"""
from __future__ import annotations

from datetime import datetime

from app.core.provenance import EvidenceKind as K


def _t(iso: str | None) -> str:
    return datetime.fromisoformat(iso).strftime("%d %b %Y %H:%M UTC") if iso else "n/a"


def build_candidate_evidence(c: dict, ctx: dict) -> dict:
    """c: candidate dict with features & feature_scores; ctx: analysis-level context."""
    f, fs = c["features"], c["feature_scores"]
    tm, b, cont = f["temporal"], f["behavior"], f["continuity"]
    stmts, sup, con, miss = [], [], [], []

    syn = bool(ctx.get("synthetic_ais"))

    def add(kind, section, text):
        if syn and section in ("AIS", "Spatial", "Trajectory", "Temporal", "Vessel type", "Behaviour",
                               "AIS continuity", "SAR/AIS at acquisition", "Assessment"):
            text = "[SYNTHETIC AIS] " + text
        stmts.append({"kind": kind.value, "section": section, "text": text})

    add(K.OBSERVED, "AIS", f"AIS reports {c['vessel']['name'] or 'an unnamed vessel'} (MMSI {c['vessel']['mmsi']}) "
                           f"with {f.get('n_positions', 'n/a')} position reports in the retrieved window "
                           f"[AIS data provenance: {ctx['ais_provenance']}].")
    d = f["source_distance_km"]
    add(K.MODEL, "Spatial", f"The track passed {'through' if d == 0 else f'within {d:.1f} km of'} the "
                            f"high-probability backtracked source region (closest approach {_t(f['closest_approach_time'])}).")
    (sup if d <= 5 else con).append(f"Closest approach to high-probability source region: {d:.1f} km")
    dst = f["spatiotemporal_distance_km"]
    add(K.MODEL, "Trajectory", f"At {_t(f['best_match_time'])} the vessel was {dst:.1f} km from the nearest "
                               f"backtracked oil particle at that same time "
                               f"({100 * f['particle_fraction_within_buffer']:.0f}% of the ensemble within "
                               f"{ctx['corridor_buffer_km']} km).")
    (sup if dst <= 3 else con).append(f"Time-matched distance to backtracked oil: {dst:.1f} km")
    if tm.get("time_difference_hours") is not None:
        spread = tm.get("oil_time_spread_hours")
        add(K.MODEL, "Temporal", f"At its best-match time ({_t(tm['vessel_time'])}) the vessel was "
                                 f"{tm['time_difference_hours']:.1f} h from the density-weighted time at which the "
                                 f"backtracked oil occupied that location ({_t(tm['oil_time_at_location'])}"
                                 + (f", ± {spread:.1f} h" if spread is not None else "") + ").")
        consistent = tm["time_difference_hours"] <= max(3.0, spread or 0)
        (sup if consistent else con).append(
            f"Temporal {'consistency' if consistent else 'mismatch'} with drift: Δt {tm['time_difference_hours']:.1f} h"
            + (f" (oil-occupancy spread ± {spread:.1f} h)" if spread is not None else ""))
    else:
        miss.append("Temporal consistency could not be evaluated: " + tm.get("note", "no drift overlap"))
    sa = f.get("sar_attached") or {}
    if sa.get("matched"):
        add(K.OBSERVED, "SAR/AIS at acquisition",
            f"At the satellite acquisition time a SAR point target {sa['target_to_slick_km']:.2f} km from the selected "
            f"slick matches this vessel's AIS position ({sa['ais_to_target_km']:.2f} km apart, AIS report "
            f"{sa['ais_time_offset_min']:+.0f} min from acquisition).")
        add(K.INFERENCE, "SAR/AIS at acquisition",
            "A vessel at the head of a dark streak is the classic signature of an operational discharge, but at low wind "
            "the streak can also be the vessel's own turbulent wake; the pattern is strong but not conclusive evidence.")
        sup.append("Vessel observed by SAR and AIS at the head of the slick at acquisition time")
    elif ctx.get("sar_attached_available"):
        con.append("Not the vessel observed at the slick by SAR at acquisition time")
    vt = f["vessel_type"]
    add(K.OBSERVED if vt != "unknown" else K.ASSUMPTION, "Vessel type",
        f"AIS metadata identifies the vessel as: {vt}. Vessel type is contextual only — any ship carries "
        f"bunker fuel and can be a source.")
    if vt == "unknown":
        miss.append("Vessel type not reported in AIS")
    if b.get("available"):
        parts = []
        if b["speed_before_kn"] is not None: parts.append(f"{b['speed_before_kn']:.1f} kn before")
        if b["speed_near_kn"] is not None: parts.append(f"{b['speed_near_kn']:.1f} kn near the matched location")
        if b["speed_after_kn"] is not None: parts.append(f"{b['speed_after_kn']:.1f} kn after")
        add(K.OBSERVED, "Behaviour", "Position-derived speed: " + ", ".join(parts) +
            (f"; net course change {b['net_course_change_deg']:.0f}°" if b["net_course_change_deg"] is not None else "") +
            (f"; {b['stop_duration_min']:.0f} min below 2 kn" if b["stop_duration_min"] else "") + ".")
        if b["slowdown_ratio"] is not None and b["slowdown_ratio"] < 0.6:
            sup.append(f"Slowdown near matched location (speed ratio {b['slowdown_ratio']:.2f})")
            add(K.INFERENCE, "Behaviour", "The slowdown is a behavioural anomaly consistent with, but not "
                                          "specific to, a discharge; it can also reflect routine operations.")
        else:
            con.append("No marked slowdown near the matched location")
    else:
        miss.append("Insufficient AIS positions near the matched location for behaviour analysis")
    if cont["relevant_gaps"]:
        g = max(cont["relevant_gaps"], key=lambda g: g["duration_min"])
        add(K.OBSERVED, "AIS continuity", f"A {g['duration_min']:.0f}-minute AIS continuity anomaly occurred "
                                          f"({_t(g['gap_start'])} – {_t(g['gap_end'])}). This is an observational "
                                          f"anomaly and is not by itself evidence of intentional behaviour.")
        sup.append(f"AIS continuity anomaly of {g['duration_min']:.0f} min near relevant time (weak indicator)")
        miss.append("Vessel position during AIS gap is unknown (not interpolated in evidence)")
    if not c["vessel"].get("imo"):
        miss.append("IMO number not available")
    miss.append("No sample-based oil fingerprinting (chemical matching) available")
    if ctx.get("sar_ship_match") is None:
        miss.append("No SAR vessel detection corroborating the vessel's position at the observation time")

    uncertainties = [
        f"Source-region uncertainty: high-probability region area {ctx['high_region_area_km2']:.1f} km² "
        f"(ensemble of {ctx['n_members']} runs × {ctx['n_particles']} particles).",
        f"Release-time uncertainty: window {_t(ctx['release_window']['start'])} – {_t(ctx['release_window']['end'])} "
        f"(uniform prior across configured offsets).",
        f"Environmental forcing provenance: {ctx['forcing_provenance']}.",
        f"AIS coverage: median report interval {f.get('median_report_interval_s') or 0:.0f} s; "
        f"{len(cont['all_gaps'])} gap(s) > {ctx['gap_threshold_min']} min on this track.",
        f"Georeferencing provenance: {ctx['georef_provenance']}.",
        "SAR/AIS matching uncertainty: " + ctx.get("sar_ais_note", "not evaluated"),
    ]
    if syn:
        miss.insert(0, "Real AIS for this region/time (the vessel tracks used here are SYNTHETIC)")
    rank_word = "the strongest" if c["rank"] == 1 else f"rank-{c['rank']}"
    summary = (f"Candidate: {c['vessel']['name'] or c['vessel']['mmsi']} — Evidence Correlation Score "
               f"{c['score']:.1f} / 100 ({rank_word} evidence correlation among {ctx['n_candidates']} retrieved "
               f"candidates). Assessment: the vessel is a candidate requiring further investigation. "
               f"The evidence does not establish that the vessel caused the spill."
               + (" This vessel is SYNTHETIC (generated for demonstration) and does not exist." if syn else ""))
    add(K.INFERENCE, "Assessment", summary)
    return {"summary": summary, "statements": stmts, "supporting_evidence": sup,
            "contradicting_evidence": con, "missing_evidence": miss, "uncertainties": uncertainties}
