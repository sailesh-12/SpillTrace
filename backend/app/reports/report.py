"""Investigator report: JSON (machine), Markdown and HTML (human).

The report is investigative decision support. It separates OBSERVED FACT,
MODEL-DERIVED RESULT, ASSUMPTION and CANDIDATE INFERENCE and never states that
a vessel caused the spill.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

LEGAL = ("This report is investigative decision support. It does not establish legal responsibility. "
         "Candidate vessels are those whose AIS tracks correlate with the modelled drift; further evidence "
         "(e.g. oil sampling and fingerprinting, inspection, logbooks) is required for attribution.")


def _try(an, name):
    try:
        return an.load(name)
    except FileNotFoundError:
        return None


def build_report(an) -> dict:
    seg = _try(an, "spill.json") or {}
    drift = _try(an, "drift/summary.json")
    fwd = _try(an, "drift/forward_summary.json")
    cand = _try(an, "candidates.json")
    spill = seg.get("spill")
    scene = seg.get("scene", {})
    assumptions = [
        "Segmentation preprocessing replicates training (RGB/255, no normalisation); threshold "
        f"{seg.get('segmentation', {}).get('threshold')} (configurable).",
    ]
    if scene.get("georef", {}) and scene["georef"].get("provenance") == "DEMO_ASSUMED":
        assumptions.append("Scene location, pixel spacing and timestamp are DEMO ASSUMPTIONS from the sidecar file.")
    if drift:
        assumptions += [f"Oil type {drift['oil']['oil_type']}; modelled as {drift['oil']['oil_model_assumption']}.",
                        f"Release time uniform over {drift['release_window']['offsets_hours']} h before observation; "
                        f"window basis: {drift['release_window'].get('basis', 'configured offsets')}.",
                        f"Wind drift factor range {[m['wind_drift_factor'] for m in drift['members']]}.",
                        "Weathering disabled in backward runs (irreversible processes); transport only."]
    limitations = [
        f"Segmentation model: validation Dice {seg.get('segmentation', {}).get('model', {}).get('metadata', {}).get('val_dice', 'n/a')}"
        " on SOS validation split (reported by training); not validated on the demo region.",
        "The SOS training dataset contains no look-alike labels; look-alike assessment is rule-based.",
        "Evidence Correlation Score weights are a prototype heuristic, not calibrated probabilities.",
        "AIS coverage may be incomplete; vessels without AIS cannot be ranked (see SAR ship detection).",
    ]
    ais_sel = (cand or {}).get("ais", {}) or {}
    if ais_sel.get("synthetic"):
        sel = ais_sel.get("selection") or {}
        assumptions.insert(0, "AIS vessel data are SYNTHETIC: generated around the spill region to demonstrate the "
                              "algorithm, as permitted by SIH26143 (\"Real AIS if available may be used else synthetic "
                              "data can be prepared for the region of oil spill to demonstrate the functioning of the "
                              "algorithm\"). No synthetic vessel is real."
                              + (f" Real AIS attempts: {sel.get('fallback_reason')}." if sel.get("fallback_reason") else ""))
    acq = _try(an, "acquisition.json") or {}
    if acq.get("provenance") == "SYNTHETIC_DEMO":
        assumptions.insert(0, "The SAR scene is SYNTHETIC (generated for the drawn area because no usable real "
                              "Sentinel-1 coverage was selected): speckle, wind texture, ships, one ship-trailing "
                              "discharge and a natural look-alike. It is not a satellite observation.")
    if drift and str(drift["forcing"]["provenance"]).startswith("SYNTHETIC"):
        assumptions.insert(0, "Currents and wind are SYNTHETIC (Indian-waters monsoon-climatology-inspired analytic "
                              "field) because real forcing was unavailable.")
    sar_src = ("SYNTHETIC SAR scene (demonstration)" if acq.get("provenance") == "SYNTHETIC_DEMO" else
               f"Sentinel-1 {acq.get('item_id')} (real, {acq.get('provenance')})" if acq else
               "SOS dataset tile (real SAR pixels)" if "sample" in str(scene.get("path", "")).lower()
               or "demo" in str(scene.get("path", "")).lower() else "user-supplied")
    data_prov = {"sar_image": sar_src,
                 "georeferencing": (scene.get("georef") or {}).get("provenance", "UNAVAILABLE"),
                 "environmental_forcing": drift["forcing"]["provenance"] if drift else "UNAVAILABLE",
                 "ais": cand["ais"]["provider"]["provenance"] if cand else "UNAVAILABLE"}
    return {
        "title": "Oil Spill Investigation Report",
        "analysis_id": an.id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": an.state.get("outcome") or an.state.get("status"),
        "error": an.state.get("error"),
        "warnings": an.state.get("warnings", []),
        "data_provenance": data_prov,
        "observed": {"scene": scene, "timestamp": scene.get("timestamp")},
        "segmentation": seg.get("segmentation"),
        "spill": spill,
        "pixel_components": seg.get("pixel_components"),
        "drift": drift,
        "forward": fwd,
        "ais": cand.get("ais") if cand else None,
        "candidates": cand.get("candidates") if cand else [],
        "candidate_message": cand.get("message") if cand else None,
        "ranking_separation": cand.get("ranking_separation") if cand else None,
        "filtered_out": cand.get("filtered_out") if cand else [],
        "sar_ship_detection": cand.get("sar_ship_detection") if cand else None,
        "sar_ais_mismatch": cand.get("sar_ais_mismatch") if cand else None,
        "lookalike": cand.get("lookalike") if cand else None,
        "multitemporal": cand.get("multitemporal") if cand else None,
        "score_disclaimer": cand.get("disclaimer") if cand else None,
        "weights": cand.get("weights") if cand else None,
        "impact": _try(an, "impact.json"),
        "assumptions": assumptions,
        "limitations": limitations,
        "legal_notice": LEGAL,
    }


def _f(v, fmt="{:.2f}"):
    return "n/a" if v is None else fmt.format(v)


def render_markdown(r: dict) -> str:
    L = [f"# {r['title']}", f"**Analysis:** `{r['analysis_id']}` · generated {r['generated_at'][:19]}Z · status **{r['status']}**", ""]
    L.append(f"> {r['legal_notice']}")
    if r["warnings"]:
        L += ["", "## ⚠ Data warnings"] + [f"- {w}" for w in r["warnings"]]
    if r.get("error"):
        L += ["", f"## Pipeline limitation\n**{r['error'].get('message')}**  \n_{r['error'].get('hint', '')}_"]
    L += ["", "## Data provenance"] + [f"- **{k}**: {v}" for k, v in r["data_provenance"].items()]
    s = r.get("spill")
    L += ["", "## 1. Observed spill  _(OBSERVED FACT + MODEL-DERIVED)_"]
    seg = r.get("segmentation") or {}
    L.append(f"- Segmentation: {seg.get('n_components')} component(s), detection confidence "
             f"{_f(seg.get('detection_confidence'))} (mean sigmoid probability over detected pixels, not calibrated)")
    if s:
        L += [f"- Timestamp (satellite acquisition): {s['timestamp']}",
              f"- Centroid: {s['centroid']['lat']:.5f} N, {s['centroid']['lon']:.5f} E",
              f"- Area: {s['area_m2'] / 1e6:.4f} km² · Perimeter: {s['perimeter_m'] / 1000:.2f} km",
              f"- Bounding box [W,S,E,N]: {[round(v, 5) for v in s['bbox']]}",
              f"- Shape: elongation {s['shape']['elongation']:.2f}, compactness {s['shape']['compactness']:.3f}"]
    la = r.get("lookalike")
    if la:
        L += ["", f"### Look-alike assessment: **{la['label']}**  _(MODEL-DERIVED, heuristic)_"] + \
             [f"- {x}" for x in la["reasons"]] + [f"- _{la['uncertainty']}_"]
    d = r.get("drift")
    if d:
        rw = d["release_window"]
        L += ["", "## 2. Backward drift & source region  _(MODEL-DERIVED)_",
              f"- Model: {d['model']}; {d['n_members']} ensemble members × {d['particles_per_member']} particles",
              f"- Forcing: {d['forcing']['provenance']} · at spill: current {d['forcing_at_spill']['current_speed']:.2f} m/s "
              f"toward {d['forcing_at_spill']['current_to_deg']:.0f}°, wind {d['forcing_at_spill']['wind_speed']:.1f} m/s "
              f"from {d['forcing_at_spill']['wind_from_deg']:.0f}°",
              f"- Oil: type {d['oil']['oil_type']}, model assumption {d['oil']['oil_model_assumption']}",
              f"- Release-time window: {rw['start']} → {rw['end']} ({rw['prior']})", "",
              "| Release time | Centre (lat, lon) | Spread (km) | High-prob. area (km²) |", "|---|---|---|---|"]
        for p in d["per_release_time"]:
            L.append(f"| {p['label']} ({p['release_time'][:16]}) | {p['center']['lat']:.4f}, {p['center']['lon']:.4f} | "
                     f"{p['spread_km']} | {p['high_region_area_km2']} |")
        L += ["", "Source probability regions (combined over window):"] + \
             [f"- **{k}** ({int(100 * v['mass_fraction'])}% mass): {v['area_km2']:.1f} km²" for k, v in d["source_regions"].items()]
        L += [f"- _{n}_" for n in d["notes"]]
    fw = r.get("forward")
    if fw:
        L += ["", "## 3. Forward drift (future flow)  _(MODEL-DERIVED)_"]
        if fw.get("available"):
            L += [f"- Forecast window: {fw['forecast_window'][0]} → {fw['forecast_window'][1]} ({fw['forcing_provenance']})",
                  f"- Predicted centre after {fw['hours']} h: {fw['end_center']['lat']:.4f}, {fw['end_center']['lon']:.4f} "
                  f"(spread {fw['end_center']['spread_km']:.1f} km); stranded particles: {fw['stranded_particles']}"]
            for k, v in fw["affected_region"].items():
                L.append(f"- Projected affected region (95% mass): {v['area_km2']:.1f} km²")
        else:
            L.append(f"- {fw.get('reason', 'Forward prediction unavailable.')}")
    if r.get("ais"):
        c = r["ais"]["cleaning"]
        L += ["", "## 4. AIS coverage  _(OBSERVED FACT)_",
              f"- Provider: {r['ais']['provider']['provider']} ({r['ais']['provider']['provenance']})",
              f"- Query: bbox {[round(v, 3) for v in r['ais']['query']['bbox']]}, {r['ais']['query']['start'][:16]} → {r['ais']['query']['end'][:16]}",
              f"- {c['input_rows']} rows → {c['positions_used']} positions / {c['vessels']} vessels; removed "
              f"{c['duplicates']} duplicates, {c['invalid_coordinates']} invalid coordinates, "
              f"{c['missing_timestamp']} missing timestamps, {c['impossible_speed']} impossible-speed fixes"]
    sm = r.get("sar_ais_mismatch")
    if sm and sm.get("available"):
        sd = r["sar_ship_detection"]
        L += ["", "## 5. SAR ship detection & SAR/AIS visibility  _(MODEL-DERIVED)_",
              f"- Detector: {sd['detector']}: {sd['n_detections']} detection(s). {sd['caveat']}",
              f"- Matched: {len(sm['matches'])}; SAR-only: {len(sm['sar_detections_without_ais'])}; "
              f"AIS-only in scene: {len(sm['ais_vessels_without_sar_detection'])}", f"- {sm['note']}"]
    L += ["", "## 6. Candidate vessels requiring investigation  _(CANDIDATE INFERENCE)_"]
    if r.get("candidate_message"):
        L.append(f"_{r['candidate_message']}_")
    if r.get("score_disclaimer"):
        L.append(f"> {r['score_disclaimer']}")
    if r.get("ranking_separation"):
        L.append(f"- **Ranking separation:** {r['ranking_separation']['note']}")
    if r.get("candidates"):
        L += ["", "| Rank | Vessel | MMSI | Type | Evidence Correlation Score | Dist. to source (km) | Δt (h) |",
              "|---|---|---|---|---|---|---|"]
        for c in r["candidates"]:
            L.append(f"| {c['rank']} | {c['vessel']['name']} | {c['vessel']['mmsi']} | {c['vessel']['type']} | "
                     f"{c['score']:.1f} | {_f(c['source_distance_km'], '{:.1f}')} | {_f(c['time_difference_hours'], '{:.1f}')} |")
        for c in r["candidates"]:
            ev = c["evidence"]
            L += ["", f"### {c['rank']}. {c['vessel']['name']} (MMSI {c['vessel']['mmsi']}) — {c['score']:.1f} / 100",
                  "Feature scores: " + ", ".join(f"{k} {v}" for k, v in c["feature_scores_pct"].items())]
            for st in ev["statements"]:
                L.append(f"- **[{st['kind']}] {st['section']}:** {st['text']}")
            L += ["", "**Supporting:** " + ("; ".join(ev["supporting_evidence"]) or "none"),
                  "**Contradicting:** " + ("; ".join(ev["contradicting_evidence"]) or "none"),
                  "**Missing:** " + ("; ".join(ev["missing_evidence"]) or "none"),
                  "**Uncertainty:** " + " ".join(ev["uncertainties"])]
    if r.get("filtered_out"):
        L += ["", "### Vessels filtered out (with reasons)"] + \
             [f"- {v['name'] or v['mmsi']} ({v['vessel_type']}): {v['reason']}" for v in r["filtered_out"]]
    imp = r.get("impact")
    if imp:
        sv, th = imp["severity"], imp["threat"]
        L += ["", "## 6b. Impact assessment & immediate response  _(MODEL-DERIVED / DECISION SUPPORT)_",
              f"**{sv['name']}: {sv['score']:.0f} / 100 — {sv['level']}** · planning estimate {sv['response_tier']['central']} "
              f"({sv['response_tier']['basis']})",
              f"- Estimated volume {sv['volume_estimate']['m3']['low']:.1f}–{sv['volume_estimate']['m3']['high']:.0f} m³ "
              f"({sv['volume_estimate']['basis']})"]
        L += [f"- {k}: {v['value'] * 100:.0f} % × weight {v['weight']:.2f} — {v['evidence']}" for k, v in sv["components"].items()]
        b = th["beaching"]
        if th.get("distance_to_coast_km") is not None:
            L.append(f"- Nearest shoreline: {th['distance_to_coast_km']:.1f} km")
        L.append(f"- Shoreline threat: " + (f"~{b['eta_hours']:.0f} h to {b['landing_place']} ({b['note']})"
                                            if b.get("eta_hours") is not None else b.get("note", "none")))
        thr = [s for s in th["sites"] if s["threatened"]]
        if thr:
            L += ["", "| Threatened receptor | Type | Sensitivity | ETA (h) |", "|---|---|---|---|"]
            L += [f"| {s['name']} | {s['type_label']} | {s['weight']}/10 | {s['eta_hours']:.0f} |" for s in thr]
        L += ["", f"**Immediate actions** (report to {imp['response']['mrcc']['name']}):", "",
              "| # | Priority | Within (h) | Action | Agency |", "|---|---|---|---|---|"]
        L += [f"| {a['id']} | {a['priority']} | {a['deadline_hours']:.0f} | {a['title']} | {a['agency']} |" for a in imp["response"]["actions"]]
        L += [f"- {n}" for n in imp["response"]["notes"]]
        if imp.get("robustness", {}).get("available"):
            L += ["", f"**Ranking robustness:** {imp['robustness']['note']}"]
        L += ["", "_Sensitive-site coordinates are approximate reference points, not official ESI maps._"]
    L += ["", "## 7. Assumptions  _(ASSUMPTION)_"] + [f"- {a}" for a in r["assumptions"]]
    L += ["", "## 8. Limitations"] + [f"- {a}" for a in r["limitations"]]
    return "\n".join(L) + "\n"


def render_html(r: dict) -> str:
    """Minimal self-contained HTML rendering of the Markdown report."""
    md = render_markdown(r)
    out, in_table, in_list = [], False, False
    import re

    def inline(t):
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        t = re.sub(r"(?<![\w*])_(.+?)_(?![\w*])", r"<em>\1</em>", t)
        return t

    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue
            if not in_table:
                out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>"); in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        if line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{inline(line[2:])}</blockquote>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table: out.append("</table>")
    if in_list: out.append("</ul>")
    css = ("body{font-family:system-ui,Segoe UI,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1b2430;"
           "line-height:1.5}h1{border-bottom:3px solid #0b5394}h2{margin-top:2rem;color:#0b5394}table{border-collapse:collapse;"
           "width:100%;margin:.5rem 0}td,th{border:1px solid #ccd;padding:.3rem .5rem;font-size:.9rem;text-align:left}"
           "th{background:#eef3fa}blockquote{background:#fff7e0;border-left:4px solid #e0a800;margin:1rem 0;padding:.5rem 1rem}"
           "code{background:#f1f3f6;padding:0 .25rem}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(r['title'])} {r['analysis_id']}</title>" \
           f"<style>{css}</style></head><body>{''.join(out)}</body></html>"
