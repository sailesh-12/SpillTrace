import { useEffect, useRef } from "react";
import maplibregl, { GeoJSONSource, Map as MLMap } from "maplibre-gl";
import type { BBox, Candidate, FC, Layers, Scene, TriageRow } from "@/api";

export type LayerKey =
  | "sar" | "slicks" | "selected" | "particles" | "heatmap" | "regions" | "ais" | "vessels" | "forward" | "sarships" | "impact";

export const LAYER_LABELS: Record<LayerKey, string> = {
  sar: "Sentinel-1 SAR image",
  slicks: "Detected dark features (triage colour)",
  selected: "Investigated slick",
  particles: "Drift particles (time-resolved)",
  heatmap: "Source probability heatmap",
  regions: "Source regions 50 / 80 / 95 %",
  ais: "AIS vessel tracks",
  vessels: "Vessel positions at cursor time",
  forward: "Forward: projected affected region",
  sarships: "SAR point targets (possible vessels)",
  impact: "Impact: sensitive sites & shoreline threat",
};

export interface TimeCursor { time: Date; kind: "backward" | "forward"; frame: number }

const EMPTY: FC = { type: "FeatureCollection", features: [] };
const STYLE: any = {
  version: 8,
  sources: { osm: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256,
                    attribution: "© OpenStreetMap contributors" } },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#0a1826" } },
    { id: "osm", type: "raster", source: "osm", paint: { "raster-opacity": 0.9, "raster-saturation": -0.45, "raster-hue-rotate": 180, "raster-contrast": 0.25,
      "raster-brightness-min": 0.72, "raster-brightness-max": 0.08 }   /* inverted OSM: dark basemap */ },
  ],
};
const LABEL_COLOR = ["match", ["get", "label"], "OIL_LIKELY", "#ff5a1f", "LOOKALIKE_LIKELY", "#4ade80", "#facc15"];

function bboxPoly(b: BBox) {
  const [w, s, e, n] = b;
  return { type: "Feature", geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] }, properties: {} };
}

function interpTrack(f: any, t: number): [number, number] | null {
  const times: number[] = f.properties._t ?? (f.properties._t = (f.properties.t ?? []).map((s: number) => s * 1000));
  const c = f.geometry.coordinates as [number, number][];
  if (t < times[0] || t > times[times.length - 1]) return null;
  let lo = 0, hi = times.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (times[mid] < t) lo = mid; else hi = mid; }
  const g = times[hi] - times[lo];
  if (g > 2 * 3600e3) return null;                  // never draw vessels through long AIS gaps
  const a = (t - times[lo]) / (g || 1);
  return [c[lo][0] + a * (c[hi][0] - c[lo][0]), c[lo][1] + a * (c[hi][1] - c[lo][1])];
}

export default function MapView(props: {
  data: Layers | null; visible: Record<LayerKey, boolean>; cursor: TimeCursor | null; probLabel: string;
  selected: Candidate | null; onSelectMmsi: (mmsi: string) => void;
  aoi: BBox | null; drawing: boolean; onAoi: (b: BBox) => void;
  scenes: Scene[]; sceneId: string | null; onSceneClick: (id: string) => void;
  checked: Set<string>; onToggleComponent: (id: string) => void; focus: { lon: number; lat: number } | null;
  fitSignal?: number;
}) {
  const { data, visible, cursor, probLabel, selected, aoi, drawing, scenes, sceneId, checked, focus } = props;
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MLMap | null>(null);
  const ready = useRef(false);
  const fitRef = useRef<maplibregl.LngLatBounds | null>(null);
  const cb = useRef(props);
  cb.current = props;

  useEffect(() => {
    const m = new maplibregl.Map({ container: el.current!, style: STYLE, center: [76.5, 15.5], zoom: 4.6,
                                   attributionControl: { compact: true } });
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    m.on("load", () => {
      ["aoi", "footprints", "slicks", "sel", "particles", "prob", "regions", "ais", "vessels", "best", "forward", "sarships", "sites", "threat"]
        .forEach((id) => m.addSource(id, { type: "geojson", data: EMPTY }));
      m.addLayer({ id: "footprints-fill", type: "fill", source: "footprints",
        paint: { "fill-color": ["case", ["get", "active"], "#38bdf8", "#94a3b8"], "fill-opacity": ["case", ["get", "active"], 0.12, 0.03] } });
      m.addLayer({ id: "footprints-line", type: "line", source: "footprints",
        paint: { "line-color": ["case", ["get", "active"], "#38bdf8", "#64748b"], "line-width": ["case", ["get", "active"], 2, 0.8] } });
      m.addLayer({ id: "prob-heat", type: "heatmap", source: "prob",
        paint: { "heatmap-weight": ["get", "w"], "heatmap-intensity": 1.1, "heatmap-opacity": 0.7,
                 "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 6, 6, 9, 22, 12, 70],
                 "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"], 0, "rgba(0,0,0,0)",
                   0.15, "#1e40af", 0.4, "#06b6d4", 0.6, "#fde047", 0.8, "#f97316", 1, "#dc2626"] } });
      m.addLayer({ id: "regions-fill", type: "fill", source: "regions",
        paint: { "fill-color": ["match", ["get", "level"], "high", "#ef4444", "medium", "#f97316", "#eab308"],
                 "fill-opacity": ["match", ["get", "level"], "high", 0.16, "medium", 0.07, 0.03] } });
      m.addLayer({ id: "regions-line", type: "line", source: "regions",
        paint: { "line-color": ["match", ["get", "level"], "high", "#ef4444", "medium", "#f97316", "#eab308"],
                 "line-width": ["match", ["get", "level"], "high", 2, 1], "line-dasharray": [3, 2] } });
      m.addLayer({ id: "forward-fill", type: "fill", source: "forward", paint: { "fill-color": "#a855f7", "fill-opacity": 0.1 } });
      m.addLayer({ id: "forward-line", type: "line", source: "forward",
        paint: { "line-color": "#c084fc", "line-width": 1.4, "line-dasharray": [1, 1.5] } });
      m.addLayer({ id: "slicks-fill", type: "fill", source: "slicks",
        paint: { "fill-color": LABEL_COLOR as any, "fill-opacity": ["case", ["get", "checked"], 0.65, 0.32] } });
      m.addLayer({ id: "slicks-line", type: "line", source: "slicks",
        paint: { "line-color": LABEL_COLOR as any, "line-width": ["case", ["get", "checked"], 2.5, 0.6] } });
      m.addLayer({ id: "sel-line", type: "line", source: "sel", paint: { "line-color": "#ffffff", "line-width": 2.5 } });
      m.addLayer({ id: "ais-line", type: "line", source: "ais",
        paint: { "line-color": ["case", ["get", "selected"], "#22d3ee", ["get", "is_candidate"], "#4ade80", "#64748b"],
                 "line-width": ["case", ["get", "selected"], 4, ["get", "is_candidate"], 1.8, 0.7],
                 "line-opacity": ["case", ["get", "selected"], 1, ["get", "is_candidate"], 0.85, 0.35] } });
      m.addLayer({ id: "particles-pt", type: "circle", source: "particles",
        paint: { "circle-radius": 1.7, "circle-color": ["case", ["==", ["get", "dir"], "forward"], "#c084fc", "#67e8f9"],
                 "circle-opacity": 0.75 } });
      m.addLayer({ id: "sarships-pt", type: "circle", source: "sarships",
        paint: { "circle-radius": 5, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": "#fde047", "circle-stroke-width": 1.6 } });
      m.addLayer({ id: "best-pt", type: "circle", source: "best",
        paint: { "circle-radius": 10, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": "#22d3ee", "circle-stroke-width": 3 } });
      m.addLayer({ id: "vessels-pt", type: "circle", source: "vessels",
        paint: { "circle-radius": ["case", ["get", "selected"], 7, ["get", "is_candidate"], 4.5, 2.5],
                 "circle-color": ["case", ["get", "selected"], "#22d3ee", ["get", "is_candidate"], "#4ade80", "#94a3b8"],
                 "circle-stroke-color": "#0a1826", "circle-stroke-width": 1.2 } });
      m.addLayer({ id: "vessels-label", type: "symbol", source: "vessels", filter: ["any", ["get", "selected"], ["get", "is_candidate"]],
        layout: { "text-field": ["get", "name"], "text-size": 11, "text-offset": [0, 1.1], "text-anchor": "top", "text-optional": true },
        paint: { "text-color": "#e2e8f0", "text-halo-color": "#0a1826", "text-halo-width": 1.5 } });
      m.addLayer({ id: "threat-line", type: "line", source: "threat", filter: ["==", ["geometry-type"], "LineString"],
        paint: { "line-color": ["match", ["get", "kind"], "landing", "#fb923c", "#94a3b8"], "line-width": 1.6, "line-dasharray": [2, 2] } });
      m.addLayer({ id: "sites-halo", type: "circle", source: "sites",
        paint: { "circle-radius": ["case", ["get", "threatened"], 14, 8], "circle-color": ["case", ["get", "threatened"], "#fb923c", "#64748b"],
                 "circle-opacity": ["case", ["get", "threatened"], 0.18, 0.08], "circle-blur": 0.4 } });
      m.addLayer({ id: "sites-pt", type: "circle", source: "sites",
        paint: { "circle-radius": 4.5, "circle-color": ["case", ["get", "threatened"], "#fb923c", "#94a3b8"],
                 "circle-stroke-color": "#0a1826", "circle-stroke-width": 1.5 } });
      m.addLayer({ id: "sites-label", type: "symbol", source: "sites",
        layout: { "text-field": ["get", "label"], "text-size": 10.5, "text-offset": [0, 1.2], "text-anchor": "top", "text-optional": true },
        paint: { "text-color": ["case", ["get", "threatened"], "#fdba74", "#cbd5e1"], "text-halo-color": "#0a1826", "text-halo-width": 1.5 } });
      m.addLayer({ id: "threat-pt", type: "circle", source: "threat", filter: ["==", ["geometry-type"], "Point"],
        paint: { "circle-radius": 7, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": "#fb923c", "circle-stroke-width": 2.5 } });
      m.addLayer({ id: "threat-label", type: "symbol", source: "threat", filter: ["==", ["geometry-type"], "Point"],
        layout: { "text-field": ["get", "label"], "text-size": 11, "text-offset": [0, -1.4], "text-anchor": "bottom" },
        paint: { "text-color": "#fdba74", "text-halo-color": "#0a1826", "text-halo-width": 1.6 } });
      m.addLayer({ id: "aoi-fill", type: "fill", source: "aoi", paint: { "fill-color": "#38bdf8", "fill-opacity": 0.06 } });
      m.addLayer({ id: "aoi-line", type: "line", source: "aoi", paint: { "line-color": "#38bdf8", "line-width": 2, "line-dasharray": [2, 1] } });

      for (const id of ["ais-line", "vessels-pt"]) {
        m.on("click", id, (e) => { const f = e.features?.[0]; if (f) cb.current.onSelectMmsi(String(f.properties.mmsi)); });
        m.on("mouseenter", id, () => (m.getCanvas().style.cursor = "pointer"));
        m.on("mouseleave", id, () => (m.getCanvas().style.cursor = ""));
      }
      m.on("click", "slicks-fill", (e) => {
        const p = e.features?.[0]?.properties;
        if (!p || cb.current.drawing) return;
        cb.current.onToggleComponent(String(p.component_id));
        new maplibregl.Popup({ closeButton: false }).setLngLat(e.lngLat).setHTML(
          `<b>#${p.rank ?? "–"} ${String(p.component_id).split("_").pop()}</b> · ${p.label ?? ""}<br/>` +
          `${Number(p.area_km2 ?? 0).toFixed(2)} km² · elongation ${Number(p.elongation ?? 0).toFixed(1)}`).addTo(m);
      });
      m.on("click", "footprints-fill", (e) => {
        if (cb.current.drawing) return;
        const f = e.features?.[0]; if (f) cb.current.onSceneClick(String(f.properties.id));
      });
      ready.current = true;
      m.fire("sih:ready" as any);
    });
    map.current = m;
    return () => m.remove();
  }, []);

  const whenReady = (fn: (m: MLMap) => void) => {
    const m = map.current; if (!m) return;
    if (ready.current) fn(m); else m.once("sih:ready" as any, () => fn(m));
  };
  const setData = (m: MLMap, id: string, d: any) => (m.getSource(id) as GeoJSONSource | undefined)?.setData(d ?? EMPTY);

  // --- AOI drawing (drag a rectangle) -----------------------------------------------------------
  useEffect(() => {
    const m = map.current;
    if (!m || !ready.current) return;
    {
      if (!drawing) { m.dragPan.enable(); m.getCanvas().style.cursor = ""; return; }
      m.dragPan.disable(); m.getCanvas().style.cursor = "crosshair";
      let start: maplibregl.LngLat | null = null;
      const down = (e: maplibregl.MapMouseEvent) => { start = e.lngLat; };
      const move = (e: maplibregl.MapMouseEvent) => {
        if (!start) return;
        const b: BBox = [Math.min(start.lng, e.lngLat.lng), Math.min(start.lat, e.lngLat.lat),
                         Math.max(start.lng, e.lngLat.lng), Math.max(start.lat, e.lngLat.lat)];
        setData(m, "aoi", { type: "FeatureCollection", features: [bboxPoly(b)] });
      };
      const up = (e: maplibregl.MapMouseEvent) => {
        if (!start) return;
        const b: BBox = [Math.min(start.lng, e.lngLat.lng), Math.min(start.lat, e.lngLat.lat),
                         Math.max(start.lng, e.lngLat.lng), Math.max(start.lat, e.lngLat.lat)];
        start = null;
        if (b[2] - b[0] > 0.01 && b[3] - b[1] > 0.01) cb.current.onAoi(b.map((v) => +v.toFixed(4)) as BBox);
      };
      m.on("mousedown", down); m.on("mousemove", move); m.on("mouseup", up);
      return () => { m.off("mousedown", down); m.off("mousemove", move); m.off("mouseup", up); };
    }
  }, [drawing]);

  useEffect(() => { whenReady((m) => setData(m, "aoi", aoi ? { type: "FeatureCollection", features: [bboxPoly(aoi)] } : null)); }, [aoi]);

  useEffect(() => {
    whenReady((m) => setData(m, "footprints", { type: "FeatureCollection", features: scenes.map((s) => ({
      type: "Feature", geometry: s.footprint, properties: { id: s.id, active: s.id === sceneId } })) }));
  }, [scenes, sceneId]);

  // --- per-analysis static layers ----------------------------------------------------------------
  useEffect(() => {
    whenReady((m) => {
      if (m.getLayer("sar-img")) m.removeLayer("sar-img");
      if (m.getSource("sar")) m.removeSource("sar");
      if (!data) return;
      const corners = data.scene?.image_corners_lonlat;
      if (corners) {
        const bust = `?v=${encodeURIComponent(data.analysis?.created_at ?? "")}`;
        m.addSource("sar", { type: "image", url: data.sar_image_url + bust, coordinates: corners });
        m.addLayer({ id: "sar-img", type: "raster", source: "sar", paint: { "raster-opacity": 0.92, "raster-fade-duration": 0 } }, "prob-heat");
      }
      setData(m, "regions", data.source_probability ? { type: "FeatureCollection",
        features: data.source_probability.features.filter((f) => f.properties.kind === "source_region") } : null);
      const fr = data.forward?.affected_region;
      setData(m, "forward", fr ? { type: "FeatureCollection", features: Object.values(fr).map((r: any) => ({
        type: "Feature", geometry: r.geometry, properties: {} })) } : null);
      const dets = data.triage?.sar_ship_detection?.detections ?? [];
      setData(m, "sarships", { type: "FeatureCollection", features: dets.filter((d: any) => d.lon != null).map((d: any) => ({
        type: "Feature", geometry: { type: "Point", coordinates: [d.lon, d.lat] }, properties: {} })) });
      setData(m, "sel", data.selected_geojson);
      const th = data.impact?.threat;
      setData(m, "sites", th ? { type: "FeatureCollection", features: th.sites.map((s: any) => ({ type: "Feature",
        geometry: { type: "Point", coordinates: [s.lon, s.lat] }, properties: { threatened: s.threatened,
          label: s.threatened && s.eta_hours != null ? `${s.name} · ${s.eta_hours < 72 ? s.eta_hours.toFixed(0) + " h" : (s.eta_hours / 24).toFixed(1) + " d"}` : s.name } })) } : null);
      const tf: any[] = [];
      const c0 = data.selected?.centroid;
      if (th?.nearest_shore_point && c0) tf.push({ type: "Feature", properties: { kind: "nearest" },
        geometry: { type: "LineString", coordinates: [[c0.lon, c0.lat], [th.nearest_shore_point.lon, th.nearest_shore_point.lat]] } });
      const bl = th?.beaching?.landing_point;
      if (bl) {
        const from = th.drift_vector?.to ?? c0;
        if (from) tf.push({ type: "Feature", properties: { kind: "landing" },
          geometry: { type: "LineString", coordinates: [[from.lon, from.lat], [bl.lon, bl.lat]] } });
        const h = th.beaching.eta_hours;
        tf.push({ type: "Feature", properties: { label: `Shore ETA ${h < 72 ? h.toFixed(0) + " h" : (h / 24).toFixed(1) + " d"}` },
          geometry: { type: "Point", coordinates: [bl.lon, bl.lat] } });
      }
      setData(m, "threat", { type: "FeatureCollection", features: tf });
      const b = new maplibregl.LngLatBounds();
      if (corners) corners.forEach((c: [number, number]) => b.extend(c));
      data.source_probability?.features.filter((f) => f.properties.kind === "source_region" && f.properties.level === "low")
        .forEach((f) => { const walk = (x: any): void => { if (typeof x[0] === "number") b.extend(x); else x.forEach(walk); }; walk(f.geometry.coordinates); });
      if (!b.isEmpty()) {
        fitRef.current = b;
        requestAnimationFrame(() => { m.resize(); m.fitBounds(b, { padding: 70, duration: 700, maxZoom: 12 }); });
      }
    });
  }, [data]);

  // --- slicks coloured by triage label; checked ones emphasised -------------------------------------
  useEffect(() => {
    whenReady((m) => {
      const rows = new Map<string, TriageRow>((data?.triage?.components ?? []).map((r) => [r.component_id, r]));
      setData(m, "slicks", data?.spill_geojson ? { type: "FeatureCollection", features: data.spill_geojson.features.map((f) => {
        const r = rows.get(f.properties.component_id);
        return { ...f, properties: { ...f.properties, label: r?.label ?? "UNCERTAIN", rank: r?.triage_rank,
          elongation: r?.elongation, area_km2: r?.area_km2, checked: checked.has(f.properties.component_id) } };
      }) } : null);
    });
  }, [data, checked]);

  useEffect(() => { if (focus) whenReady((m) => m.flyTo({ center: [focus.lon, focus.lat], zoom: Math.max(m.getZoom(), 11), duration: 600 })); }, [focus]);
  useEffect(() => {
    const bp = selected?.features?.best_match_position;
    if (bp) whenReady((m) => m.flyTo({ center: [bp.lon, bp.lat], zoom: Math.max(m.getZoom(), 9.5), duration: 700 }));
  }, [selected]);
  useEffect(() => {
    if (props.fitSignal) whenReady((m) => fitRef.current && m.fitBounds(fitRef.current, { padding: 70, duration: 600, maxZoom: 12 }));
  }, [props.fitSignal]);

  useEffect(() => {
    whenReady((m) => {
      const fc = data?.ais_tracks;
      setData(m, "ais", fc ? { ...fc, features: fc.features.map((f) => ({ ...f,
        properties: { ...f.properties, selected: f.properties.mmsi === selected?.vessel.mmsi } })) } : null);
      const bp = selected?.features?.best_match_position;
      setData(m, "best", bp ? { type: "FeatureCollection", features: [{ type: "Feature",
        geometry: { type: "Point", coordinates: [bp.lon, bp.lat] }, properties: {} }] } : null);
    });
  }, [data, selected]);

  useEffect(() => {
    whenReady((m) => {
      const cells = (data?.source_probability?.features ?? []).filter(
        (f) => f.properties.kind === "probability_cell" && f.properties.map === probLabel);
      const max = Math.max(1e-12, ...cells.map((f) => f.properties.probability));
      setData(m, "prob", { type: "FeatureCollection", features: cells.map((f) => ({ ...f, properties: { w: f.properties.probability / max } })) });
    });
  }, [data, probLabel]);

  useEffect(() => {
    whenReady((m) => {
      if (!data || !cursor) { setData(m, "particles", null); setData(m, "vessels", null); return; }
      const fr = cursor.kind === "backward" ? data.backward_particles : data.forward_particles;
      setData(m, "particles", { type: "FeatureCollection", features: [{ type: "Feature",
        geometry: { type: "MultiPoint", coordinates: fr?.frames[cursor.frame] ?? [] }, properties: { dir: cursor.kind } }] });
      const t = cursor.time.getTime();
      const vs = (data.ais_tracks?.features ?? []).map((f) => {
        const p = interpTrack(f, t);
        return p && { type: "Feature", geometry: { type: "Point", coordinates: p }, properties: { mmsi: f.properties.mmsi,
          name: f.properties.name ?? f.properties.mmsi, is_candidate: f.properties.is_candidate, selected: f.properties.mmsi === selected?.vessel.mmsi } };
      }).filter(Boolean);
      setData(m, "vessels", { type: "FeatureCollection", features: vs });
    });
  }, [data, cursor, selected]);

  useEffect(() => {
    whenReady((m) => {
      const groups: Record<LayerKey, string[]> = {
        sar: ["sar-img"], slicks: ["slicks-fill", "slicks-line"], selected: ["sel-line"], particles: ["particles-pt"],
        heatmap: ["prob-heat"], regions: ["regions-fill", "regions-line"], ais: ["ais-line", "best-pt"],
        vessels: ["vessels-pt", "vessels-label"], forward: ["forward-fill", "forward-line"], sarships: ["sarships-pt"],
        impact: ["threat-line", "sites-halo", "sites-pt", "sites-label", "threat-pt", "threat-label"],
      };
      (Object.keys(groups) as LayerKey[]).forEach((k) => groups[k].forEach((id) => {
        if (m.getLayer(id)) m.setLayoutProperty(id, "visibility", visible[k] ? "visible" : "none");
      }));
    });
  }, [visible, data]);

  // maplibre forces position:relative on its container, so the sizing wrapper must be separate
  return <div className="absolute inset-0"><div ref={el} className="h-full w-full" /></div>;
}
