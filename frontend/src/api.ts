// Typed client for the FastAPI backend. All science happens server-side.

export type Feature = { type: "Feature"; geometry: any; properties: Record<string, any> };
export type FC = { type: "FeatureCollection"; features: Feature[] };
export type BBox = [number, number, number, number];

export interface ProgressEvent {
  time: string; stage: string; status: string; message: string; progress: number | null; hint?: string;
}

export interface Scene {
  id: string; datetime: string; platform: string; orbit_state: string; relative_orbit?: number;
  footprint: any; aoi_coverage: number; thumbnail?: string | null; provenance: string;
}

export interface TriageRow {
  component_id: string; triage_rank: number; label: "OIL_LIKELY" | "LOOKALIKE_LIKELY" | "UNCERTAIN";
  triage_score: number; area_km2: number; length_km: number; elongation: number; solidity: number;
  contrast_db: number | null; nearest_sar_target_km: number | null; ship_trail: boolean;
  mean_probability: number; centroid: { lat: number; lon: number }; reasons: string[];
}

export interface Frames {
  times: string[]; frames: [number, number][][]; direction: "backward" | "forward";
  n_members: number; n_particles_per_member: number;
}

export interface Statement { kind: string; section: string; text: string }

export interface Candidate {
  rank: number; score: number; synthetic?: boolean;
  vessel: { mmsi: string; imo?: string | null; name?: string | null; type: string };
  feature_scores_pct: Record<string, number>;
  source_distance_km: number | null; time_difference_hours: number | null; features: any;
  evidence: { summary: string; statements: Statement[]; supporting_evidence: string[];
              contradicting_evidence: string[]; missing_evidence: string[]; uncertainties: string[] };
}

export interface Layers {
  analysis: any; scene: any; segmentation: any; spill: any; acquisition: any;
  triage: { wind_at_acquisition: any; sar_ship_detection: any; components: TriageRow[]; note: string } | null;
  selected: any; selected_geojson: FC | null;
  sar_image_url: string; mask_url: string; probability_url: string;
  spill_geojson: FC | null; drift: any; forward: any; source_probability: FC | null;
  backward_particles: Frames | null; forward_particles: Frames | null;
  ais_tracks: FC | null; candidates: any; report_html_url: string; report_md_url: string;
  impact?: any; evidence_seal?: { root: string; chain: string; version: number; sealed_at: string; n_files: number; algorithm: string } | null;
}

export interface Source {
  id: string; name: string; kind: string; key_required: boolean; env: string[]; integrated: boolean;
  coverage: string; notes: string; configured: boolean; state: "ACTIVE" | "READY_TO_INTEGRATE" | "NOT_CONFIGURED";
}

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const b = await r.json();
      detail = b?.error?.message ?? (typeof b?.detail === "string" ? b.detail : JSON.stringify(b?.detail)) ?? detail;
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

const post = (url: string, body: unknown) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  health: () => fetch("/api/health").then((r) => j<any>(r)),
  config: () => fetch("/api/v1/config").then((r) => j<any>(r)),
  sources: () => fetch("/api/v1/sources").then((r) => j<Source[]>(r)),
  list: () => fetch("/api/v1/analyses").then((r) => j<any[]>(r)),
  queue: () => fetch("/api/v1/queue").then((r) => j<{ running: string | null; waiting: string[] }>(r)),
  layers: (id: string) => fetch(`/api/spill/${id}/layers`).then((r) => j<Layers>(r)),
  impact: (id: string) => fetch(`/api/spill/${id}/impact`, { method: "POST" }).then((r) => j<any>(r)),
  verify: (id: string) => fetch(`/api/spill/${id}/verify`).then((r) => j<any>(r)),
  scenes: (bbox: BBox, start: string, end: string) =>
    fetch(`/api/v1/scenes?bbox=${bbox.map((v) => v.toFixed(4)).join(",")}&start=${start}&end=${end}`).then((r) => j<Scene[]>(r)),
  detect: (scene_id: string, aoi: BBox, resolution_m: number, threshold?: number) =>
    post("/api/v1/analyses", { scene_id, aoi, resolution_m, threshold }).then((r) => j<{ analysis_id: string }>(r)),
  upload: (form: FormData) =>
    fetch("/api/v1/analyses/upload", { method: "POST", body: form }).then((r) => j<{ analysis_id: string }>(r)),
  investigate: (id: string, component_ids: string[], ais_mode?: "auto" | "real" | "synthetic", particles?: number, members?: number) =>
    post(`/api/v1/analyses/${id}/investigate`, { component_ids, ais_mode, particles, members }).then((r) => j<{ analysis_id: string }>(r)),
  events: (id: string, since: number, onEvent: (e: ProgressEvent) => void, onDone: (d: any) => void, onError: () => void) => {
    const es = new EventSource(`/api/v1/analyses/${id}/events?since=${since}`);
    es.addEventListener("progress", (m) => onEvent(JSON.parse((m as MessageEvent).data)));
    es.addEventListener("done", (m) => { onDone(JSON.parse((m as MessageEvent).data)); es.close(); });
    es.onerror = () => { onError(); es.close(); };
    return () => es.close();
  },
};

export const fmtT = (s?: string | null) => (s ? new Date(s).toISOString().replace("T", " ").slice(0, 16) + " UTC" : "n/a");
export const n = (v: any, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "n/a" : Number(v).toFixed(d));
