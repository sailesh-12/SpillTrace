import { useState } from "react";
import {
  AlertTriangle, CalendarRange, CheckCircle2, ChevronDown, CircleDashed, Crosshair, Database, Info, KeyRound, Loader2,
  MousePointerSquareDashed, Pause, Play, Radar, Search, Ship, Upload, Waves, Wind, XCircle,
} from "lucide-react";
import type { BBox, Candidate, Layers, ProgressEvent, Scene, Source, TriageRow } from "@/api";
import { fmtT, n } from "@/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { LAYER_LABELS, LayerKey } from "./MapView";

// ---------------------------------------------------------------------------------------------------
export function Prov({ p }: { p?: string | null }) {
  const v = p ?? "UNAVAILABLE";
  const map: Record<string, [string, string]> = {
    LIVE_EXTERNAL: ["LIVE", "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"],
    SYNTHETIC_DEMO: ["SYNTHETIC", "bg-amber-500/20 text-amber-300 border-amber-500/50"],
    LOCAL_FILE: ["LOCAL FILE", "bg-sky-500/15 text-sky-300 border-sky-500/30"],
    UNAVAILABLE: ["UNAVAILABLE", "bg-rose-500/15 text-rose-300 border-rose-500/30"],
  };
  const [label, cls] = map[v] ?? ["ASSUMED / DEMO", "bg-amber-500/15 text-amber-300 border-amber-500/30"];
  return <Badge variant="outline" className={cn("text-[10px] font-semibold tracking-wide", cls)} title={v}>{label}</Badge>;
}

const LABEL_STYLE: Record<string, string> = {
  OIL_LIKELY: "bg-orange-500/15 text-orange-300 border-orange-500/40",
  LOOKALIKE_LIKELY: "bg-green-500/15 text-green-300 border-green-500/40",
  UNCERTAIN: "bg-yellow-500/15 text-yellow-200 border-yellow-500/40",
};
export const TriageBadge = ({ l }: { l: string }) =>
  <Badge variant="outline" className={cn("text-[10px]", LABEL_STYLE[l])}>{l.replace("_", " ")}</Badge>;

const Field = ({ label, children }: { label: string; children: React.ReactNode }) =>
  <div className="grid gap-1.5"><Label className="text-xs text-muted-foreground">{label}</Label>{children}</div>;

const Stat = ({ icon: Icon, label, value, sub }: { icon: any; label: string; value: React.ReactNode; sub?: React.ReactNode }) => (
  <div className="group relative overflow-hidden rounded-xl border border-slate-700/50 bg-gradient-to-b from-slate-800/50 to-slate-900/40 p-2.5 transition hover:border-cyan-500/40">
    <div className="pointer-events-none absolute -right-4 -top-4 size-12 rounded-full bg-cyan-500/10 blur-xl transition group-hover:bg-cyan-500/20" />
    <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-slate-400">
      <span className="grid size-5 place-items-center rounded-md bg-cyan-500/10 text-cyan-300"><Icon className="size-3" /></span>{label}</div>
    <div className="mt-1.5 font-mono text-[15px] font-semibold text-slate-50">{value}</div>
    {sub && <div className="truncate text-[10.5px] text-slate-400">{sub}</div>}
  </div>
);

/** Compact, collapsible notice (replaces tall alert boxes). */
export function Callout({ tone = "amber", title, children, defaultOpen = false }: {
  tone?: "amber" | "sky" | "rose"; title: React.ReactNode; children?: React.ReactNode; defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const c = { amber: "border-amber-500/35 bg-amber-500/[0.07] text-amber-200", sky: "border-sky-500/30 bg-sky-500/[0.06] text-sky-200",
    rose: "border-rose-500/35 bg-rose-500/[0.07] text-rose-200" }[tone];
  const I = tone === "sky" ? Info : AlertTriangle;
  return (
    <div className={cn("rounded-lg border text-xs", c)}>
      <button className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left" onClick={() => children && setOpen(!open)}>
        <I className="size-3.5 shrink-0" /><span className="flex-1 font-medium">{title}</span>
        {children && <ChevronDown className={cn("size-3.5 shrink-0 opacity-70 transition", open && "rotate-180")} />}
      </button>
      {open && children && <div className="px-2.5 pb-2 pl-8 text-[11px] leading-relaxed text-slate-300/90">{children}</div>}
    </div>
  );
}

export const SectionLabel = ({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) => (
  <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-slate-400">
    <span className="size-1 rounded-full bg-cyan-400" />{children}<span className="h-px flex-1 bg-slate-700/60" />{right}</div>
);

// ---------------------------------------------------------------------------------------------------
export function SearchPanel(props: {
  aoi: BBox | null; setAoi: (b: BBox | null) => void; drawing: boolean; setDrawing: (b: boolean) => void;
  scenes: Scene[]; sceneId: string | null; setSceneId: (id: string) => void; busy: boolean;
  onSearch: (start: string, end: string) => void; searching: boolean;
  onDetect: (res: number) => void; onUpload: (f: FormData) => void; searched: boolean; onSynthetic: () => void;
}) {
  const today = new Date();
  const iso = (d: Date) => d.toISOString().slice(0, 10);
  const [start, setStart] = useState(iso(new Date(today.getTime() - 21 * 86400e3)));
  const [end, setEnd] = useState(iso(today));
  const [res, setRes] = useState("10");
  const [txt, setTxt] = useState(props.aoi?.join(", ") ?? "");
  const [file, setFile] = useState<File | null>(null);
  const aoiArea = props.aoi ? ((props.aoi[2] - props.aoi[0]) * (props.aoi[3] - props.aoi[1])) : 0;
  return (
    <div className="grid gap-3">
      <Card size="sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm"><MousePointerSquareDashed className="size-4 text-primary" />1 · Area of interest</CardTitle>
          <CardDescription className="text-xs">Draw a box over Indian waters (≤ 1° per side). Real Sentinel-1 and Open-Meteo data are used where available; missing data are replaced by clearly labelled SYNTHETIC data.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-2.5">
          <div className="flex gap-2">
            <Button size="sm" variant={props.drawing ? "default" : "outline"} onClick={() => props.setDrawing(!props.drawing)}>
              <Crosshair />{props.drawing ? "Drag on map…" : "Draw on map"}</Button>
            <Input className="h-7 text-xs" placeholder="W, S, E, N" value={props.aoi?.join(", ") ?? txt}
              onChange={(e) => { setTxt(e.target.value); const v = e.target.value.split(",").map(Number);
                if (v.length === 4 && v.every(Number.isFinite) && v[0] < v[2] && v[1] < v[3]) props.setAoi(v as BBox); }} />
          </div>
          {props.aoi && <p className="text-[11px] text-muted-foreground">
            {n(props.aoi[2] - props.aoi[0], 2)}° × {n(props.aoi[3] - props.aoi[1], 2)}°{aoiArea > 1 &&
              <span className="text-amber-300"> — large AOIs take longer to download</span>}</p>}
        </CardContent>
      </Card>
      <Card size="sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm"><CalendarRange className="size-4 text-primary" />2 · Sentinel-1 scenes</CardTitle>
          <CardDescription className="text-xs">Microsoft Planetary Computer · no API key</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-2.5">
          <div className="grid grid-cols-2 gap-2">
            <Field label="From"><Input type="date" className="h-7 text-xs" value={start} onChange={(e) => setStart(e.target.value)} /></Field>
            <Field label="To"><Input type="date" className="h-7 text-xs" value={end} onChange={(e) => setEnd(e.target.value)} /></Field>
          </div>
          <Button size="sm" disabled={!props.aoi || props.searching} onClick={() => props.onSearch(start, end)}>
            {props.searching ? <Loader2 className="animate-spin" /> : <Search />}Search scenes</Button>
          {props.scenes.length > 0 && <div className="grid max-h-72 gap-1.5 overflow-y-auto pr-1">
            {props.scenes.map((s) => (
              <button key={s.id} onClick={() => props.setSceneId(s.id)}
                className={cn("flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-900/40 p-1.5 text-left transition hover:border-cyan-500/40 hover:bg-slate-800/50",
                  props.sceneId === s.id && "border-cyan-400/70 bg-cyan-500/10 shadow-[0_0_16px_rgb(34_211_238/0.15)]")}>
                {s.thumbnail ? <img src={s.thumbnail} className="size-11 rounded object-cover opacity-90" alt="" />
                  : <div className="size-11 rounded bg-muted" />}
                <div className="min-w-0 text-xs">
                  <div className="font-medium">{fmtT(s.datetime)}</div>
                  <div className="text-muted-foreground">{s.platform} · {s.orbit_state}</div>
                  <div className="mt-0.5 flex items-center gap-1"><Prov p="LIVE_EXTERNAL" />
                    <span className={cn(s.aoi_coverage < 0.5 ? "text-amber-300" : "text-emerald-300")}>
                      covers {Math.round(100 * s.aoi_coverage)}% of your box</span></div>
                </div>
              </button>))}
          </div>}
          {props.scenes.length > 0 && <div className="flex items-end gap-2">
            <Field label="Pixel size">
              <select className="h-7 rounded-lg border bg-transparent px-2 text-xs" value={res} onChange={(e) => setRes(e.target.value)}>
                <option value="10">10 m native (recommended, calibrated)</option><option value="20">20 m (faster)</option>
              </select></Field>
            <Button size="sm" className="flex-1" disabled={!props.sceneId || props.busy} onClick={() => props.onDetect(+res)}>
              <Radar />Acquire & detect</Button>
          </div>}
          {props.searched && (() => {
            const best = Math.max(0, ...props.scenes.map((s) => s.aoi_coverage));
            if (best >= 0.5) return null;
            return <Callout defaultOpen title={props.scenes.length ? `Real Sentinel-1 covers at most ${Math.round(100 * best)}% of this box`
                : "No real Sentinel-1 scene for this box and period"}>Sentinel-1 revisits Indian waters less often than Europe. A partially
                covering real scene is clipped to the covered part. To demonstrate the full chain here, use a SYNTHETIC SAR
                scene (clearly labelled, not an observation).</Callout>;
          })()}
          <Button size="sm" variant="outline" disabled={!props.aoi || props.busy} onClick={props.onSynthetic}
            className="border-amber-500/40 text-amber-200">
            <Radar />Use SYNTHETIC SAR scene for this box (demo)</Button>
        </CardContent>
      </Card>
      <Card size="sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm"><Upload className="size-4 text-primary" />Or upload a GeoTIFF</CardTitle>
          <CardDescription className="text-xs">Georeferenced SAR with an acquisition-time tag. Plain images are refused for drift/AIS.</CardDescription>
        </CardHeader>
        <CardContent className="flex gap-2">
          <Input type="file" accept=".tif,.tiff" className="h-8 text-xs" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <Button size="sm" variant="outline" disabled={!file || props.busy}
            onClick={() => { const f = new FormData(); f.append("file", file!); props.onUpload(f); }}>Analyse</Button>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------
export function TriagePanel(props: { d: Layers; checked: Set<string>; toggle: (id: string) => void;
  onFocus: (r: TriageRow) => void; onInvestigate: (aisMode: "auto" | "real" | "synthetic") => void; busy: boolean }) {
  const [aisMode, setAisMode] = useState<"auto" | "real" | "synthetic">("auto");
  const t = props.d.triage;
  const [filter, setFilter] = useState<"all" | "trail" | "oil">("all");
  if (!t) return <Empty text={props.d.analysis?.error?.message ?? "Run a detection first."} />;
  const rows = t.components.filter((r) => filter === "all" || (filter === "trail" ? r.ship_trail : r.label === "OIL_LIKELY"));
  const synScene = props.d.acquisition?.provenance === "SYNTHETIC_DEMO";
  const w = t.wind_at_acquisition;
  const lowWind = w && w.wind_speed_ms < 3;
  return (
    <div className="grid gap-3">
      <div className="grid grid-cols-3 gap-2">
        <Stat icon={Wind} label="Wind at acquisition" value={w ? `${n(w.wind_speed_ms, 1)} m/s` : "n/a"} sub={w?.source} />
        <Stat icon={Waves} label="Dark features" value={t.components.length} sub={`${t.components.filter((r) => r.label === "OIL_LIKELY").length} oil-likely`} />
        <Stat icon={Ship} label="SAR point targets" value={t.sar_ship_detection?.n_detections ?? 0} sub="untrained baseline detector" />
      </div>
      {synScene && <Callout title="SYNTHETIC SAR scene — not a satellite observation">Generated for this box because no usable
        real Sentinel-1 coverage was selected: speckle, wind texture, ships, one ship-trailing discharge and a natural look-alike.
        The detection model and triage run exactly as on real data.</Callout>}
      {lowWind && <Callout title={`Low wind (${n(w.wind_speed_ms, 1)} m/s) — look-alikes likely`}>Below ~3 m/s natural films and
        calm zones look like oil. Treat detections with caution.</Callout>}
      <SectionLabel right={<span className="normal-case tracking-normal text-cyan-300">{props.checked.size} selected</span>}>Dark features</SectionLabel>
      <div className="flex items-center gap-1 rounded-lg border border-slate-700/50 bg-slate-900/40 p-0.5">
        {(["all", "trail", "oil"] as const).map((f) => (
          <button key={f} onClick={() => setFilter(f)} className={cn("flex-1 rounded-md px-2 py-1 text-[11px] transition",
            filter === f ? "bg-cyan-500/15 text-cyan-200 shadow-[inset_0_0_0_1px_rgb(34_211_238/0.3)]" : "text-slate-400 hover:text-slate-100")}>
            {f === "all" ? `All (${t.components.length})` : f === "trail" ? "Ship trails" : "Oil-likely"}</button>))}
      </div>
      <div className="max-h-[42vh] overflow-y-auto rounded-xl border border-slate-700/50 bg-slate-900/30">
        <Table>
          <TableHeader><TableRow>
            <TableHead className="w-8" /><TableHead>#</TableHead><TableHead>Triage</TableHead>
            <TableHead className="text-right">km²</TableHead><TableHead className="text-right">Elong.</TableHead><TableHead className="text-right">dB</TableHead>
          </TableRow></TableHeader>
          <TableBody>
            {rows.slice(0, 150).map((r) => (
              <TableRow key={r.component_id} className="cursor-pointer" data-state={props.checked.has(r.component_id) ? "selected" : undefined}
                onClick={() => props.onFocus(r)}>
                <TableCell onClick={(e) => e.stopPropagation()}>
                  <Checkbox checked={props.checked.has(r.component_id)} onCheckedChange={() => props.toggle(r.component_id)} /></TableCell>
                <TableCell className="text-xs text-muted-foreground">{r.triage_rank}</TableCell>
                <TableCell><div className="flex items-center gap-1"><TriageBadge l={r.label} />
                  {r.ship_trail && <Ship className="size-3.5 text-orange-300" />}</div></TableCell>
                <TableCell className="text-right text-xs">{n(r.area_km2, 2)}</TableCell>
                <TableCell className="text-right text-xs">{n(r.elongation, 1)}</TableCell>
                <TableCell className="text-right text-xs">{n(r.contrast_db, 1)}</TableCell>
              </TableRow>))}
          </TableBody>
        </Table>
      </div>
      <p className="text-[11px] leading-snug text-muted-foreground">{t.note} Click a row to zoom; tick the slick(s) you judge to be oil.</p>
      <SectionLabel>Investigate</SectionLabel>
      <Field label="AIS source">
        <select className="h-8 rounded-lg border bg-transparent px-2 text-xs" value={aisMode}
          onChange={(e) => setAisMode(e.target.value as "auto" | "real" | "synthetic")}>
          <option value="auto">Auto — real AIS if available, else SYNTHETIC (labelled)</option>
          <option value="real">Real AIS only (no synthetic fallback)</option>
          <option value="synthetic">SYNTHETIC AIS (demonstration, SIH26143-permitted)</option>
        </select></Field>
      <Button disabled={!props.checked.size || props.busy} onClick={() => props.onInvestigate(aisMode)}
        className="shadow-[0_0_20px_rgb(34_211_238/0.25)]">
        <Radar />Investigate {props.checked.size || ""} selected slick{props.checked.size === 1 ? "" : "s"}</Button>
    </div>
  );
}

export function SlickDetail({ r }: { r: TriageRow | null }) {
  if (!r) return null;
  return (
    <Card size="sm">
      <CardHeader><CardTitle className="flex items-center gap-2 text-sm">Slick #{r.triage_rank}<TriageBadge l={r.label} /></CardTitle>
        <CardDescription className="text-xs">{r.component_id}</CardDescription></CardHeader>
      <CardContent className="grid gap-1 text-xs">
        <div className="grid grid-cols-3 gap-2 text-muted-foreground">
          <span>{n(r.area_km2, 2)} km²</span><span>{n(r.length_km, 1)} km long</span><span>p̄ {n(r.mean_probability, 2)}</span></div>
        <ul className="mt-1 list-disc space-y-0.5 pl-4">{r.reasons.map((x, i) => <li key={i}>{x}</li>)}</ul>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------------------------------
export function InvestigationPanel({ d }: { d: Layers }) {
  const s = d.selected, dr = d.drift, fw = d.forward;
  if (!dr) return <Empty text={d.analysis?.error?.message ?? "Select slick(s) in Triage and start the investigation."} />;
  const f = dr.forcing_at_spill;
  const synForcing = String(dr.forcing?.provenance ?? "").startsWith("SYNTHETIC");
  return (
    <div className="grid gap-3">
      {synForcing && <Callout title="SYNTHETIC currents & wind">Real forcing (Open-Meteo) was unavailable, so a
        monsoon-climatology-inspired analytic field for Indian waters was used. Drift results illustrate the method only.</Callout>}
      <Card size="sm">
        <CardHeader><CardTitle className="text-sm">Investigated slick</CardTitle>
          <CardDescription className="text-xs">{s?.selected_component_ids?.map((c: string) => c.split("_").pop()).join(", ")}</CardDescription></CardHeader>
        <CardContent className="grid grid-cols-2 gap-2">
          <Stat icon={Waves} label="Area" value={`${n(s?.area_m2 / 1e6, 3)} km²`} sub={`perimeter ${n(s?.perimeter_m / 1000, 1)} km`} />
          <Stat icon={Crosshair} label="Centroid" value={`${n(s?.centroid?.lat, 4)}°N`} sub={`${n(s?.centroid?.lon, 4)}°E`} />
          <Stat icon={CalendarRange} label="Observed" value={fmtT(s?.timestamp)} sub={d.acquisition?.platform} />
          <Stat icon={Radar} label="Detection confidence" value={n(s?.detection_confidence, 2)} sub="mean sigmoid, uncalibrated" />
        </CardContent>
      </Card>
      <Card size="sm">
        <CardHeader><CardTitle className="flex items-center gap-2 text-sm">Backward drift <Prov p={dr.forcing?.provenance} /></CardTitle>
          <CardDescription className="text-xs">{dr.model} · {dr.n_members} members × {dr.particles_per_member} particles</CardDescription></CardHeader>
        <CardContent className="grid gap-2 text-xs">
          <div className="grid grid-cols-2 gap-2">
            <Stat icon={Wind} label="Wind at slick" value={`${n(f.wind_speed, 1)} m/s`} sub={`from ${n(f.wind_from_deg, 0)}°`} />
            <Stat icon={Waves} label="Current at slick" value={`${n(f.current_speed, 2)} m/s`} sub={`toward ${n(f.current_to_deg, 0)}°`} />
          </div>
          <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-2">
            <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400">Release window</div>
            <div className="mt-0.5 font-mono text-[11.5px] text-slate-100">{fmtT(dr.release_window.start)} → {fmtT(dr.release_window.end)}</div>
            <div className="text-[11px] text-muted-foreground">{dr.release_window.basis ?? dr.release_window.prior}</div></div>
          <div className="flex flex-wrap gap-1.5">{Object.entries(dr.source_regions).map(([k, v]: any) => (
            <Badge key={k} variant="outline" className={cn("text-[10px]", k === "high" ? "text-red-300" : k === "medium" ? "text-orange-300" : "text-yellow-200")}>
              {k} {Math.round(v.mass_fraction * 100)}% · {n(v.area_km2, 0)} km²</Badge>))}</div>
          <div><span className="text-muted-foreground">Oil:</span> {dr.oil.oil_type} → modelled as {dr.oil.oil_model_assumption}</div>
          <Separator />
          <div><span className="text-muted-foreground">Forward {fw?.hours ?? ""} h:</span> {fw?.available
            ? <>centre {n(fw.end_center.lat, 3)}, {n(fw.end_center.lon, 3)} · affected {n((Object.values(fw.affected_region)[0] as any)?.area_km2, 0)} km² · stranded {fw.stranded_particles}</>
            : (fw?.reason ?? "unavailable")}</div>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------
export function CandidatesPanel(props: { d: Layers; selected: Candidate | null; select: (c: Candidate) => void }) {
  const r = props.d.candidates;
  const [showFiltered, setShowFiltered] = useState(false);
  if (!r) return <Empty text="No candidate analysis yet." />;
  return (
    <div className="grid gap-3">
      {r.ais?.synthetic && <SyntheticBanner reason={r.ais?.selection?.fallback_reason} />}
      <Callout tone="sky" title="Candidates for investigation — not a finding of responsibility">{r.disclaimer}</Callout>
      <SectionLabel right={<Prov p={r.ais?.provider?.provenance} />}>{r.candidates.length} ranked vessel{r.candidates.length === 1 ? "" : "s"}</SectionLabel>
      {r.ranking_separation && <p className={cn("text-[11px]", r.ranking_separation.separable ? "text-emerald-300/90" : "text-amber-200/90")}>
        {r.ranking_separation.note}</p>}
      <div className="grid max-h-[50vh] gap-1.5 overflow-y-auto pr-0.5">
        {r.candidates.map((c: Candidate) => {
          const sel = props.selected?.vessel.mmsi === c.vessel.mmsi;
          const rb = props.d.impact?.robustness?.candidates?.find((x: any) => x.mmsi === c.vessel.mmsi);
          return (
            <button key={c.vessel.mmsi} onClick={() => props.select(c)}
              className={cn("group grid grid-cols-[28px_1fr_auto] items-center gap-2.5 rounded-xl border p-2 text-left transition",
                sel ? "border-cyan-400/70 bg-cyan-500/10 shadow-[0_0_18px_rgb(34_211_238/0.18)]"
                  : "border-slate-700/50 bg-slate-900/40 hover:border-cyan-500/40 hover:bg-slate-800/50")}>
              <span className={cn("grid size-7 place-items-center rounded-lg font-mono text-xs font-bold",
                c.rank === 1 ? "bg-gradient-to-br from-cyan-300 to-sky-500 text-slate-950" : "border border-slate-600 text-slate-300")}>{c.rank}</span>
              <div className="min-w-0">
                <div className="flex items-center gap-1.5 truncate text-[12.5px] font-semibold text-slate-100">{c.vessel.name ?? c.vessel.mmsi}
                  {c.features?.sar_attached?.matched && <span title="Seen by SAR at the slick" className="inline-flex items-center gap-0.5 rounded bg-cyan-500/15 px-1 text-[9px] font-medium text-cyan-300"><Ship className="size-2.5" />SAR</span>}
                  {c.synthetic && <span className="rounded border border-amber-500/40 px-1 text-[9px] font-medium text-amber-300">SYN</span>}</div>
                <div className="truncate font-mono text-[10.5px] text-slate-400">{c.vessel.type ?? "Unknown"} · {c.vessel.mmsi}</div>
                <div className="mt-1 h-1 overflow-hidden rounded-full bg-slate-800">
                  <div className="h-full rounded-full bg-gradient-to-r from-cyan-500 via-sky-400 to-indigo-400" style={{ width: `${c.score}%` }} /></div>
              </div>
              <div className="text-right">
                <div className="font-mono text-base font-bold leading-none text-slate-50">{n(c.score, 0)}</div>
                <div className="mt-1 font-mono text-[10px] text-slate-400">{n(c.source_distance_km, 1)} km · {n(c.time_difference_hours, 1)} h</div>
                {rb && <div className="font-mono text-[9.5px] text-cyan-300/80" title="Share of weight perturbations in which this vessel ranks first">P(#1) {n(100 * rb.p_rank1, 0)}%</div>}
              </div>
            </button>);
        })}
      </div>
      <div className="flex items-center gap-2 text-xs">
        <Switch checked={showFiltered} onCheckedChange={setShowFiltered} /> Show {r.filtered_out?.length ?? 0} filtered vessels & reasons</div>
      {showFiltered && <ul className="max-h-48 list-disc space-y-0.5 overflow-y-auto pl-4 text-[11px] text-muted-foreground">
        {r.filtered_out.slice(0, 300).map((v: any) => <li key={v.mmsi}><b className="text-foreground">{v.name ?? v.mmsi}</b> ({v.vessel_type}): {v.reason}</li>)}</ul>}
      {r.sar_ais_mismatch?.available && <p className="text-[11px] text-muted-foreground">
        SAR/AIS at acquisition: {r.sar_ais_mismatch.matches.length} matched · {r.sar_ais_mismatch.sar_detections_without_ais.length} SAR-only ·
        {" "}{r.sar_ais_mismatch.ais_vessels_without_sar_detection.length} AIS-only. {r.sar_ais_mismatch.note}</p>}
    </div>
  );
}

const KIND: Record<string, [string, string]> = {
  OBSERVED_FACT: ["Observed", "bg-sky-500/15 text-sky-300 border-sky-500/30"],
  MODEL_DERIVED: ["Model", "bg-violet-500/15 text-violet-300 border-violet-500/30"],
  ASSUMPTION: ["Assumption", "bg-amber-500/15 text-amber-300 border-amber-500/30"],
  CANDIDATE_INFERENCE: ["Inference", "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"],
};

export function SyntheticBanner({ reason }: { reason?: string | null }) {
  return (
    <Callout title="SYNTHETIC AIS — not real vessels">Vessel tracks were generated around the spill region to demonstrate the
      algorithm (permitted by SIH26143 when real AIS is unavailable). No synthetic vessel exists.
      {reason ? ` Real AIS: ${reason}.` : ""}</Callout>
  );
}

function ScoreRing({ v }: { v: number }) {
  const r = 26, c = 2 * Math.PI * r;
  return (
    <div className="relative size-16 shrink-0">
      <svg viewBox="0 0 64 64" className="size-16 -rotate-90">
        <circle cx="32" cy="32" r={r} fill="none" stroke="rgb(51 65 85 / 0.6)" strokeWidth="5" />
        <circle cx="32" cy="32" r={r} fill="none" stroke="url(#ring)" strokeWidth="5" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - v / 100)} style={{ transition: "stroke-dashoffset 1s ease" }} />
        <defs><linearGradient id="ring"><stop offset="0" stopColor="#22d3ee" /><stop offset="1" stopColor="#818cf8" /></linearGradient></defs>
      </svg>
      <div className="absolute inset-0 grid place-items-center"><div className="text-center leading-none">
        <div className="font-mono text-lg font-bold text-slate-50">{n(v, 0)}</div><div className="text-[8px] tracking-wider text-slate-400">SCORE</div></div></div>
    </div>
  );
}

export function EvidencePanel({ c }: { c: Candidate | null }) {
  if (!c) return <Empty text="Select a candidate vessel to inspect its evidence." />;
  const e = c.evidence;
  return (
    <div className="grid gap-3">
      {c.synthetic && <SyntheticBanner />}
      <div className="flex items-center gap-3 rounded-xl border border-slate-700/50 bg-gradient-to-br from-slate-800/60 to-slate-900/40 p-3">
        <ScoreRing v={c.score} />
        <div className="min-w-0">
          <div className="flex items-center gap-2"><span className="grid size-5 place-items-center rounded bg-cyan-500/15 font-mono text-[10px] text-cyan-300">#{c.rank}</span>
            <span className="truncate text-sm font-semibold text-slate-50">{c.vessel.name ?? c.vessel.mmsi}</span></div>
          <div className="mt-0.5 font-mono text-[11px] text-slate-400">MMSI {c.vessel.mmsi}{c.vessel.imo ? ` · IMO ${c.vessel.imo}` : ""}</div>
          <div className="text-[11px] text-slate-400">{c.vessel.type} · {n(c.source_distance_km, 1)} km from source · Δt {n(c.time_difference_hours, 1)} h</div>
        </div>
      </div>
      <SectionLabel>Evidence factors</SectionLabel>
      <div className="grid gap-1.5">{Object.entries(c.feature_scores_pct).map(([k, v]) => (
        <div key={k} className="grid grid-cols-[104px_1fr_30px] items-center gap-2 text-[11px]">
          <span className="capitalize text-slate-400">{k.replace(/_/g, " ")}</span>
          <div className="h-1.5 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-indigo-400"
            style={{ width: `${v}%`, transition: "width .8s ease" }} /></div>
          <span className="text-right font-mono text-slate-200">{n(v, 0)}</span></div>))}</div>
      <SectionLabel>Statements</SectionLabel>
      <ul className="grid gap-2">{e.statements.map((s, i) => (
        <li key={i} className="text-xs leading-snug"><Badge variant="outline" className={cn("mr-1.5 text-[10px]", KIND[s.kind]?.[1])}>
          {KIND[s.kind]?.[0] ?? s.kind}</Badge><span className="text-muted-foreground">{s.section}: </span>{s.text}</li>))}</ul>
      <div className="grid gap-2 text-xs">
        <EvList icon={CheckCircle2} cls="text-emerald-300" title="Supporting" items={e.supporting_evidence} />
        <EvList icon={XCircle} cls="text-rose-300" title="Contradicting" items={e.contradicting_evidence} />
        <EvList icon={CircleDashed} cls="text-amber-300" title="Missing" items={e.missing_evidence} />
      </div>
      <details className="text-xs"><summary className="cursor-pointer text-primary">Uncertainty</summary>
        <ul className="mt-1 list-disc space-y-0.5 pl-4 text-muted-foreground">{e.uncertainties.map((u, i) => <li key={i}>{u}</li>)}</ul></details>
      <p className="rounded-lg border border-emerald-500/30 bg-emerald-500/[0.06] p-2.5 text-xs leading-relaxed">{e.summary}</p>
    </div>
  );
}

const EvList = ({ icon: Icon, cls, title, items }: { icon: any; cls: string; title: string; items: string[] }) => (
  <div><div className={cn("flex items-center gap-1 font-medium", cls)}><Icon className="size-3.5" />{title}</div>
    <ul className="mt-0.5 list-disc space-y-0.5 pl-5 text-muted-foreground">{items.length ? items.map((x, i) => <li key={i}>{x}</li>) : <li>none identified</li>}</ul></div>
);

export const Empty = ({ text }: { text: string }) =>
  <div className="grid place-items-center gap-2 rounded-xl border border-dashed border-slate-700/70 p-8 text-center text-xs text-muted-foreground">
    <div className="relative grid size-10 place-items-center"><span className="absolute size-10 animate-ping rounded-full border border-cyan-500/30" />
      <Radar className="size-5 text-cyan-400/70" /></div>{text}</div>;

// ---------------------------------------------------------------------------------------------------
export function ProgressCard({ events, stages, busy }: { events: ProgressEvent[]; stages: string[]; busy: boolean }) {
  const last = events[events.length - 1];
  if (!last) return null;
  const idx = Math.max(0, stages.indexOf(last.stage));
  const pct = last.status === "COMPLETED" ? 100 : Math.round((100 * idx) / Math.max(stages.length - 1, 1));
  return (
    <Card size="sm" className="w-[min(380px,calc(100vw-420px))] border-slate-700/60 bg-slate-950/80 backdrop-blur-md">
      <CardContent className="grid gap-2">
        <div className="flex items-center gap-2 text-xs">
          {busy ? <Loader2 className="size-3.5 animate-spin text-primary" /> : <CheckCircle2 className="size-3.5 text-emerald-400" />}
          <span className="font-mono font-semibold tracking-wide">{last.stage.replace(/_/g, " ")}</span>
          <Badge variant="outline" className="ml-auto font-mono text-[10px]">{busy ? `${pct}%` : last.status.replace(/_/g, " ")}</Badge></div>
        <div className="flex gap-0.5">{stages.map((st, i) => (
          <span key={st} title={st.replace(/_/g, " ")} className={cn("h-1 flex-1 rounded-full transition-colors",
            i < idx || last.status === "COMPLETED" ? "bg-cyan-400" : i === idx ? (busy ? "animate-pulse bg-cyan-300" : "bg-cyan-400") : "bg-slate-700")} />))}</div>
        <div className="line-clamp-2 text-[11px] text-muted-foreground">{last.message}</div>
      </CardContent>
    </Card>
  );
}

export function Timeline(props: { times: { time: Date; kind: string }[]; idx: number; setIdx: (i: number) => void;
  playing: boolean; setPlaying: (b: boolean) => void; obs: Date | null }) {
  const { times, idx, obs } = props;
  if (!times.length) return null;
  const cur = times[idx];
  const dh = obs ? (cur.time.getTime() - obs.getTime()) / 3600e3 : 0;
  const rel = Math.abs(dh) < 0.01 ? "T (observation)" : `T ${dh < 0 ? "−" : "+"} ${Math.abs(dh).toFixed(0)} h`;
  return (
    <Card size="sm" className="w-full max-w-[760px] border-slate-700/60 bg-slate-950/80 backdrop-blur-md">
      <CardContent className="flex items-center gap-3">
        <Button size="icon-sm" onClick={() => props.setPlaying(!props.playing)}>{props.playing ? <Pause /> : <Play />}</Button>
        <div className="grid flex-1 gap-1">
          <input type="range" min={0} max={times.length - 1} value={idx} onChange={(e) => props.setIdx(+e.target.value)}
            className="w-full accent-[var(--primary)]" />
          <div className="flex items-center gap-2 text-xs">
            <b className="font-mono text-cyan-200">{rel}</b><span className="font-mono text-muted-foreground">{fmtT(cur.time.toISOString())}</span>
            <Badge variant="outline" className={cn("text-[10px]", cur.kind === "backward" ? "text-cyan-300" : "text-purple-300")}>
              {cur.kind === "backward" ? "hindcast" : "forecast"}</Badge></div>
        </div>
      </CardContent>
    </Card>
  );
}

export function LayerMenu(props: { visible: Record<LayerKey, boolean>; set: (k: LayerKey, v: boolean) => void;
  probLabel: string; probLabels: string[]; setProbLabel: (s: string) => void; follow: boolean; setFollow: (b: boolean) => void }) {
  return (
    <Card size="sm" className="w-64 border-slate-700/60 bg-slate-950/85 backdrop-blur-md">
      <CardHeader><CardTitle className="text-xs">Map layers</CardTitle></CardHeader>
      <CardContent className="grid gap-1.5">
        {(Object.keys(LAYER_LABELS) as LayerKey[]).map((k) => (
          <label key={k} className="flex items-center gap-2 text-xs"><Switch checked={props.visible[k]} onCheckedChange={(v: boolean) => props.set(k, v)} />
            {LAYER_LABELS[k]}</label>))}
        <Separator className="my-1" />
        <Field label="Source-probability map">
          <select className="h-7 rounded-lg border bg-transparent px-2 text-xs" value={props.probLabel} onChange={(e) => props.setProbLabel(e.target.value)}>
            {props.probLabels.map((l) => <option key={l} value={l}>{l === "combined" ? "combined (release window)" : `release at ${l}`}</option>)}
          </select></Field>
        <label className="flex items-center gap-2 text-xs"><Switch checked={props.follow} onCheckedChange={props.setFollow} />heatmap follows timeline</label>
      </CardContent>
    </Card>
  );
}

export function SourcesSheet({ open, setOpen, sources }: { open: boolean; setOpen: (b: boolean) => void; sources: Source[] }) {
  const cls: Record<string, string> = { ACTIVE: "text-emerald-300 border-emerald-500/40", READY_TO_INTEGRATE: "text-sky-300 border-sky-500/40",
                                        NOT_CONFIGURED: "text-muted-foreground" };
  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetContent side="right" className="w-[480px] sm:max-w-[480px]">
        <SheetHeader><SheetTitle className="flex items-center gap-2"><Database className="size-4" />Data sources</SheetTitle>
          <SheetDescription>Key-free sources are used automatically. Optional keys go in a local <code>.env</code> file
            (see <code>.env.example</code>) — they are never shown here.</SheetDescription></SheetHeader>
        <div className="grid gap-2 overflow-y-auto px-4 pb-4">
          {sources.map((s) => (
            <div key={s.id} className="rounded-lg border p-3 text-xs">
              <div className="flex items-center gap-2"><span className="font-semibold">{s.name}</span>
                <Badge variant="outline" className={cn("ml-auto text-[10px]", cls[s.state])}>{s.state.replace(/_/g, " ")}</Badge></div>
              <div className="mt-1 text-muted-foreground">{s.kind} · {s.coverage}</div>
              <div className="mt-1 flex items-center gap-1.5">{s.key_required
                ? <><KeyRound className="size-3.5 text-amber-300" />{s.env.join(", ")} {s.configured ? "— set" : "— not set"}</>
                : <><CheckCircle2 className="size-3.5 text-emerald-400" />No key required</>}</div>
              <div className="mt-1 text-muted-foreground">{s.notes}</div>
            </div>))}
        </div>
      </SheetContent>
    </Sheet>
  );
}
