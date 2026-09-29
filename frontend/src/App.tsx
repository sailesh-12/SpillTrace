import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Anchor, Check, ChevronDown, Database, FileText, Home, Layers as LayersIcon, ListTree, Maximize2, Radar, ShieldAlert, Ship, Waves } from "lucide-react";
import ImpactPanel from "@/components/ImpactPanel";
import { Brand, LivePill, UtcClock } from "@/components/chrome";
import { Toaster, toast } from "sonner";
import { api, BBox, Candidate, Layers, ProgressEvent, Scene, Source, TriageRow } from "@/api";
import MapView, { LayerKey, TimeCursor } from "@/components/MapView";
import {
  CandidatesPanel, EvidencePanel, InvestigationPanel, LayerMenu, ProgressCard, Prov, SearchPanel, SlickDetail,
  SourcesSheet, Timeline, TriagePanel,
} from "@/components/panels";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

const VISIBLE: Record<LayerKey, boolean> = {
  sar: true, slicks: true, selected: true, particles: true, heatmap: true, regions: true, ais: true, vessels: true,
  forward: true, sarships: true, impact: true,
};
const STEPS = [
  { key: "search", label: "Area & scene", icon: Radar, hint: "Choose a box over Indian waters and a Sentinel-1 pass (or a SYNTHETIC scene)." },
  { key: "triage", label: "Detection & triage", icon: Waves, hint: "Review dark features; tick the slick(s) you judge to be oil." },
  { key: "investigation", label: "Hindcast", icon: Anchor, hint: "Backward OpenOil drift with currents and wind; probable source regions." },
  { key: "candidates", label: "Candidates", icon: Ship, hint: "Vessels whose AIS tracks correlate with the backtracked oil (max 10)." },
  { key: "evidence", label: "Evidence", icon: FileText, hint: "Per-factor score, statements, supporting / contradicting / missing evidence." },
  { key: "impact", label: "Impact & response", icon: ShieldAlert, hint: "Severity index, shoreline threat, immediate actions, intercepts and evidence seal." },
] as const;
type Tab = typeof STEPS[number]["key"];

export default function App({ initialStep }: { initialStep?: string }) {
  const [health, setHealth] = useState<any>(null);
  const [stages, setStages] = useState<string[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [analyses, setAnalyses] = useState<any[]>([]);
  const [aid, setAid] = useState<string | null>(null);
  const [data, setData] = useState<Layers | null>(null);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<Tab>((STEPS.find((s) => s.key === initialStep)?.key ?? "search") as Tab);
  useEffect(() => { const k = STEPS.find((s) => s.key === initialStep)?.key; if (k) setTab(k); }, [initialStep]);
  const [aoi, setAoi] = useState<BBox | null>([71.5, 18.6, 72.2, 19.1]);   // off Mumbai (Arabian Sea shipping lanes)
  const [searched, setSearched] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [sceneId, setSceneId] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [focusRow, setFocusRow] = useState<TriageRow | null>(null);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [visible, setVisible] = useState(VISIBLE);
  const [layersOpen, setLayersOpen] = useState(false);
  const [fitSignal, setFitSignal] = useState(0);
  const [noticesOpen, setNoticesOpen] = useState(false);
  const [legendOpen, setLegendOpen] = useState(true);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [probLabel, setProbLabel] = useState("combined");
  const [follow, setFollow] = useState(true);
  const stopStream = useRef<null | (() => void)>(null);
  const [queuePos, setQueuePos] = useState<number | null>(null);     // 0 = running, n = n jobs ahead
  useEffect(() => {
    if (!busy || !aid) { setQueuePos(null); return; }
    const tick = () => api.queue().then((q) => {
      if (q.running === aid) setQueuePos(0);
      else { const i = q.waiting.indexOf(aid); setQueuePos(i >= 0 ? i + 1 : null); }
    }).catch(() => {});
    tick();
    const h = setInterval(tick, 3000);
    return () => clearInterval(h);
  }, [busy, aid]);

  const refreshList = useCallback(() => api.list().then(setAnalyses).catch(() => {}), []);
  useEffect(() => {
    api.health().then(setHealth).catch((e) => toast.error(`Backend unreachable: ${e.message}`));
    api.config().then((c) => setStages(c.stages)).catch(() => {});
    api.sources().then(setSources).catch(() => {});
    refreshList();
  }, [refreshList]);

  const load = useCallback((id: string, goTo?: Tab) => {
    api.layers(id).then((d) => {
      setData(d);
      setEvents(d.analysis?.events ?? []);
      setSelected(d.candidates?.candidates?.[0] ?? null);
      setChecked(new Set(d.selected?.selected_component_ids ?? []));
      if (goTo) setTab(goTo);
      else setTab(d.candidates ? "candidates" : d.triage ? "triage" : "search");
    }).catch((e) => toast.error(e.message));
  }, []);

  const follow_ = (id: string, since: number, goTo: Tab) => {
    setBusy(true);
    stopStream.current?.();
    stopStream.current = api.events(id, since, (e) => setEvents((xs) => [...xs, e]),
      (d) => {
        setBusy(false); refreshList(); load(id, d.status === "COMPLETED" ? goTo : undefined);
        if (d.status === "FAILED") toast.error(d.error?.message ?? "Analysis failed");
        else toast.success(d.status === "AWAITING_SLICK_SELECTION" ? "Detection complete — review the triage" : `Analysis ${d.status.toLowerCase()}`);
      },
      () => { setBusy(false); toast.error("Lost the progress stream — reopen the analysis from the list."); });
  };

  const onSearch = (start: string, end: string) => {
    if (!aoi) return;
    setSearching(true);
    api.scenes(aoi, start, end).then((s) => { setScenes(s); setSearched(true);
      setSceneId([...s].sort((a, b) => b.aoi_coverage - a.aoi_coverage)[0]?.id ?? null);
      if (!s.length) toast.message("No Sentinel-1 IW scenes for this area and period."); })
      .catch((e) => toast.error(e.message)).finally(() => setSearching(false));
  };
  const onSynthetic = () => {
    if (!aoi) return;
    setData(null); setEvents([]); setSelected(null); setChecked(new Set());
    api.detect("SYNTHETIC", aoi, 20).then(({ analysis_id }) => { setAid(analysis_id); follow_(analysis_id, 0, "triage"); })
      .catch((e) => toast.error(e.message));
  };
  const onDetect = (res: number) => {
    if (!sceneId || !aoi) return;
    setData(null); setEvents([]); setSelected(null); setChecked(new Set());
    api.detect(sceneId, aoi, res).then(({ analysis_id }) => { setAid(analysis_id); follow_(analysis_id, 0, "triage"); })
      .catch((e) => toast.error(e.message));
  };
  const onUpload = (f: FormData) => {
    setData(null); setEvents([]);
    api.upload(f).then(({ analysis_id }) => { setAid(analysis_id); follow_(analysis_id, 0, "triage"); }).catch((e) => toast.error(e.message));
  };
  const onInvestigate = (aisMode: "auto" | "real" | "synthetic") => {
    if (!aid) return;
    const since = events.length;
    api.investigate(aid, [...checked], aisMode).then(() => { setTab("investigation"); follow_(aid, since, "candidates"); })
      .catch((e) => toast.error(e.message));
  };
  const openAnalysis = (id: string) => { setAid(id); stopStream.current?.(); setBusy(false); load(id); };

  const toggle = (id: string) => setChecked((s) => { const x = new Set(s); x.has(id) ? x.delete(id) : x.add(id); return x; });

  // ---- unified timeline (hindcast ascending, then forecast) ----------------------------------------
  const obs = data?.selected?.timestamp ? new Date(data.selected.timestamp) : data?.spill?.timestamp ? new Date(data.spill.timestamp) : null;
  const timeline = useMemo(() => {
    const out: TimeCursor[] = [];
    const b = data?.backward_particles;
    if (b) for (let i = b.times.length - 1; i >= 0; i--) out.push({ time: new Date(b.times[i]), kind: "backward", frame: i });
    const f = data?.forward_particles;
    if (f) f.times.forEach((t, i) => { if (i > 0) out.push({ time: new Date(t), kind: "forward", frame: i }); });
    return out;
  }, [data]);
  useEffect(() => { setIdx(0); setPlaying(false); }, [timeline]);
  useEffect(() => {
    if (!playing || !timeline.length) return;
    const h = setInterval(() => setIdx((i) => (i + 1) % timeline.length), 300);
    return () => clearInterval(h);
  }, [playing, timeline]);
  const cursor = timeline[idx] ?? null;
  const probLabels = useMemo(() => {
    const s = new Set<string>();
    data?.source_probability?.features.forEach((f) => f.properties.kind === "probability_cell" && s.add(f.properties.map));
    return ["combined", ...[...s].filter((x) => x !== "combined").sort((a, b) => parseFloat(a.slice(2)) - parseFloat(b.slice(2)))];
  }, [data]);
  const effProb = useMemo(() => {
    if (!follow || !cursor || !obs || cursor.kind !== "backward") return probLabel;
    const h = (obs.getTime() - cursor.time.getTime()) / 3600e3;
    const offs = probLabels.filter((l) => l !== "combined").map((l) => ({ l, h: parseFloat(l.slice(2)) }));
    return offs.length ? offs.reduce((a, b) => (Math.abs(b.h - h) < Math.abs(a.h - h) ? b : a)).l : probLabel;
  }, [follow, cursor, obs, probLabel, probLabels]);

  const selectCandidate = (c: Candidate) => {
    setSelected(c); setTab("evidence");
    const t = c.features?.best_match_time ? Date.parse(c.features.best_match_time) : null;
    if (t && timeline.length) {
      let best = 0;
      timeline.forEach((x, i) => { if (Math.abs(x.time.getTime() - t) < Math.abs(timeline[best].time.getTime() - t)) best = i; });
      setIdx(best); setPlaying(false);
    }
  };
  const stepDone = (k: Tab) => ({ search: !!data, triage: !!data?.triage, investigation: !!data?.drift,
    candidates: !!data?.candidates, evidence: !!selected, impact: !!data?.impact }[k]);
  const warnings: string[] = data?.analysis?.warnings ?? [];

  return (
    <div className="flex h-screen flex-col bg-[#070b14] text-foreground">
      <Toaster theme="dark" position="bottom-right" richColors />
      {/* ---------------- top bar ---------------- */}
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-slate-800/80 bg-[#070b14]/90 px-3 backdrop-blur-md">
        <a href="#/" title="Back to the project overview"><Brand /></a>
        <a href="#/" className="ml-2 hidden items-center gap-1.5 rounded-md px-2 py-1 text-xs text-slate-400 transition-colors hover:bg-slate-800/60 hover:text-cyan-300 md:flex">
          <Home className="size-3.5" />Overview</a>
        <div className="ml-auto flex items-center gap-2">
          <span className="hidden xl:inline"><UtcClock /></span>
          <LivePill ok={health ? health.status === "ok" : null} />
          <select className="h-8 max-w-56 rounded-lg border border-slate-700/60 bg-slate-900/70 px-2 text-xs" value={aid ?? ""}
            onChange={(e) => e.target.value && openAnalysis(e.target.value)}>
            <option value="">Open investigation…</option>
            {analyses.map((a) => <option key={a.analysis_id} value={a.analysis_id}>{a.analysis_id.replace("SPILL_", "")} · {a.status}</option>)}
          </select>
          <Button size="sm" variant="outline" onClick={() => setSourcesOpen(true)}><Database /><span className="hidden sm:inline">Data sources</span></Button>
          {data?.candidates && <Button size="sm" variant="outline" render={<a href={data.report_html_url} target="_blank" />}><FileText />Report</Button>}
          {health && <Badge variant="outline" className="hidden font-mono text-[10px] lg:inline-flex">GPU {health.cuda ? "✓" : "✗"} · {health.mode?.toUpperCase()}</Badge>}
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* ---------------- step rail ---------------- */}
        <nav className="hidden w-[84px] shrink-0 flex-col items-stretch gap-1 border-r border-slate-800/80 bg-[#070b14]/80 px-2 py-3 md:flex">
          {STEPS.map((s, i) => {
            const active = tab === s.key, done = stepDone(s.key);
            return (
              <button key={s.key} onClick={() => setTab(s.key)} title={s.label}
                className={cn("group relative flex flex-col items-center gap-1 rounded-lg px-1 py-2.5 text-[10px] leading-tight transition-colors",
                  active ? "bg-cyan-500/10 text-cyan-300 shadow-[inset_0_0_0_1px_rgb(34_211_238/0.35)]" : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-100")}>
                {active && <span className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-cyan-400" />}
                <span className="relative"><s.icon className="size-4.5" />
                  {done && <Check className="absolute -right-2 -top-1.5 size-3 rounded-full bg-emerald-500 p-px text-slate-950" />}</span>
                <span className="text-center"><span className="font-mono text-slate-500">{i + 1}</span> {s.label}</span>
              </button>);
          })}
          <div className="mt-auto grid gap-2 border-t border-slate-800/80 pt-3 font-mono text-[9px] text-slate-500">
            {[["MODEL", health?.model_present], ["GPU", health?.cuda], ["QUEUE", !busy]].map(([k, ok]) => (
              <div key={String(k)} className="flex items-center justify-between px-1">{k}
                <span className={cn("size-1.5 rounded-full", ok === undefined ? "bg-slate-600" : ok ? "bg-emerald-400" : "bg-amber-400")} /></div>))}
          </div>
        </nav>

        {/* ---------------- workspace ---------------- */}
        <aside className="flex w-[340px] shrink-0 flex-col border-r border-slate-800/80 bg-gradient-to-b from-[#0c1422] to-[#080d18] xl:w-[430px]">
          {/* step header */}
          {(() => {
            const i = STEPS.findIndex((x) => x.key === tab), st = STEPS[i];
            return (
              <div className="relative overflow-hidden border-b border-slate-800/80 px-4 pb-3 pt-3.5">
                <div className="pointer-events-none absolute -right-8 -top-10 size-32 rounded-full bg-cyan-500/10 blur-2xl" />
                <div className="flex items-center gap-2 font-mono text-[10px] tracking-[0.2em] text-cyan-400">STEP {i + 1} / {STEPS.length}
                  {aid && <span className="ml-auto truncate tracking-normal text-slate-500">{aid.replace("SPILL_", "")}</span>}</div>
                <div className="mt-1 flex items-center gap-2 text-base font-semibold text-slate-50"><st.icon className="size-4.5 text-cyan-300" />{st.label}</div>
                <p className="mt-0.5 text-[11.5px] leading-snug text-slate-400">{st.hint}</p>
                <div className="mt-2.5 flex gap-1">{STEPS.map((x, j) => (
                  <button key={x.key} onClick={() => setTab(x.key)} title={x.label} className={cn("h-1 flex-1 rounded-full transition-colors",
                    j === i ? "bg-cyan-400 shadow-[0_0_8px_rgb(34_211_238/0.8)]" : stepDone(x.key) ? "bg-cyan-700" : "bg-slate-700/80")} />))}</div>
              </div>);
          })()}
          {data && <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-slate-800/80 bg-slate-950/30 px-4 py-2 font-mono text-[10px] uppercase tracking-wider text-slate-400">
            <span className="flex items-center gap-1.5">SAR<Prov p={data.acquisition?.provenance ?? data.scene?.georef?.provenance} /></span>
            <span className="flex items-center gap-1.5">Forcing<Prov p={data.drift?.forcing?.provenance} /></span>
            <span className="flex items-center gap-1.5">AIS<Prov p={data.candidates?.ais?.provider?.provenance} /></span>
          </div>}
          {warnings.length > 0 && <div className="border-b border-slate-800/80 px-3 py-2">
            <button onClick={() => setNoticesOpen(!noticesOpen)}
              className="flex w-full items-center gap-2 rounded-lg border border-amber-500/35 bg-amber-500/[0.07] px-2.5 py-1.5 text-left text-[11.5px] text-amber-200">
              <AlertTriangle className="size-3.5 shrink-0" /><span className="flex-1 truncate">
                <b>{warnings.length} data notice{warnings.length === 1 ? "" : "s"}</b> · {warnings[0]}</span>
              <ChevronDown className={cn("size-3.5 shrink-0 transition", noticesOpen && "rotate-180")} /></button>
            {noticesOpen && <ul className="mt-1.5 grid gap-1 pl-1 text-[11px] leading-snug text-amber-100/80">
              {warnings.map((w, i) => <li key={i} className="flex gap-1.5"><span className="text-amber-400">•</span>{w}</li>)}</ul>}
          </div>}
          <Tabs value={tab} onValueChange={(v) => setTab(v as Tab)} className="min-h-0 flex-1 gap-0">
            <TabsList variant="line" className="w-full justify-start border-b px-2 md:hidden">
              {STEPS.map((s) => <TabsTrigger key={s.key} value={s.key} className="text-xs">{s.label}</TabsTrigger>)}
            </TabsList>
            <div key={tab} className="fade-up min-h-0 flex-1 overflow-y-auto p-3" style={{ animationDuration: ".35s" }}>
              <TabsContent value="search">
                <SearchPanel aoi={aoi} setAoi={setAoi} drawing={drawing} setDrawing={setDrawing} scenes={scenes} sceneId={sceneId}
                  setSceneId={setSceneId} busy={busy} onSearch={onSearch} searching={searching} onDetect={onDetect} onUpload={onUpload}
                  searched={searched} onSynthetic={onSynthetic} />
              </TabsContent>
              <TabsContent value="triage" className="grid gap-3">
                {data ? <TriagePanel d={data} checked={checked} toggle={toggle} busy={busy} onInvestigate={onInvestigate}
                  onFocus={(r) => setFocusRow(r)} /> : <p className="text-xs text-muted-foreground">No detection loaded.</p>}
                <SlickDetail r={focusRow} />
              </TabsContent>
              <TabsContent value="investigation">{data ? <InvestigationPanel d={data} /> : null}</TabsContent>
              <TabsContent value="candidates">{data ? <CandidatesPanel d={data} selected={selected} select={selectCandidate} /> : null}</TabsContent>
              <TabsContent value="evidence"><EvidencePanel c={selected} /></TabsContent>
              <TabsContent value="impact">{data ? <ImpactPanel d={data} reload={() => aid && load(aid, "impact")} /> : null}</TabsContent>
            </div>
          </Tabs>
          <p className="border-t px-3 py-2 text-[10px] leading-snug text-muted-foreground">Investigative decision support. Rankings reflect evidence
            correlation under stated assumptions and do not establish that any vessel caused a spill.</p>
        </aside>

        {/* ---------------- map ---------------- */}
        <main className="relative min-w-0 flex-1">
          <MapView data={data} visible={visible} cursor={cursor} probLabel={effProb} selected={selected}
            onSelectMmsi={(m) => { const c = data?.candidates?.candidates?.find((x: Candidate) => x.vessel.mmsi === m); if (c) selectCandidate(c); }}
            aoi={aoi} drawing={drawing} onAoi={(b) => { setAoi(b); setDrawing(false); }}
            scenes={tab === "search" ? scenes : []} sceneId={sceneId} onSceneClick={setSceneId}
            checked={checked} onToggleComponent={toggle} focus={focusRow ? focusRow.centroid : null} fitSignal={fitSignal} />
          <div className="pointer-events-none absolute inset-x-3 top-3 flex items-start justify-between gap-3">
            <div className="pointer-events-auto grid gap-2">
              {busy && queuePos !== null && queuePos > 0 && <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                Queued — {queuePos} analysis job{queuePos === 1 ? "" : "s"} ahead (one job runs at a time on this machine)</div>}
              {events.length > 0 && <ProgressCard events={events} stages={stages} busy={busy} />}
            </div>
            <div className="pointer-events-auto mr-12 grid justify-items-end gap-2">
              <div className="flex gap-2">
                {data && <Button size="sm" variant="secondary" onClick={() => setFitSignal((x) => x + 1)}><Maximize2 />Zoom to scene</Button>}
                <Button size="sm" variant="secondary" onClick={() => setLayersOpen(!layersOpen)}><LayersIcon />Layers</Button>
              </div>
              {layersOpen && <LayerMenu visible={visible} set={(k, v) => setVisible((s) => ({ ...s, [k]: v }))} probLabel={probLabel}
                probLabels={probLabels} setProbLabel={setProbLabel} follow={follow} setFollow={setFollow} />}
            </div>
          </div>
          <div className="absolute bottom-3 left-3 rounded-xl border border-slate-700/60 bg-slate-950/80 text-[11px] backdrop-blur-md">
            <button onClick={() => setLegendOpen(!legendOpen)} className="flex w-full items-center gap-1.5 px-2.5 py-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-400 hover:text-slate-100">
              <ListTree className="size-3.5" />Legend<ChevronDown className={cn("ml-auto size-3.5 transition", !legendOpen && "-rotate-90")} /></button>
            {legendOpen && <div className="grid gap-1 px-2.5 pb-2.5">
              {[["#ff5a1f", "Oil-likely"], ["#facc15", "Uncertain"], ["#4ade80", "Look-alike likely"], ["#67e8f9", "Hindcast particles"],
                ["#c084fc", "Forecast / affected area"], ["#ef4444", "50 % source region"], ["#22d3ee", "Selected vessel"],
                ["#fb923c", "Threatened receptor / shore ETA"]].map(([c, l]) =>
                <div key={l} className="flex items-center gap-2 text-slate-300"><span className="h-2 w-3.5 rounded-sm" style={{ background: c, boxShadow: `0 0 6px ${c}` }} />{l}</div>)}
            </div>}
          </div>
          {data && !layersOpen && (() => {
            const tri = data.triage, cands = data.candidates?.candidates ?? [], top = cands[0];
            const items: [string, React.ReactNode][] = [
              ["Observed", obs ? obs.toISOString().slice(0, 16).replace("T", " ") + "Z" : "—"],
              ["Features", tri ? `${tri.components.length} · ${tri.components.filter((r: TriageRow) => r.label === "OIL_LIKELY").length} oil-likely` : "—"],
              ["Wind", tri?.wind_at_acquisition ? `${tri.wind_at_acquisition.wind_speed_ms.toFixed(1)} m/s` : "—"],
              ["Hindcast", data.drift ? `${data.drift.n_members} members` : "—"],
              ["Candidates", data.candidates ? String(cands.length) : "—"],
              ["Top", top ? `${top.vessel.name ?? top.vessel.mmsi} · ${top.score.toFixed(0)}` : "—"],
              ["Severity", data.impact ? `${data.impact.severity.score.toFixed(0)} · ${data.impact.severity.level}` : "—"],
              ["Shore ETA", data.impact?.threat?.beaching?.eta_hours != null ? `${data.impact.threat.beaching.eta_hours.toFixed(0)} h` : "—"],
            ];
            return (
              <div className="pointer-events-none absolute right-14 top-14 hidden w-60 rounded-xl border border-slate-700/60 bg-slate-950/80 p-3 backdrop-blur-md xl:block">
                <div className="mb-2 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.2em] text-cyan-400">
                  <span className="relative flex size-2"><span className="absolute inline-flex size-full animate-ping rounded-full bg-cyan-400 opacity-60" />
                    <span className="relative inline-flex size-2 rounded-full bg-cyan-400" /></span>Investigation</div>
                <div className="grid gap-1 text-[11px]">{items.map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2"><span className="text-slate-400">{k}</span>
                    <span className="truncate font-mono text-slate-100">{v}</span></div>))}</div>
              </div>);
          })()}
          <div className="pointer-events-none absolute bottom-3 left-48 right-3 flex justify-center [&>*]:pointer-events-auto">
            <Timeline times={timeline} idx={idx} setIdx={setIdx} playing={playing} setPlaying={setPlaying} obs={obs} /></div>
          <div className="pointer-events-none absolute inset-0 shadow-[inset_0_0_80px_rgb(2_6_16/0.75)]" />
          {drawing && <div className="absolute left-1/2 top-3 -translate-x-1/2 rounded-full bg-primary px-3 py-1 text-xs text-primary-foreground shadow-[0_0_20px_rgb(34_211_238/0.5)]">
            Drag a rectangle over water to set the area of interest</div>}
        </main>
      </div>
      <SourcesSheet open={sourcesOpen} setOpen={setSourcesOpen} sources={sources} />
    </div>
  );
}
