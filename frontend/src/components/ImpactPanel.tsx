import { useState } from "react";
import {
  Anchor, BadgeCheck, CheckCircle2, Clock, Copy, Download, Factory, Fish, Fingerprint, Loader2, MapPin, Palmtree,
  ShieldAlert, ShieldCheck, Ship, Sprout, Trees, Waves, XCircle,
} from "lucide-react";
import { toast } from "sonner";
import type { Layers } from "@/api";
import { api, fmtT, n } from "@/api";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import { Callout, Empty, SectionLabel } from "./panels";

const LEVEL: Record<string, string> = {
  LOW: "text-emerald-300 border-emerald-500/40 bg-emerald-500/10",
  MODERATE: "text-yellow-200 border-yellow-500/40 bg-yellow-500/10",
  HIGH: "text-orange-300 border-orange-500/40 bg-orange-500/10",
  CRITICAL: "text-red-300 border-red-500/50 bg-red-500/15",
};
const PRI: Record<string, string> = {
  P1: "bg-red-500/15 text-red-300 border-red-500/40", P2: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  P3: "bg-sky-500/15 text-sky-300 border-sky-500/40",
};
const SITE_ICON: Record<string, any> = {
  coral: Sprout, coral_mangrove: Sprout, mangrove: Trees, turtle_mangrove: Trees, lagoon: Waves, intake: Factory,
  port: Anchor, fishing: Fish, tourism: Palmtree,
};
const hrs = (h: number | null | undefined) => h == null ? "—" : h < 72 ? `${n(h, 0)} h` : `${n(h / 24, 1)} d`;

function Gauge({ v, level }: { v: number; level: string }) {
  const r = 58, c = Math.PI * r;
  return (
    <div className="relative mx-auto h-[86px] w-[150px]">
      <svg viewBox="0 0 150 86" className="h-full w-full">
        <defs><linearGradient id="sev" x1="0" x2="1"><stop offset="0" stopColor="#34d399" /><stop offset="0.45" stopColor="#facc15" />
          <stop offset="0.75" stopColor="#fb923c" /><stop offset="1" stopColor="#ef4444" /></linearGradient></defs>
        <path d="M17 78 A58 58 0 0 1 133 78" fill="none" stroke="rgb(51 65 85 / 0.6)" strokeWidth="11" strokeLinecap="round" />
        <path d="M17 78 A58 58 0 0 1 133 78" fill="none" stroke="url(#sev)" strokeWidth="11" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - v / 100)} style={{ transition: "stroke-dashoffset 1.2s ease" }} />
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center leading-none">
        <div className="font-mono text-3xl font-bold text-slate-50">{n(v, 0)}</div>
        <div className={cn("mx-auto mt-1 w-fit rounded border px-1.5 py-0.5 font-mono text-[9px] tracking-widest", LEVEL[level])}>{level}</div>
      </div>
    </div>
  );
}

export default function ImpactPanel({ d, reload }: { d: Layers; reload: () => void }) {
  const imp = d.impact;
  const aid: string = d.analysis?.analysis_id;
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<Set<number>>(new Set());
  const [ver, setVer] = useState<any>(null);
  const [verifying, setVerifying] = useState(false);
  const [showPol, setShowPol] = useState(false);
  const compute = () => { setBusy(true); api.impact(aid).then(() => { toast.success("Impact assessment updated"); reload(); })
    .catch((e) => toast.error(e.message)).finally(() => setBusy(false)); };
  if (!imp) return (
    <div className="grid gap-3">
      <Empty text={d.selected ? "No impact assessment for this investigation yet (it was run before this feature existed)." :
        "Investigate a slick first — the impact assessment runs automatically at the end."} />
      {d.selected && <Button onClick={compute} disabled={busy}>{busy ? <Loader2 className="animate-spin" /> : <ShieldAlert />}Compute impact assessment</Button>}
    </div>);
  const sev = imp.severity, th = imp.threat, plan = imp.response, rob = imp.robustness;
  const syn = Object.entries(imp.synthetic_inputs ?? {}).filter(([, v]) => v).map(([k]) => k);
  const threatened = th.sites.filter((s: any) => s.threatened);
  const cands = d.candidates?.candidates ?? [];
  const toggle = (id: number) => setDone((s) => { const x = new Set(s); x.has(id) ? x.delete(id) : x.add(id); return x; });

  return (
    <div className="grid gap-3">
      {syn.length > 0 && <Callout title={`Inputs include SYNTHETIC data (${syn.join(", ")}) — exercise only`}>The severity
        index and response plan demonstrate the method; they must not be acted upon operationally.</Callout>}

      {/* ---- severity ---- */}
      <div className="rounded-xl border border-slate-700/50 bg-gradient-to-br from-slate-800/60 to-slate-900/40 p-3">
        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-slate-400">
          <ShieldAlert className="size-3.5 text-cyan-300" />{sev.name}</div>
        <div className="mt-2 grid grid-cols-[150px_1fr] items-center gap-3">
          <Gauge v={sev.score} level={sev.level} />
          <div className="grid gap-1.5 text-[11px]">
            <div className="flex items-center justify-between"><span className="text-slate-400">Response tier</span>
              <span className="rounded border border-cyan-500/40 bg-cyan-500/10 px-1.5 font-mono text-cyan-200">{sev.response_tier.central}</span></div>
            <div className="flex items-center justify-between"><span className="text-slate-400">Est. volume</span>
              <span className="font-mono text-slate-100">{n(sev.volume_estimate.m3.low, 1)}–{n(sev.volume_estimate.m3.high, 0)} m³</span></div>
            <div className="flex items-center justify-between"><span className="text-slate-400">Nearest shore</span>
              <span className="font-mono text-slate-100">{th.distance_to_coast_km != null ? `${n(th.distance_to_coast_km, 1)} km` : "> 300 km"}</span></div>
            <div className="flex items-center justify-between"><span className="text-slate-400">Shoreline ETA</span>
              <span className={cn("font-mono", th.beaching.eta_hours != null && th.beaching.eta_hours < 48 ? "text-red-300" : "text-slate-100")}>
                {hrs(th.beaching.eta_hours)}</span></div>
          </div>
        </div>
        <div className="mt-3 grid gap-1.5">{Object.entries(sev.components).map(([k, c]: any) => (
          <div key={k} title={c.evidence} className="grid grid-cols-[76px_1fr_34px] items-center gap-2 text-[11px]">
            <span className="capitalize text-slate-400">{k}</span>
            <div className="h-1.5 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-amber-400 to-red-500"
              style={{ width: `${100 * c.value}%`, transition: "width .8s ease" }} /></div>
            <span className="text-right font-mono text-slate-300">+{n(c.points, 0)}</span></div>))}</div>
        <details className="mt-2 text-[11px] text-slate-400"><summary className="cursor-pointer text-cyan-300">Why this score</summary>
          <ul className="mt-1 grid gap-0.5 pl-3">{Object.entries(sev.components).map(([k, c]: any) =>
            <li key={k}><b className="capitalize text-slate-300">{k}:</b> {c.evidence}</li>)}</ul>
          <p className="mt-1">{sev.volume_estimate.basis}. {sev.response_tier.basis}. {sev.disclaimer}</p></details>
      </div>

      {/* ---- shoreline threat ---- */}
      <SectionLabel right={<span className="normal-case tracking-normal text-slate-500">{threatened.length} threatened</span>}>Shoreline & receptors</SectionLabel>
      <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-2.5 text-[11.5px] leading-relaxed">
        {th.beaching.eta_hours != null
          ? <>Oil is forecast to reach the shore near <b className="text-slate-50">{th.beaching.landing_place}</b> in about{" "}
              <b className="text-orange-300">{hrs(th.beaching.eta_hours)}</b> <span className="text-slate-400">({th.beaching.method === "forward_model"
                ? "OpenOil forward forecast" : "extrapolated from the forecast drift"})</span>.</>
          : <span className="text-emerald-300">{th.beaching.note}.</span>}
        {th.drift_vector && <div className="mt-0.5 font-mono text-[10.5px] text-slate-400">drift {n(th.drift_vector.speed_kmh, 2)} km/h toward {n(th.drift_vector.bearing_deg, 0)}°</div>}
      </div>
      <div className="grid gap-1.5">{th.sites.slice(0, 6).map((s: any) => {
        const I = SITE_ICON[s.type] ?? MapPin;
        return (
          <div key={s.id} className={cn("flex items-center gap-2.5 rounded-lg border p-2", s.threatened ? "border-orange-500/40 bg-orange-500/[0.06]" : "border-slate-700/50 bg-slate-900/30")}>
            <span className={cn("grid size-7 shrink-0 place-items-center rounded-md", s.threatened ? "bg-orange-500/15 text-orange-300" : "bg-slate-800 text-slate-400")}><I className="size-4" /></span>
            <div className="min-w-0 flex-1">
              <div className="truncate text-[12px] font-medium text-slate-100">{s.name}</div>
              <div className="truncate text-[10.5px] text-slate-400">{s.type_label} · {s.receptor}</div></div>
            <div className="text-right">
              <div className={cn("font-mono text-[12px] font-semibold", s.threatened ? "text-orange-300" : "text-slate-400")}>{s.threatened ? hrs(s.eta_hours) : `${n(s.distance_now_km, 0)} km`}</div>
              <div className="flex justify-end gap-0.5">{Array.from({ length: 5 }, (_, i) =>
                <span key={i} className={cn("size-1 rounded-full", i < Math.round(s.weight / 2) ? "bg-cyan-400" : "bg-slate-700")} />)}</div></div>
          </div>);
      })}</div>

      {/* ---- response plan ---- */}
      <SectionLabel right={<span className="normal-case tracking-normal text-cyan-300">{done.size}/{plan.actions.length} done</span>}>
        Immediate response · {plan.mrcc.name}</SectionLabel>
      <div className="grid gap-1.5">{plan.actions.map((a: any) => (
        <div key={a.id} className={cn("rounded-lg border p-2 transition", done.has(a.id) ? "border-emerald-500/30 bg-emerald-500/[0.04] opacity-60" : "border-slate-700/50 bg-slate-900/40")}>
          <div className="flex items-start gap-2">
            <Checkbox className="mt-0.5" checked={done.has(a.id)} onCheckedChange={() => toggle(a.id)} />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className={cn("rounded border px-1 font-mono text-[9px] font-bold", PRI[a.priority])}>{a.priority}</span>
                <span className={cn("text-[12px] font-medium text-slate-100", done.has(a.id) && "line-through")}>{a.title}</span></div>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[10.5px] text-slate-400">
                <span className="flex items-center gap-1 font-mono text-amber-200/90"><Clock className="size-3" />within {hrs(a.deadline_hours)}</span>
                <span>{a.agency}</span></div>
              <details className="text-[11px] text-slate-400"><summary className="cursor-pointer text-cyan-300/80">details</summary>
                <p className="mt-0.5 text-slate-300">{a.detail}</p><p className="mt-0.5 italic">Why: {a.why}</p></details>
            </div>
          </div>
        </div>))}</div>
      <p className="text-[10.5px] text-slate-500">{plan.notes.join(" ")}</p>

      {/* ---- intercepts ---- */}
      {cands.length > 0 && <>
        <SectionLabel>Suspect vessels · next-port intercept</SectionLabel>
        <div className="grid gap-1.5">{cands.slice(0, 5).map((c: any) => {
          const ic = imp.intercepts?.[c.vessel.mmsi];
          const r = rob?.candidates?.find((x: any) => x.mmsi === c.vessel.mmsi);
          return (
            <div key={c.vessel.mmsi} className="grid grid-cols-[1fr_auto] gap-2 rounded-lg border border-slate-700/50 bg-slate-900/40 p-2 text-[11px]">
              <div className="min-w-0">
                <div className="flex items-center gap-1.5 truncate font-medium text-slate-100"><Ship className="size-3.5 text-cyan-300" />#{c.rank} {c.vessel.name ?? c.vessel.mmsi}</div>
                <div className="truncate text-slate-400">{ic?.port ? <>→ <b className="text-slate-200">{ic.port}</b> · ETA {fmtT(ic.eta_time)} · {n(ic.speed_kn, 1)} kn</> : (ic?.note ?? "no AIS track")}</div>
                {ic?.ais_silence && <div className="truncate text-amber-300/90">⚠ {ic.ais_silence}</div>}
              </div>
              {r && <div className="text-right"><div className="font-mono text-[12px] font-semibold text-slate-50">{n(100 * r.p_rank1, 0)}%</div>
                <div className="text-[9.5px] text-slate-400">P(#1)</div></div>}
            </div>);
        })}</div>
        {rob?.available && <p className={cn("text-[11px]", rob.verdict === "ROBUST" ? "text-emerald-300/90" : rob.verdict === "CONTESTED" ? "text-amber-200/90" : "text-slate-300")}>
          <b>Ranking {rob.verdict.toLowerCase()}:</b> {rob.note}</p>}
      </>}

      {/* ---- evidence seal ---- */}
      <SectionLabel>Chain of custody</SectionLabel>
      <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-2.5 text-[11px]">
        {d.evidence_seal ? <>
          <div className="flex items-center gap-2"><Fingerprint className="size-4 text-cyan-300" />
            <span className="text-slate-300">Evidence sealed · {d.evidence_seal.n_files} files · v{d.evidence_seal.version}</span>
            <Button size="xs" variant="outline" className="ml-auto" disabled={verifying}
              onClick={() => { setVerifying(true); api.verify(aid).then(setVer).catch((e) => toast.error(e.message)).finally(() => setVerifying(false)); }}>
              {verifying ? <Loader2 className="animate-spin" /> : <ShieldCheck />}Verify</Button></div>
          <div className="mt-1 truncate font-mono text-[10px] text-slate-500" title={d.evidence_seal.root}>SHA-256 root {d.evidence_seal.root}</div>
          <div className="font-mono text-[10px] text-slate-500">sealed {fmtT(d.evidence_seal.sealed_at)}</div>
          {ver && <div className={cn("mt-1.5 flex items-center gap-1.5", ver.ok ? "text-emerald-300" : "text-red-300")}>
            {ver.ok ? <BadgeCheck className="size-3.5" /> : <XCircle className="size-3.5" />}{ver.message}
            {!ver.ok && ver.modified?.length > 0 && <span className="text-slate-400">({ver.modified.slice(0, 3).join(", ")})</span>}</div>}
        </> : <span className="text-slate-400">Not sealed yet — recompute to seal.</span>}
      </div>

      {/* ---- POLREP ---- */}
      <SectionLabel>POLREP draft</SectionLabel>
      <div className="flex gap-2">
        <Button size="sm" variant="outline" onClick={() => setShowPol(!showPol)}>{showPol ? "Hide" : "Preview"} POLREP</Button>
        <Button size="sm" variant="outline" onClick={() => navigator.clipboard.writeText(imp.polrep).then(() => toast.success("POLREP copied"))}><Copy />Copy</Button>
        <Button size="sm" variant="outline" render={<a href={`/api/spill/${aid}/polrep`} />}><Download />.txt</Button>
      </div>
      {showPol && <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded-lg border border-slate-700/50 bg-slate-950/70 p-2.5 font-mono text-[10.5px] leading-relaxed text-slate-300">{imp.polrep}</pre>}
      <p className="text-[10.5px] text-slate-500">Draft only — the system never sends messages. {imp.gazetteer_note}</p>
      <Button size="sm" variant="ghost" onClick={compute} disabled={busy} className="justify-self-start text-slate-400">
        {busy ? <Loader2 className="animate-spin" /> : <CheckCircle2 />}Recompute & re-seal</Button>
    </div>
  );
}
