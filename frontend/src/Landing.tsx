import { useEffect, useState } from "react";
import {
  AlertTriangle, Anchor, ArrowRight, BookOpen, Braces, Cpu, Database, FileText, Gauge, Globe2, HeartPulse,
  Layers, ListOrdered, Radar, Satellite, Scale, Ship, ShieldAlert, ShieldCheck, Terminal, Waves as WavesIcon, Wind,
} from "lucide-react";
import ParticleField from "@/components/ParticleField";
import OceanMap from "@/components/OceanMap";
import Waves from "@/components/Waves";
import { CountUp, Reveal, Typewriter, useScrollY, useSectionProgress, WordsIn } from "@/components/motion";
import { Brand, glass, LivePill, Meter, UtcClock } from "@/components/chrome";
import { cn } from "@/lib/utils";

export const goConsole = (step?: string) => { window.location.hash = step ? `#/console?step=${step}` : "#/console"; };

const PROBLEM = [
  { icon: Globe2, k: <CountUp to={11098} suffix=" km" />, v: "Indian coastline (2023 re-measurement) bordering one of the world's busiest tanker routes" },
  { icon: Ship, k: <CountUp to={95} prefix="~" suffix=" %" />, v: "of India's trade by volume moves by sea past the west and east coasts" },
  { icon: WavesIcon, k: <CountUp to={2.0} decimals={1} suffix=" M km²" />, v: "Exclusive Economic Zone to police for illegal discharges (MARPOL Annex I)" },
  { icon: AlertTriangle, k: "hours → days", v: "between a discharge and the satellite pass that reveals the slick — the ship is long gone" },
];

const TERMINAL = [
  "sentinel-1 IW GRD · Arabian Sea · dark feature → OIL_LIKELY (ship-trail pattern)",
  "openoil hindcast · 48 h backwards · ensemble of currents + wind · source regions per hour",
  "ais correlation · vessels in the backtracked corridor · up to 10 candidates ranked",
  "evidence report · per-factor score · supporting / contradicting / missing evidence",
];

const WHY = [
  { t: "Slicks are seen, polluters are not", d: "Synthetic Aperture Radar sees oil day and night through cloud, but only as a dark patch. It does not say which vessel released it." },
  { t: "The oil has moved", d: "By the time of the satellite pass, currents and wind have carried the slick kilometres away from where it was released." },
  { t: "Too many vessels to check", d: "Dozens to hundreds of AIS tracks cross any busy area in two days. Checking them by hand is slow and misses AIS gaps and slow-downs." },
  { t: "Evidence must stand scrutiny", d: "Enforcement agencies (Coast Guard, DG Shipping) need a traceable chain: data sources, assumptions, uncertainties, and no over-claiming." },
];

const STEPS = [
  { key: "search", icon: Radar, t: "Area & scene", d: "Pick a box over Indian waters. Find Sentinel-1 radar scenes (real), or a clearly labelled SYNTHETIC scene where coverage is missing." },
  { key: "triage", icon: WavesIcon, t: "Detection & triage", d: "DeepLabV3+ segments dark slicks; wind, shape and ship-attachment checks separate oil from natural look-alikes." },
  { key: "investigation", icon: Anchor, t: "Hindcast", d: "OpenDrift/OpenOil runs the slick backwards in time with real currents and wind, giving probable source regions per hour." },
  { key: "candidates", icon: Ship, t: "Candidates", d: "AIS tracks (real, or SYNTHETIC for demonstration) are matched to the backtracked oil in space and time; up to 10 vessels are ranked." },
  { key: "evidence", icon: FileText, t: "Evidence", d: "An explainable Evidence Correlation Score, per-factor breakdown, uncertainties and a downloadable report." },
  { key: "impact", icon: ShieldAlert, t: "Impact & response", d: "Severity index, shoreline ETA to sensitive sites, prioritised actions for ICG, next-port intercept, sealed evidence and a POLREP draft." },
];

const DATA = [
  { icon: Satellite, t: "SAR imagery", real: "Sentinel-1 GRD · Microsoft Planetary Computer", syn: "SYNTHETIC scene (speckle, wind texture, ships, one discharge trail, one look-alike)" },
  { icon: Wind, t: "Currents & wind", real: "Open-Meteo marine currents + ERA5 / forecast wind", syn: "SYNTHETIC monsoon-season climatology (WICC/EICC, SW/NE monsoon)" },
  { icon: Ship, t: "AIS vessel tracks", real: "Global Fishing Watch (token) · Danish Maritime Authority · local files", syn: "SYNTHETIC Indian traffic (MMSI 419…, SYN- names, max 10 vessels)" },
];

type Ep = { method: "GET" | "POST"; path: string; href?: string; d: string; icon: any; check?: string };
const ENDPOINTS: Ep[] = [
  { method: "GET", path: "/#/console", href: "#/console", d: "Investigation console (dashboard)", icon: Terminal },
  { method: "GET", path: "/docs", href: "/docs", d: "Interactive API reference (Swagger UI)", icon: BookOpen, check: "/docs" },
  { method: "GET", path: "/openapi.json", href: "/openapi.json", d: "OpenAPI schema", icon: Braces, check: "/openapi.json" },
  { method: "GET", path: "/api/health", href: "/api/health", d: "Health: model, GPU, providers, modes", icon: HeartPulse, check: "/api/health" },
  { method: "GET", path: "/api/v1/sources", href: "/api/v1/sources", d: "Data sources and whether keys are configured", icon: Database, check: "/api/v1/sources" },
  { method: "GET", path: "/api/v1/scenes", d: "Search Sentinel-1 scenes for a bbox and period", icon: Satellite },
  { method: "POST", path: "/api/v1/analyses", d: "Phase 1 — acquire scene (or SYNTHETIC) and detect", icon: Radar },
  { method: "POST", path: "/api/v1/analyses/{id}/investigate", d: "Phase 2 — hindcast, AIS correlation, ranking", icon: Anchor },
  { method: "GET", path: "/api/v1/analyses", href: "/api/v1/analyses", d: "All investigations and their status", icon: ListOrdered, check: "/api/v1/analyses" },
  { method: "GET", path: "/api/v1/queue", href: "/api/v1/queue", d: "Job queue (one analysis runs at a time)", icon: Layers, check: "/api/v1/queue" },
  { method: "GET", path: "/api/spill/{id}/report", d: "Evidence report (HTML / JSON / Markdown)", icon: FileText },
  { method: "POST", path: "/api/spill/{id}/impact", d: "Severity index, shoreline threat, response plan (re-seals)", icon: ShieldAlert },
  { method: "GET", path: "/api/spill/{id}/verify", d: "Verify the SHA-256 evidence seal (chain of custody)", icon: ShieldCheck },
  { method: "GET", path: "/api/spill/{id}/polrep", d: "Draft pollution report (POLREP) for the MRCC", icon: FileText },
];

function scrollTo(id: string) { document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" }); }

function useEndpointStatus() {
  const [st, setSt] = useState<Record<string, { ok: boolean; ms: number } | undefined>>({});
  useEffect(() => {
    ENDPOINTS.filter((e) => e.check).forEach((e) => {
      const t0 = performance.now();
      fetch(e.check!).then((r) => setSt((s) => ({ ...s, [e.path]: { ok: r.ok, ms: performance.now() - t0 } })))
        .catch(() => setSt((s) => ({ ...s, [e.path]: { ok: false, ms: 0 } })));
    });
  }, []);
  return st;
}

export default function Landing() {
  const [health, setHealth] = useState<any>(null);
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [runs, setRuns] = useState<any[]>([]);
  const [queue, setQueue] = useState<any>(null);
  const ep = useEndpointStatus();
  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then((h) => { setHealth(h); setHealthOk(h.status === "ok"); }).catch(() => setHealthOk(false));
    fetch("/api/v1/analyses").then((r) => r.json()).then(setRuns).catch(() => {});
    const q = () => fetch("/api/v1/queue").then((r) => r.json()).then(setQueue).catch(() => {});
    q(); const h = setInterval(q, 5000); return () => clearInterval(h);
  }, []);
  const done = runs.filter((r) => r.status === "COMPLETED").length;
  const y = useScrollY();
  const maxY = Math.max(document.documentElement.scrollHeight - window.innerHeight, 1);
  const pageP = Math.min(y / maxY, 1);
  const [howRef, howP] = useSectionProgress<HTMLElement>();
  const howFill = Math.min(howP * 1.4, 1);

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#070b14] text-slate-100">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_top,rgb(8_47_73/0.55),transparent_60%)]" />
      <div className="fixed inset-0" style={{ transform: `translateY(${y * -0.05}px)` }}><ParticleField density={1.1} /></div>
      {/* deep-sea gradient that darkens as you scroll down ("diving") */}
      <div className="pointer-events-none fixed inset-0 bg-gradient-to-b from-transparent to-[#02050b]" style={{ opacity: pageP * 0.8 }} />

      {/* ---------------- nav ---------------- */}
      <header className="sticky top-0 z-30 border-b border-slate-800/70 bg-[#070b14]/70 backdrop-blur-md">
        <div className="absolute bottom-0 left-0 h-px bg-gradient-to-r from-cyan-400 via-sky-400 to-indigo-400 shadow-[0_0_8px_rgb(34_211_238/0.8)]"
          style={{ width: `${(pageP * 100).toFixed(2)}%` }} />
        <div className="mx-auto flex h-16 max-w-7xl items-center gap-6 px-4 sm:px-6">
          <Brand />
          <nav className="mx-auto hidden items-center gap-1 text-sm text-slate-400 md:flex">
            {[["why", "Why it matters"], ["how", "How it works"], ["data", "Data"], ["endpoints", "Endpoints"]].map(([id, l]) =>
              <button key={id} onClick={() => scrollTo(id)} className="relative rounded-md px-3 py-1.5 transition-colors hover:bg-slate-800/60 hover:text-cyan-300">{l}</button>)}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            <span className="hidden sm:inline"><UtcClock /></span>
            <LivePill ok={healthOk} />
            <button onClick={() => goConsole()} className="hidden rounded-lg bg-cyan-500 px-3 py-1.5 text-sm font-semibold text-slate-950 shadow-[0_0_20px_rgb(34_211_238/0.35)] transition hover:bg-cyan-400 sm:inline-flex">
              Launch console</button>
          </div>
        </div>
      </header>

      <main className="relative z-10">
        {/* ---------------- hero ---------------- */}
        <section className="relative">
          <div className="mx-auto grid max-w-7xl items-center gap-10 px-4 pb-4 pt-12 sm:px-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:pt-14"
            style={{ opacity: Math.max(1 - y / 900, 0.15) }}>
            <div style={{ transform: `translateY(${y * 0.12}px)` }}>
              <div className="fade-up mb-5 inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 font-mono text-[11px] tracking-wider text-cyan-300">
                <span className="relative flex size-2"><span className="absolute inline-flex size-full animate-ping rounded-full bg-cyan-400 opacity-75" />
                  <span className="relative inline-flex size-2 rounded-full bg-cyan-400" /></span>
                SMART INDIA HACKATHON 2026 · PROBLEM STATEMENT SIH26143</div>
              <h1 className="text-4xl font-bold leading-[1.1] tracking-tight [perspective:600px] sm:text-5xl xl:text-6xl">
                <WordsIn text="From an oil slick on a radar image" start={150} /><br />
                <WordsIn text="to the vessels that were there." start={650}
                  wordClassName="bg-gradient-to-r from-cyan-300 via-sky-400 to-indigo-400 bg-clip-text text-transparent" />
              </h1>
              <p className="fade-up mt-6 max-w-2xl text-base leading-relaxed text-slate-300 sm:text-lg" style={{ animationDelay: "1.1s" }}>
                A decision-support system for maritime oil-pollution investigation in Indian waters. It detects slicks in Sentinel-1
                SAR imagery, traces them back in time with ocean-drift physics, and correlates the backtrack with AIS vessel tracks —
                ranking the vessels that should be investigated first, with every data source, assumption and uncertainty shown.
              </p>
              <div className="fade-up mt-5 flex items-center gap-2 overflow-hidden rounded-lg border border-slate-700/60 bg-slate-950/60 px-3 py-2 font-mono text-[12px] text-slate-300"
                style={{ animationDelay: "1.35s" }}>
                <Terminal className="size-3.5 shrink-0 text-cyan-400" /><span className="text-cyan-400">$</span>
                <Typewriter lines={TERMINAL} className="truncate" /></div>
              <div className="fade-up mt-7 flex flex-wrap gap-3" style={{ animationDelay: "1.55s" }}>
                <button onClick={() => goConsole()} className="group relative inline-flex items-center gap-2 overflow-hidden rounded-lg bg-cyan-500 px-5 py-2.5 font-semibold text-slate-950 shadow-[0_0_28px_rgb(34_211_238/0.35)] transition hover:bg-cyan-400">
                  <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/40 to-transparent transition-transform duration-700 group-hover:translate-x-full" />
                  <span className="relative">Start an investigation</span><ArrowRight className="relative size-4 transition group-hover:translate-x-0.5" /></button>
                <button onClick={() => scrollTo("endpoints")} className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/60 px-5 py-2.5 font-medium text-slate-200 transition hover:border-cyan-500/50 hover:text-cyan-300">
                  <Braces className="size-4" />Explore the endpoints</button>
              </div>
              <p className="fade-up mt-5 flex items-start gap-2 text-xs text-slate-500" style={{ animationDelay: "1.7s" }}><Scale className="mt-0.5 size-3.5 shrink-0" />
                Rankings express evidence correlation under stated assumptions. They do not establish that any vessel caused a spill.</p>
            </div>

            {/* animated ops map of Indian waters */}
            <div className="fade-up relative" style={{ animationDelay: "0.4s" }}>
              <div style={{ transform: `translateY(${y * -0.06}px)` }}>
                <div className="absolute -inset-6 rounded-[32px] bg-cyan-500/10 blur-3xl" />
                <OceanMap className="relative w-full drop-shadow-[0_0_40px_rgb(8_145_178/0.25)]" />
                <div className={cn(glass, "float-y absolute -left-2 top-4 px-3 py-2 sm:-left-5")}>
                  <div className="font-mono text-[9px] tracking-[0.2em] text-slate-400">SYSTEM TIME</div><UtcClock /></div>
                <div className={cn(glass, "float-y absolute -right-1 bottom-12 px-3 py-2 text-[11px] sm:-right-4")} style={{ animationDelay: "-3s" }}>
                  <div className="mb-1 flex items-center gap-2"><LivePill ok={healthOk} /><span className="font-mono text-slate-400">backend</span></div>
                  <div className="font-mono text-slate-300">model {health ? (health.model_present ? "loaded" : "missing") : "…"} · {health ? (health.cuda ? "GPU" : "CPU") : "…"} · AIS {health?.ais_mode ?? "…"}</div></div>
                <div className="mt-3 flex flex-wrap items-center justify-center gap-3 font-mono text-[10px] text-slate-500">
                  <span className="flex items-center gap-1"><span className="inline-block h-0.5 w-4 bg-sky-400/60" />shipping lane</span>
                  <span className="flex items-center gap-1"><span className="inline-block size-0 border-y-[3px] border-l-[6px] border-y-transparent border-l-cyan-300" />vessel</span>
                  <span className="flex items-center gap-1"><span className="inline-block size-2 rounded-full border border-orange-400" />slick</span>
                  <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 bg-cyan-400/30" />SAR swath</span>
                  <span className="text-slate-600">· illustrative</span></div>
              </div>
            </div>
          </div>
          <Waves shipX={pageP} className="-mt-2" />
        </section>

        {/* ---------------- problem stats ---------------- */}
        <section className="mx-auto max-w-7xl px-4 pt-6 sm:px-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {PROBLEM.map(({ icon: I, k, v }, i) => (
              <Reveal key={i} delay={i * 110} className={cn(glass, "group p-5 transition duration-300 hover:-translate-y-1 hover:border-cyan-500/40")}>
                <I className="size-5 text-cyan-400 transition group-hover:scale-110" />
                <div className="mt-3 font-mono text-2xl font-semibold text-slate-50">{k}</div>
                <p className="mt-1 text-sm leading-snug text-slate-400">{v}</p></Reveal>))}
          </div>
        </section>

        {/* ---------------- why ---------------- */}
        <section id="why" className="mx-auto max-w-7xl scroll-mt-20 px-4 py-20 sm:px-6">
          <SectionTitle kicker="THE NEED" title="Why this project is necessary" />
          <Reveal as="p" variant="blur" className="max-w-3xl text-slate-300">Operational and illegal discharges — bilge water, tank washings, sludge — are a chronic source
            of oil at sea, alongside accidents such as the 2010 MSC Chitra collision off Mumbai, the 2017 Ennore spill near Chennai and
            the 2025 sinking of MSC Elsa 3 off Kochi. Detecting a slick is only half the task; attributing it quickly and defensibly is what
            enables enforcement and deters polluters.</Reveal>
          <div className="mt-8 grid gap-4 md:grid-cols-2">
            {WHY.map((w, i) => (
              <Reveal key={w.t} variant={i % 2 ? "right" : "left"} delay={i * 90} className={cn(glass, "flex gap-4 p-5 transition hover:border-cyan-500/40")}>
                <div className="grid size-9 shrink-0 place-items-center rounded-lg border border-cyan-500/30 bg-cyan-500/10 font-mono text-sm text-cyan-300">0{i + 1}</div>
                <div><div className="font-semibold text-slate-100">{w.t}</div><p className="mt-1 text-sm leading-relaxed text-slate-400">{w.d}</p></div>
              </Reveal>))}
          </div>
        </section>

        {/* ---------------- how ---------------- */}
        <section id="how" ref={howRef} className="mx-auto max-w-7xl scroll-mt-20 px-4 pb-20 sm:px-6">
          <SectionTitle kicker="THE PIPELINE" title="How it works — six steps, one console" />
          <div className="relative mb-5 hidden h-1 overflow-hidden rounded-full bg-slate-800/80 md:block">
            <div className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-sky-400 to-orange-400 shadow-[0_0_12px_rgb(34_211_238/0.7)]"
              style={{ width: `${(howFill * 100).toFixed(1)}%` }} />
          </div>
          <div className="grid gap-4 md:grid-cols-3 xl:grid-cols-6">
            {STEPS.map((s, i) => {
              const lit = howFill >= (i + 0.5) / STEPS.length;
              return (
                <Reveal key={s.key} delay={i * 120}>
                  <button onClick={() => goConsole(s.key)}
                    className={cn(glass, "group relative h-full w-full p-5 text-left transition duration-500 hover:-translate-y-1 hover:border-cyan-500/50 hover:shadow-[0_0_30px_rgb(34_211_238/0.12)]",
                      lit && "border-cyan-500/40 shadow-[0_0_24px_rgb(34_211_238/0.10)]")}>
                    <div className="flex items-center justify-between">
                      <span className={cn("grid size-9 place-items-center rounded-lg border transition duration-500",
                        lit ? "border-cyan-400/60 bg-cyan-400/15 text-cyan-300" : "border-slate-700 text-slate-500")}><s.icon className="size-5" /></span>
                      <span className="font-mono text-[11px] text-slate-500">STEP {i + 1}</span></div>
                    <div className="mt-4 font-semibold text-slate-100">{s.t}</div>
                    <p className="mt-1.5 text-[13px] leading-relaxed text-slate-400">{s.d}</p>
                    <div className="mt-4 flex items-center gap-1 text-xs text-cyan-300 opacity-0 transition group-hover:opacity-100">
                      Open in console<ArrowRight className="size-3.5" /></div>
                  </button>
                </Reveal>);
            })}
          </div>
        </section>

        {/* ---------------- data ---------------- */}
        <section id="data" className="mx-auto max-w-7xl scroll-mt-20 px-4 pb-20 sm:px-6">
          <SectionTitle kicker="REAL FIRST" title="Real data where it exists, labelled SYNTHETIC where it does not" />
          <Reveal as="p" variant="blur" className="mb-8 max-w-3xl text-slate-300">Sentinel-1 revisits Indian waters less often than Europe, and open AIS for India is limited.
            So every step uses real data when available and otherwise a realistic, clearly labelled synthetic stand-in (permitted by SIH26143)
            — the detection model, drift physics and scoring run unchanged either way.</Reveal>
          <div className="grid gap-4 md:grid-cols-3">
            {DATA.map((d, i) => (
              <Reveal key={d.t} variant="scale" delay={i * 120} className={cn(glass, "p-5 transition hover:border-cyan-500/40")}>
                <div className="flex items-center gap-2 font-semibold"><d.icon className="size-4.5 text-cyan-400" />{d.t}</div>
                <div className="mt-4 grid gap-3 text-sm">
                  <div><span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">REAL</span>
                    <p className="mt-1.5 text-slate-300">{d.real}</p></div>
                  <div><span className="rounded border border-amber-500/50 bg-amber-500/15 px-1.5 py-0.5 font-mono text-[10px] text-amber-300">SYNTHETIC FALLBACK</span>
                    <p className="mt-1.5 text-slate-400">{d.syn}</p></div>
                </div>
              </Reveal>))}
          </div>
        </section>

        {/* ---------------- endpoints ---------------- */}
        <section id="endpoints" className="mx-auto max-w-7xl scroll-mt-20 px-4 pb-24 sm:px-6">
          <SectionTitle kicker="ENDPOINTS" title="Where to go next" />
          <div className="mb-6 grid gap-4 md:grid-cols-3">
            <BigLink delay={0} icon={Terminal} t="Investigation console" d="Draw an area, detect, hindcast, rank candidates and read the evidence." onClick={() => goConsole()} primary />
            <BigLink delay={120} icon={BookOpen} t="API reference" d="Every REST endpoint with schemas; try requests in the browser." href="/docs" />
            <BigLink delay={240} icon={ShieldCheck} t="Health & data sources" d="What is loaded, which providers are live and which need a key." href="/api/v1/sources" />
          </div>
          <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
            <Reveal className={cn(glass, "overflow-hidden")}>
              <div className="flex items-center justify-between border-b border-slate-700/50 px-4 py-3 text-sm">
                <span className="flex items-center gap-2 font-semibold"><Braces className="size-4 text-cyan-400" />Available endpoints</span>
                <span className="font-mono text-[11px] text-slate-500">base: {window.location.origin}</span></div>
              <ul className="divide-y divide-slate-800/80">
                {ENDPOINTS.map((e) => {
                  const s = ep[e.path];
                  const row = (
                    <div className="flex items-center gap-3 px-4 py-2.5 text-sm">
                      <span className={cn("w-12 shrink-0 rounded px-1.5 py-0.5 text-center font-mono text-[10px] font-bold",
                        e.method === "GET" ? "bg-cyan-500/15 text-cyan-300" : "bg-violet-500/15 text-violet-300")}>{e.method}</span>
                      <e.icon className="hidden size-4 shrink-0 text-slate-500 sm:block" />
                      <code className="min-w-0 shrink-0 truncate font-mono text-[13px] text-slate-200 sm:w-72">{e.path}</code>
                      <span className="hidden min-w-0 flex-1 truncate text-slate-400 md:block">{e.d}</span>
                      <span className="ml-auto shrink-0 font-mono text-[11px]">
                        {e.check ? (s ? <span className={s.ok ? "text-emerald-400" : "text-red-400"}>● {s.ok ? `${Math.round(s.ms)} ms` : "error"}</span>
                          : <span className="text-slate-500">● …</span>) : <span className="text-slate-600">{e.href ? "open" : "via console / API"}</span>}</span>
                    </div>);
                  return <li key={e.path}>{e.href ? <a href={e.href} className="block transition hover:bg-cyan-500/5">{row}</a> : row}</li>;
                })}
              </ul>
            </Reveal>
            <Reveal variant="right" delay={150} className="grid content-start gap-4">
              <div className={cn(glass, "relative overflow-hidden p-5")}>
                <div className="pointer-events-none absolute -right-12 -top-12 size-40 rounded-full border border-cyan-500/20">
                  <div className="radar-sweep absolute inset-0 rounded-full bg-[conic-gradient(from_0deg,rgb(34_211_238/0.28),transparent_25%)]" /></div>
                <UtcClock big /></div>
              <div className={cn(glass, "p-5")}>
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm font-semibold"><Gauge className="size-4 text-cyan-400" />System status</div>
                  <LivePill ok={healthOk} /></div>
                <div className="grid grid-cols-2 gap-2.5 text-xs">
                  {[
                    ["Detection model", health ? (health.model_present ? "Loaded" : "Missing") : "…", Cpu],
                    ["GPU", health ? (health.cuda ? "CUDA" : "CPU only") : "…", Cpu],
                    ["Forcing", health?.forcing_provider ?? "…", Wind],
                    ["AIS mode", health?.ais_mode ?? "…", Ship],
                  ].map(([k, v, I]: any) => (
                    <div key={k} className="rounded-lg border border-slate-700/50 bg-slate-800/40 p-2.5">
                      <div className="flex items-center gap-1.5 text-slate-400"><I className="size-3.5" />{k}</div>
                      <div className="mt-1 font-mono text-sm text-slate-100">{v}</div></div>))}
                </div>
                <div className="mt-4 grid gap-3">
                  <Meter label="Investigations completed" value={runs.length ? (100 * done) / runs.length : 0} />
                  <Meter label="Worker load" value={queue ? (queue.running ? 100 : 0) : 0} tone="violet" />
                </div>
                <div className="mt-3 font-mono text-[11px] text-slate-400">
                  {runs.length} investigations · {queue?.running ? `running ${queue.running.replace("SPILL_", "")}` : "worker idle"}
                  {queue?.waiting?.length ? ` · ${queue.waiting.length} queued` : ""}</div>
              </div>
            </Reveal>
          </div>
        </section>
      </main>

      <footer className="relative z-10">
        <Waves />
        <div className="border-t border-slate-800/70 bg-[#070b14]">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-6 text-xs text-slate-500 sm:px-6">
            <Brand compact />
            <span>Investigative decision support · not a legal determination of responsibility</span>
            <button onClick={() => goConsole()} className="text-cyan-400 hover:text-cyan-300">Open console →</button>
          </div>
        </div>
      </footer>
    </div>
  );
}

function SectionTitle({ kicker, title }: { kicker: string; title: string }) {
  return (
    <Reveal className="mb-6">
      <div className="flex items-center gap-3 font-mono text-[11px] tracking-[0.25em] text-cyan-400">
        <span className="relative flex size-3 items-center justify-center"><span className="sonar absolute size-3 rounded-full border border-cyan-400" />
          <span className="size-1.5 rounded-full bg-cyan-400" /></span>{kicker}
        <span className="h-px w-16 bg-gradient-to-r from-cyan-400/60 to-transparent" /></div>
      <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-50 sm:text-3xl">{title}</h2>
    </Reveal>
  );
}

function BigLink({ icon: I, t, d, href, onClick, primary, delay = 0 }: {
  icon: any; t: string; d: string; href?: string; onClick?: () => void; primary?: boolean; delay?: number;
}) {
  const cls = cn(glass, "group block h-full w-full p-5 text-left transition hover:-translate-y-0.5 hover:border-cyan-500/50",
    primary && "border-cyan-500/40 bg-cyan-500/[0.07] shadow-[0_0_30px_rgb(34_211_238/0.12)]");
  const body = (<>
    <div className="flex items-center justify-between"><I className="size-5 text-cyan-400" />
      <ArrowRight className="size-4 text-slate-500 transition group-hover:translate-x-0.5 group-hover:text-cyan-300" /></div>
    <div className="mt-3 font-semibold text-slate-100">{t}</div>
    <p className="mt-1 text-sm text-slate-400">{d}</p></>);
  return <Reveal delay={delay}>{href ? <a href={href} className={cls}>{body}</a> : <button onClick={onClick} className={cls}>{body}</button>}</Reveal>;
}
