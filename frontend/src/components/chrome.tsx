import { useEffect, useState } from "react";
import { Hexagon } from "lucide-react";
import { cn } from "@/lib/utils";

/** Shared console chrome: brand mark, UTC clock and a live/offline pill. */
export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="relative grid size-9 place-items-center">
        <Hexagon className="absolute size-9 text-cyan-400" strokeWidth={1.4} />
        <span className="font-mono text-[10px] font-bold text-cyan-300">OS</span>
      </div>
      <div className="leading-tight">
        <div className="bg-gradient-to-r from-cyan-300 to-sky-500 bg-clip-text text-sm font-bold tracking-wide text-transparent">
          OIL SPILL INVESTIGATION</div>
        {!compact && <div className="font-mono text-[10px] tracking-wider text-slate-400">SIH26143 · SAR · DRIFT · AIS</div>}
      </div>
    </div>
  );
}

export function useUtcClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => { const h = setInterval(() => setNow(new Date()), 1000); return () => clearInterval(h); }, []);
  return now;
}

export function UtcClock({ big = false }: { big?: boolean }) {
  const now = useUtcClock();
  const t = now.toISOString().slice(11, 19);
  if (!big) return <span className="font-mono text-xs tabular-nums text-cyan-300">{t} <span className="text-slate-500">UTC</span></span>;
  return (
    <div className="text-center">
      <div className="font-mono text-[10px] tracking-[0.2em] text-slate-400">SYSTEM TIME · UTC</div>
      <div className="font-mono text-3xl font-light tabular-nums text-cyan-300 [text-shadow:0_0_18px_rgb(34_211_238/0.45)]">{t}</div>
      <div className="text-xs text-slate-400">{now.toUTCString().slice(0, 16)}</div>
    </div>
  );
}

export function LivePill({ ok, label }: { ok: boolean | null; label?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[10px] tracking-wider",
      ok === null ? "border-slate-600 text-slate-400" : ok ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
        : "border-red-500/40 bg-red-500/10 text-red-300")}>
      <span className={cn("size-1.5 rounded-full", ok === null ? "bg-slate-500" : ok ? "animate-pulse bg-cyan-400" : "bg-red-400")} />
      {label ?? (ok === null ? "CHECKING" : ok ? "LIVE" : "OFFLINE")}
    </span>
  );
}

export function Meter({ label, value, tone = "cyan" }: { label: string; value: number; tone?: "cyan" | "emerald" | "violet" | "amber" }) {
  const bar = { cyan: "from-cyan-500 to-sky-500", emerald: "from-emerald-500 to-teal-500", violet: "from-violet-500 to-indigo-500",
    amber: "from-amber-500 to-orange-500" }[tone];
  return (
    <div>
      <div className="mb-1 flex justify-between text-[11px]"><span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-300">{Math.round(value)}%</span></div>
      <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
        <div className={cn("h-full rounded-full bg-gradient-to-r", bar)} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} /></div>
    </div>
  );
}

export const glass = "rounded-xl border border-slate-700/50 bg-slate-900/55 backdrop-blur-md";
