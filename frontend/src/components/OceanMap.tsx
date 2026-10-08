import { INDIA_PATH, INDIA_VIEWBOX, project } from "./indiaShape";

/** Animated "ops map" of Indian waters: real GSHHG coastline, main shipping lanes with vessels moving along
 *  them, major ports, a Sentinel-1 swath sweeping over the sea and a detected slick off Mumbai with its
 *  hindcast trail. Decorative/illustrative — not live data. */

type LL = [number, number];
const smooth = (pts: LL[]) => {
  const p = pts.map(([lo, la]) => project(lo, la));
  let d = `M${p[0][0].toFixed(1)} ${p[0][1].toFixed(1)}`;
  for (let i = 0; i < p.length - 1; i++) {
    const p0 = p[i - 1] ?? p[i], p1 = p[i], p2 = p[i + 1], p3 = p[i + 2] ?? p2;
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C${c1[0].toFixed(1)} ${c1[1].toFixed(1)} ${c2[0].toFixed(1)} ${c2[1].toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
  }
  return d;
};

const LANES: { id: string; pts: LL[] }[] = [
  { id: "west", pts: [[68.3, 22.4], [70.8, 20.4], [72.2, 18.6], [72.8, 16.0], [73.9, 13.0], [75.2, 10.0], [76.4, 7.5], [78.2, 6.3], [80.6, 5.4], [85, 5.7], [96, 6.3]] },
  { id: "arabian", pts: [[62, 13.5], [67, 11.5], [72, 8.9], [76.4, 7.5], [78.2, 6.3]] },
  { id: "east", pts: [[81.6, 5.6], [82.4, 8.5], [81.3, 12.0], [81.0, 13.5], [82.8, 16.1], [85.4, 18.8], [87.4, 20.2], [88.2, 21.3]] },
  { id: "bay", pts: [[81.0, 13.5], [86, 12.5], [92, 9.5], [97, 7.5]] },
];
const SHIPS = [
  { lane: "west", dur: 46, begin: 0, c: "#67e8f9" }, { lane: "west", dur: 46, begin: -23, c: "#a5f3fc" },
  { lane: "arabian", dur: 30, begin: -8, c: "#67e8f9" }, { lane: "east", dur: 38, begin: -5, c: "#a5f3fc" },
  { lane: "east", dur: 38, begin: -24, c: "#67e8f9" }, { lane: "bay", dur: 34, begin: -12, c: "#a5f3fc" },
];
const PORTS: [string, number, number, "l" | "r"][] = [
  ["Kandla", 70.2, 23.0, "r"], ["Mumbai", 72.83, 18.95, "r"], ["Mormugao", 73.8, 15.4, "r"], ["Kochi", 76.25, 9.95, "r"],
  ["Chennai", 80.3, 13.1, "l"], ["Visakhapatnam", 83.3, 17.7, "l"], ["Paradip", 86.7, 20.3, "l"], ["Haldia", 88.1, 22.0, "l"],
];
const SLICK: LL = [71.85, 18.75];
const BACKTRACK: LL[] = [[71.85, 18.75], [71.6, 18.95], [71.45, 19.3], [71.1, 19.55]];

export default function OceanMap({ className }: { className?: string }) {
  const { w, h } = INDIA_VIEWBOX;
  const [sx, sy] = project(...SLICK);
  const grid = [];
  for (let lo = 65; lo <= 95; lo += 5) { const [x] = project(lo, 0); grid.push(<line key={`x${lo}`} x1={x} x2={x} y1={0} y2={h} />); }
  for (let la = 5; la <= 25; la += 5) { const [, y] = project(0, la); grid.push(<line key={`y${la}`} x1={0} x2={w} y1={y} y2={y} />); }
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className={className} role="img" aria-label="Animated map of Indian waters with shipping lanes">
      <defs>
        <linearGradient id="land" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#0d1726" /><stop offset="1" stopColor="#0a1320" /></linearGradient>
        <radialGradient id="sea" cx="0.45" cy="0.7" r="0.8">
          <stop offset="0" stopColor="#0b5a78" stopOpacity="0.75" /><stop offset="1" stopColor="#070b14" stopOpacity="0" /></radialGradient>
        <linearGradient id="swath" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#22d3ee" stopOpacity="0" /><stop offset="0.85" stopColor="#22d3ee" stopOpacity="0.10" />
          <stop offset="1" stopColor="#67e8f9" stopOpacity="0.45" /></linearGradient>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.2" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        <clipPath id="frame"><rect width={w} height={h} rx="18" /></clipPath>
        {LANES.map((l) => <path key={l.id} id={`lane-${l.id}`} d={smooth(l.pts)} />)}
        <path id="backtrack" d={smooth(BACKTRACK)} />
      </defs>
      <g clipPath="url(#frame)">
        <rect width={w} height={h} fill="url(#sea)" />
        <g stroke="#1e3a4f" strokeWidth="0.5" strokeDasharray="2 4" opacity="0.7">{grid}</g>
        {/* sea labels */}
        <g fill="#38bdf8" opacity="0.28" fontSize="10" letterSpacing="4" fontFamily="ui-monospace, monospace">
          <text x={project(64.6, 15)[0]} y={project(64.6, 15)[1]}>ARABIAN SEA</text>
          <text x={project(84.2, 14.8)[0]} y={project(84.2, 14.8)[1]}>BAY OF BENGAL</text>
          <text x={project(70.5, 4.7)[0]} y={project(70.5, 4.7)[1]}>INDIAN OCEAN</text>
        </g>
        {/* Sentinel-1 swath sweeping west -> east */}
        <g className="swath"><rect x={-120} y={-40} width={120} height={h + 80} fill="url(#swath)" transform={`skewX(-12)`} /></g>
        {/* land */}
        <path d={INDIA_PATH} fill="url(#land)" fillOpacity="0.9" />
        <path d={INDIA_PATH} fill="none" stroke="#22d3ee" strokeWidth="1" strokeOpacity="0.75" filter="url(#glow)"
          pathLength={1} className="coast-draw" />
        {/* lanes + ships */}
        {LANES.map((l) => <use key={l.id} href={`#lane-${l.id}`} fill="none" stroke="#38bdf8" strokeOpacity="0.35" strokeWidth="1"
          strokeDasharray="3 5" className="lane-flow" />)}
        {SHIPS.map((s, i) => (
          <g key={i} filter="url(#glow)">
            <path d="M5 0 L-4 -3 L-2 0 L-4 3 Z" fill={s.c}>
              <animateMotion dur={`${s.dur}s`} begin={`${s.begin}s`} repeatCount="indefinite" rotate="auto">
                <mpath href={`#lane-${s.lane}`} /></animateMotion></path>
          </g>))}
        {/* ports */}
        {PORTS.map(([name, lo, la, side]) => {
          const [x, y] = project(lo, la);
          return (
            <g key={name}>
              <circle cx={x} cy={y} r="2" fill="#e0f2fe" />
              <circle cx={x} cy={y} r="2" fill="none" stroke="#67e8f9" strokeWidth="1">
                <animate attributeName="r" values="2;9" dur="3s" begin={`${(lo % 3).toFixed(1)}s`} repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.9;0" dur="3s" begin={`${(lo % 3).toFixed(1)}s`} repeatCount="indefinite" /></circle>
              <text x={side === "r" ? x + 6 : x - 6} y={y + 3} textAnchor={side === "r" ? "start" : "end"} fontSize="8.5"
                fill="#94a3b8" fontFamily="ui-monospace, monospace">{name}</text>
            </g>);
        })}
        {/* detected slick + hindcast trail */}
        <use href="#backtrack" fill="none" stroke="#f97316" strokeWidth="1.2" strokeDasharray="2 3" strokeOpacity="0.8" />
        {[0, -1.3, -2.6].map((b) => (
          <circle key={b} r="1.6" fill="#fdba74">
            <animateMotion dur="3.9s" begin={`${b}s`} repeatCount="indefinite"><mpath href="#backtrack" /></animateMotion>
            <animate attributeName="opacity" values="1;0" dur="3.9s" begin={`${b}s`} repeatCount="indefinite" /></circle>))}
        <ellipse cx={sx} cy={sy} rx="7" ry="3" transform={`rotate(-35 ${sx} ${sy})`} fill="#1c1917" stroke="#fb923c" strokeWidth="1.2" />
        <circle cx={sx} cy={sy} r="6" fill="none" stroke="#fb923c" strokeWidth="1">
          <animate attributeName="r" values="6;20" dur="2.2s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.9;0" dur="2.2s" repeatCount="indefinite" /></circle>
        <g fontFamily="ui-monospace, monospace" fontSize="8.5">
          <line x1={sx - 8} y1={sy + 4} x2={sx - 58} y2={sy + 34} stroke="#fb923c" strokeOpacity="0.6" strokeWidth="0.8" />
          <text x={sx - 60} y={sy + 44} textAnchor="end" fill="#fdba74">SLICK · OIL_LIKELY</text>
          <text x={sx - 60} y={sy + 55} textAnchor="end" fill="#94a3b8">hindcast −48 h ↖</text>
        </g>
      </g>
      <rect width={w} height={h} rx="18" fill="none" stroke="#1e3a4f" strokeOpacity="0.8" />
    </svg>
  );
}
