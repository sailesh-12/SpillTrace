/** Layered animated ocean waves; an optional ship rides the front wave at `shipX` (0..1, scroll-linked). */
export default function Waves({ shipX, className = "" }: { shipX?: number; className?: string }) {
  const wave = (amp: number, len: number, y: number) => {
    let d = `M0 ${y}`;
    for (let x = 0; x <= 2880; x += len) d += ` q ${len / 4} ${-amp} ${len / 2} 0 t ${len / 2} 0`;
    return d + ` V 140 H 0 Z`;
  };
  return (
    <div className={`pointer-events-none relative h-28 w-full overflow-hidden ${className}`} aria-hidden>
      <svg className="wave-x absolute bottom-0 h-full w-[200%]" style={{ animationDuration: "26s" }} viewBox="0 0 2880 140" preserveAspectRatio="none">
        <path d={wave(14, 360, 58)} fill="#0e7490" fillOpacity="0.10" /></svg>
      <svg className="wave-x absolute bottom-0 h-full w-[200%]" style={{ animationDuration: "18s", animationDirection: "reverse" }} viewBox="0 0 2880 140" preserveAspectRatio="none">
        <path d={wave(10, 240, 76)} fill="#0891b2" fillOpacity="0.12" /></svg>
      <svg className="wave-x absolute bottom-0 h-full w-[200%]" style={{ animationDuration: "12s" }} viewBox="0 0 2880 140" preserveAspectRatio="none">
        <path d={wave(7, 180, 92)} fill="#070b14" fillOpacity="0.95" />
        <path d={wave(7, 180, 92).split(" V")[0]} fill="none" stroke="#22d3ee" strokeOpacity="0.35" strokeWidth="1.5" /></svg>
      {shipX !== undefined && (
        <div className="bob absolute bottom-[38px]" style={{ left: `calc(${(shipX * 100).toFixed(2)}% - 40px)` }}>
          <svg width="80" height="34" viewBox="0 0 80 34">
            {/* tanker silhouette with wake */}
            <path d="M-40 29 Q 0 26 12 28" stroke="#67e8f9" strokeOpacity="0.35" strokeWidth="1.2" fill="none" />
            <path d="M8 22 H70 L64 30 H14 Z" fill="#0f172a" stroke="#22d3ee" strokeWidth="1" />
            <rect x="52" y="12" width="10" height="10" fill="#0f172a" stroke="#22d3ee" strokeWidth="1" />
            <rect x="56" y="6" width="3" height="6" fill="#22d3ee" />
            <path d="M16 22 V18 H48 V22" fill="none" stroke="#22d3ee" strokeOpacity="0.6" strokeWidth="1" />
          </svg>
        </div>)}
    </div>
  );
}
