import { useEffect, useRef } from "react";

/** Animated backdrop: tracer particles drifting in a slowly rotating current field (the same idea as the
 *  hindcast particles), with faint links between near neighbours. Pauses when the tab is hidden and draws a
 *  single static frame when the user prefers reduced motion. */
export default function ParticleField({ density = 1, className = "" }: { density?: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current!;
    const ctx = cv.getContext("2d")!;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0, h = 0, raf = 0, t = 0;
    type P = { x: number; y: number; r: number; a: number };
    let ps: P[] = [];
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = cv.clientWidth; h = cv.clientHeight;
      cv.width = w * dpr; cv.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const n = Math.round((w * h) / 9000 * density);
      ps = Array.from({ length: n }, () => ({ x: Math.random() * w, y: Math.random() * h,
        r: Math.random() * 1.4 + 0.4, a: Math.random() * 0.5 + 0.2 }));
    };
    const flow = (x: number, y: number) => {
      const ang = Math.sin(x * 0.0023 + t * 0.00017) + Math.cos(y * 0.0031 - t * 0.00011);
      return [Math.cos(ang) * 0.25 + 0.12, Math.sin(ang) * 0.25];
    };
    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      for (const p of ps) {
        const [vx, vy] = flow(p.x, p.y);
        p.x += vx; p.y += vy;
        if (p.x > w + 5) p.x = -5; if (p.x < -5) p.x = w + 5;
        if (p.y > h + 5) p.y = -5; if (p.y < -5) p.y = h + 5;
      }
      ctx.lineWidth = 0.6;
      for (let i = 0; i < ps.length; i++) for (let j = i + 1; j < ps.length; j++) {
        const dx = ps[i].x - ps[j].x, dy = ps[i].y - ps[j].y, d2 = dx * dx + dy * dy;
        if (d2 < 6400) {
          ctx.strokeStyle = `rgba(34, 211, 238, ${0.08 * (1 - d2 / 6400)})`;
          ctx.beginPath(); ctx.moveTo(ps[i].x, ps[i].y); ctx.lineTo(ps[j].x, ps[j].y); ctx.stroke();
        }
      }
      for (const p of ps) {
        ctx.fillStyle = `rgba(103, 232, 249, ${p.a})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      }
    };
    const loop = () => { t += 16; draw(); raf = requestAnimationFrame(loop); };
    resize();
    const ro = new ResizeObserver(resize); ro.observe(cv);
    if (reduce) draw(); else raf = requestAnimationFrame(loop);
    const vis = () => { cancelAnimationFrame(raf); if (!document.hidden && !reduce) raf = requestAnimationFrame(loop); };
    document.addEventListener("visibilitychange", vis);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); document.removeEventListener("visibilitychange", vis); };
  }, [density]);
  return <canvas ref={ref} aria-hidden className={`pointer-events-none absolute inset-0 h-full w-full ${className}`} />;
}
