import { CSSProperties, ReactNode, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/** Small motion toolkit for the landing page (no dependencies): reveal-on-scroll, count-up numbers,
 *  a cycling typewriter line and scroll trackers. Everything respects prefers-reduced-motion via CSS. */

export function useInView<T extends Element>(opts: IntersectionObserverInit = { threshold: 0.18 }, once = true) {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setInView(true); if (once) io.disconnect(); } else if (!once) setInView(false);
    }, opts);
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return [ref, inView] as const;
}

type Variant = "up" | "left" | "right" | "scale" | "blur";
export function Reveal({ children, delay = 0, variant = "up", className, as: Tag = "div", style }: {
  children: ReactNode; delay?: number; variant?: Variant; className?: string; as?: any; style?: CSSProperties;
}) {
  const [ref, inView] = useInView<HTMLDivElement>();
  return (
    <Tag ref={ref} data-reveal={variant} data-shown={inView ? "" : undefined}
      style={{ ...style, transitionDelay: `${delay}ms` }} className={cn("reveal", className)}>{children}</Tag>
  );
}

/** Headline that assembles word by word. */
export function WordsIn({ text, className, wordClassName, start = 0, step = 70 }: {
  text: string; className?: string; wordClassName?: string; start?: number; step?: number;
}) {
  return (
    <span className={className}>
      {text.split(" ").map((w, i) => (
        <span key={i} className={cn("word-in inline-block pr-[0.25em]", wordClassName)} style={{ animationDelay: `${start + i * step}ms` }}>{w}</span>))}
    </span>
  );
}

export function CountUp({ to, decimals = 0, prefix = "", suffix = "", ms = 1600 }: {
  to: number; decimals?: number; prefix?: string; suffix?: string; ms?: number;
}) {
  const [ref, inView] = useInView<HTMLSpanElement>({ threshold: 0.5 });
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!inView) return;
    let raf = 0; const t0 = performance.now();
    const tick = (t: number) => {
      const p = Math.min((t - t0) / ms, 1);
      setV(to * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, to, ms]);
  return <span ref={ref} className="tabular-nums">{prefix}{v.toLocaleString("en-IN", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}{suffix}</span>;
}

/** Types each line, holds it, erases it, moves to the next. */
export function Typewriter({ lines, className }: { lines: string[]; className?: string }) {
  const [i, setI] = useState(0);
  const [n, setN] = useState(0);
  const [del, setDel] = useState(false);
  useEffect(() => {
    const line = lines[i];
    const h = setTimeout(() => {
      if (!del && n < line.length) setN(n + 1);
      else if (!del) setDel(true);
      else if (n > 0) setN(n - 2 < 0 ? 0 : n - 2);
      else { setDel(false); setI((i + 1) % lines.length); }
    }, !del && n === line.length ? 2200 : del ? 12 : 28);
    return () => clearTimeout(h);
  }, [i, n, del, lines]);
  return <span className={className}>{lines[i].slice(0, n)}<span className="caret">▍</span></span>;
}

export function useScrollY() {
  const [y, setY] = useState(0);
  useEffect(() => {
    let raf = 0;
    const f = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => setY(window.scrollY)); };
    window.addEventListener("scroll", f, { passive: true }); f();
    return () => { window.removeEventListener("scroll", f); cancelAnimationFrame(raf); };
  }, []);
  return y;
}

/** 0 when the element's top reaches the bottom of the viewport, 1 when its bottom reaches the middle. */
export function useSectionProgress<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const y = useScrollY();
  const [p, setP] = useState(0);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const r = el.getBoundingClientRect(), vh = window.innerHeight;
    setP(Math.min(Math.max((vh - r.top) / (r.height + vh / 2), 0), 1));
  }, [y]);
  return [ref, p] as const;
}
