import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "maplibre-gl/dist/maplibre-gl.css";
import "./index.css";
import App from "./App";
import Landing from "./Landing";

// hash routes: "" -> landing page, "#/console[?step=<tab>]" -> investigation console
function useHashRoute() {
  const [h, setH] = useState(window.location.hash);
  useEffect(() => { const f = () => setH(window.location.hash); window.addEventListener("hashchange", f); return () => window.removeEventListener("hashchange", f); }, []);
  return h;
}

function Root() {
  const h = useHashRoute();
  useEffect(() => { window.scrollTo(0, 0); }, [h.startsWith("#/console")]);
  if (h.startsWith("#/console")) {
    const step = new URLSearchParams(h.split("?")[1] ?? "").get("step") ?? undefined;
    return <App initialStep={step} />;
  }
  return <Landing />;
}

createRoot(document.getElementById("root")!).render(<Root />);
