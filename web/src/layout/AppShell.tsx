import { NavLink, Outlet } from "react-router-dom";
import { useStore } from "../api/store";
import { badgeClass } from "../lib/ui";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/runtime", label: "Runtime" },
  { to: "/images", label: "Images" },
  { to: "/wallpapers", label: "Wallpapers" },
  { to: "/shots", label: "Shots" },
  { to: "/jobs", label: "Jobs" },
  { to: "/models", label: "Models" },
  { to: "/downloads", label: "Downloads" },
];

export function AppShell() {
  const { status, error } = useStore();
  const state = status?.loadState || (status ? "STOPPED" : "CONNECTING");
  const gpuName = status ? status.gpu?.name || "GPU box" : "Connecting…";
  const depth = status?.queue?.queueDepth ?? 0;

  return (
    <div className="shell">
      <aside className="nav" aria-label="Primary">
        <NavLink className="brand" to="/">GPU control</NavLink>
        <nav>
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end}>
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <div className="stage">
        <header className="top">
          <div>
            <p className="sub" id="gpu-name">{gpuName}</p>
          </div>
          <div className="header-meta">
            <span className="chip">Queue {depth}</span>
            <span className={`badge ${badgeClass(status ? state : "STOPPED")}`}>
              {status ? state : "CONNECTING"}
            </span>
          </div>
        </header>
        {error ? <p className="error banner">{error}</p> : null}
        <main>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
