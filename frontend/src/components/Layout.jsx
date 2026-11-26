import { NavLink, Outlet } from "react-router-dom";
import { useEffect, useState } from "react";

const LINKS = [
  ["/", "Live Monitor"],
  ["/review", "Review Queue"],
  ["/entities", "Entity Risk"],
  ["/insights", "Model Insights"],
  ["/cost", "Cost Simulator"],
  ["/try", "Try It"],
  ["/about", "About"],
];

export default function Layout() {
  const [dark, setDark] = useState(true);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    document.documentElement.classList.toggle("light", !dark);
  }, [dark]);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-white/10 bg-ink-950/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-4 px-4 py-3">
          <div className="mr-2 animate-rise">
            <div className="font-display text-2xl font-bold tracking-tight text-white">
              Fraud <span className="text-mint-400">Guard</span>
            </div>
            <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-slate-400">
              Real-time risk · Bayesian · Explainable
            </div>
          </div>
          <nav className="flex flex-1 flex-wrap gap-1">
            {LINKS.map(([to, label]) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-1.5 text-sm transition ${
                    isActive ? "bg-mint-500/15 text-mint-400" : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
          <button className="btn-ghost" onClick={() => setDark((d) => !d)} type="button">
            {dark ? "Light" : "Dark"}
          </button>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
