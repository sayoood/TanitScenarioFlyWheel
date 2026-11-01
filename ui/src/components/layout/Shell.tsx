import { motion } from "framer-motion";
import {
  Activity,
  Boxes,
  Brain,
  Clapperboard,
  FlaskConical,
  LayoutDashboard,
  Settings as SettingsIcon,
  Wifi,
  WifiOff,
} from "lucide-react";
import { Suspense, useEffect, type ReactNode } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router";
import clsx from "clsx";
import { useHealth } from "../../lib/hooks";
import { useApi } from "../../lib/provider";
import { useSettings } from "../../lib/settings";
import { LogoMark } from "../ui/Logo";
import { Skeleton } from "../ui/primitives";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/studio", label: "Studio", icon: Clapperboard },
  { to: "/runs", label: "Runs", icon: Boxes },
  { to: "/harness", label: "Harness", icon: Brain },
  { to: "/kpis", label: "KPIs", icon: Activity },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

export function ModeBadge({ compact }: { compact?: boolean }) {
  const { kind, probe, reprobe } = useApi();
  const [settings] = useSettings();
  const live = kind === "live";
  const reachable = probe.ok;
  const label = live ? "LIVE" : "MOCK";
  const detail = live
    ? reachable
      ? `Connected to ${settings.baseUrl} (${probe.ms ?? "?"} ms)`
      : `Live mode forced, but ${settings.baseUrl} is not answering`
    : settings.mode === "mock"
      ? "Mock mode forced in Settings"
      : `Thor API not reachable (${probe.error ?? "probing"}). Showing mock data. Retrying automatically.`;
  const bad = live && !reachable;
  return (
    <button
      onClick={reprobe}
      title={`${detail}. Click to re-probe.`}
      aria-label={`${label} data source. ${detail}`}
      className={clsx(
        "inline-flex h-8 items-center justify-center gap-2 rounded-full border px-3 text-[11.5px] font-bold tracking-[0.12em] transition-colors",
        compact && "w-9 !px-0 xl:w-auto xl:!px-3",
        live && !bad && "border-ok/40 bg-ok/10 text-ok",
        bad && "border-bad/40 bg-bad/10 text-bad",
        !live && "border-warn/45 bg-warn/10 text-warn",
      )}
    >
      {live ? <Wifi size={14} /> : <WifiOff size={14} />}
      <span className={clsx(compact && "hidden xl:inline")}>{label}</span>
      {!compact && live && reachable && probe.ms !== undefined && <span className="mono font-medium tracking-normal opacity-80">{probe.ms} ms</span>}
      {!compact && !live && <span className="hidden font-medium tracking-normal opacity-80 2xl:inline">demo data</span>}
    </button>
  );
}

function SchedulerChip() {
  const health = useHealth();
  const s = health.data?.services.scheduler;
  if (!s) return <Skeleton className="h-8 w-40" />;
  return (
    <Link
      to={s.running ? `/runs/${s.running}` : "/runs"}
      className="hidden h-8 items-center gap-2 rounded-full border border-line bg-bg-1/70 px-3 text-[12px] text-fg-1 hover:border-line-2 md:inline-flex"
      title="GPU scheduler: running job and queue length"
    >
      <span className={clsx("live-dot", s.running ? "text-accent-2" : "text-fg-3")} style={s.running ? undefined : { animation: "none" }} />
      {s.running ? (
        <span>
          Running <span className="mono text-fg">{s.running}</span>
        </span>
      ) : (
        <span>Scheduler idle</span>
      )}
      <span className="text-fg-3">|</span>
      <span>
        queue <span className="num text-fg">{s.queue}</span>
      </span>
    </Link>
  );
}

function Sidebar() {
  return (
    <aside className="sticky top-0 z-20 flex h-screen w-[72px] shrink-0 flex-col border-r border-line bg-bg-1/70 backdrop-blur-xl xl:w-[248px]">
      <Link to="/" className="flex h-[68px] items-center gap-3 px-[18px] xl:px-5" aria-label="Tanit Scenario Flywheel home">
        <LogoMark size={34} />
        <div className="hidden leading-tight xl:block">
          <div className="font-display text-[17px] font-semibold tracking-tight text-fg">Tanit</div>
          <div className="text-[11.5px] tracking-wide text-fg-2">Scenario Flywheel</div>
        </div>
      </Link>
      <nav className="mt-2 flex flex-1 flex-col gap-1 px-3" aria-label="Main">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.end} className="nav-link" title={n.label}>
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-[10px] border border-line-2 bg-[linear-gradient(90deg,rgb(139_124_246/0.22),rgb(34_211_238/0.06))]"
                    transition={{ type: "spring", stiffness: 420, damping: 36 }}
                  />
                )}
                <n.icon size={19} className={clsx("relative shrink-0", isActive ? "text-accent-2" : "text-fg-2")} />
                <span className="relative hidden xl:inline">{n.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="p-3">
        <div className="hidden rounded-xl border border-line bg-bg/60 p-3 xl:block">
          <div className="eyebrow mb-2 flex items-center gap-1.5">
            <FlaskConical size={12} /> Flywheel v0.1
          </div>
          <p className="m-0 text-[12px] leading-relaxed text-fg-2">Five use cases, one typed IR, hard verifiers as reward.</p>
        </div>
        <div className="mt-3 flex justify-center xl:justify-start">
          <ModeBadge compact />
        </div>
      </div>
    </aside>
  );
}

function PageFallback() {
  return (
    <div className="space-y-4" aria-busy>
      <Skeleton className="h-9 w-72" />
      <Skeleton className="h-64 w-full" />
      <div className="grid grid-cols-3 gap-4">
        <Skeleton className="h-40" />
        <Skeleton className="h-40" />
        <Skeleton className="h-40" />
      </div>
    </div>
  );
}

export function Shell(): ReactNode {
  const loc = useLocation();
  const [settings] = useSettings();
  useEffect(() => {
    document.documentElement.classList.toggle("reduce-motion", settings.reduceMotion);
  }, [settings.reduceMotion]);
  const section = loc.pathname.split("/")[1] ?? "";
  return (
    <div className="relative z-10 flex min-h-screen">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex h-[68px] items-center justify-between gap-4 border-b border-line bg-bg/70 px-6 backdrop-blur-xl xl:px-8">
          <div className="eyebrow">{NAV.find((n) => (n.to === "/" ? section === "" : section === n.to.slice(1)))?.label ?? "Run"}</div>
          <div className="flex items-center gap-3">
            <SchedulerChip />
            <ModeBadge />
            <Link to="/studio" className="btn btn-primary btn-sm">
              <Clapperboard size={15} /> New run
            </Link>
          </div>
        </header>
        <motion.main
          key={section}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.28, ease: [0.2, 0.7, 0.2, 1] }}
          className="mx-auto w-full max-w-[1680px] flex-1 px-6 py-7 xl:px-8"
        >
          <Suspense fallback={<PageFallback />}>
            <Outlet />
          </Suspense>
        </motion.main>
      </div>
    </div>
  );
}
