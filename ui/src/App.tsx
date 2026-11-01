import { MotionConfig } from "framer-motion";
import { lazy } from "react";
import { Route, Routes } from "react-router";
import { Shell } from "./components/layout/Shell";
import { ApiProvider } from "./lib/provider";
import { useSettings } from "./lib/settings";

// Route-level code splitting keeps three.js and the chart library out of the initial bundle.
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Studio = lazy(() => import("./pages/Studio"));
const Runs = lazy(() => import("./pages/Runs"));
const RunDetailPage = lazy(() => import("./pages/RunDetail"));
const Harness = lazy(() => import("./pages/Harness"));
const Kpis = lazy(() => import("./pages/Kpis"));
const SettingsPage = lazy(() => import("./pages/Settings"));

function NotFound() {
  return (
    <div className="panel p-10 text-center">
      <h1 className="text-xl font-semibold">Page not found</h1>
      <p className="mt-2 text-fg-2">This route does not exist in the operator console.</p>
    </div>
  );
}

export function App() {
  const [settings] = useSettings();
  return (
    <MotionConfig reducedMotion={settings.reduceMotion ? "always" : "user"}>
      <ApiProvider>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<Dashboard />} />
            <Route path="studio" element={<Studio />} />
            <Route path="runs" element={<Runs />} />
            <Route path="runs/:id" element={<RunDetailPage />} />
            <Route path="harness" element={<Harness />} />
            <Route path="kpis" element={<Kpis />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </ApiProvider>
    </MotionConfig>
  );
}
