import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { QueryClient, QueryClientProvider, useQueryClient } from "@tanstack/react-query";
import type { ApiKind, FlywheelApi } from "./api";
import { pushEvent } from "./events";
import { LiveApi, probe } from "./live";
import { getMockApi } from "./mock/backend";
import { useSettings } from "./settings";

export interface ProbeInfo {
  ok: boolean;
  ms?: number;
  error?: string;
  at?: number;
  checking: boolean;
}

interface ApiCtx {
  api: FlywheelApi;
  kind: ApiKind;
  probe: ProbeInfo;
  eventsConnected: boolean;
  reprobe: () => void;
}

const Ctx = createContext<ApiCtx | null>(null);

export function useApi(): ApiCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error("ApiProvider missing");
  return v;
}

function Inner({ children }: { children: ReactNode }) {
  const [settings] = useSettings();
  const qc = useQueryClient();
  const [info, setInfo] = useState<ProbeInfo>({ ok: false, checking: true });
  const [eventsConnected, setEventsConnected] = useState(false);
  const fails = useRef(0);
  const [nonce, setNonce] = useState(0);

  // ---- probe the live API (auto and live modes)
  useEffect(() => {
    if (settings.mode === "mock") {
      setInfo((i) => ({ ...i, ok: false, checking: false, error: "Mock mode forced" }));
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let misses = 0;
    let lastAt = 0;
    const run = async () => {
      setInfo((i) => ({ ...i, checking: true }));
      const r = await probe(settings.baseUrl);
      if (cancelled) return;
      lastAt = Date.now();
      if (r.ok) {
        fails.current = 0;
        misses = 0;
        setInfo({ ok: true, ms: r.ms, at: lastAt, checking: false });
      } else {
        fails.current += 1;
        misses += 1;
        // tolerate a single blip once connected
        setInfo((i) => (i.ok && fails.current < 2 ? { ...i, checking: false } : { ok: false, error: r.error, at: lastAt, checking: false }));
      }
      // back off while the Thor is away (browsers log every failed request, so keep it sparse)
      const delay = r.ok ? 15000 : Math.min(15000 * 2 ** Math.min(misses - 1, 3), 120000);
      timer = setTimeout(run, delay);
    };
    void run();
    const onVisible = () => {
      if (document.visibilityState === "visible" && Date.now() - lastAt > 20000) {
        if (timer) clearTimeout(timer);
        void run();
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [settings.mode, settings.baseUrl, nonce]);

  const kind: ApiKind = settings.mode === "mock" ? "mock" : settings.mode === "live" ? "live" : info.ok ? "live" : "mock";

  const api = useMemo<FlywheelApi>(() => (kind === "live" ? new LiveApi(settings.baseUrl) : getMockApi()), [kind, settings.baseUrl]);

  // ---- live event stream
  useEffect(() => {
    setEventsConnected(false);
    const off = api.subscribe(
      (e) => {
        pushEvent(e);
        if (e.type === "run") {
          void qc.invalidateQueries({ queryKey: [api.kind, api.base, "runs"] });
          if (e.run_id) void qc.invalidateQueries({ queryKey: [api.kind, api.base, "run", e.run_id] });
        } else if (e.type === "stage" && e.run_id) {
          void qc.invalidateQueries({ queryKey: [api.kind, api.base, "run", e.run_id] });
        } else if (e.type === "health") {
          void qc.invalidateQueries({ queryKey: [api.kind, api.base, "health"] });
        }
      },
      setEventsConnected,
    );
    return off;
  }, [api, qc]);

  const reprobe = useCallback(() => setNonce((n) => n + 1), []);
  const value = useMemo<ApiCtx>(() => ({ api, kind, probe: info, eventsConnected, reprobe }), [api, kind, info, eventsConnected, reprobe]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function ApiProvider({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 1000 } },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <Inner>{children}</Inner>
    </QueryClientProvider>
  );
}
