import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "./client";
import type { Capabilities, DownloadsPayload, Job, ModelsPayload, Operation, Status } from "./types";

const POLL_MS = 2000;

type Store = {
  status: Status | null;
  models: ModelsPayload | null;
  operations: Operation[];
  capabilities: Capabilities | null;
  jobs: Job[];
  downloads: DownloadsPayload | null;
  error: string | null;
  busy: boolean;
  refresh: () => Promise<void>;
  run: (fn: () => Promise<void>) => Promise<void>;
};

const StoreContext = createContext<Store | null>(null);

export function StoreProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status | null>(null);
  const [models, setModels] = useState<ModelsPayload | null>(null);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [downloads, setDownloads] = useState<DownloadsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [nextStatus, nextModels, nextDownloads, nextOps, nextJobs] = await Promise.all([
        api<Status>("/v1/status"),
        api<ModelsPayload>("/v1/models").catch(() => null),
        api<DownloadsPayload>("/v1/downloads").catch(() => null),
        api<{ operations?: Operation[]; capabilities?: Capabilities }>("/v1/image/operations").catch(() => ({ operations: [] as Operation[], capabilities: undefined })),
        api<{ data?: Job[] }>("/v1/jobs?limit=40").catch(() => ({ data: [] })),
      ]);
      setStatus(nextStatus);
      if (nextModels) setModels(nextModels);
      if (nextDownloads) setDownloads(nextDownloads);
      setOperations(nextOps.operations || []);
      if (nextOps.capabilities) setCapabilities(nextOps.capabilities);
      setJobs(nextJobs.data || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const run = useCallback(async (fn: () => Promise<void>) => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await fn();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [busy, refresh]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      if (!busy) void refresh();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [busy, refresh]);

  const value = useMemo(
    () => ({ status, models, operations, capabilities, jobs, downloads, error, busy, refresh, run }),
    [status, models, operations, capabilities, jobs, downloads, error, busy, refresh, run],
  );

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStore(): Store {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStore must be used inside StoreProvider");
  return ctx;
}
