export type GpuInfo = {
  name?: string | null;
  memoryUsed?: string | null;
  memoryTotal?: string | null;
  utilization?: string | null;
};

export type GpuProcess = {
  pid: string;
  name: string;
  memory: string;
};

export type QueueInfo = {
  queueDepth?: number;
  activeJobId?: string | null;
  busy?: boolean;
};

export type Status = {
  loadState?: string;
  profile?: string;
  model?: string;
  apiState?: string;
  gpu?: GpuInfo;
  gpuProcesses?: GpuProcess[];
  queue?: QueueInfo;
};

export type Aspect = {
  id: string;
  width: number;
  height: number;
};

export type StylePreset = {
  id: string;
  label: string;
};

export type Capabilities = {
  samplers: string[];
  schedulers: string[];
  aspects: Aspect[];
  stylePresets?: StylePreset[];
};

export type Operation = {
  id: string;
  label: string;
  description: string;
  available: boolean;
  missingBundles?: string[];
  requiredBundles?: string[];
  promptRequired?: boolean;
  requiresMask?: boolean;
  requiresImage?: boolean;
  referenceSlots?: string[];
  defaultWidth?: number;
  defaultHeight?: number;
  defaultSteps?: number;
  defaultCfg?: number;
  defaultShift?: number;
};

export type JobError = {
  message?: string;
  code?: string;
};

export type Job = {
  job_id: string;
  operation: string;
  status: string;
  phase?: string;
  progress?: number;
  queue_position?: number;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  error?: JobError | null;
  asset_ids?: string[];
  seed?: number | null;
  inputs?: Record<string, unknown>;
};

export type BundleFile = {
  name: string;
  state?: string;
  optional?: boolean;
};

export type Bundle = {
  id: string;
  status?: string;
  dest?: string;
  files_ok?: number;
  files_required?: number;
  size_human?: string;
  files?: BundleFile[];
};

export type CatalogModel = {
  id: string;
  runtime?: string;
  state?: string;
};

export type ModelsPayload = {
  bundles?: Bundle[];
  models?: CatalogModel[];
};

export type DownloadsPayload = {
  status?: string;
  running?: boolean;
  currentBundle?: string;
  currentFile?: string;
  logTail?: string[];
  progress?: {
    percentComplete?: number;
    bundlesComplete?: number;
    bundlesTotal?: number;
  };
};
