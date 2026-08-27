/**
 * Typed client for the two FastAPI backends.
 *
 * Every function below maps to a route that genuinely exists in
 * FINAL/CCTV/netra/api.py. Anything the console wants but the backend does
 * NOT provide is collected under `MISSING_ENDPOINTS` at the bottom of this
 * file as an explicit TODO - it is never faked.
 *
 * Two base URLs, because CCTV and DRONE run as two separate processes:
 *   NEXT_PUBLIC_CCTV_API   default http://localhost:8000
 *   NEXT_PUBLIC_DRONE_API  default http://localhost:8011
 */

import type {
  Backend,
  Camera,
  DashboardSnapshot,
  Health,
  Incident,
  IncidentStatus,
  Job,
  MetricsResponse,
  ProblemVideo,
  ReportFile,
  RoadGraphGeoJSON,
  RouteResponse,
  Summary,
} from "./types";

export const CCTV_API =
  process.env.NEXT_PUBLIC_CCTV_API?.replace(/\/+$/, "") || "http://localhost:8000";

export const DRONE_API =
  process.env.NEXT_PUBLIC_DRONE_API?.replace(/\/+$/, "") || "http://localhost:8011";

export function baseUrl(backend: Backend = "cctv"): string {
  return backend === "drone" ? DRONE_API : CCTV_API;
}

/** Absolute URL for a backend path - use for <img>/<video> src attributes. */
export function apiUrl(path: string, backend: Backend = "cctv"): string {
  return `${baseUrl(backend)}${path.startsWith("/") ? path : `/${path}`}`;
}

/* -------------------------------------------------------------------------
 * Transport
 * ---------------------------------------------------------------------- */

export class ApiError extends Error {
  readonly status: number;
  readonly backend: Backend;
  readonly path: string;

  constructor(message: string, status: number, backend: Backend, path: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.backend = backend;
    this.path = path;
  }

  /** True when the backend process itself is unreachable (not an HTTP error). */
  get isOffline(): boolean {
    return this.status === 0;
  }
}

interface RequestOptions {
  backend?: Backend;
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
  headers?: Record<string, string>;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { backend = "cctv", method = "GET", body, signal, headers = {} } = options;
  const url = apiUrl(path, backend);

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      signal,
      cache: "no-store",
      headers: body === undefined ? headers : { "content-type": "application/json", ...headers },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (cause) {
    // A dead backend process and a CORS rejection both land here. Status 0
    // lets the UI say "backend not reachable" instead of showing a fake zero.
    throw new ApiError(
      `${backend.toUpperCase()} backend not reachable at ${baseUrl(backend)}`,
      0,
      backend,
      path,
    );
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const text = await response.text();
      if (text) {
        try {
          const parsed = JSON.parse(text) as { detail?: unknown };
          detail = typeof parsed.detail === "string" ? parsed.detail : text;
        } catch {
          detail = text;
        }
      }
    } catch {
      /* keep the status-line fallback */
    }
    throw new ApiError(detail, response.status, backend, path);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const qs = search.toString();
  return qs ? `?${qs}` : "";
}

/* -------------------------------------------------------------------------
 * Health, summary, dashboard      GET /api/health /api/summary /api/dashboard
 * ---------------------------------------------------------------------- */

export const getHealth = (backend: Backend = "cctv", signal?: AbortSignal) =>
  request<Health>("/api/health", { backend, signal });

export const getSummary = (backend: Backend = "cctv", signal?: AbortSignal) =>
  request<Summary>("/api/summary", { backend, signal });

/** One consistent snapshot: summary + incidents + cameras in a single scan. */
export const getDashboard = (backend: Backend = "cctv", signal?: AbortSignal) =>
  request<DashboardSnapshot>("/api/dashboard", { backend, signal });

/* -------------------------------------------------------------------------
 * Cameras
 * ---------------------------------------------------------------------- */

export const getCameras = (backend: Backend = "cctv", signal?: AbortSignal) =>
  request<Camera[]>("/api/cameras", { backend, signal });

export const getCamera = (cameraId: string, backend: Backend = "cctv", signal?: AbortSignal) =>
  request<Camera>(`/api/cameras/${encodeURIComponent(cameraId)}`, { backend, signal });

/**
 * First frame of the camera's source - the calibration canvas.
 * Returns a URL because this is an image endpoint, not JSON.
 */
export const cameraFrameUrl = (cameraId: string, backend: Backend = "cctv") =>
  apiUrl(`/api/cameras/${encodeURIComponent(cameraId)}/frame`, backend);

/** POST /api/cameras/{id}/calibration - body is {camera_id, payload}. */
export const saveCalibration = (
  cameraId: string,
  payload: Record<string, unknown>,
  backend: Backend = "cctv",
) =>
  request<{ saved: string; corridors: number; zones: number }>(
    `/api/cameras/${encodeURIComponent(cameraId)}/calibration`,
    { backend, method: "POST", body: { camera_id: cameraId, payload } },
  );

/* -------------------------------------------------------------------------
 * Incidents
 * ---------------------------------------------------------------------- */

export interface IncidentQuery {
  camera_id?: string;
  status?: IncidentStatus | "";
  event_type?: string;
  /** Backend caps this at 2000 (Query(200, le=2000)). */
  limit?: number;
  latest_only?: boolean;
}

export const getIncidents = (
  params: IncidentQuery = {},
  backend: Backend = "cctv",
  signal?: AbortSignal,
) => request<Incident[]>(`/api/incidents${query({ ...params })}`, { backend, signal });

export const getIncident = (id: number, backend: Backend = "cctv", signal?: AbortSignal) =>
  request<Incident>(`/api/incidents/${id}`, { backend, signal });

/** POST /api/incidents/{id}/verify - moves status to "verified". */
export const verifyIncident = (id: number, actor = "operator", backend: Backend = "cctv") =>
  request<unknown>(`/api/incidents/${id}/verify`, {
    backend,
    method: "POST",
    body: { status: "verified", actor },
  });

/**
 * POST /api/incidents/{id}/reject.
 * `reason` MUST be one of REJECTION_REASONS or the backend returns 400.
 * A rejection is labelled training data, not a delete.
 */
export const rejectIncident = (
  id: number,
  reason: string,
  comment = "",
  actor = "operator",
  backend: Backend = "cctv",
) =>
  request<unknown>(`/api/incidents/${id}/reject`, {
    backend,
    method: "POST",
    body: { reason, actor, comment },
  });

/** POST /api/incidents/{id}/assign - the backend also sets status="assigned". */
export const assignIncident = (
  id: number,
  owner: string,
  team = "",
  backend: Backend = "cctv",
) =>
  request<unknown>(`/api/incidents/${id}/assign`, {
    backend,
    method: "POST",
    body: { owner, team },
  });

/** PATCH /api/incidents/{id}/status - the generic workflow transition. */
export const setIncidentStatus = (
  id: number,
  status: IncidentStatus,
  opts: { actor?: string; reason?: string; comment?: string } = {},
  backend: Backend = "cctv",
) =>
  request<unknown>(`/api/incidents/${id}/status`, {
    backend,
    method: "PATCH",
    body: {
      status,
      actor: opts.actor ?? "operator",
      reason: opts.reason ?? "",
      comment: opts.comment ?? "",
    },
  });

/**
 * GET /api/incidents/{id}/evidence/{name}
 *
 * NOTE the route is nested under the incident - there is no flat
 * /api/evidence/{name}. `name` is a bare filename from the incident's
 * evidence manifest (evidence.annotated_frame / evidence.clip).
 */
export const evidenceUrl = (incidentId: number, name: string, backend: Backend = "cctv") =>
  apiUrl(`/api/incidents/${incidentId}/evidence/${encodeURIComponent(name)}`, backend);

/* -------------------------------------------------------------------------
 * Road graph + routing
 * ---------------------------------------------------------------------- */

export const getGraph = (backend: Backend = "cctv", signal?: AbortSignal) =>
  request<RoadGraphGeoJSON>("/api/graph", { backend, signal });

/**
 * GET /api/route?source=&target=
 * Both are road-graph NODE ids. The backend has no "nearest node to this
 * incident" helper, so the caller must choose them - see MISSING_ENDPOINTS.
 */
export const getRoute = (
  source: string,
  target: string,
  backend: Backend = "cctv",
  signal?: AbortSignal,
) => request<RouteResponse>(`/api/route${query({ source, target })}`, { backend, signal });

/** Operator confirms the carriageway is blocked; the edge leaves the graph. */
export const closeRoad = (incidentId: number, backend: Backend = "cctv") =>
  request<{ closed_edge: string; closed: string[] }>(
    `/api/incidents/${incidentId}/close_road`,
    { backend, method: "POST" },
  );

export const reopenRoad = (incidentId: number, backend: Backend = "cctv") =>
  request<{ reopened_edge: string | null }>(`/api/incidents/${incidentId}/reopen_road`, {
    backend,
    method: "POST",
  });

/* -------------------------------------------------------------------------
 * Metrics / reports
 * ---------------------------------------------------------------------- */

export const getMetrics = (runId?: string, backend: Backend = "cctv", signal?: AbortSignal) =>
  request<MetricsResponse>(`/api/metrics${query({ run_id: runId })}`, { backend, signal });

export const getReports = (backend: Backend = "cctv", signal?: AbortSignal) =>
  request<ReportFile[]>("/api/reports", { backend, signal });

/* -------------------------------------------------------------------------
 * Upload jobs
 * ---------------------------------------------------------------------- */

export const getJobs = (backend: Backend = "cctv", signal?: AbortSignal) =>
  request<Job[]>("/api/jobs", { backend, signal });

export const getJob = (jobId: string, backend: Backend = "cctv", signal?: AbortSignal) =>
  request<Job>(`/api/jobs/${encodeURIComponent(jobId)}`, { backend, signal });

export const jobVideoUrl = (jobId: string, backend: Backend = "cctv") =>
  apiUrl(`/api/jobs/${encodeURIComponent(jobId)}/video`, backend);

export const jobPosterUrl = (jobId: string, backend: Backend = "cctv") =>
  apiUrl(`/api/jobs/${encodeURIComponent(jobId)}/poster`, backend);

/**
 * POST /api/jobs
 *
 * The backend streams the RAW request body to disk and reads the original
 * filename from the `x-filename` header - it is deliberately NOT a multipart
 * form. XMLHttpRequest is used instead of fetch() purely because fetch gives
 * no upload-progress events, and the job tray needs them.
 */
export function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void,
  backend: Backend = "cctv",
): { promise: Promise<Job>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<Job>((resolve, reject) => {
    xhr.open("POST", apiUrl("/api/jobs", backend));
    xhr.setRequestHeader("x-filename", encodeURIComponent(file.name));
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as Job);
        } catch (cause) {
          reject(new ApiError("upload succeeded but the response was not JSON", xhr.status, backend, "/api/jobs"));
        }
      } else {
        let detail = `${xhr.status} ${xhr.statusText}`;
        try {
          const parsed = JSON.parse(xhr.responseText) as { detail?: unknown };
          if (typeof parsed.detail === "string") detail = parsed.detail;
        } catch {
          /* keep the status line */
        }
        reject(new ApiError(detail, xhr.status, backend, "/api/jobs"));
      }
    };
    xhr.onerror = () =>
      reject(new ApiError(`${backend.toUpperCase()} backend not reachable`, 0, backend, "/api/jobs"));
    xhr.onabort = () => reject(new ApiError("upload cancelled", 0, backend, "/api/jobs"));
    xhr.send(file);
  });
  return { promise, abort: () => xhr.abort() };
}

/* -------------------------------------------------------------------------
 * ProblemSet review videos
 * ---------------------------------------------------------------------- */

/**
 * Reads CCTV/ProblemSet/Results_release_candidate/summary.json server-side.
 * That directory is absent from the FINAL/CCTV port, so this returns [] until
 * the results are placed there. Callers render an empty state, not mock rows.
 */
export const getProblemVideos = (backend: Backend = "cctv", signal?: AbortSignal) =>
  request<ProblemVideo[]>("/api/problem-videos", { backend, signal });

export const problemVideoUrl = (stem: string, backend: Backend = "cctv") =>
  apiUrl(`/api/problem-videos/${encodeURIComponent(stem)}/video`, backend);

export const problemPosterUrl = (stem: string, backend: Backend = "cctv") =>
  apiUrl(`/api/problem-videos/${encodeURIComponent(stem)}/poster`, backend);

/* -------------------------------------------------------------------------
 * TODO - endpoints this console wants that the backend does NOT expose
 * -------------------------------------------------------------------------
 *
 * Each of these renders an explicit, labelled empty state in the UI. None of
 * them is mocked. Adding any of them is a backend change, not a UI change.
 */
export const MISSING_ENDPOINTS = [
  {
    wanted: "GET /api/incidents/{id}/route",
    why:
      "Per-incident diversion geometry. /api/route exists but needs two road-graph " +
      "NODE ids chosen by hand; nothing maps an incident's road_edge_id to a " +
      "source/target pair. The map page therefore asks the operator to pick the " +
      "two nodes instead of guessing them.",
    ui: "app/map/page.tsx - manual source/target selectors",
  },
  {
    wanted: "GET /api/incidents/{id}/access-route",
    why:
      "Simulated responder access route (dashed blue). No responder-facility " +
      "table and no access-route solver exist in netra/. Drawing one would be " +
      "fabricated geometry, so the layer stays empty and says so.",
    ui: "app/map/page.tsx - access-route legend row marked NOT AVAILABLE",
  },
  {
    wanted: "GET /api/incidents/{id}  on the DRONE backend",
    why:
      "DRONE/scripts/api.py now runs a real FastAPI process (health, status, " +
      "process, results, results/{name}, dashboard) and GET /api/dashboard " +
      "synthesises Incident-shaped rows from whatever is in results/ - but " +
      "there is no per-incident route. Those synthetic ids are also not " +
      "stable identifiers (they are renumbered 1, 2, 3... from disk on every " +
      "request), so a drone incident card links to /incidents/{id}?backend=drone " +
      "and gets an honest 'not found' from the DRONE process rather than " +
      "silently rendering a same-numbered CCTV incident's real data.",
    ui: "app/incidents/[id]/page.tsx via components/IncidentCard.tsx (backend=\"drone\")",
  },
  {
    wanted: "GET /api/cameras, /api/graph, /api/route  on the DRONE backend",
    why:
      "DRONE has no camera catalogue, scene calibration, or road graph of its " +
      "own - it is a hover-dispatch verifier for an already-confirmed CCTV " +
      "incident, not a second independent camera network. /calibrate and " +
      "/map stay CCTV-only; the drone page shows only what GET /api/health, " +
      "/api/dashboard and /api/results genuinely return.",
    ui: "app/drone/page.tsx",
  },
  {
    wanted: "Live camera stream (MJPEG/WebRTC)",
    why:
      "The pipeline is offline/batch: it analyses files and writes evidence " +
      "clips. There is no live-frame endpoint beyond the single calibration " +
      "still at /api/cameras/{id}/frame. The jury demo is seed-data replay.",
    ui: "app/cctv/page.tsx - shows the calibration still + recorded evidence",
  },
  {
    wanted: "Auto-calibration trigger over HTTP",
    why:
      "scripts/autocalibrate.py is a CLI. The API only accepts an already-built " +
      "scene model via POST /api/cameras/{id}/calibration. The calibrate page " +
      "reviews and confirms geometry; it cannot kick off a fresh auto-calibration.",
    ui: "app/calibrate/page.tsx - CLI command shown instead of a button",
  },
] as const;
