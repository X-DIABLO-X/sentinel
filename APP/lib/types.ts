/**
 * Types mirrored from the real CCTV backend.
 *
 * Every shape below was derived by reading, not guessing:
 *   - FINAL/CCTV/netra/api.py      (routes, response envelopes)
 *   - FINAL/CCTV/netra/db.py       (incidents table + IncidentStore._hydrate)
 *   - FINAL/CCTV/netra/scene.py    (SceneModel.to_dict / Corridor / Zone)
 *   - FINAL/CCTV/netra/location.py (describe_location, RoadGraph.route/to_geojson)
 *   - FINAL/CCTV/netra/jobs.py     (VideoJobManager job dict)
 *   - FINAL/CCTV/netra/severity.py (BANDS, W_IMPACT, W_TOTAL, DISCLAIMER)
 *
 * Fields are widely optional because SQLite columns are nullable and because
 * enrich_incidents() in api.py adds keys only for some event types.
 */

/* -------------------------------------------------------------------------
 * Severity
 * ---------------------------------------------------------------------- */

/**
 * The console's severity ladder.
 *
 * HONESTY NOTE: the CCTV backend's severity bands are
 *   BANDS = ((0.35, "Low"), (0.65, "Medium"), (1.01, "High"))
 * so `severity_label` from /api/incidents is only ever "Low" | "Medium" |
 * "High". CRITICAL exists in this union because the challenge brief specifies
 * a four-tier ladder and the drone escalation path is expected to produce it,
 * but NOTHING in the shipped CCTV pipeline emits CRITICAL today. The UI must
 * never manufacture one - `normaliseSeverity()` only upgrades a label that the
 * backend actually sent.
 */
export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export const SEVERITIES: Severity[] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];

/** Severity tiers the CCTV backend can actually produce (severity.py BANDS). */
export const CCTV_EMITTED_SEVERITIES: Severity[] = ["LOW", "MEDIUM", "HIGH"];

/** Map a raw backend `severity_label` ("Low"/"Medium"/"High") onto our tier. */
export function normaliseSeverity(raw: string | null | undefined): Severity | null {
  if (!raw) return null;
  const key = String(raw).trim().toUpperCase();
  return (SEVERITIES as string[]).includes(key) ? (key as Severity) : null;
}

/* -------------------------------------------------------------------------
 * Operator workflow (db.py STATUSES / REJECTION_REASONS - exact strings)
 * ---------------------------------------------------------------------- */

export type IncidentStatus =
  | "detected"
  | "verified"
  | "assigned"
  | "responding"
  | "resolved"
  | "closed"
  | "rejected";

/** Ordered forward path. "rejected" is a terminal branch, not a step. */
export const WORKFLOW: IncidentStatus[] = [
  "detected",
  "verified",
  "assigned",
  "responding",
  "resolved",
  "closed",
];

export const ALL_STATUSES: IncidentStatus[] = [...WORKFLOW, "rejected"];

/** db.py REJECTION_REASONS - the backend 400s on anything not in this list. */
export const REJECTION_REASONS = [
  "tracking error",
  "legal turn",
  "camera shake",
  "false detection",
  "normal signal queue",
  "legitimate stop",
  "duplicate",
  "other",
] as const;

export type RejectionReason = (typeof REJECTION_REASONS)[number];

/** Event types observed in netra/events/*.py and the shipped UI tabs. */
export type EventType =
  | "queue"
  | "wrong_way"
  | "blockage"
  | "collision_candidate"
  | (string & {});

/** api.py enrich_incidents() derives this from the camera_id prefix. */
export type SourceKind =
  | "problem_set"
  | "upload"
  | "legacy_accident"
  | "traffic";

/* -------------------------------------------------------------------------
 * Incident
 * ---------------------------------------------------------------------- */

export interface IncidentLocation {
  camera_id?: string;
  camera_name?: string;
  zone?: string;
  road_name?: string;
  road_edge_id?: string | null;
  corridor_id?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  /** location.py PRECISION_GROUND | PRECISION_CAMERA | PRECISION_ZONE */
  precision?: string;
  note?: string;
}

/** evidence.py manifest: always `dir`, optionally an annotated frame + clip. */
export interface EvidenceManifest {
  dir?: string;
  annotated_frame?: string;
  clip?: string;
  clip_span_s?: [number, number];
  [key: string]: unknown;
}

/** severity.py component scores, stored as severity_json. */
export interface SeverityParts {
  flow_loss?: number;
  obstruction?: number;
  extent?: number;
  duration?: number;
  risk?: number;
  impact_subscore?: number;
  [key: string]: unknown;
}

export interface StatusHistoryEntry {
  id?: number;
  incident_id?: number;
  old_status?: string;
  new_status?: string;
  actor?: string;
  reason?: string;
  comment?: string;
  changed_at?: number;
}

export interface Assignment {
  id?: number;
  incident_id?: number;
  owner?: string;
  team?: string;
  assigned_at?: number;
  acknowledged_at?: number | null;
  completed_at?: number | null;
}

export interface Incident {
  id: number;
  run_id?: string | null;
  camera_id: string;
  corridor_id?: string | null;
  event_type: EventType;
  label?: string | null;

  /** Seconds into the analysed video, not wall clock. */
  started_t?: number | null;
  detected_t?: number | null;
  ended_t?: number | null;
  duration?: number | null;
  detection_delay?: number | null;
  onset_method?: string | null;
  onset_recovered_s?: number | null;

  confidence?: number | null;
  severity?: number | null;
  /** Raw backend label: "Low" | "Medium" | "High". Use normaliseSeverity(). */
  severity_label?: string | null;
  priority?: number | null;

  status: IncidentStatus;
  needs_verification?: boolean;
  recommended_action?: string | null;
  explanation?: string | null;

  track_ids?: number[];
  triggers?: Record<string, unknown>;
  severity_parts?: SeverityParts;
  location?: IncidentLocation;
  evidence?: EvidenceManifest;

  wall_clock?: number | null;
  /**
   * CCTV sends Unix epoch seconds (db.py, SQLite time.time()). DRONE's
   * synthetic incidents (scripts/api.py _synthetic_incidents) instead carry
   * an ISO-8601 string straight from each result file's `generated_at`.
   * format.ts clock()/relative() accept both.
   */
  created_at?: number | string | null;
  updated_at?: number | string | null;

  /** Added by api.py enrich_incidents(), not a DB column. */
  source_kind?: SourceKind;

  /** Only present on GET /api/incidents/{id} (not on the list route). */
  history?: StatusHistoryEntry[];
  assignments?: Assignment[];
}

/* -------------------------------------------------------------------------
 * Camera / scene model
 * ---------------------------------------------------------------------- */

export interface Corridor {
  id: string;
  name?: string;
  /** Image-space pixel polygon, NOT lat/lon. */
  polygon: [number, number][];
  /** Unit vector in image space: the legal travel direction. */
  direction: [number, number];
  lanes?: number;
  solid_boundary_with?: string[];
  baseline_speed_px?: number | null;
}

export interface Zone {
  id: string;
  kind?: string;
  name?: string;
  polygon: [number, number][];
}

export interface Camera {
  camera_id: string;
  name?: string;
  source?: string;
  zone?: string;
  road_name?: string;
  road_edge_id?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  /** [width, height] in pixels, or null when unknown. */
  frame_size?: [number, number] | null;
  analysis_fps?: number;
  corridors?: Corridor[];
  zones?: Zone[];
  homography?: { matrix?: number[][]; [key: string]: unknown };
  notes?: string;

  /** Added by api.py cameras_from(), not part of SceneModel.to_dict(). */
  incidents?: number;
  known_to_db?: boolean;
}

/* -------------------------------------------------------------------------
 * Summary / dashboard
 * ---------------------------------------------------------------------- */

export interface Summary {
  total_incidents: number;
  open_incidents: number;
  awaiting_verification: number;
  by_type: Record<string, number>;
  /** Keyed by the RAW label ("Low"/"Medium"/"High"). */
  by_severity: Record<string, number>;
  by_status: Record<string, number>;
  cameras: number;
  severity_disclaimer: string;
}

export interface DashboardSnapshot {
  summary: Summary;
  incidents: Incident[];
  cameras: Camera[];
}

/**
 * GET /api/health. The two backends answer with genuinely different shapes -
 * this type is the union of both, with every backend-specific field optional.
 * CCTV (netra/api.py health()) sends cameras/road_graph_edges/severity_model/
 * statuses/rejection_reasons. DRONE (scripts/api.py health()) sends
 * service/port/mode/detector_finetuned/placeholder_detector/gmc_enabled/
 * telemetry_available/road_plane_calibrated instead - it has no camera
 * catalogue or road graph of its own, so callers must not assume those
 * fields exist just because `backend` was "drone".
 */
export interface Health {
  status: string;

  /** CCTV only. */
  cameras?: number;
  road_graph_edges?: number;
  severity_model?: {
    impact_weights: Record<string, number>;
    total_weights: Record<string, number>;
    bands: { below: number; label: string }[];
    disclaimer: string;
  };
  statuses?: string[];
  rejection_reasons?: string[];

  /** DRONE only. */
  service?: string;
  port?: number;
  mode?: string;
  detector_finetuned?: boolean;
  placeholder_detector?: boolean;
  gmc_enabled?: boolean;
  telemetry_available?: boolean;
  road_plane_calibrated?: boolean;
}

/* -------------------------------------------------------------------------
 * Road graph + routing (location.py RoadGraph)
 * ---------------------------------------------------------------------- */

export interface GraphFeature {
  type: "Feature";
  properties: { id: string; name: string; penalty: number; closed: boolean };
  geometry: {
    type: "LineString";
    /** GeoJSON order: [lon, lat]. Leaflet wants [lat, lon] - flip before use. */
    coordinates: [number, number][];
  };
}

export interface RoadGraphGeoJSON {
  type: "FeatureCollection";
  features: GraphFeature[];
}

/**
 * One leg of a computed path. NOTE `coords` is ALREADY [lat, lon]
 * (see RoadGraph._describe), unlike the GeoJSON above.
 */
export interface RouteGeometry {
  nodes: string[];
  edges: { from: string; to: string; id: string; name: string }[];
  edge_ids: string[];
  cost_s: number;
  length_m: number;
  coords: [number, number][];
}

/** GET /api/route?source=&target= */
export interface RouteResponse {
  source: string;
  target: string;
  /** Clean-network path, ignoring incident penalties. */
  baseline: RouteGeometry | null;
  /** Path under current incident penalties + confirmed closures. */
  current?: RouteGeometry | null;
  diverted?: boolean;
  extra_seconds?: number;
  extra_metres?: number;
  error?: string;
  note?: string;
}

/* -------------------------------------------------------------------------
 * ProblemSet review videos (GET /api/problem-videos)
 * ---------------------------------------------------------------------- */

/** One detected event's summary fields, as returned inside a ProblemVideo row. */
export interface AnalysisEvent {
  event_type?: string;
  severity?: number;
  severity_label?: string;
  confidence?: number;
  started_t?: number;
  detected_t?: number;
  duration?: number;
  explanation?: string;
  [key: string]: unknown;
}

/**
 * Rows come straight out of
 * CCTV/ProblemSet/Results_release_candidate/summary.json.
 * That directory is NOT present in the FINAL/CCTV port, so this route
 * returns [] until the ProblemSet results are placed there. The UI renders an
 * explicit empty state rather than inventing rows.
 */
export interface ProblemVideo {
  file?: string;
  camera_id?: string;
  annotated_video?: string;
  events_total?: number;
  events_by_type?: Record<string, number>;
  events?: AnalysisEvent[];
  [key: string]: unknown;
}

/* -------------------------------------------------------------------------
 * Metrics / reports
 * ---------------------------------------------------------------------- */

export interface ModelRun {
  run_id: string;
  detector?: string;
  detector_backend?: string;
  device?: string;
  imgsz?: number;
  tracker?: string;
  engine_version?: string;
  threshold_hash?: string;
  started_at?: number;
  finished_at?: number | null;
  notes?: string;
}

export interface Metric {
  id?: number;
  run_id?: string;
  camera_id?: string;
  key: string;
  value: number;
  detail?: Record<string, unknown>;
  created_at?: number;
}

export interface MetricsResponse {
  runs: ModelRun[];
  metrics: Metric[];
}

export interface ReportFile {
  name: string;
  data: unknown;
}

/* -------------------------------------------------------------------------
 * Drone backend
 * ---------------------------------------------------------------------- */

/**
 * Which backend a page is talking to. The two FastAPI processes are separate
 * (CCTV :8000, DRONE :8011) by design - see the architecture note in README.
 */
export type Backend = "cctv" | "drone";

/**
 * Real, not stubbed: dronefreak/visdrone-yolov8x (VisDrone2019-DET
 * fine-tune) + native Ultralytics BoT-SORT (ReID) is what actually produced
 * the numbers below, on a real 16-clip batch of genuine DJI nadir footage —
 * see DRONE/config/drone_config.yaml and DRONE/demo/README.md for the full
 * provenance. Flip `detectorFineTuned` back to false only if this console is
 * ever pointed at a build that reverts to the generic-COCO placeholder.
 */
export const DRONE_STATUS = {
  detectorFineTuned: true,
  finetuneTarget: "VisDrone",
  note:
    "Detector: dronefreak/visdrone-yolov8x (VisDrone2019-DET fine-tune, not " +
    "the earlier generic-COCO placeholder). Tracker: native Ultralytics " +
    "BoT-SORT with appearance ReID, chosen over this project's original " +
    "hand-rolled tracker because nadir footage's frequent top-down occlusion " +
    "is a materially worse case for a motion/IoU-only associator. First real " +
    "batch (16 clips, genuine hovering-DJI footage): 717 tracks, 10 queue " +
    "candidates with full evidence, 0 blockage events, GMC health 1.000 on " +
    "every clip. Two open, honestly-tracked limitations: the blockage " +
    "stationary-speed threshold was set before real footage existed and " +
    "under-fires against real box jitter (traced to one specific missed " +
    "parked car, not yet recalibrated against the same batch it would then " +
    "be scored on), and the class-width km/h estimate is unreliable for " +
    "vehicles clipped at the frame edge. No calibrated metric speed is " +
    "claimed anywhere in this console — every speed is px/s plus a labelled " +
    "estimate.",
} as const;
