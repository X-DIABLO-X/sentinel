import type { Severity } from "@/lib/types";
import { normaliseSeverity } from "@/lib/types";

const STYLES: Record<Severity, string> = {
  LOW: "border-sev-low/50 bg-sev-low/10 text-sev-low",
  MEDIUM: "border-sev-medium/50 bg-sev-medium/10 text-sev-medium",
  HIGH: "border-sev-high/50 bg-sev-high/10 text-sev-high",
  CRITICAL: "border-sev-critical/60 bg-sev-critical/15 text-sev-critical",
};

/** Hex values, for the map pins and anywhere Tailwind classes cannot reach. */
export const SEVERITY_HEX: Record<Severity, string> = {
  LOW: "#3fb950",
  MEDIUM: "#d29922",
  HIGH: "#f0883e",
  CRITICAL: "#e5484d",
};

export const UNKNOWN_SEVERITY_HEX = "#5d6775";

export default function SeverityBadge({
  severity,
  score,
  className = "",
}: {
  /** Raw backend label ("Low"/"Medium"/"High") or an already-normalised tier. */
  severity: string | null | undefined;
  /** The 0..1 numeric severity, shown alongside the tier when present. */
  score?: number | null;
  className?: string;
}) {
  const tier = normaliseSeverity(severity);

  if (!tier) {
    return (
      <span
        className={`pill border-line text-ink-3 ${className}`}
        title="The backend recorded no severity label for this incident."
      >
        severity unscored
      </span>
    );
  }

  return (
    <span className={`pill ${STYLES[tier]} ${className}`}>
      {tier}
      {score !== null && score !== undefined && !Number.isNaN(score) ? (
        <span className="opacity-70">{Number(score).toFixed(2)}</span>
      ) : null}
    </span>
  );
}
