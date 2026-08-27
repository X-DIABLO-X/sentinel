import type { ReactNode } from "react";

/**
 * The console's single honest fallback.
 *
 * Every place the backend has no data - or has no endpoint at all - renders
 * one of these with a real reason. Nothing in this console ever draws
 * plausible-looking placeholder data.
 */
export default function EmptyState({
  title,
  detail,
  tone = "neutral",
  children,
}: {
  title: string;
  detail?: ReactNode;
  /**
   * "stub" is reserved for capabilities that do not exist yet.
   * "ok" is for a capability that used to be a stub and now has real,
   * measured output behind it — same dashed-panel honesty pattern, positive
   * framing instead of a red "NOT IMPLEMENTED" prefix.
   */
  tone?: "neutral" | "warn" | "stub" | "ok";
  children?: ReactNode;
}) {
  const toneClass =
    tone === "stub"
      ? "border-sev-critical/40 bg-sev-critical/[0.06]"
      : tone === "warn"
        ? "border-sev-medium/40 bg-sev-medium/[0.06]"
        : tone === "ok"
          ? "border-sev-low/40 bg-sev-low/[0.06]"
          : "border-line bg-panel-2/40";

  const titleClass =
    tone === "stub"
      ? "text-sev-critical"
      : tone === "warn"
        ? "text-sev-medium"
        : tone === "ok"
          ? "text-sev-low"
          : "text-ink-1";

  return (
    <div className={`rounded-lg border border-dashed px-5 py-6 ${toneClass}`}>
      <p className={`text-[14px] font-semibold ${titleClass}`}>
        {tone === "stub" ? "NOT IMPLEMENTED — " : ""}
        {title}
      </p>
      {detail ? (
        <div className="mt-1.5 max-w-2xl text-[12.5px] leading-relaxed text-ink-2">{detail}</div>
      ) : null}
      {children ? <div className="mt-3">{children}</div> : null}
    </div>
  );
}
