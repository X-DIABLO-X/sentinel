import type { ReactNode } from "react";

export default function KpiCard({
  label,
  value,
  hint,
  tone = "neutral",
  href,
}: {
  label: string;
  /** Pass the string "—" (not 0) when the number is genuinely unknown. */
  value: ReactNode;
  hint?: ReactNode;
  tone?: "neutral" | "low" | "medium" | "high" | "critical" | "accent";
  href?: string;
}) {
  const accentClass = {
    neutral: "text-ink-0",
    accent: "text-accent",
    low: "text-sev-low",
    medium: "text-sev-medium",
    high: "text-sev-high",
    critical: "text-sev-critical",
  }[tone];

  const body = (
    <div className="panel h-full px-4 py-3 transition-colors hover:border-line/80">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">{label}</div>
      <div className={`mt-1.5 font-mono text-[26px] leading-none ${accentClass}`}>{value}</div>
      {hint ? <div className="mt-1.5 text-[11.5px] leading-snug text-ink-3">{hint}</div> : null}
    </div>
  );

  if (!href) return body;
  return (
    <a href={href} className="block h-full">
      {body}
    </a>
  );
}
