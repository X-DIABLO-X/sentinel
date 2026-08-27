import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import NavLinks from "@/components/NavLinks";
import BackendStatus from "@/components/BackendStatus";

export const metadata: Metadata = {
  title: "ELCIA Operator Console",
  description:
    "Traffic incident operator console - CCTV backbone with drone escalation. " +
    "ELCIA Smart City Drone-AI Challenge 2026.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap"
        />
      </head>
      <body className="min-h-screen font-sans antialiased">
        <div className="flex min-h-screen flex-col">
          <header className="sticky top-0 z-[500] flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-line bg-panel-1/95 px-5 py-2.5 backdrop-blur">
            <Link href="/" className="flex items-baseline gap-2.5">
              <span className="text-[17px] font-bold tracking-tight text-ink-0">NETRA</span>
              <span className="hidden text-[12px] text-ink-3 sm:inline">
                traffic incident operator console
              </span>
            </Link>
            <NavLinks />
            <BackendStatus />
          </header>

          <main className="flex-1">{children}</main>

          <footer className="border-t border-line px-5 py-3 text-[11px] leading-relaxed text-ink-3">
            Severity shown here is <strong className="text-ink-2">traffic-impact</strong> severity
            computed from observable variables (flow loss, obstruction, extent, duration,
            exposure). It is <strong className="text-ink-2">not</strong> injury or casualty
            severity, which cannot be inferred from RGB video. Camera coordinates are the{" "}
            <em>camera&rsquo;s</em> position, not the vehicle&rsquo;s.
          </footer>
        </div>
      </body>
    </html>
  );
}
