import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark operator-console palette. Named by role, not by hue, so the
        // whole console can be re-themed from one place.
        ink: {
          0: "#f2f5f8",
          1: "#c9d2dc",
          2: "#8b95a5",
          3: "#5d6775",
        },
        panel: {
          0: "#0b0e13",
          1: "#11151c",
          2: "#171c25",
          3: "#1f2632",
        },
        line: "#232b38",
        accent: "#4da3ff",
        // Severity tiers. CRITICAL is defined here for completeness, but see
        // the note in lib/types.ts: the CCTV backend's severity bands only
        // emit Low / Medium / High today.
        sev: {
          low: "#3fb950",
          medium: "#d29922",
          high: "#f0883e",
          critical: "#e5484d",
        },
        // Map route conventions, kept as tokens so the legend and the
        // polylines can never drift apart.
        diversion: "#e5484d", // solid red   - diversion for other traffic
        access: "#4da3ff", // dashed blue - simulated responder access
      },
      fontFamily: {
        sans: ["'Source Sans 3'", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
