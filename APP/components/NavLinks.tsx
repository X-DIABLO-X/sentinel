"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS: { href: string; label: string }[] = [
  { href: "/", label: "Overview" },
  { href: "/incidents", label: "Incidents" },
  { href: "/map", label: "Map" },
  { href: "/cctv", label: "CCTV" },
  { href: "/drone", label: "Drone" },
];

export default function NavLinks() {
  const pathname = usePathname();

  return (
    <nav className="flex flex-wrap items-center gap-1">
      {LINKS.map((link) => {
        const active =
          link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
        return (
          <Link
            key={link.href}
            href={link.href}
            className={[
              "rounded-md px-2.5 py-1 text-[13px] transition-colors",
              active
                ? "bg-accent/15 text-accent"
                : "text-ink-2 hover:bg-panel-2 hover:text-ink-0",
            ].join(" ")}
          >
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
