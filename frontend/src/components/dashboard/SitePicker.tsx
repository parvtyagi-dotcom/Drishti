import { ChevronDown, MapPinned } from "lucide-react";
import type { SiteSummary } from "../../api/map";

/** A compact select over the configured sites; hidden when there is only one. */
export function SitePicker({ sites, value, onChange }: { sites: SiteSummary[]; value: string | undefined; onChange: (id: string) => void }) {
  if (sites.length < 2) return null;
  return (
    <label className="relative inline-flex items-center">
      <MapPinned size={13} className="pointer-events-none absolute left-2.5 z-10 text-text-tertiary" />
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label="Site"
        className="max-w-[220px] appearance-none truncate rounded-lg border border-border-2 bg-bg-1/85 py-1.5 pl-8 pr-7 text-[12px] text-text-primary backdrop-blur hover:border-border-strong focus:outline-none"
      >
        {sites.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name}
          </option>
        ))}
      </select>
      <ChevronDown size={13} className="pointer-events-none absolute right-2 z-10 text-text-tertiary" />
    </label>
  );
}
