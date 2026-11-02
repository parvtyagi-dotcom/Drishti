import type { LucideIcon } from "lucide-react";

type Accent = "default" | "red" | "amber" | "green" | "blue";

const accentClass: Record<Accent, string> = {
  default: "text-text-primary",
  red: "text-accent-red",
  amber: "text-accent-amber",
  green: "text-accent-green",
  blue: "text-accent-blue",
};

export function MetricCard({
  label,
  value,
  sub,
  accent = "default",
  icon: Icon,
}: {
  label: string;
  value: string | number;
  sub?: string;
  accent?: Accent;
  icon?: LucideIcon;
}) {
  const isEmpty = value === "" || value == null;
  return (
    <div className="surface rounded-xl border border-border-1 px-5 py-4">
      <div className="flex items-center gap-1.5 text-[12px] font-medium text-text-secondary">
        {Icon && <Icon size={13} className="text-text-tertiary" />}
        {label}
      </div>
      <div className={`mt-2.5 tabular-nums text-[30px] font-semibold leading-none tracking-[-0.03em] ${isEmpty ? "text-text-disabled" : accentClass[accent]}`}>
        {isEmpty ? "—" : value}
      </div>
      {sub && <div className="mt-1 text-[11px] text-text-tertiary">{sub}</div>}
    </div>
  );
}
