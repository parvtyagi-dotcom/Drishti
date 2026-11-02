import {
  AlertTriangle, Car, Clock, Crosshair, Eye, Footprints, Gauge, MapPinOff,
  Megaphone, Navigation, Shovel, Plane, RadioTower, ShieldAlert, Sparkles, TriangleAlert, Users, type LucideIcon,
} from "lucide-react";

export type Severity = "red" | "amber" | "green" | "blue" | "neutral";

export interface EventMeta {
  label: string;
  severity: Severity;
  icon: LucideIcon;
}

// Every event type the backend actually emits (core/events.py, zone/state.py,
// behavior/state.py, anpr/pipeline.py, orchestrator.py's TARGET_ZONE_INTRUSION
// / LOW_LIGHT). Unknown types fall back to a neutral generic entry rather
// than crashing the UI.
const REGISTRY: Record<string, EventMeta> = {
  TARGET_CONFIRMED: { label: "Target Confirmed", severity: "red", icon: Crosshair },
  TARGET_REACQUIRED: { label: "Target Reacquired", severity: "red", icon: Crosshair },
  TARGET_LOST: { label: "Target Lost", severity: "amber", icon: Eye },
  VEHICLE_DETECTED: { label: "Vehicle Detected", severity: "blue", icon: Car },
  PLATE_DETECTED: { label: "Plate Detected", severity: "blue", icon: Car },
  PLATE_READ: { label: "Plate Read", severity: "blue", icon: Car },
  PLATE_CONFIRMED: { label: "Plate Confirmed", severity: "green", icon: Car },
  TARGET_VEHICLE_FOUND: { label: "Target Vehicle Found", severity: "red", icon: Car },
  TARGET_VEHICLE_REACQUIRED: { label: "Target Vehicle Reacquired", severity: "red", icon: Car },
  TARGET_VEHICLE_LOST: { label: "Target Vehicle Lost", severity: "amber", icon: Car },
  // The warning that precedes an intrusion: amber, because it has not
  // happened yet. The crossing itself is red and sounds the siren.
  ZONE_APPROACH: { label: "Approaching Zone", severity: "amber", icon: Navigation },
  ZONE_ENTRY: { label: "Zone Intrusion", severity: "red", icon: ShieldAlert },
  ZONE_EXIT: { label: "Zone Exit", severity: "green", icon: MapPinOff },
  LONG_DWELL: { label: "Long Dwell", severity: "amber", icon: Clock },
  TARGET_ZONE_INTRUSION: { label: "Target Zone Intrusion", severity: "red", icon: ShieldAlert },
  LOITERING: { label: "Loitering", severity: "amber", icon: Footprints },
  GROUP_GATHERING: { label: "Group Gathering", severity: "amber", icon: Users },
  SUDDEN_DIRECTION_CHANGE: { label: "Sudden Direction Change", severity: "amber", icon: TriangleAlert },
  ABNORMAL_SPEED: { label: "Abnormal Speed", severity: "amber", icon: Gauge },
  REPEATED_BACK_AND_FORTH: { label: "Repeated Pacing", severity: "amber", icon: Footprints },
  // Raised either by the confirmed target OR by anyone behaving oddly inside a
  // restricted zone, so the label stays neutral - the event's own reason line
  // says which of the two actually happened.
  TARGET_BEHAVIOR_ALERT: { label: "Behaviour Escalation", severity: "red", icon: ShieldAlert },
  FOG_DETECTED: { label: "Fog Detected", severity: "blue", icon: Sparkles },
  LOW_LIGHT: { label: "Low-Light Detected", severity: "blue", icon: Sparkles },
  // Sentinel (backend/live/tamper.py and the camera health monitor).
  CAMERA_TAMPERED: { label: "Camera Tampering", severity: "red", icon: ShieldAlert },
  CAMERA_SABOTAGE_SUSPECTED: { label: "Possible Sabotage", severity: "red", icon: ShieldAlert },
  CAMERA_VIEW_RESTORED: { label: "Camera View Restored", severity: "green", icon: Eye },
  // Sensor fusion (backend/live/sensors.py): a sensor trip checked on camera.
  SENSOR_TRIGGERED: { label: "Sensor Tripped", severity: "amber", icon: RadioTower },
  SENSOR_ALARM_CONFIRMED: { label: "Sensor Alarm Confirmed", severity: "red", icon: RadioTower },
  SENSOR_ALARM_UNVERIFIED: { label: "Sensor Alarm Unchecked", severity: "red", icon: RadioTower },
  SENSOR_ALARM_UNCONFIRMED: { label: "Sensor Trip, Nothing Seen", severity: "neutral", icon: RadioTower },
  TALKDOWN_ISSUED: { label: "Spoken Warning Played", severity: "blue", icon: Megaphone },
  DIGGING_SUSPECTED: { label: "Possible Digging / Tunnel", severity: "red", icon: Shovel },
  DRONE_SUSPECTED: { label: "Possible Drone", severity: "red", icon: Plane },
  // AI analyst records (backend/agent). Always informational: the analyst
  // explains and proposes, it never raises an alarm.
  AGENT_BRIEF: { label: "AI Brief", severity: "blue", icon: Sparkles },
  AGENT_ANSWER: { label: "AI Copilot Answer", severity: "blue", icon: Sparkles },
  AGENT_PROPOSAL: { label: "AI Proposal", severity: "blue", icon: Sparkles },
  AGENT_APPROVED: { label: "AI Proposal Approved", severity: "green", icon: Sparkles },
  AGENT_REJECTED: { label: "AI Proposal Rejected", severity: "neutral", icon: Sparkles },
};

const FALLBACK: EventMeta = { label: "Event", severity: "neutral", icon: AlertTriangle };

export function eventMeta(type: string): EventMeta {
  return REGISTRY[type] ?? { ...FALLBACK, label: type };
}
