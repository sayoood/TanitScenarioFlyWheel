import type { StageName, UseCaseId, VerifierLevel } from "./types";

export interface UcMeta {
  id: UseCaseId;
  short: string;
  name: string;
  blurb: string;
  inputLabel: string;
  /** CSS variable holding the accent colour of this use case. */
  color: string;
}

export const USE_CASES: UcMeta[] = [
  { id: "uc1", short: "UC1", name: "Text to scenario", blurb: "Natural-language query lifted into TSIR.", inputLabel: "Query", color: "var(--color-uc1)" },
  { id: "uc2", short: "UC2", name: "Dashcam video", blurb: "Reconstruct a drive and replay it as a scenario.", inputLabel: "Video", color: "var(--color-uc2)" },
  { id: "uc3", short: "UC3", name: "Incident to positive twin", blurb: "Counterfactual twin of an IMDAD NEGATIVE record.", inputLabel: "Incident", color: "var(--color-uc3)" },
  { id: "uc4", short: "UC4", name: "Sensor snippet augmentation", blurb: "Semantic variants of a logged snippet.", inputLabel: "Snippet", color: "var(--color-uc4)" },
  { id: "uc5", short: "UC5", name: "Negative snippet to corrective data", blurb: "Positive twin plus variations for a failure.", inputLabel: "Snippet", color: "var(--color-uc5)" },
];

export const ucMeta = (id: string): UcMeta =>
  USE_CASES.find((u) => u.id === id) ?? { id: "uc1", short: id.toUpperCase(), name: id, blurb: "", inputLabel: "Input", color: "var(--color-fg-1)" };

export const STAGE_LABEL: Record<StageName, string> = {
  analyse: "Analyse",
  propose: "Propose",
  verify: "Verify",
  repair: "Repair",
  document: "Document",
  simulate: "Simulate",
  wsm: "WSM render",
  cosmos_transfer: "Cosmos Transfer",
  augment: "Augment",
  closed_loop: "Closed loop",
};

export const STAGE_HINT: Record<StageName, string> = {
  analyse: "Parse intent, retrieve lessons, reconstruct inputs",
  propose: "Lifter agent emits TSIR (constrained decoding)",
  verify: "Verifier ladder L0 to L5",
  repair: "Diagnostics fed back to the agent",
  document: "Compile xodr / xosc / osc2 and package",
  simulate: "TanitSim 2D, metrics",
  wsm: "RDS-HQ export and WSM render, LiDAR range maps",
  cosmos_transfer: "Cosmos 3 multi-camera and LiDAR generation",
  augment: "Semantic augmentation strategies, re-verified",
  closed_loop: "Policy under test in the loop",
};

export const LEVEL_INFO: Record<VerifierLevel, { title: string; tool: string }> = {
  L0: { title: "Schema", tool: "TSIR schema + referential integrity" },
  L1: { title: "Standards", tool: "OpenDRIVE / OSC 1.3 XSD, ASAM QC" },
  L2: { title: "Cross-check", tool: "TanitSim reads the generated xodr" },
  L3: { title: "Oracle", tool: "Simulate and check TSIR assertions" },
  L4: { title: "Intent", tool: "VLM judges query vs. simulation summary" },
  L5: { title: "Plausibility", tool: "Kinematic limits, jerk, TTC priors" },
};

export const AUGMENT_STRATEGIES = [
  { id: "weather", label: "Weather", description: "Rain, fog, cloud cover via prompt and TSIR environment" },
  { id: "time_of_day", label: "Time of day", description: "Dawn, dusk and night lighting" },
  { id: "behavior", label: "Behaviour", description: "Coverage-guided parameter sweep, criticality steering" },
  { id: "object", label: "Objects", description: "Category, size and colour of surrounding actors" },
  { id: "map", label: "Map", description: "Lane width, curvature and lane count" },
];

export const POLICIES = [
  { id: "refe_sft4", label: "REFe SFT-4", description: "Learned policy, NAVSIM virtual cameras (960x512)" },
  { id: "const_vel", label: "Constant velocity", description: "Holds the initial speed" },
  { id: "accel_decel", label: "Accelerate / decelerate", description: "Speeds up, then brakes to standstill" },
];

export const RESOLUTIONS = ["480p", "720p", "1080p"];

export const UC1_EXAMPLES = [
  "Generate data from a scenario where the ego vehicle changes the lane to the left to turn left in an intersection with traffic light at day in good weather conditions with vehicles surrounding",
  "Ego drives 50 km/h on a two-lane urban road at dusk in light rain; a delivery van parked on the right and a pedestrian steps out from behind it, ego must brake",
  "Highway, three lanes, fog: a truck on the right lane cuts in 20 m ahead of the ego at 80 km/h, ego has to decelerate without leaving its lane",
];

export const UC_EXAMPLE_INPUTS: Record<Exclude<UseCaseId, "uc1">, string[]> = {
  uc2: ["youtube:qdAI_quNcqw", "youtube:amuldGGgNc4", "youtube:dToIIqqvSt0"],
  uc3: ["IMDAD-2025-NEG-0037", "IMDAD-2023-NEG-0033", "IMDAD-2026-NEG-0014"],
  uc4: ["pai_av:b3681088-1163-447b-8a66-d8277ffec933", "pai_av:0cd05bb5-5459-465f-8b59-08300162b292"],
  uc5: ["pai_av:bcc52cd1-9507-425e-a5a6-5f8a795ed525", "pai_av:f33a4bdb-871c-4e67-8ac5-73cfaf97cb1d"],
};
