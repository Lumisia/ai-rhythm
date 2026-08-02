import judgmentJson from "@contracts/judgment/judgment-v1.json";

import type {
  JudgmentConfig,
  JudgmentName,
  JudgmentPreset,
  JudgmentWindows,
} from "./types";

const orderedWindows: Array<keyof JudgmentWindows> = ["PERFECT", "GREAT", "GOOD", "BAD"];

export function loadJudgmentConfig(): JudgmentConfig {
  return structuredClone(judgmentJson) as JudgmentConfig;
}

export function classifyError(
  errorMs: number,
  preset: JudgmentPreset,
  config: JudgmentConfig = loadJudgmentConfig(),
): JudgmentName {
  const absoluteErrorMs = Math.abs(errorMs);
  const windows = config.presets[preset];

  for (const judgment of orderedWindows) {
    if (absoluteErrorMs <= windows[judgment]) {
      return judgment;
    }
  }

  return "MISS";
}
