import type { LaneSemantic } from "./types";

export interface LaneGeometry {
  index: number;
  semantic: LaneSemantic;
  x: number;
  width: number;
  color: number;
  backgroundColor: number;
}

function weightFor(semantic: LaneSemantic): number {
  if (semantic.startsWith("SIDE_")) return 1.24;
  if (semantic === "CENTER") return 1.18;
  return 1;
}

function colorFor(semantic: LaneSemantic): number {
  if (semantic.startsWith("SIDE_")) return 0x78cbb8;
  if (semantic === "CENTER") return 0xe27268;
  if (semantic === "MAIN_2" || semantic === "MAIN_3") return 0x7f9fc2;
  return 0xdce3e8;
}

export function layoutLanes(
  totalWidth: number,
  semantics: readonly LaneSemantic[],
): readonly LaneGeometry[] {
  if (!Number.isFinite(totalWidth) || totalWidth <= 0) {
    throw new Error("lane layout width must be positive");
  }
  if (semantics.length === 0) throw new Error("at least one lane semantic is required");

  const weights = semantics.map(weightFor);
  const unit = totalWidth / weights.reduce((sum, weight) => sum + weight, 0);
  let x = 0;
  return semantics.map((semantic, index) => {
    const width = index === semantics.length - 1 ? totalWidth - x : weights[index] * unit;
    const lane = {
      index,
      semantic,
      x,
      width,
      color: colorFor(semantic),
      backgroundColor: index % 2 === 0 ? 0x181e29 : 0x1d2330,
    };
    x += width;
    return lane;
  });
}
