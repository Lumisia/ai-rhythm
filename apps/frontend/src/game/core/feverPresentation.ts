export const FEVER_GLOW_FADE_MS = 400;

export function feverGlowAlpha(
  active: boolean,
  endedAtMs: number | null,
  songTimeMs: number,
  reduceMotion: boolean,
): number {
  if (active) return 1;
  if (reduceMotion || endedAtMs === null) return 0;
  const elapsedMs = Math.max(0, songTimeMs - endedAtMs);
  return Math.max(0, 1 - elapsedMs / FEVER_GLOW_FADE_MS);
}

export function feverGaugeVisible(enabled: boolean): boolean {
  return enabled;
}
