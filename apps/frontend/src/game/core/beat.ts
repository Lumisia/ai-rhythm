export const DEFAULT_BEAT_MS = 500;

export function beatDurationMs(bpm: number | undefined): number {
  return bpm !== undefined && Number.isFinite(bpm) && bpm > 0
    ? 60_000 / bpm
    : DEFAULT_BEAT_MS;
}
