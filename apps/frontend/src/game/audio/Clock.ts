export interface Clock {
  songTimeMs(): number;
  performanceToSongTimeMs(performanceMs: number): number;
}
