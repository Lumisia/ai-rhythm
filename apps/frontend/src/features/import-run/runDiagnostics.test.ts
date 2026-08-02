import { describe, expect, it } from "vitest";

import type { ChartDocument, Difficulty, KeyMode } from "../../game/core/types";
import { diagnoseCharts } from "./runDiagnostics";

function chart(
  difficulty: Difficulty,
  rating: number,
  overrides: Partial<ChartDocument> = {},
): ChartDocument {
  return {
    schemaVersion: 1,
    chartId: "10000000-0000-4000-8000-000000000001",
    songVersionId: "20000000-0000-4000-8000-000000000001",
    gameAudioAssetId: "30000000-0000-4000-8000-000000000001",
    audioSha256: "a".repeat(64),
    keyMode: 4 as KeyMode,
    difficulty,
    laneSemantics: ["MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4"],
    offsetMs: 0,
    durationMs: 8_000,
    bpmEvents: [{ timeMs: 0, bpm: 120 }],
    bpmSource: "BEAT_THIS",
    notes: [
      { id: 1, lane: 0, timeMs: 1_000, type: "TAP", durationMs: null },
      { id: 2, lane: 1, timeMs: 2_000, type: "HOLD", durationMs: 500 },
    ],
    autoPlayOnsets: [],
    metrics: {
      noteCount: 2,
      holdCount: 1,
      avgNps: 0.25,
      p95Nps: 1,
      peakNps: 1,
      chordRatio: 0,
      maxJack: 1,
      projectRating: rating,
      projectTier: difficulty,
      patternEntropy: 0,
      drumCoverage: 0,
      drumPrecision: 0,
      meanAbsErrMs: 0,
      sideNoteRatio: 0,
      sideHoldRatio: 0,
      movedNoteRatio: 0,
    },
    generator: {
      name: "fixture",
      version: "fixture",
      analysisVersion: "fixture",
      postprocessVersion: "fixture",
      seed: 7,
    },
    ...overrides,
  };
}

describe("diagnoseCharts", () => {
  it("reports note-boundary and metric inconsistencies as errors", () => {
    const invalid = chart("NORMAL", 2.6, {
      notes: [
        { id: 1, lane: 4, timeMs: 8_000, type: "TAP", durationMs: null },
        { id: 2, lane: 1, timeMs: 7_900, type: "HOLD", durationMs: 200 },
      ],
      metrics: { ...chart("NORMAL", 2.6).metrics, noteCount: 3, holdCount: 0 },
    });

    const report = diagnoseCharts([invalid]);

    expect(report.errors).toEqual(
      expect.arrayContaining([
        expect.stringMatching(/note 1.*time.*duration/i),
        expect.stringMatching(/note 1.*lane/i),
        expect.stringMatching(/note 2.*HOLD.*duration/i),
        expect.stringMatching(/noteCount/i),
        expect.stringMatching(/holdCount/i),
      ]),
    );
  });

  it("reports target, tier, and rating-order concerns as warnings", () => {
    const easy = chart("EASY", 3.0, {
      metrics: { ...chart("EASY", 3.0).metrics, projectRating: 3.0, projectTier: "NORMAL" },
    });
    const normal = chart("NORMAL", 2.4);

    const report = diagnoseCharts([easy, normal]);

    expect(report.errors).toEqual([]);
    expect(report.warnings).toEqual(
      expect.arrayContaining([
        "4K EASY: rating 3.000 exceeds target 1.500",
        "4K EASY: measured tier is NORMAL",
        "4K rating inversion: EASY 3.000 > NORMAL 2.400",
      ]),
    );
  });
});
