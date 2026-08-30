import { describe, expect, it } from "vitest";

import type { BoundaryLabelV1, BoundaryLabelV2 } from "../../game/core/types";
import {
  validateBoundaryLabel,
  validateBoundaryLabelV2,
  validatePlaytestRunV3,
} from "./schemas";

function validPlaytestRunV3(): Record<string, unknown> {
  return {
    version: 3,
    runId: "30000000-0000-4000-8000-000000000001",
    title: "v3 fixture",
    generatedAt: "2026-08-30T00:00:00Z",
    workerVersion: "fixture",
    audio: {
      game: { path: "audio/game.flac", sha256: "a".repeat(64) },
      noDrums: null,
      keys: null,
    },
    charts: [4, 6, 7].flatMap((keyMode) =>
      ["EASY", "NORMAL", "HARD", "EXPERT"].map((difficulty) => ({
        path: `charts/${keyMode}k-${difficulty.toLowerCase()}.chart.json`,
        sha256: "b".repeat(64),
        keyMode,
        difficulty,
        provenance: "PRIMARY",
        familyAssignmentKind: "ORIGINAL",
        sourceDifficulty: null,
        familyResolutionState: "RESOLVED",
        familyResolutionReasons: [],
        playabilityTier: null,
        coverageSummary: null,
      })),
    ),
    missingCharts: [],
    keysoundManifestPath: null,
    generationReport: {
      path: "generation-report.json",
      sha256: "c".repeat(64),
    },
    outcome: {
      execution: "SUCCEEDED",
      completeness: "COMPLETE",
      quality: "PASS",
      failureCategory: "NONE",
      publishableStrict: true,
    },
    strictBlockers: [],
    publication: {
      policyVersion: "PUBLICATION_POLICY_V2",
      decision: "ALLOW_PRODUCTION",
      reasonCodes: [],
    },
  };
}

function validBoundaryLabel(): BoundaryLabelV1 {
  return {
    version: 1,
    labelId: "10000000-0000-4000-8000-000000000001",
    createdAt: "2026-08-10T00:00:00Z",
    reviewerId: "reviewer-01",
    run: {
      runId: "20000000-0000-4000-8000-000000000001",
      title: "fixture song",
      songVersionId: "30000000-0000-4000-8000-000000000001",
      gameAudioAssetId: "40000000-0000-4000-8000-000000000001",
    },
    audio: {
      sha256: "a".repeat(64),
      durationMs: 8000,
    },
    generationReport: {
      path: "generation-report.json",
      sha256: "b".repeat(64),
    },
    group: {
      groupId: "recording-fixture-01",
      relation: "EXACT_RECORDING",
      confirmed: true,
    },
    automaticEvidence: {
      availability: "UNAVAILABLE",
      unavailableReason: "musicBounds must be an object",
      evaluationVersion: null,
      policyState: null,
      policyConfidence: null,
      enforcementMode: null,
      observationSha256: null,
      lastDetectedOnsetMs: null,
      lastActiveRmsEndMs: null,
      lastEvidenceMs: null,
      provisionalMaxNoteStartMs: null,
      provisionalReleaseEndMs: null,
      effectiveMaxNoteStartMs: null,
      effectiveReleaseEndMs: null,
    },
    annotation: {
      lastMeaningfulAttack: { earliestMs: 7500, latestMs: 7550 },
      lastAcceptableRelease: { earliestMs: 7700, latestMs: 7800 },
      provisionalBoundaryVerdict: "NOT_AVAILABLE",
      tailCharacters: ["FADE_OR_REVERB"],
      confidence: "MEDIUM",
      comment: "",
    },
  };
}

function validBoundaryLabelV2(): BoundaryLabelV2 {
  const v1 = validBoundaryLabel();
  return {
    ...v1,
    version: 2,
    annotation: {
      lastPlayableAttack: { earliestMs: 7300, latestMs: 7400 },
      primaryContentEnd: { earliestMs: 7500, latestMs: 7550 },
      acceptableReleaseEnd: { earliestMs: 7700, latestMs: 7800 },
      provisionalBoundaryVerdict: "NOT_AVAILABLE",
      tailCharacters: ["FADE_OR_REVERB"],
      confidence: "MEDIUM",
      comment: "",
    },
  };
}

describe("boundary-label-v1 contract", () => {
  it("accepts a structurally valid label", () => {
    expect(() => validateBoundaryLabel(validBoundaryLabel())).not.toThrow();
  });

  it("rejects fields that are outside the exported Python schema", () => {
    const document = { ...validBoundaryLabel(), unexpected: true };

    expect(() => validateBoundaryLabel(document, "fixture.json")).toThrow(
      /fixture\.json schema validation failed.*additional properties/i,
    );
  });
});

describe("boundary-label-v2 contract", () => {
  it("accepts all three explicitly named human intervals", () => {
    expect(() => validateBoundaryLabelV2(validBoundaryLabelV2())).not.toThrow();
  });

  it("does not accept a v1 document as v2", () => {
    expect(() => validateBoundaryLabelV2(validBoundaryLabel())).toThrow(/version/i);
  });
});

describe("playtest-run-v3 contract", () => {
  it("accepts factual chart trust fields without chart-level authority", () => {
    expect(() => validatePlaytestRunV3(validPlaytestRunV3())).not.toThrow();
  });

  it.each(["productionEligible", "distributionTier"])(
    "rejects removed chart authority field %s",
    (field) => {
      const document = validPlaytestRunV3();
      const charts = document.charts as Array<Record<string, unknown>>;
      charts[0][field] = field === "productionEligible" ? true : "PRODUCTION_CANDIDATE";

      expect(() => validatePlaytestRunV3(document)).toThrow(/additional properties/i);
    },
  );

  it.each([
    "provenance",
    "familyAssignmentKind",
    "sourceDifficulty",
    "familyResolutionState",
    "familyResolutionReasons",
    "playabilityTier",
    "coverageSummary",
  ])("rejects an omitted explicit chart trust fact %s", (field) => {
    const document = validPlaytestRunV3();
    const charts = document.charts as Array<Record<string, unknown>>;
    delete charts[0][field];

    expect(() => validatePlaytestRunV3(document)).toThrow(/required/i);
  });
});
