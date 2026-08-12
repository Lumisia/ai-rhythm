import { describe, expect, it } from "vitest";

import type { BoundaryAutomaticEvidence } from "../../game/core/types";
import type { BoundaryLabelContext } from "../import-run/importRun";
import {
  boundaryLabelFileName,
  boundaryLabelFileNameV2,
  buildBoundaryLabel,
  buildBoundaryLabelV2,
  serializeBoundaryLabel,
  serializeBoundaryLabelV2,
  type BoundaryLabelDraft,
  type BoundaryLabelDraftV2,
} from "./boundaryLabel";

const unavailableEvidence: BoundaryAutomaticEvidence = {
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
};

function context(): BoundaryLabelContext {
  return {
    available: true,
    unavailableReason: null,
    songVersionId: "10000000-0000-4000-8000-000000000001",
    gameAudioAssetId: "20000000-0000-4000-8000-000000000001",
    audioDurationMs: 10_000,
    generationReport: {
      path: "generation-report.json",
      sha256: "b".repeat(64),
    },
    automaticEvidence: unavailableEvidence,
  };
}

function draft(): BoundaryLabelDraft {
  return {
    reviewerId: "  reviewer-a  ",
    groupId: "  recording-group-a  ",
    relation: "UNKNOWN",
    groupConfirmed: false,
    lastMeaningfulAttack: { earliestMs: 8800, latestMs: 9000 },
    lastAcceptableRelease: { earliestMs: 9000, latestMs: 9500 },
    provisionalBoundaryVerdict: "NOT_AVAILABLE",
    tailCharacters: ["FADE_OR_REVERB"],
    confidence: "LOW",
    comment: "  uncertain decay  ",
  };
}

function draftV2(): BoundaryLabelDraftV2 {
  return {
    reviewerId: "  reviewer-a  ",
    groupId: "  recording-group-a  ",
    relation: "UNKNOWN",
    groupConfirmed: false,
    lastPlayableAttack: { earliestMs: 8200, latestMs: 8400 },
    primaryContentEnd: { earliestMs: 8800, latestMs: 9000 },
    acceptableReleaseEnd: { earliestMs: 9200, latestMs: 9500 },
    provisionalBoundaryVerdict: "NOT_AVAILABLE",
    tailCharacters: ["FADE_OR_REVERB"],
    confidence: "LOW",
    comment: "  three distinct meanings  ",
  };
}

const identity = {
  runId: "40000000-0000-4000-8000-000000000001",
  title: "Fixture Song",
  audioSha256: "a".repeat(64),
};

const environment = {
  createUuid: () => "50000000-0000-4000-8000-000000000001",
  now: () => new Date("2026-08-10T00:00:00.000Z"),
};

describe("boundary label builder", () => {
  it("builds a deterministic label and trims human text", () => {
    const label = buildBoundaryLabel(draft(), context(), identity, environment);

    expect(label).toMatchObject({
      version: 1,
      labelId: "50000000-0000-4000-8000-000000000001",
      createdAt: "2026-08-10T00:00:00.000Z",
      reviewerId: "reviewer-a",
      run: {
        runId: identity.runId,
        title: identity.title,
        songVersionId: "10000000-0000-4000-8000-000000000001",
        gameAudioAssetId: "20000000-0000-4000-8000-000000000001",
      },
      audio: { sha256: identity.audioSha256, durationMs: 10_000 },
      generationReport: context().generationReport,
      group: {
        groupId: "recording-group-a",
        relation: "UNKNOWN",
        confirmed: false,
      },
      automaticEvidence: unavailableEvidence,
      annotation: {
        provisionalBoundaryVerdict: "NOT_AVAILABLE",
        comment: "uncertain decay",
      },
    });
  });

  it("rejects a context without a bound generation report", () => {
    const invalid = { ...context(), generationReport: null };

    expect(() => buildBoundaryLabel(draft(), invalid, identity, environment)).toThrow(
      /generation report SHA-256 binding is required/i,
    );
  });

  it("rejects a reversed uncertainty interval", () => {
    const invalid = draft();
    invalid.lastMeaningfulAttack = { earliestMs: 9001, latestMs: 9000 };

    expect(() => buildBoundaryLabel(invalid, context(), identity, environment)).toThrow(
      /last meaningful attack.*earliestMs.*latestMs/i,
    );
  });

  it("rejects unavailable evidence with a comparable verdict", () => {
    const invalid = draft();
    invalid.provisionalBoundaryVerdict = "ACCEPTABLE";

    expect(() => buildBoundaryLabel(invalid, context(), identity, environment)).toThrow(
      /unavailable automatic evidence requires NOT_AVAILABLE verdict/i,
    );
  });

  it("rejects fractional milliseconds instead of rounding them", () => {
    const invalid = draft();
    invalid.lastMeaningfulAttack.latestMs = 9000.5;

    expect(() => buildBoundaryLabel(invalid, context(), identity, environment)).toThrow(
      /last meaningful attack latestMs must be a non-negative integer/i,
    );
  });

  it("serializes with stable indentation, trailing newline, and a deterministic filename", () => {
    const label = buildBoundaryLabel(draft(), context(), identity, environment);
    const serialized = serializeBoundaryLabel(label);

    expect(serialized).toBe(`${JSON.stringify(label, null, 2)}\n`);
    expect(boundaryLabelFileName(label)).toBe(
      "40000000-0000-4000-8000-000000000001-boundary-label-v1.json",
    );
  });

  it("refuses to serialize a typed object that bypassed the builder's interval checks", () => {
    const label = buildBoundaryLabel(draft(), context(), identity, environment);
    label.annotation.lastMeaningfulAttack = { earliestMs: 9001, latestMs: 9000 };

    expect(() => serializeBoundaryLabel(label)).toThrow(
      /last meaningful attack earliestMs must not exceed latestMs/i,
    );
  });
});

describe("boundary label v2 builder", () => {
  it("preserves playable attack, primary content end, and acceptable release separately", () => {
    const label = buildBoundaryLabelV2(draftV2(), context(), identity, environment);

    expect(label).toMatchObject({
      version: 2,
      annotation: {
        lastPlayableAttack: { earliestMs: 8200, latestMs: 8400 },
        primaryContentEnd: { earliestMs: 8800, latestMs: 9000 },
        acceptableReleaseEnd: { earliestMs: 9200, latestMs: 9500 },
        comment: "three distinct meanings",
      },
    });
  });

  it("rejects an attack interval that is entirely after primary content", () => {
    const invalid = draftV2();
    invalid.lastPlayableAttack = { earliestMs: 9100, latestMs: 9200 };

    expect(() => buildBoundaryLabelV2(invalid, context(), identity, environment)).toThrow(
      /last playable attack.*primary content end/i,
    );
  });

  it("serializes as v2 without changing the v1 filename contract", () => {
    const label = buildBoundaryLabelV2(draftV2(), context(), identity, environment);

    expect(serializeBoundaryLabelV2(label)).toBe(`${JSON.stringify(label, null, 2)}\n`);
    expect(boundaryLabelFileNameV2(label)).toBe(
      "40000000-0000-4000-8000-000000000001-boundary-label-v2.json",
    );
    expect(boundaryLabelFileName(buildBoundaryLabel(draft(), context(), identity, environment))).toBe(
      "40000000-0000-4000-8000-000000000001-boundary-label-v1.json",
    );
  });
});
