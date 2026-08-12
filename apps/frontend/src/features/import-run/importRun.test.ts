import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import type {
  ChartDocument,
  Difficulty,
  KeyMode,
  KeysoundManifest,
  LaneSemantic,
  OutcomeStatusSnapshot,
  PlaytestRunManifest,
  PublicationDecisionSnapshot,
  PublicationStrictBlocker,
  RunChartRef,
} from "../../game/core/types";
import { LocalDirectoryAssetSource } from "../../shared/local-files/LocalDirectoryAssetSource";
import { importPlaytestRun, importRun } from "./importRun";

const encoder = new TextEncoder();
const keyModes: KeyMode[] = [4, 6, 7];
const difficulties: Difficulty[] = ["EASY", "NORMAL", "HARD", "EXPERT"];
const songVersionId = "10000000-0000-4000-8000-000000000001";
const gameAudioAssetId = "20000000-0000-4000-8000-000000000001";

interface RunOptions {
  badChartHash?: boolean;
  chartPath?: string;
  duplicateCombination?: boolean;
  includeBothManifests?: boolean;
  invalidChartSchema?: boolean;
  manifestPublicationMissingReason?: boolean;
  manifestVersion?: 1 | 2;
  missingReport?: boolean;
  omitFirstChart?: boolean;
  partial?: boolean;
  reportOutcomeMismatch?: boolean;
  reportPublicationMismatch?: boolean;
  reportPublishableMismatch?: boolean;
  reportStrictBlockerMismatch?: boolean;
  review?: boolean;
  tamperReportHash?: boolean;
  uncalibratedBoundary?: boolean;
  varyFirstChartDuration?: boolean;
  withBoundaryEvidence?: boolean;
  withKeysounds?: boolean;
}

function digest(body: string | Uint8Array): string {
  return createHash("sha256").update(body).digest("hex");
}

function laneSemantics(keyMode: KeyMode): LaneSemantic[] {
  if (keyMode === 4) return ["MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4"];
  if (keyMode === 6) {
    return ["SIDE_LEFT", "MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4", "SIDE_RIGHT"];
  }
  return ["SIDE_LEFT", "MAIN_1", "MAIN_2", "CENTER", "MAIN_3", "MAIN_4", "SIDE_RIGHT"];
}

function makeFile(relativePath: string, body: string | Uint8Array): File {
  let part: BlobPart;
  if (typeof body === "string") {
    part = body;
  } else {
    const buffer = new ArrayBuffer(body.byteLength);
    new Uint8Array(buffer).set(body);
    part = buffer;
  }
  const file = new File([part], relativePath.split("/").at(-1) ?? relativePath);
  Object.defineProperty(file, "webkitRelativePath", {
    configurable: true,
    value: `selected-run/${relativePath}`,
  });
  return file;
}

function makeChart(index: number, keyMode: KeyMode, difficulty: Difficulty, audioSha256: string) {
  const chart: ChartDocument = {
    schemaVersion: 1,
    chartId: `30000000-0000-4000-8000-${String(index + 1).padStart(12, "0")}`,
    songVersionId,
    gameAudioAssetId,
    audioSha256,
    keyMode,
    difficulty,
    laneSemantics: laneSemantics(keyMode),
    offsetMs: 0,
    durationMs: 8000,
    bpmEvents: [{ timeMs: 0, bpm: 120 }],
    bpmSource: "BEAT_THIS",
    notes: [{ id: 1, lane: 0, timeMs: 1000, type: "TAP", durationMs: null }],
    autoPlayOnsets: [],
    metrics: {
      noteCount: 1,
      holdCount: 0,
      avgNps: 0.125,
      p95Nps: 1,
      peakNps: 1,
      chordRatio: 0,
      maxJack: 1,
      projectRating: index + 1,
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
      version: "1",
      analysisVersion: "1",
      postprocessVersion: "1",
      seed: 7,
    },
  };
  return chart;
}

function makeRunFiles(options: RunOptions = {}): File[] {
  const gameAudio = new Uint8Array([0x66, 0x4c, 0x61, 0x43, 1, 2, 3]);
  const gameSha = digest(gameAudio);
  const chartFiles: File[] = [];
  const chartRefs: RunChartRef[] = [];

  let index = 0;
  for (const keyMode of keyModes) {
    for (const difficulty of difficulties) {
      const chart = makeChart(index, keyMode, difficulty, gameSha);
      if (options.varyFirstChartDuration && index === 0) {
        chart.durationMs = 7999;
      }
      if (options.invalidChartSchema && index === 0) {
        Object.assign(chart, { keyMode: 5 });
      }
      const body = JSON.stringify(chart);
      const path = index === 0 && options.chartPath ? options.chartPath : `charts/${keyMode}k-${difficulty}.json`;
      if (!(options.partial && index === 0)) {
        chartRefs.push({ path, sha256: digest(body), keyMode, difficulty });
      }
      if (!(options.omitFirstChart && index === 0) && !(options.partial && index === 0)) {
        chartFiles.push(makeFile(`charts/${keyMode}k-${difficulty}.json`, options.badChartHash && index === 0 ? "{}" : body));
      }
      index += 1;
    }
  }

  if (options.duplicateCombination) {
    chartRefs[1] = { ...chartRefs[1], keyMode: chartRefs[0].keyMode, difficulty: chartRefs[0].difficulty };
  }

  const audio: PlaytestRunManifest["audio"] = {
    game: { path: "audio/game.flac", sha256: gameSha },
    noDrums: null,
    keys: null,
  };
  let keysoundManifestPath: string | null = null;
  const files = [makeFile("audio/game.flac", gameAudio), ...chartFiles];

  if (options.withKeysounds) {
    const noDrums = new Uint8Array([10, 11, 12]);
    const keys = new Uint8Array([20, 21, 22]);
    audio.noDrums = { path: "audio/no_drums.flac", sha256: digest(noDrums) };
    audio.keys = { path: "audio/drums.flac", sha256: digest(keys) };
    keysoundManifestPath = "keysound-manifest.json";
    const keysoundManifest: KeysoundManifest = {
      schemaVersion: 1,
      songVersionId,
      bgmAssetId: "50000000-0000-4000-8000-000000000001",
      keysAssetId: "60000000-0000-4000-8000-000000000001",
      sliceSec: 0.3,
      prerollSec: 0.012,
      snapWindowMs: 50,
      drumOnsets: [1000, 2000],
    };
    files.push(
      makeFile("audio/no_drums.flac", noDrums),
      makeFile("audio/drums.flac", keys),
      makeFile("keysound-manifest.json", JSON.stringify(keysoundManifest)),
    );
  }

  const legacyManifest: PlaytestRunManifest = {
    version: 1,
    runId: "40000000-0000-4000-8000-000000000001",
    title: "fixture song",
    generatedAt: "2026-08-02T00:00:00Z",
    workerVersion: "0.1.0",
    audio,
    charts: chartRefs,
    ...(options.partial
      ? { missingCharts: [{ keyMode: 4, difficulty: "EASY", reason: "FIXTURE_MISSING" }] }
      : {}),
    keysoundManifestPath,
    generationReportPath: "generation-report.json",
  };

  if (options.manifestVersion === 1 || options.includeBothManifests) {
    files.push(makeFile("playtest-run-v1.json", JSON.stringify(legacyManifest)));
  }

  if (options.manifestVersion !== 1 || options.includeBothManifests) {
    const outcome: OutcomeStatusSnapshot = options.partial
      ? {
          execution: "SUCCEEDED",
          completeness: "PARTIAL",
          quality: "REVIEW",
          failureCategory: "NONE",
          publishableStrict: false,
        }
      : options.review
        ? {
            execution: "SUCCEEDED",
            completeness: "COMPLETE",
            quality: "REVIEW",
            failureCategory: "NONE",
            publishableStrict: false,
          }
        : {
            execution: "SUCCEEDED",
            completeness: "COMPLETE",
            quality: "PASS",
            failureCategory: "NONE",
            publishableStrict: true,
          };
    const strictBlockers: PublicationStrictBlocker[] = options.uncalibratedBoundary
      ? ["BOUNDARY_POLICY_UNCALIBRATED"]
      : [];
    const publication: PublicationDecisionSnapshot = options.partial
      ? {
          policyVersion: "PUBLICATION_POLICY_V2",
          decision: "PLAYTEST_ONLY",
          reasonCodes: [
            "INCOMPLETE_CHART_SET",
            "QUALITY_REVIEW_REQUIRED",
            "STRICT_OUTCOME_FALSE",
          ],
        }
      : options.review
        ? {
            policyVersion: "PUBLICATION_POLICY_V2",
            decision: "PLAYTEST_ONLY",
            reasonCodes: ["QUALITY_REVIEW_REQUIRED", "STRICT_OUTCOME_FALSE"],
          }
        : {
            policyVersion: "PUBLICATION_POLICY_V2",
            decision: "ALLOW_PRODUCTION",
            reasonCodes: [],
          };
    if (strictBlockers.length > 0) {
      publication.decision = "PLAYTEST_ONLY";
      publication.reasonCodes = [...strictBlockers, ...publication.reasonCodes].sort();
    }
    if (options.manifestPublicationMissingReason) {
      publication.reasonCodes = publication.reasonCodes.filter(
        (reason) => reason !== "STRICT_OUTCOME_FALSE",
      );
    }
    const reportOutcome = options.reportOutcomeMismatch
      ? {
          execution: "SUCCEEDED",
          completeness: "COMPLETE",
          quality: "REVIEW",
          failureCategory: "NONE",
          publishableStrict: false,
        }
      : outcome;
    const reportPublication = options.reportOutcomeMismatch || options.reportPublicationMismatch
      ? {
          policyVersion: "PUBLICATION_POLICY_V2",
          decision: "PLAYTEST_ONLY",
          reasonCodes: ["QUALITY_REVIEW_REQUIRED", "STRICT_OUTCOME_FALSE"],
        }
      : publication;
    const reportStrictBlockers = options.reportStrictBlockerMismatch
      ? []
      : strictBlockers;
    const reportBody = JSON.stringify({
      publishable: options.reportPublishableMismatch
        ? reportPublication.decision !== "ALLOW_PRODUCTION"
        : reportPublication.decision === "ALLOW_PRODUCTION",
      outcomeStatusV2: reportOutcome,
      strictBlockers: reportStrictBlockers,
      publicationDecision: reportPublication,
      ...(options.withBoundaryEvidence
        ? {
            musicBounds: {
              outroObservation: {
                lastDetectedOnsetMs: 7_500,
                lastActiveRmsEndMs: 7_700,
                lastEvidenceMs: 7_700,
              },
              boundaryPolicyEvaluation: {
                version: "BOUNDARY_POLICY_EVALUATION_V1",
                policyState: "PROVISIONAL",
                confidence: "LOW",
                enforcementMode: "SHADOW",
                observationSha256: "a".repeat(64),
                provisionalContract: {
                  maxNoteStartMs: 7_800,
                  releaseEndMs: 8_000,
                },
                effectiveContract: {
                  maxNoteStartMs: 8_000,
                  releaseEndMs: 8_000,
                },
              },
            },
          }
        : {}),
    });
    const reportFileBody = options.tamperReportHash
      ? JSON.stringify({ ...JSON.parse(reportBody), tampered: true })
      : reportBody;
    if (!options.missingReport) {
      files.push(makeFile("generation-report.json", reportFileBody));
    }
    files.push(
      makeFile(
        "playtest-run-v2.json",
        JSON.stringify({
          version: 2,
          runId: legacyManifest.runId,
          title: legacyManifest.title,
          generatedAt: legacyManifest.generatedAt,
          workerVersion: legacyManifest.workerVersion,
          audio,
          charts: chartRefs,
          missingCharts: legacyManifest.missingCharts ?? [],
          keysoundManifestPath,
          generationReport: {
            path: "generation-report.json",
            sha256: digest(reportBody),
          },
          outcome,
          strictBlockers,
          publication,
        }),
      ),
    );
  }
  return files;
}

describe("importRun", () => {
  it("imports a valid v2 run in production mode", async () => {
    const imported = await importRun(makeRunFiles(), "PRODUCTION");
    expect(imported.manifest.charts).toHaveLength(12);
    expect(imported.charts).toHaveLength(12);
    expect(imported.publicationState).toBe("PRODUCTION_VERIFIED");
    expect(imported.publicationReasons).toEqual([]);
    expect(imported.boundaryLabelContext).toMatchObject({
      available: true,
      unavailableReason: null,
      songVersionId,
      gameAudioAssetId,
      audioDurationMs: 8000,
      generationReport: {
        path: "generation-report.json",
      },
      automaticEvidence: {
        availability: "UNAVAILABLE",
      },
    });
    expect(Array.from(new Uint8Array(imported.audio.game).slice(0, 4))).toEqual(
      Array.from(encoder.encode("fLaC")),
    );
  });

  it("projects verified v2 boundary evidence without making it a human verdict", async () => {
    const imported = await importRun(
      makeRunFiles({ withBoundaryEvidence: true }),
      "PRODUCTION",
    );

    expect(imported.boundaryLabelContext.automaticEvidence).toMatchObject({
      availability: "AVAILABLE",
      unavailableReason: null,
      policyState: "PROVISIONAL",
      enforcementMode: "SHADOW",
      lastEvidenceMs: 7_700,
      provisionalMaxNoteStartMs: 7_800,
      effectiveMaxNoteStartMs: 8_000,
    });
  });

  it("keeps v2 human labeling available when automatic boundary evidence is absent", async () => {
    const imported = await importRun(makeRunFiles(), "PRODUCTION");

    expect(imported.boundaryLabelContext.available).toBe(true);
    expect(imported.boundaryLabelContext.automaticEvidence).toMatchObject({
      availability: "UNAVAILABLE",
      unavailableReason: "musicBounds must be an object",
    });
  });

  it("disables labeling when chart durations do not identify one audio timeline", async () => {
    const imported = await importRun(
      makeRunFiles({ varyFirstChartDuration: true }),
      "PRODUCTION",
    );

    expect(imported.boundaryLabelContext).toMatchObject({
      available: false,
      unavailableReason: "published charts disagree on audio durationMs",
      audioDurationMs: 0,
    });
  });

  it("blocks an uncalibrated boundary in production but preserves it in playtest", async () => {
    const files = makeRunFiles({ uncalibratedBoundary: true });

    await expect(importRun(files, "PRODUCTION")).rejects.toThrow(
      /production.*PLAYTEST_ONLY/i,
    );
    const imported = await importRun(files, "PLAYTEST");
    expect(imported.publicationState).toBe("PLAYTEST_ONLY");
    expect(imported.publicationReasons).toEqual([
      "BOUNDARY_POLICY_UNCALIBRATED",
    ]);
  });

  it("rejects a report that drops manifest boundary blockers", async () => {
    await expect(
      importRun(
        makeRunFiles({
          uncalibratedBoundary: true,
          reportStrictBlockerMismatch: true,
        }),
        "PLAYTEST",
      ),
    ).rejects.toThrow(/strict blockers/i);
  });

  it("rejects a missing run manifest", async () => {
    const files = makeRunFiles().filter((file) => file.name !== "playtest-run-v2.json");
    await expect(importRun(files, "PLAYTEST")).rejects.toThrow(/run manifest/i);
  });

  it.each([{ review: true }, { partial: true }])(
    "rejects playtest-only v2 outcomes in production mode: %o",
    async (options) => {
      await expect(importRun(makeRunFiles(options), "PRODUCTION")).rejects.toThrow(
        /production.*PLAYTEST_ONLY/i,
      );
    },
  );

  it("imports a review outcome only in playtest mode with explicit reasons", async () => {
    const imported = await importRun(makeRunFiles({ review: true }), "PLAYTEST");
    expect(imported.publicationState).toBe("PLAYTEST_ONLY");
    expect(imported.publicationReasons).toEqual([
      "QUALITY_REVIEW_REQUIRED",
      "STRICT_OUTCOME_FALSE",
    ]);
  });

  it("imports an explicitly partial run only in playtest mode", async () => {
    const imported = await importRun(makeRunFiles({ partial: true }), "PLAYTEST");
    expect(imported.charts).toHaveLength(11);
    expect(imported.charts.length + (imported.manifest.missingCharts ?? []).length).toBe(12);
    expect(imported.publicationState).toBe("PLAYTEST_ONLY");
    expect(imported.publicationReasons).toContain("INCOMPLETE_CHART_SET");
    expect(imported.boundaryLabelContext.available).toBe(true);
  });

  it("rejects a v2 generation report hash mismatch in playtest mode", async () => {
    await expect(
      importRun(makeRunFiles({ tamperReportHash: true }), "PLAYTEST"),
    ).rejects.toThrow(/SHA-256.*generation-report/i);
  });

  it("rejects report and manifest outcome disagreement in playtest mode", async () => {
    await expect(
      importRun(makeRunFiles({ reportOutcomeMismatch: true }), "PLAYTEST"),
    ).rejects.toThrow(/outcome.*disagree/i);
  });

  it("rejects report and manifest publication disagreement in playtest mode", async () => {
    await expect(
      importRun(makeRunFiles({ reportPublicationMismatch: true }), "PLAYTEST"),
    ).rejects.toThrow(/publication decision.*disagree/i);
  });

  it("rejects a legacy publishable flag that disagrees with strict outcome", async () => {
    await expect(
      importRun(makeRunFiles({ reportPublishableMismatch: true }), "PLAYTEST"),
    ).rejects.toThrow(/publishable flag.*disagree/i);
  });

  it("rejects a manifest publication decision with a missing reason", async () => {
    await expect(
      importRun(
        makeRunFiles({ review: true, manifestPublicationMissingReason: true }),
        "PLAYTEST",
      ),
    ).rejects.toThrow(/manifest publication decision.*disagree/i);
  });

  it("accepts v1 only as legacy-unverified playtest input", async () => {
    const files = makeRunFiles({ manifestVersion: 1 });
    const imported = await importRun(files, "PLAYTEST");
    expect(imported.publicationState).toBe("LEGACY_UNVERIFIED");
    expect(imported.publicationReasons).toEqual(["LEGACY_MANIFEST_UNVERIFIED"]);
    expect(imported.boundaryLabelContext).toMatchObject({
      available: false,
      unavailableReason: "v1 run manifest does not bind generation-report SHA-256",
      generationReport: null,
      automaticEvidence: {
        availability: "UNAVAILABLE",
      },
    });
    await expect(importRun(files, "PRODUCTION")).rejects.toThrow(/v1.*production/i);
  });

  it("provides an explicit local-playtest wrapper with no production fallback", async () => {
    const imported = await importPlaytestRun(makeRunFiles({ manifestVersion: 1 }));
    expect(imported.publicationState).toBe("LEGACY_UNVERIFIED");
  });

  it("rejects ambiguous directories containing both manifest versions", async () => {
    await expect(
      importRun(makeRunFiles({ includeBothManifests: true }), "PLAYTEST"),
    ).rejects.toThrow(/both.*manifest/i);
  });

  it("rejects a v2 run whose generation report is absent", async () => {
    await expect(
      importRun(makeRunFiles({ missingReport: true }), "PLAYTEST"),
    ).rejects.toThrow(/missing file.*generation-report/i);
  });

  it("rejects a missing chart with its file name", async () => {
    await expect(importRun(makeRunFiles({ omitFirstChart: true }), "PRODUCTION")).rejects.toThrow(
      /charts\/4k-EASY\.json/,
    );
  });

  it("rejects a chart that fails its JSON Schema", async () => {
    await expect(importRun(makeRunFiles({ invalidChartSchema: true }), "PRODUCTION")).rejects.toThrow(
      /charts\/4k-EASY\.json.*schema/i,
    );
  });

  it("rejects a chart whose sha256 differs from the run manifest", async () => {
    await expect(
      importRun(makeRunFiles({ badChartHash: true }), "PRODUCTION"),
    ).rejects.toThrow(/SHA-256.*4k-EASY/);
  });

  it("rejects a parent-directory path", async () => {
    await expect(
      importRun(makeRunFiles({ chartPath: "../chart.json" }), "PRODUCTION"),
    ).rejects.toThrow(/relative path/);
  });

  it("rejects a duplicate key-mode and difficulty combination", async () => {
    await expect(
      importRun(makeRunFiles({ duplicateCombination: true }), "PRODUCTION"),
    ).rejects.toThrow(/duplicate.*4K.*EASY/i);
  });

  it("imports and validates optional keysound assets", async () => {
    const imported = await importRun(makeRunFiles({ withKeysounds: true }), "PRODUCTION");
    expect(imported.keysoundManifest?.drumOnsets).toEqual([1000, 2000]);
    expect(imported.audio.noDrums?.byteLength).toBe(3);
    expect(imported.audio.keys?.byteLength).toBe(3);
  });

  it("rejects case-insensitive duplicate paths", () => {
    expect(
      () =>
        new LocalDirectoryAssetSource([
          makeFile("charts/chart.json", "one"),
          makeFile("Charts/CHART.json", "two"),
        ]),
    ).toThrow(/duplicate path/i);
  });
});
