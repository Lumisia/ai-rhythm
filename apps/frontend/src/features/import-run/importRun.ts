import type {
  AudioFileRef,
  BoundaryAutomaticEvidence,
  ChartDocument,
  Difficulty,
  KeyMode,
  KeysoundManifest,
  PlaytestRunManifest,
  PlaytestRunManifestV2,
  PublicationReasonCode,
  ReportFileRef,
  RunChartRef,
} from "../../game/core/types";
import {
  extractBoundaryEvidence,
  unavailableBoundaryEvidence,
} from "../boundary-label/boundaryEvidence";
import {
  validateChart,
  validateKeysoundManifest,
  validatePlaytestRunV1,
  validatePlaytestRunV2,
} from "../../shared/contracts/schemas";
import type { AssetSource } from "../../shared/local-files/AssetSource";
import {
  LocalDirectoryAssetSource,
  normalizeRelativePath,
} from "../../shared/local-files/LocalDirectoryAssetSource";
import { sha256Hex } from "../../shared/local-files/hash";
import { derivePublication } from "./publication";

const RUN_MANIFEST_V1_PATH = "playtest-run-v1.json";
const RUN_MANIFEST_V2_PATH = "playtest-run-v2.json";
const keyModes: KeyMode[] = [4, 6, 7];
const difficulties: Difficulty[] = ["EASY", "NORMAL", "HARD", "EXPERT"];

export type ImportMode = "PRODUCTION" | "PLAYTEST";
export type PublicationState =
  | "PRODUCTION_VERIFIED"
  | "PLAYTEST_ONLY"
  | "LEGACY_UNVERIFIED";
export type ImportPublicationReason = PublicationReasonCode | "LEGACY_MANIFEST_UNVERIFIED";

export interface ImportedChart {
  ref: RunChartRef;
  document: ChartDocument;
  bytes: ArrayBuffer;
}

export interface ImportedRun {
  source: AssetSource;
  manifest: PlaytestRunManifest;
  publicationState: PublicationState;
  publicationReasons: ImportPublicationReason[];
  charts: ImportedChart[];
  keysoundManifest: KeysoundManifest | null;
  audio: {
    game: ArrayBuffer;
    noDrums: ArrayBuffer | null;
    keys: ArrayBuffer | null;
  };
  boundaryLabelContext: BoundaryLabelContext;
}

export interface BoundaryLabelContext {
  available: boolean;
  unavailableReason: string | null;
  songVersionId: string;
  gameAudioAssetId: string;
  audioDurationMs: number;
  generationReport: ReportFileRef | null;
  automaticEvidence: BoundaryAutomaticEvidence;
}

interface LoadedManifest {
  manifest: PlaytestRunManifest;
  publicationState: PublicationState;
  publicationReasons: ImportPublicationReason[];
  generationReportDocument: Record<string, unknown> | null;
}

function parseJson(text: string, fileName: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new Error(`${fileName} is not valid JSON: ${reason}`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  }
  if (isRecord(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value) ?? "undefined";
}

function valuesAgree(left: unknown, right: unknown): boolean {
  return canonicalJson(left) === canonicalJson(right);
}

async function readVerifiedAsset(source: AssetSource, ref: AudioFileRef): Promise<ArrayBuffer> {
  const path = normalizeRelativePath(ref.path);
  const bytes = await source.readBytes(path);
  const actualSha256 = await sha256Hex(bytes);
  if (actualSha256 !== ref.sha256) {
    throw new Error(`SHA-256 mismatch for ${path}: expected ${ref.sha256}, received ${actualSha256}`);
  }
  return bytes;
}

async function validateV2Publication(
  source: AssetSource,
  manifest: PlaytestRunManifestV2,
  mode: ImportMode,
): Promise<LoadedManifest> {
  const expectedPublication = derivePublication(
    manifest.outcome,
    manifest.charts.length,
    12,
    manifest.strictBlockers,
  );
  if (!valuesAgree(manifest.publication, expectedPublication)) {
    throw new Error("manifest publication decision disagrees with its outcome snapshot");
  }

  const reportPath = normalizeRelativePath(manifest.generationReport.path);
  const reportBytes = await readVerifiedAsset(source, manifest.generationReport);
  const report = parseJson(
    new TextDecoder("utf-8", { fatal: true }).decode(reportBytes),
    reportPath,
  );
  if (!isRecord(report)) {
    throw new Error(`${reportPath} must contain a JSON object`);
  }
  if (!valuesAgree(report.outcomeStatusV2, manifest.outcome)) {
    throw new Error("generation report outcome disagrees with the v2 manifest snapshot");
  }
  if (!valuesAgree(report.publicationDecision, manifest.publication)) {
    throw new Error("generation report publication decision disagrees with the v2 manifest snapshot");
  }
  if (!valuesAgree(report.strictBlockers, manifest.strictBlockers)) {
    throw new Error("generation report strict blockers disagree with the v2 manifest snapshot");
  }
  const productionAllowed = manifest.publication.decision === "ALLOW_PRODUCTION";
  if (report.publishable !== productionAllowed) {
    throw new Error("generation report publishable flag disagrees with the v2 publication decision");
  }

  if (manifest.publication.decision === "REJECTED") {
    throw new Error("rejected publication decisions cannot be imported for gameplay");
  }
  if (mode === "PRODUCTION" && manifest.publication.decision !== "ALLOW_PRODUCTION") {
    throw new Error(
      `production import rejected publication decision ${manifest.publication.decision}`,
    );
  }

  return {
    manifest,
    publicationState:
      manifest.publication.decision === "ALLOW_PRODUCTION"
        ? "PRODUCTION_VERIFIED"
        : "PLAYTEST_ONLY",
    publicationReasons: [...manifest.publication.reasonCodes],
    generationReportDocument: report,
  };
}

async function loadManifest(source: AssetSource, mode: ImportMode): Promise<LoadedManifest> {
  const hasV1 = source.has(RUN_MANIFEST_V1_PATH);
  const hasV2 = source.has(RUN_MANIFEST_V2_PATH);
  if (hasV1 && hasV2) {
    throw new Error("selected directory contains both v1 and v2 run manifests");
  }
  if (!hasV1 && !hasV2) {
    throw new Error("selected directory does not contain a run manifest");
  }

  if (hasV2) {
    const document = parseJson(
      await source.readText(RUN_MANIFEST_V2_PATH),
      RUN_MANIFEST_V2_PATH,
    );
    validatePlaytestRunV2(document, RUN_MANIFEST_V2_PATH);
    return validateV2Publication(source, document, mode);
  }

  const document = parseJson(
    await source.readText(RUN_MANIFEST_V1_PATH),
    RUN_MANIFEST_V1_PATH,
  );
  validatePlaytestRunV1(document, RUN_MANIFEST_V1_PATH);
  if (mode === "PRODUCTION") {
    throw new Error("v1 run manifests are not accepted for production import");
  }
  return {
    manifest: document,
    publicationState: "LEGACY_UNVERIFIED",
    publicationReasons: ["LEGACY_MANIFEST_UNVERIFIED"],
    generationReportDocument: null,
  };
}

function buildBoundaryLabelContext(
  loaded: LoadedManifest,
  charts: ImportedChart[],
): BoundaryLabelContext {
  const firstChart = charts[0]?.document;
  if (!firstChart) {
    throw new Error("cannot build boundary label context without a published chart");
  }

  const legacyReason = "v1 run manifest does not bind generation-report SHA-256";
  if (loaded.manifest.version === 1) {
    return {
      available: false,
      unavailableReason: legacyReason,
      songVersionId: firstChart.songVersionId,
      gameAudioAssetId: firstChart.gameAudioAssetId,
      audioDurationMs: firstChart.durationMs,
      generationReport: null,
      automaticEvidence: unavailableBoundaryEvidence(legacyReason),
    };
  }

  const durations = new Set(charts.map((chart) => chart.document.durationMs));
  const automaticEvidence = extractBoundaryEvidence(loaded.generationReportDocument);
  if (durations.size !== 1) {
    return {
      available: false,
      unavailableReason: "published charts disagree on audio durationMs",
      songVersionId: firstChart.songVersionId,
      gameAudioAssetId: firstChart.gameAudioAssetId,
      audioDurationMs: 0,
      generationReport: loaded.manifest.generationReport,
      automaticEvidence,
    };
  }

  return {
    available: true,
    unavailableReason: null,
    songVersionId: firstChart.songVersionId,
    gameAudioAssetId: firstChart.gameAudioAssetId,
    audioDurationMs: firstChart.durationMs,
    generationReport: loaded.manifest.generationReport,
    automaticEvidence,
  };
}

function validateChartSet(manifest: PlaytestRunManifest): void {
  if (manifest.charts.length === 0) {
    throw new Error('run manifest contains no charts');
  }

  const combinations = new Set<string>();
  for (const chart of manifest.charts) {
    normalizeRelativePath(chart.path);
    const combination = `${chart.keyMode}K ${chart.difficulty}`;
    if (combinations.has(combination)) {
      throw new Error(`duplicate chart combination: ${combination}`);
    }
    combinations.add(combination);
  }

  // A run may publish fewer than 12 charts when a combination failed, but it
  // must say which ones are absent. Silently missing charts would look like a
  // corrupt run directory.
  const declaredMissing = new Set<string>();
  for (const missing of manifest.missingCharts ?? []) {
    const combination = `${missing.keyMode}K ${missing.difficulty}`;
    if (combinations.has(combination)) {
      throw new Error(`chart is both published and missing: ${combination}`);
    }
    declaredMissing.add(combination);
  }

  for (const keyMode of keyModes) {
    for (const difficulty of difficulties) {
      const combination = `${keyMode}K ${difficulty}`;
      if (!combinations.has(combination) && !declaredMissing.has(combination)) {
        throw new Error(`missing chart combination: ${combination}`);
      }
    }
  }
}

function validateChartIdentity(
  chart: ChartDocument,
  ref: RunChartRef,
  manifest: PlaytestRunManifest,
  commonSongVersionId: string | null,
  commonGameAudioAssetId: string | null,
): void {
  if (chart.keyMode !== ref.keyMode || chart.difficulty !== ref.difficulty) {
    throw new Error(`${ref.path} identity differs from its run manifest entry`);
  }
  if (chart.audioSha256 !== manifest.audio.game.sha256) {
    throw new Error(`${ref.path} audioSha256 differs from audio/game SHA-256`);
  }
  if (commonSongVersionId !== null && chart.songVersionId !== commonSongVersionId) {
    throw new Error(`${ref.path} has a different songVersionId`);
  }
  if (commonGameAudioAssetId !== null && chart.gameAudioAssetId !== commonGameAudioAssetId) {
    throw new Error(`${ref.path} has a different gameAudioAssetId`);
  }
  if (chart.laneSemantics.length !== chart.keyMode) {
    throw new Error(`${ref.path} laneSemantics length does not match ${chart.keyMode}K`);
  }
}

async function importCharts(
  source: AssetSource,
  manifest: PlaytestRunManifest,
): Promise<ImportedChart[]> {
  const imported: ImportedChart[] = [];
  let commonSongVersionId: string | null = null;
  let commonGameAudioAssetId: string | null = null;

  for (const ref of manifest.charts) {
    const bytes = await readVerifiedAsset(source, ref);
    const document = parseJson(new TextDecoder("utf-8", { fatal: true }).decode(bytes), ref.path);
    validateChart(document, ref.path);
    validateChartIdentity(
      document,
      ref,
      manifest,
      commonSongVersionId,
      commonGameAudioAssetId,
    );
    commonSongVersionId ??= document.songVersionId;
    commonGameAudioAssetId ??= document.gameAudioAssetId;
    imported.push({ ref, document, bytes });
  }

  return imported;
}

async function importKeysounds(
  source: AssetSource,
  manifest: PlaytestRunManifest,
  songVersionId: string,
): Promise<{
  manifest: KeysoundManifest | null;
  noDrums: ArrayBuffer | null;
  keys: ArrayBuffer | null;
}> {
  const { noDrums, keys } = manifest.audio;
  const manifestPath = manifest.keysoundManifestPath;
  const hasAnyKeysoundPart = Boolean(noDrums || keys || manifestPath);
  const hasEveryKeysoundPart = Boolean(noDrums && keys && manifestPath);

  if (hasAnyKeysoundPart && !hasEveryKeysoundPart) {
    throw new Error("keysound run requires noDrums, keys, and keysoundManifestPath together");
  }
  if (!hasEveryKeysoundPart || !noDrums || !keys || !manifestPath) {
    return { manifest: null, noDrums: null, keys: null };
  }

  const normalizedManifestPath = normalizeRelativePath(manifestPath);
  const keysoundDocument = parseJson(
    await source.readText(normalizedManifestPath),
    normalizedManifestPath,
  );
  validateKeysoundManifest(keysoundDocument, normalizedManifestPath);
  if (keysoundDocument.songVersionId !== songVersionId) {
    throw new Error(`${normalizedManifestPath} has a different songVersionId`);
  }

  return {
    manifest: keysoundDocument,
    noDrums: await readVerifiedAsset(source, noDrums),
    keys: await readVerifiedAsset(source, keys),
  };
}

export async function importRun(files: File[], mode: ImportMode): Promise<ImportedRun> {
  const source = new LocalDirectoryAssetSource(files);
  const loaded = await loadManifest(source, mode);
  const manifestDocument = loaded.manifest;
  validateChartSet(manifestDocument);
  if (manifestDocument.version === 1) {
    normalizeRelativePath(manifestDocument.generationReportPath);
  }
  normalizeRelativePath(manifestDocument.audio.game.path);
  if (manifestDocument.audio.noDrums) normalizeRelativePath(manifestDocument.audio.noDrums.path);
  if (manifestDocument.audio.keys) normalizeRelativePath(manifestDocument.audio.keys.path);

  const game = await readVerifiedAsset(source, manifestDocument.audio.game);
  const charts = await importCharts(source, manifestDocument);
  const keysounds = await importKeysounds(
    source,
    manifestDocument,
    charts[0].document.songVersionId,
  );
  const boundaryLabelContext = buildBoundaryLabelContext(loaded, charts);

  return {
    source,
    manifest: manifestDocument,
    publicationState: loaded.publicationState,
    publicationReasons: loaded.publicationReasons,
    charts,
    keysoundManifest: keysounds.manifest,
    audio: { game, noDrums: keysounds.noDrums, keys: keysounds.keys },
    boundaryLabelContext,
  };
}

export function importPlaytestRun(files: File[]): Promise<ImportedRun> {
  return importRun(files, "PLAYTEST");
}
