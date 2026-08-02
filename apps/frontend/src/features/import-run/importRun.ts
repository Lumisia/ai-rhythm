import type {
  AudioFileRef,
  ChartDocument,
  Difficulty,
  KeyMode,
  KeysoundManifest,
  PlaytestRunManifest,
  RunChartRef,
} from "../../game/core/types";
import {
  validateChart,
  validateKeysoundManifest,
  validatePlaytestRun,
} from "../../shared/contracts/schemas";
import type { AssetSource } from "../../shared/local-files/AssetSource";
import {
  LocalDirectoryAssetSource,
  normalizeRelativePath,
} from "../../shared/local-files/LocalDirectoryAssetSource";
import { sha256Hex } from "../../shared/local-files/hash";

const RUN_MANIFEST_PATH = "playtest-run-v1.json";
const keyModes: KeyMode[] = [4, 6, 7];
const difficulties: Difficulty[] = ["EASY", "NORMAL", "HARD", "EXPERT"];

export interface ImportedChart {
  ref: RunChartRef;
  document: ChartDocument;
  bytes: ArrayBuffer;
}

export interface ImportedRun {
  source: AssetSource;
  manifest: PlaytestRunManifest;
  charts: ImportedChart[];
  keysoundManifest: KeysoundManifest | null;
  audio: {
    game: ArrayBuffer;
    noDrums: ArrayBuffer | null;
    keys: ArrayBuffer | null;
  };
}

function parseJson(text: string, fileName: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new Error(`${fileName} is not valid JSON: ${reason}`);
  }
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

function validateChartSet(manifest: PlaytestRunManifest): void {
  if (manifest.charts.length !== 12) {
    throw new Error(`expected 12 chart combinations, received ${manifest.charts.length}`);
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

  for (const keyMode of keyModes) {
    for (const difficulty of difficulties) {
      const combination = `${keyMode}K ${difficulty}`;
      if (!combinations.has(combination)) {
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

export async function importRun(files: File[]): Promise<ImportedRun> {
  const source = new LocalDirectoryAssetSource(files);
  const manifestDocument = parseJson(
    await source.readText(RUN_MANIFEST_PATH),
    RUN_MANIFEST_PATH,
  );
  validatePlaytestRun(manifestDocument);
  validateChartSet(manifestDocument);
  normalizeRelativePath(manifestDocument.generationReportPath);
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

  return {
    source,
    manifest: manifestDocument,
    charts,
    keysoundManifest: keysounds.manifest,
    audio: { game, noDrums: keysounds.noDrums, keys: keysounds.keys },
  };
}
