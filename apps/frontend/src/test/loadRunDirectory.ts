import { readFileSync } from "node:fs";
import { basename, isAbsolute, relative, resolve } from "node:path";

interface ManifestFileRef {
  path: string;
}

interface ManifestPaths {
  audio: {
    game: ManifestFileRef;
    noDrums: ManifestFileRef | null;
    keys: ManifestFileRef | null;
  };
  charts: ManifestFileRef[];
  keysoundManifestPath: string | null;
  generationReportPath: string;
}

function checkedPath(root: string, candidate: string): string {
  const normalized = candidate.replaceAll("\\", "/");
  const segments = normalized.split("/");
  if (
    !normalized ||
    normalized.startsWith("/") ||
    /^[a-zA-Z]:\//.test(normalized) ||
    segments.some((segment) => !segment || segment === "." || segment === "..")
  ) {
    throw new Error(`invalid run-relative path: ${candidate}`);
  }

  const absolutePath = resolve(root, ...segments);
  const fromRoot = relative(root, absolutePath);
  if (fromRoot.startsWith("..") || isAbsolute(fromRoot)) {
    throw new Error(`run asset is outside the selected directory: ${candidate}`);
  }
  return absolutePath;
}

function fileFromPath(root: string, relativePath: string): File {
  const path = checkedPath(root, relativePath);
  const raw = readFileSync(path);
  const body = new ArrayBuffer(raw.byteLength);
  new Uint8Array(body).set(raw);
  const file = new File([body], basename(path));
  Object.defineProperty(file, "webkitRelativePath", {
    configurable: true,
    value: `playtest-run/${relativePath.replaceAll("\\", "/")}`,
  });
  return file;
}

export function loadRunDirectory(root: string): File[] {
  const absoluteRoot = resolve(root);
  const manifestPath = checkedPath(absoluteRoot, "playtest-run-v1.json");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf-8")) as ManifestPaths;
  const referencedPaths = new Set([
    "playtest-run-v1.json",
    manifest.generationReportPath,
    manifest.audio.game.path,
    ...manifest.charts.map((chart) => chart.path),
  ]);
  if (manifest.audio.noDrums) referencedPaths.add(manifest.audio.noDrums.path);
  if (manifest.audio.keys) referencedPaths.add(manifest.audio.keys.path);
  if (manifest.keysoundManifestPath) referencedPaths.add(manifest.keysoundManifestPath);
  return [...referencedPaths].map((relativePath) => fileFromPath(absoluteRoot, relativePath));
}
