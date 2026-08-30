import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { loadRunDirectory } from "./loadRunDirectory";

const createdRoots: string[] = [];

function createRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "ai-rhythm-run-loader-"));
  createdRoots.push(root);
  mkdirSync(join(root, "audio"));
  writeFileSync(join(root, "audio", "game.bin"), "audio");
  writeFileSync(join(root, "generation-report.json"), "{}");
  return root;
}

function commonManifest(): Record<string, unknown> {
  return {
    audio: {
      game: { path: "audio/game.bin" },
      noDrums: null,
      keys: null,
    },
    charts: [],
    keysoundManifestPath: null,
  };
}

afterEach(() => {
  while (createdRoots.length > 0) {
    rmSync(createdRoots.pop()!, { recursive: true, force: true });
  }
});

describe("loadRunDirectory", () => {
  it("loads a v1 manifest and its unbound generation report", () => {
    const root = createRoot();
    writeFileSync(
      join(root, "playtest-run-v1.json"),
      JSON.stringify({
        ...commonManifest(),
        generationReportPath: "generation-report.json",
      }),
    );

    const paths = loadRunDirectory(root).map((file) => file.webkitRelativePath);

    expect(paths).toEqual([
      "playtest-run/playtest-run-v1.json",
      "playtest-run/generation-report.json",
      "playtest-run/audio/game.bin",
    ]);
  });

  it("loads a v2 manifest and its SHA-bound generation report", () => {
    const root = createRoot();
    writeFileSync(
      join(root, "playtest-run-v2.json"),
      JSON.stringify({
        ...commonManifest(),
        generationReport: { path: "generation-report.json" },
      }),
    );

    const paths = loadRunDirectory(root).map((file) => file.webkitRelativePath);

    expect(paths).toEqual([
      "playtest-run/playtest-run-v2.json",
      "playtest-run/generation-report.json",
      "playtest-run/audio/game.bin",
    ]);
  });

  it("loads a v3 manifest and its SHA-bound generation report", () => {
    const root = createRoot();
    writeFileSync(
      join(root, "playtest-run-v3.json"),
      JSON.stringify({
        ...commonManifest(),
        generationReport: { path: "generation-report.json" },
      }),
    );

    const paths = loadRunDirectory(root).map((file) => file.webkitRelativePath);

    expect(paths).toEqual([
      "playtest-run/playtest-run-v3.json",
      "playtest-run/generation-report.json",
      "playtest-run/audio/game.bin",
    ]);
  });

  it("rejects ambiguous v1, v2, and v3 manifests independent of order", () => {
    const root = createRoot();
    writeFileSync(join(root, "playtest-run-v1.json"), "{}");
    writeFileSync(join(root, "playtest-run-v2.json"), "{}");
    writeFileSync(join(root, "playtest-run-v3.json"), "{}");

    expect(() => loadRunDirectory(root)).toThrow(
      "expected exactly one run manifest, found 3",
    );
  });
});
