import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { findRunManifestNames } from "./run-manifest-paths.mjs";

const roots = [];

function createRoot() {
  const root = mkdtempSync(join(tmpdir(), "ai-rhythm-manifest-discovery-"));
  roots.push(root);
  return root;
}

afterEach(() => {
  while (roots.length > 0) rmSync(roots.pop(), { recursive: true, force: true });
});

describe("run manifest discovery", () => {
  it("recognizes a sole V3 manifest", () => {
    const root = createRoot();
    writeFileSync(join(root, "playtest-run-v3.json"), "{}");

    expect(findRunManifestNames(root)).toEqual(["playtest-run-v3.json"]);
  });

  it("returns every present version so the caller can reject ambiguity", () => {
    const root = createRoot();
    writeFileSync(join(root, "playtest-run-v1.json"), "{}");
    writeFileSync(join(root, "playtest-run-v3.json"), "{}");

    expect(findRunManifestNames(root)).toEqual([
      "playtest-run-v1.json",
      "playtest-run-v3.json",
    ]);
  });
});
