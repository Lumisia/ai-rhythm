import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { loadRunDirectory } from "../../test/loadRunDirectory";
import { validateChart } from "../../shared/contracts/schemas";
import { importRun } from "./importRun";

const fixtureRoot = resolve(process.cwd(), "src/test/fixtures/playtest-run");

describe("chart worker output contract", () => {
  it("imports the deterministic worker fixture as a complete run", async () => {
    const imported = await importRun(loadRunDirectory(fixtureRoot));

    expect(imported.manifest.version).toBe(1);
    expect(imported.manifest.charts).toHaveLength(12);
    expect(imported.charts).toHaveLength(12);
    expect(new Set(imported.charts.map(({ document }) => document.schemaVersion))).toEqual(
      new Set([1]),
    );
    expect(
      imported.charts.map(({ ref }) => `${ref.keyMode}K:${ref.difficulty}`).sort(),
    ).toEqual(
      [
        "4K:EASY",
        "4K:EXPERT",
        "4K:HARD",
        "4K:NORMAL",
        "6K:EASY",
        "6K:EXPERT",
        "6K:HARD",
        "6K:NORMAL",
        "7K:EASY",
        "7K:EXPERT",
        "7K:HARD",
        "7K:NORMAL",
      ],
    );
    expect(new TextDecoder().decode(imported.audio.game.slice(0, 4))).toBe("fLaC");
    expect(imported.audio.game.byteLength).toBeGreaterThan(1_000);
    expect(imported.keysoundManifest).toBeNull();

    const report = JSON.parse(
      await imported.source.readText(imported.manifest.generationReportPath),
    ) as { charts: unknown[] };
    expect(report.charts).toHaveLength(12);
  });

  it("accepts Mapperatorinator timing that begins before audio zero", async () => {
    const imported = await importRun(loadRunDirectory(fixtureRoot));
    const document = {
      ...imported.charts[0].document,
      bpmEvents: [{ timeMs: -120, bpm: 120 }],
    };

    expect(() => validateChart(document, "negative-timing.chart.json")).not.toThrow();
  });
});
