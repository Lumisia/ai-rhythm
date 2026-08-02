import { describe, expect, it } from "vitest";

import { loadRunDirectory } from "../../test/loadRunDirectory";
import { importRun } from "./importRun";
import { diagnoseCharts } from "./runDiagnostics";

const runDirectory = process.env.PLAYTEST_RUN_DIR?.trim();

if (!runDirectory) {
  describe.skip("local worker run", () => {
    it("requires PLAYTEST_RUN_DIR", () => undefined);
  });
} else {
  describe("local worker run", () => {
    it("passes the frontend importer and gameplay constraints", async () => {
      const imported = await importRun(loadRunDirectory(runDirectory));
      const report = diagnoseCharts(imported.charts.map(({ document }) => document));

      expect(imported.charts).toHaveLength(12);
      expect(imported.audio.game.byteLength).toBeGreaterThan(1_000_000);
      expect(report.errors).toEqual([]);
      expect(
        imported.charts.every(({ document }) =>
          document.generator.name.toLocaleLowerCase("en-US").includes("mapperatorinator"),
        ),
      ).toBe(true);

      console.info(
        JSON.stringify(
          {
            runId: imported.manifest.runId,
            title: imported.manifest.title,
            audioBytes: imported.audio.game.byteLength,
            charts: report.charts,
            warnings: report.warnings,
          },
          null,
          2,
        ),
      );
    });
  });
}
