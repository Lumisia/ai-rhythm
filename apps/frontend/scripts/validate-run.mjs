import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const runDirectory = process.argv[2] ? resolve(process.argv[2]) : null;
if (!runDirectory) {
  console.error('Usage: npm run validate:run -- "<playtest-run-directory>"');
  process.exitCode = 2;
} else {
  const manifestPaths = ["playtest-run-v1.json", "playtest-run-v2.json"].filter((name) =>
    existsSync(resolve(runDirectory, name)),
  );
  if (manifestPaths.length !== 1) {
    console.error(
      `Expected exactly one playtest run manifest in ${runDirectory}; found ${manifestPaths.length}`,
    );
    process.exitCode = 2;
  } else {
    const vitest = fileURLToPath(new URL("../node_modules/vitest/vitest.mjs", import.meta.url));
    const testFile = fileURLToPath(
      new URL("../src/features/import-run/localRun.test.ts", import.meta.url),
    );
    const result = spawnSync(
      process.execPath,
      [vitest, "run", testFile, "--reporter=verbose"],
      {
        cwd: fileURLToPath(new URL("..", import.meta.url)),
        env: { ...process.env, PLAYTEST_RUN_DIR: runDirectory },
        stdio: "inherit",
      },
    );
    process.exitCode = result.status ?? 1;
  }
}
