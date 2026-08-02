import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const runDirectory = process.argv[2] ? resolve(process.argv[2]) : null;
if (!runDirectory) {
  console.error('Usage: npm run validate:run -- "<playtest-run-directory>"');
  process.exitCode = 2;
} else if (!existsSync(resolve(runDirectory, "playtest-run-v1.json"))) {
  console.error(`playtest-run-v1.json was not found in ${runDirectory}`);
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
