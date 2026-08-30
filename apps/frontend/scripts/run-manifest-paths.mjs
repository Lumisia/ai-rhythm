import { existsSync } from "node:fs";
import { resolve } from "node:path";

export const RUN_MANIFEST_NAMES = Object.freeze([
  "playtest-run-v1.json",
  "playtest-run-v2.json",
  "playtest-run-v3.json",
]);

export function findRunManifestNames(runDirectory) {
  return RUN_MANIFEST_NAMES.filter((name) => existsSync(resolve(runDirectory, name)));
}
