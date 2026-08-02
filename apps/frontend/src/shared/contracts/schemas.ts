import chartSchema from "@contracts/chart-schema/chart-v1.schema.json";
import keysoundSchema from "@contracts/chart-schema/keysound-manifest-v1.schema.json";
import playtestRunSchema from "@contracts/chart-schema/playtest-run-v1.schema.json";
import Ajv2020, { type ErrorObject, type ValidateFunction } from "ajv/dist/2020.js";
import addFormats from "ajv-formats";

import type {
  ChartDocument,
  KeysoundManifest,
  PlaytestRunManifest,
} from "../../game/core/types";

const ajv = new Ajv2020({ allErrors: true, strict: true });
addFormats(ajv);

const runValidator = ajv.compile<PlaytestRunManifest>(playtestRunSchema);
const chartValidator = ajv.compile<ChartDocument>(chartSchema);
const keysoundValidator = ajv.compile<KeysoundManifest>(keysoundSchema);

function describeErrors(errors: ErrorObject[] | null | undefined): string {
  return (errors ?? [])
    .map((error) => `${error.instancePath || "/"} ${error.message ?? "is invalid"}`)
    .join("; ");
}

function validate<T>(validator: ValidateFunction<T>, value: unknown, fileName: string): asserts value is T {
  if (!validator(value)) {
    throw new Error(`${fileName} schema validation failed: ${describeErrors(validator.errors)}`);
  }
}

export function validatePlaytestRun(value: unknown, fileName = "playtest-run-v1.json"): asserts value is PlaytestRunManifest {
  validate(runValidator, value, fileName);
}

export function validateChart(value: unknown, fileName: string): asserts value is ChartDocument {
  validate(chartValidator, value, fileName);
}

export function validateKeysoundManifest(value: unknown, fileName: string): asserts value is KeysoundManifest {
  validate(keysoundValidator, value, fileName);
}
