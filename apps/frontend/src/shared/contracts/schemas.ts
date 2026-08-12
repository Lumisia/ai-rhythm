import chartSchema from "@contracts/chart-schema/chart-v1.schema.json";
import boundaryLabelSchema from "@contracts/chart-schema/boundary-label-v1.schema.json";
import boundaryLabelV2Schema from "@contracts/chart-schema/boundary-label-v2.schema.json";
import keysoundSchema from "@contracts/chart-schema/keysound-manifest-v1.schema.json";
import playtestRunV1Schema from "@contracts/chart-schema/playtest-run-v1.schema.json";
import playtestRunV2Schema from "@contracts/chart-schema/playtest-run-v2.schema.json";
import Ajv2020, { type ErrorObject, type ValidateFunction } from "ajv/dist/2020.js";
import addFormats from "ajv-formats";

import type {
  BoundaryLabelV1,
  BoundaryLabelV2,
  ChartDocument,
  KeysoundManifest,
  PlaytestRunManifestV1,
  PlaytestRunManifestV2,
} from "../../game/core/types";

const ajv = new Ajv2020({ allErrors: true, strict: true });
addFormats(ajv);

const runV1Validator = ajv.compile<PlaytestRunManifestV1>(playtestRunV1Schema);
const runV2Validator = ajv.compile<PlaytestRunManifestV2>(playtestRunV2Schema);
const boundaryLabelValidator = ajv.compile<BoundaryLabelV1>(boundaryLabelSchema);
const boundaryLabelV2Validator = ajv.compile<BoundaryLabelV2>(boundaryLabelV2Schema);
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

export function validatePlaytestRunV1(
  value: unknown,
  fileName = "playtest-run-v1.json",
): asserts value is PlaytestRunManifestV1 {
  validate(runV1Validator, value, fileName);
}

export function validatePlaytestRunV2(
  value: unknown,
  fileName = "playtest-run-v2.json",
): asserts value is PlaytestRunManifestV2 {
  validate(runV2Validator, value, fileName);
}

export function validateChart(value: unknown, fileName: string): asserts value is ChartDocument {
  validate(chartValidator, value, fileName);
}

export function validateBoundaryLabel(
  value: unknown,
  fileName = "boundary-label-v1.json",
): asserts value is BoundaryLabelV1 {
  validate(boundaryLabelValidator, value, fileName);
}

export function validateBoundaryLabelV2(
  value: unknown,
  fileName = "boundary-label-v2.json",
): asserts value is BoundaryLabelV2 {
  validate(boundaryLabelV2Validator, value, fileName);
}

export function validateKeysoundManifest(value: unknown, fileName: string): asserts value is KeysoundManifest {
  validate(keysoundValidator, value, fileName);
}
