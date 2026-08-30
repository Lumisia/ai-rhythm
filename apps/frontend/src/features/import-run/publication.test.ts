import { describe, expect, it } from "vitest";

import { derivePublication } from "./publication";

describe("derivePublication", () => {
  it("allows only a complete strict pass in production", () => {
    expect(
      derivePublication(
        {
          execution: "SUCCEEDED",
          completeness: "COMPLETE",
          quality: "PASS",
          failureCategory: "NONE",
          publishableStrict: true,
        },
        12,
        12,
        [],
      ),
    ).toEqual({
      policyVersion: "PUBLICATION_POLICY_V2",
      decision: "ALLOW_PRODUCTION",
      reasonCodes: [],
    });
  });

  it("keeps an uncalibrated boundary in playtest even when chart quality passes", () => {
    expect(
      derivePublication(
        {
          execution: "SUCCEEDED",
          completeness: "COMPLETE",
          quality: "PASS",
          failureCategory: "NONE",
          publishableStrict: true,
        },
        12,
        12,
        ["BOUNDARY_POLICY_UNCALIBRATED"],
      ),
    ).toEqual({
      policyVersion: "PUBLICATION_POLICY_V2",
      decision: "PLAYTEST_ONLY",
      reasonCodes: ["BOUNDARY_POLICY_UNCALIBRATED"],
    });
  });

  it("marks review and partial outcomes as playtest-only with literal reasons", () => {
    expect(
      derivePublication(
        {
          execution: "SUCCEEDED",
          completeness: "COMPLETE",
          quality: "REVIEW",
          failureCategory: "NONE",
          publishableStrict: false,
        },
        12,
        12,
        [],
      ).reasonCodes,
    ).toEqual(["QUALITY_REVIEW_REQUIRED", "STRICT_OUTCOME_FALSE"]);

    expect(
      derivePublication(
        {
          execution: "SUCCEEDED",
          completeness: "PARTIAL",
          quality: "PASS",
          failureCategory: "NONE",
          publishableStrict: false,
        },
        11,
        12,
        [],
      ).reasonCodes,
    ).toEqual(["INCOMPLETE_CHART_SET", "STRICT_OUTCOME_FALSE"]);
  });

  it("rejects outcome snapshots that disagree with slot counts or strictness", () => {
    expect(() =>
      derivePublication(
        {
          execution: "SUCCEEDED",
          completeness: "COMPLETE",
          quality: "PASS",
          failureCategory: "NONE",
          publishableStrict: true,
        },
        11,
        12,
        [],
      ),
    ).toThrow(/completeness/i);

    expect(() =>
      derivePublication(
        {
          execution: "SUCCEEDED",
          completeness: "COMPLETE",
          quality: "PASS",
          failureCategory: "NONE",
          publishableStrict: false,
        },
        12,
        12,
        [],
      ),
    ).toThrow(/publishableStrict/);
  });

  it("rejects failure categories that disagree with execution", () => {
    expect(() =>
      derivePublication(
        {
          execution: "SUCCEEDED",
          completeness: "COMPLETE",
          quality: "PASS",
          failureCategory: "VALIDATION",
          publishableStrict: true,
        },
        12,
        12,
        [],
      ),
    ).toThrow(/failureCategory/);

    expect(() =>
      derivePublication(
        {
          execution: "FAILED",
          completeness: "EMPTY",
          quality: "UNKNOWN",
          failureCategory: "NONE",
          publishableStrict: false,
        },
        0,
        12,
        [],
      ),
    ).toThrow(/failureCategory/);
  });
});
