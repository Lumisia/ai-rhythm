import type {
  CompletenessStatus,
  OutcomeStatusSnapshot,
  PublicationDecisionName,
  PublicationDecisionSnapshot,
  PublicationReasonCode,
  PublicationStrictBlocker,
} from "../../game/core/types";

export const PUBLICATION_POLICY_VERSION = "PUBLICATION_POLICY_V2" as const;

function completenessForSlots(publishedSlots: number, expectedSlots: number): CompletenessStatus {
  if (!Number.isInteger(expectedSlots) || expectedSlots <= 0) {
    throw new Error("expectedSlots must be a positive integer");
  }
  if (
    !Number.isInteger(publishedSlots) ||
    publishedSlots < 0 ||
    publishedSlots > expectedSlots
  ) {
    throw new Error("publishedSlots must be an integer within expectedSlots");
  }
  if (publishedSlots === 0) return "EMPTY";
  if (publishedSlots === expectedSlots) return "COMPLETE";
  return "PARTIAL";
}

export function derivePublication(
  outcome: OutcomeStatusSnapshot,
  publishedSlots: number,
  expectedSlots: number,
  strictBlockers: readonly PublicationStrictBlocker[],
): PublicationDecisionSnapshot {
  const actualCompleteness = completenessForSlots(publishedSlots, expectedSlots);
  if (outcome.completeness !== actualCompleteness) {
    throw new Error("outcome completeness disagrees with published slot count");
  }
  if ((outcome.execution === "SUCCEEDED") !== (outcome.failureCategory === "NONE")) {
    throw new Error("outcome failureCategory disagrees with execution");
  }

  const expectedStrict =
    outcome.execution === "SUCCEEDED" &&
    outcome.completeness === "COMPLETE" &&
    outcome.quality === "PASS";
  if (outcome.publishableStrict !== expectedStrict) {
    throw new Error("outcome publishableStrict is internally inconsistent");
  }
  const canonicalBlockers = [...new Set(strictBlockers)].sort();
  if (
    canonicalBlockers.length !== strictBlockers.length ||
    canonicalBlockers.some((blocker, index) => blocker !== strictBlockers[index])
  ) {
    throw new Error("publication strict blockers must be sorted and unique");
  }

  const reasonCodes: PublicationReasonCode[] = [...strictBlockers];
  if (outcome.execution === "FAILED") reasonCodes.push("EXECUTION_FAILED");
  if (outcome.completeness !== "COMPLETE") reasonCodes.push("INCOMPLETE_CHART_SET");
  if (outcome.quality === "REVIEW") {
    reasonCodes.push("QUALITY_REVIEW_REQUIRED");
  } else if (outcome.quality === "REJECTED") {
    reasonCodes.push("QUALITY_REJECTED");
  } else if (outcome.quality === "UNKNOWN") {
    reasonCodes.push("QUALITY_UNKNOWN");
  }
  if (!outcome.publishableStrict) reasonCodes.push("STRICT_OUTCOME_FALSE");
  reasonCodes.sort();

  let decision: PublicationDecisionName;
  if (reasonCodes.length === 0) {
    decision = "ALLOW_PRODUCTION";
  } else if (
    outcome.execution === "FAILED" ||
    outcome.quality === "REJECTED" ||
    outcome.quality === "UNKNOWN"
  ) {
    decision = "REJECTED";
  } else {
    decision = "PLAYTEST_ONLY";
  }

  return {
    policyVersion: PUBLICATION_POLICY_VERSION,
    decision,
    reasonCodes,
  };
}
