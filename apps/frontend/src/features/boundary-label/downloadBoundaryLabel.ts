import type { BoundaryLabelV1, BoundaryLabelV2 } from "../../game/core/types";
import {
  boundaryLabelFileName,
  boundaryLabelFileNameV2,
  serializeBoundaryLabel,
  serializeBoundaryLabelV2,
} from "./boundaryLabel";

export function downloadBoundaryLabel(label: BoundaryLabelV1): void {
  const blob = new Blob([serializeBoundaryLabel(label)], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = boundaryLabelFileName(label);

  try {
    document.body.append(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    URL.revokeObjectURL(url);
  }
}

export function downloadBoundaryLabelV2(label: BoundaryLabelV2): void {
  const blob = new Blob([serializeBoundaryLabelV2(label)], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = boundaryLabelFileNameV2(label);

  try {
    document.body.append(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    URL.revokeObjectURL(url);
  }
}
