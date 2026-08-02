"""keysound-manifest-v1 계약."""

from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from chart_worker.schema.chart import CamelModel


class KeysoundManifest(CamelModel):
    schema_version: Literal[1] = 1
    song_version_id: UUID
    bgm_asset_id: UUID
    keys_asset_id: UUID
    slice_sec: float = Field(default=0.30, gt=0)
    preroll_sec: float = Field(default=0.012, ge=0)
    snap_window_ms: int = Field(default=50, ge=0)
    drum_onsets: list[int]

    @model_validator(mode="after")
    def _check_onsets(self) -> Self:
        if self.drum_onsets != sorted(set(self.drum_onsets)):
            raise ValueError("drumOnsets must be sorted without duplicates")
        if self.drum_onsets and self.drum_onsets[0] < 0:
            raise ValueError("drumOnsets must be non-negative")
        return self
