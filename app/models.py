from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProfileSection(str, Enum):
    SCAN = "scan"
    VOICE = "voice"
    OTHER = "other"


class ScanAsset(BaseModel):
    id: str
    path: str
    caption: str = ""
    quality_score: float = Field(ge=0, le=1, default=1.0)
    yaw_deg: float | None = None  # measured head turn when the photo was taken (+ = toward screen right)
    pitch_deg: float | None = None  # + = looking up
    tracked: bool = True  # False when the angle was estimated because the face was out of view (profile)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VoicePhase(BaseModel):
    id: str
    label: str
    path: str
    waveform_peaks: list[float] = Field(default_factory=list)
    duration_sec: float = 0.0
    quality_score: float = Field(ge=0, le=1, default=1.0)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OtherData(BaseModel):
    display_name: str = ""
    notes: str = ""
    preferences: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserProfile(BaseModel):
    canonical_name: str
    display_name: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scan_complete: bool = False
    # The person chose "no face to scan" (a character, a pet, a voice-only avatar): the scan section counts as done.
    scan_skipped: bool = False
    voice_complete: bool = False
    other_complete: bool = False
    scan_assets: list[ScanAsset] = Field(default_factory=list)
    voice_phases: list[VoicePhase] = Field(default_factory=list)
    other: OtherData = Field(default_factory=OtherData)
    avatar_ready: bool = False
    saved: bool = False

    def section_complete(self, section: ProfileSection) -> bool:
        from app.profile_logic import MIN_FACE_ANGLES, face_assets

        if section == ProfileSection.SCAN:
            return self.scan_skipped or len(face_assets(self)) >= MIN_FACE_ANGLES
        if section == ProfileSection.VOICE:
            return any((p.path or "").strip() for p in self.voice_phases)
        return bool(
            (self.other.notes or "").strip()
            or (
                (self.other.display_name or "").strip()
                and self.other.display_name.strip() != self.canonical_name.strip()
            )
        )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)