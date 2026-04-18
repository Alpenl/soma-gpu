from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegisteredMocapSegment:
    segment_id: str
    basename_suffix: str
    start_fidx: int
    end_fidx: int
    category: str
    description: str
    mocap_path_suffix: tuple[str, ...]

    def matches_mocap(self, mocap_fname):
        mocap_path = Path(str(mocap_fname))
        parts = mocap_path.as_posix().split("/")
        suffix_len = len(self.mocap_path_suffix)
        if len(parts) < suffix_len:
            return False
        return tuple(parts[-suffix_len:]) == self.mocap_path_suffix


REAL_MCP_SEGMENTS = {
    "wolf001-stable-hold-300f": RegisteredMocapSegment(
        segment_id="wolf001-stable-hold-300f",
        basename_suffix="wolf001_stable_hold_300f",
        start_fidx=12720,
        end_fidx=13020,
        category="stable",
        description="Fully visible low-motion hold window used as the long-sequence stability anchor.",
        mocap_path_suffix=("input", "wolf001", "4090-haonan-73.mcp"),
    ),
    "wolf001-fast-turn-300f": RegisteredMocapSegment(
        segment_id="wolf001-fast-turn-300f",
        basename_suffix="wolf001_fast_turn_300f",
        start_fidx=17100,
        end_fidx=17400,
        category="fast_turn",
        description="High-motion turn window with large centroid and foot-marker velocity.",
        mocap_path_suffix=("input", "wolf001", "4090-haonan-73.mcp"),
    ),
    "wolf001-dirty-recovery-300f": RegisteredMocapSegment(
        segment_id="wolf001-dirty-recovery-300f",
        basename_suffix="wolf001_dirty_recovery_300f",
        start_fidx=15300,
        end_fidx=15600,
        category="dirty_recovery",
        description="Moderate-motion window with localized marker spikes consistent with dirty data or recovery.",
        mocap_path_suffix=("input", "wolf001", "4090-haonan-73.mcp"),
    ),
}


def apply_segment_overrides(overrides, *, segment_id=None, mocap_fname=None):
    updated = dict(overrides)
    if segment_id is None:
        return updated

    segment = REAL_MCP_SEGMENTS[segment_id]
    if not segment.matches_mocap(mocap_fname):
        suffix = "/".join(segment.mocap_path_suffix)
        raise ValueError(
            f"--segment-id {segment_id} is only registered for mocap paths ending with {suffix}"
        )

    if "mocap.start_fidx" in updated or "mocap.end_fidx" in updated:
        raise ValueError("--segment-id cannot be combined with --cfg mocap.start_fidx/mocap.end_fidx")

    updated["mocap.start_fidx"] = str(segment.start_fidx)
    updated["mocap.end_fidx"] = str(segment.end_fidx)

    basename = str(updated.get("mocap.basename", Path(str(mocap_fname)).stem))
    suffix = segment.basename_suffix
    segment_basename = basename if basename.endswith(f"_{suffix}") else f"{basename}_{suffix}"
    updated["mocap.basename"] = segment_basename
    return updated
