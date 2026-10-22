"""SALAD loop retrieval for long-sequence reconstruction."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from long_reconstruction.runner import ChunkRange, LoopPair


@dataclass(frozen=True)
class SaladRetrievalConfig:
    salad_checkpoint: Path
    dinov2_checkpoint: Path
    repo_root: Path = Path(__file__).resolve().parents[1]
    image_size: tuple[int, int] = (336, 336)
    batch_size: int = 32
    similarity_threshold: float = 0.85
    top_k: int = 5
    use_nms: bool = True
    nms_threshold: int = 25
    descriptor_cache_root: Path | None = None


def frame_to_chunk(frame: int, ranges: Sequence[ChunkRange]) -> int:
    """Match original ``find_chunk_index`` semantics: latest eligible chunk wins."""
    for index in range(len(ranges) - 1, -1, -1):
        item = ranges[index]
        if item.start <= frame < item.end:
            return index
    raise ValueError(f"frame {frame} is outside all chunk ranges")


def map_retrievals_to_loop_pairs(
    retrievals: Sequence[Sequence[Any]], ranges: Sequence[ChunkRange]
) -> list[LoopPair]:
    """Convert frame retrievals to unique cross-chunk loop constraints.

    SALAD returns candidates in descending similarity order. Keep the first
    candidate for each chunk pair to select its highest-similarity match.
    """
    pairs: list[LoopPair] = []
    seen: set[tuple[int, int]] = set()
    for item in retrievals:
        frame_a, frame_b = int(item[0]), int(item[1])
        chunk_a = frame_to_chunk(frame_a, ranges)
        chunk_b = frame_to_chunk(frame_b, ranges)
        if chunk_a == chunk_b:
            continue
        key = (chunk_a, chunk_b)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            LoopPair(
                chunk_a=chunk_a,
                chunk_b=chunk_b,
                frame_a=frame_a,
                frame_b=frame_b,
                similarity=float(item[2]) if len(item) >= 3 else None,
            )
        )
    return pairs


def retrieve_salad_loop_pairs(
    image_dir: str | Path,
    ranges: Sequence[ChunkRange],
    config: SaladRetrievalConfig,
    output_dir: str | Path,
) -> tuple[list[LoopPair], list[tuple[Any, ...]]]:
    """Run the exact upstream SALAD detector and map its frame pairs to chunks."""
    repo_root = config.repo_root.resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"bundled SALAD source root not found: {repo_root}")
    for path in (config.salad_checkpoint, config.dinov2_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"loop retrieval checkpoint not found: {path}")

    sys.path.insert(0, str(repo_root))
    try:
        from loop_models.loop_model import LoopDetector
    finally:
        if sys.path[0] == str(repo_root):
            sys.path.pop(0)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    salad_config: dict[str, Any] = {
        "Weights": {
            "SALAD": str(config.salad_checkpoint),
            "DNIO": str(config.dinov2_checkpoint),
        },
        "Loop": {
            "SALAD": {
                "image_size": list(config.image_size),
                "batch_size": config.batch_size,
                "similarity_threshold": config.similarity_threshold,
                "top_k": config.top_k,
                "use_nms": config.use_nms,
                "nms_threshold": config.nms_threshold,
                "cache": {
                    "enable": config.descriptor_cache_root is not None,
                    "root": None if config.descriptor_cache_root is None else str(config.descriptor_cache_root),
                },
            }
        },
    }
    detector = LoopDetector(
        image_dir=str(image_dir),
        output=str(output_root / "loop_closures.txt"),
        config=salad_config,
    )
    detector.run()
    retrievals = detector.get_loop_list(include_similarity=True, include_metadata=True)
    return map_retrievals_to_loop_pairs(retrievals, ranges), retrievals


__all__ = [
    "SaladRetrievalConfig",
    "frame_to_chunk",
    "map_retrievals_to_loop_pairs",
    "retrieve_salad_loop_pairs",
]
