"""Deterministic subword-token to semantic-region alignment."""

from __future__ import annotations

from src.regions.extract import Region


def byte_to_character_offsets(source: str) -> dict[int, int]:
    """Map every UTF-8 byte boundary in source to its character offset."""
    result = {0: 0}
    byte_offset = 0
    for character_offset, character in enumerate(source, start=1):
        byte_offset += len(character.encode("utf-8"))
        result[byte_offset] = character_offset
    return result


def align_token_offsets(
    source: str,
    offsets: list[tuple[int, int]],
    regions: tuple[Region, ...],
) -> list[int | None]:
    """Assign tokens to the smallest meaningful enclosing region or UNMAPPED."""
    boundary = byte_to_character_offsets(source)
    spans = [
        (boundary[region.start_byte], boundary[region.end_byte], region.region_id)
        for region in regions
    ]
    assignments: list[int | None] = []
    for start, end in offsets:
        if start == end:
            assignments.append(None)
            continue
        candidates = [
            (region_end - region_start, region_id)
            for region_start, region_end, region_id in spans
            if region_start <= start and end <= region_end
        ]
        assignments.append(min(candidates)[1] if candidates else None)
    return assignments


def align_prompt_offsets(
    source: str,
    prompt_offsets: list[tuple[int, int]],
    source_span: tuple[int, int],
    regions: tuple[Region, ...],
) -> list[tuple[int, int | None]]:
    """Align only tokens contained in the source span of a serialized prompt."""
    source_start, source_end = source_span
    source_token_indexes: list[int] = []
    relative_offsets: list[tuple[int, int]] = []
    for token_index, (start, end) in enumerate(prompt_offsets):
        if start < source_start or end > source_end or start == end:
            continue
        source_token_indexes.append(token_index)
        relative_offsets.append((start - source_start, end - source_start))
    assignments = align_token_offsets(source, relative_offsets, regions)
    return list(zip(source_token_indexes, assignments, strict=False))
