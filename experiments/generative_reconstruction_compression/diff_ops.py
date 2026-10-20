"""Pure helpers for the .grczip opcode stream.

``simple_grc`` writes this format. Nothing in this module loads a model, so
unit tests can exercise coalescing and the 16-bit count field directly.
"""

from __future__ import annotations

MAX_RUN = (1 << 16) - 1
LEN_BITS = 16
BLOB_BITS = 16
SEED_BITS = 32
MIN_EQUAL_TOKENS = 12

OPCODE_EQUAL = (0, 0)
OPCODE_DELETE = (1, 0)
OPCODE_INSERT = (1, 1)
OPCODE_EOS = (0, 1)


def write_uint(bitout, value: int, num_bits: int) -> None:
    """Write ``value`` as a big-endian unsigned integer of ``num_bits``."""
    if num_bits < 0:
        raise ValueError("num_bits must be non-negative")
    if value < 0 or value.bit_length() > num_bits:
        raise ValueError(f"{value} does not fit in {num_bits} bits")
    for i in range(num_bits - 1, -1, -1):
        bitout.write((value >> i) & 1)


def write_count_opcode(bitout, opcode: tuple[int, int], count: int) -> None:
    """Write an EQUAL or DELETE opcode, splitting runs that exceed 16 bits.

    The on-wire count is 16 bits. A longer span is emitted as several opcodes
    whose counts sum to ``count``. A zero count is preserved as one opcode so
    callers that intentionally write an empty span stay explicit.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    remaining = count
    while True:
        chunk = remaining if remaining <= MAX_RUN else MAX_RUN
        for bit in opcode:
            bitout.write(bit)
        write_uint(bitout, chunk, LEN_BITS)
        remaining -= chunk
        if remaining <= 0:
            break


def coalesce_opcodes(
    opcodes: list,
    ref_tokens: list,
    tgt_tokens: list,
    min_equal: int = MIN_EQUAL_TOKENS,
) -> list:
    """Fold short EQUAL runs that sit between edits into one REPLACE span.

    ``ref_tokens`` and ``tgt_tokens`` are accepted so callers can pass the same
    arguments they use for encoding. The merge itself only needs the opcode
    index ranges, which difflib already aligns with those sequences.

    An EQUAL shorter than ``min_equal`` tokens between two edits is absorbed
    into a single REPLACE covering the combined range. Adjacent REPLACE
    opcodes are then merged.
    """
    del ref_tokens, tgt_tokens
    if not opcodes:
        return opcodes

    merged = list(opcodes)
    changed = True

    while changed:
        changed = False
        new = []
        i = 0
        while i < len(merged):
            if (
                i + 2 < len(merged)
                and merged[i][0] in ("insert", "replace", "delete")
                and merged[i + 1][0] == "equal"
                and merged[i + 2][0] in ("insert", "replace", "delete")
                and (merged[i + 1][2] - merged[i + 1][1]) < min_equal
            ):
                a = merged[i]
                b = merged[i + 2]
                new.append(("replace", a[1], b[2], a[3], b[4]))
                i += 3
                changed = True
            else:
                new.append(merged[i])
                i += 1
        merged = new

    final: list = []
    for op in merged:
        if (
            final
            and final[-1][0] == "replace"
            and op[0] == "replace"
            and final[-1][2] == op[1]
            and final[-1][4] == op[3]
        ):
            prev = final.pop()
            final.append(("replace", prev[1], op[2], prev[3], op[4]))
        else:
            final.append(op)

    return final
