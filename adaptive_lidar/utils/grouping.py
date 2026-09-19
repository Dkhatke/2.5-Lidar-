"""
Vectorised grouping primitives — the one permitted way to group points.

THE PATTERN
-----------
Everywhere in this codebase that needs "do something per cell / per tile /
per voxel / per instance", the answer is:

    keys                       = pack_keys_2d(ix, iy)
    uniq, starts, order        = group_by_key(keys)
    sums                       = segment_reduce(values[order], starts, "sum")

A Python ``for`` loop over points is a bug.
A Python loop over groups that rescans all points is a bug.

``group_by_key`` deliberately returns CSR-style *offsets* rather than a list of
arrays: ``np.split`` allocates one Python object per group (~100k objects at
120k points), which costs more than the reduction it enables.  Every reduction
below runs on the flat sorted array with ``np.add.reduceat`` and friends, so
the per-group cost is zero Python-level work.

All functions are pure NumPy and allocation-conscious.  No loops over points.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Coordinate bias so negative grid indices pack into unsigned bit fields and
# still sort in the correct order.  21 bits per axis → ±1,048,576 cells, which
# at 5 cm is ±52 km.  Comfortably beyond any LiDAR range.
_BITS = 21
_BIAS = 1 << (_BITS - 1)          # 1,048,576
_MASK = (1 << _BITS) - 1


# ────────────────────────────────────────────────────────────────
# Key packing
# ────────────────────────────────────────────────────────────────
def pack_keys_2d(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Pack two integer grid indices into one int64 key.

    Layout: ``[ ix+bias : 21 bits ][ iy+bias : 21 bits ]``.
    Keys are unique per (ix, iy) and monotone in ix then iy, so sorting the
    keys is equivalent to a row-major sort of the grid.
    """
    a = (np.asarray(ix, dtype=np.int64) + _BIAS) & _MASK
    b = (np.asarray(iy, dtype=np.int64) + _BIAS) & _MASK
    return (a << _BITS) | b


def unpack_keys_2d(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_keys_2d`."""
    k = np.asarray(keys, dtype=np.int64)
    ix = ((k >> _BITS) & _MASK) - _BIAS
    iy = (k & _MASK) - _BIAS
    return ix.astype(np.int32), iy.astype(np.int32)


def pack_keys_3d(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """Pack three integer grid indices into one int64 key.

    Layout: ``[ ix : 21 ][ iy : 21 ][ iz : 21 ]`` — 63 bits, fits int64.
    """
    a = (np.asarray(ix, dtype=np.int64) + _BIAS) & _MASK
    b = (np.asarray(iy, dtype=np.int64) + _BIAS) & _MASK
    c = (np.asarray(iz, dtype=np.int64) + _BIAS) & _MASK
    return (a << (2 * _BITS)) | (b << _BITS) | c


def unpack_keys_3d(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_keys_3d`."""
    k = np.asarray(keys, dtype=np.int64)
    ix = ((k >> (2 * _BITS)) & _MASK) - _BIAS
    iy = ((k >> _BITS) & _MASK) - _BIAS
    iz = (k & _MASK) - _BIAS
    return ix.astype(np.int32), iy.astype(np.int32), iz.astype(np.int32)


# ────────────────────────────────────────────────────────────────
# 2D Morton code — used by the adaptive map (Phase 4)
# ────────────────────────────────────────────────────────────────
def _part1by1(x: np.ndarray) -> np.ndarray:
    """Interleave a 32-bit integer's bits with zeros (bit spreading)."""
    x = x.astype(np.int64) & 0xFFFFFFFF
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    return x


def _compact1by1(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_part1by1`."""
    x = x.astype(np.int64) & 0x5555555555555555
    x = (x | (x >> 1)) & 0x3333333333333333
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFF
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF
    return x


def morton_2d(ix: np.ndarray, iy: np.ndarray, level: int = 0) -> np.ndarray:
    """2D Morton (Z-order) code from signed grid indices *at that level*.

    ``ix``/``iy`` are the grid indices at the given level (i.e. already
    ``floor(x / cell_size(level))``).  Indices are biased into the
    non-negative range first — by ``_BIAS >> level``, so that the bias itself
    coarsens along with the grid — which is what makes the shift identity hold
    for negative coordinates too.

    THE IDENTITY THAT MAKES THE MAP ALIGNMENT-SAFE::

        morton_2d(ix0 >> l, iy0 >> l, level=l) == morton_2d(ix0, iy0) >> (2 * l)

    A level-l cell therefore contains exactly the 4^l level-0 cells whose codes
    share its high bits.  No float boundary case, no point in two cells.
    """
    bias = _BIAS >> level
    a = (np.asarray(ix, dtype=np.int64) + bias) & _MASK
    b = (np.asarray(iy, dtype=np.int64) + bias) & _MASK
    return (_part1by1(a) << 1) | _part1by1(b)


def morton_2d_inverse(code: np.ndarray, level: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Recover (ix, iy) at ``level`` from a level-``level`` Morton code."""
    c = np.asarray(code, dtype=np.int64)
    ix = _compact1by1(c >> 1) - (_BIAS >> level)
    iy = _compact1by1(c) - (_BIAS >> level)
    return ix.astype(np.int32), iy.astype(np.int32)


# ────────────────────────────────────────────────────────────────
# Grouping
# ────────────────────────────────────────────────────────────────
def group_by_key(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group elements by integer key.

    Parameters
    ----------
    keys : (N,) integer array

    Returns
    -------
    unique_keys : (M,) — sorted distinct keys
    group_starts: (M+1,) int64 — CSR offsets into the sorted order.
                  Group *j* is ``order[group_starts[j] : group_starts[j+1]]``.
                  The trailing element is N, so ``np.diff`` gives group sizes
                  directly and ``reduceat`` can use ``group_starts[:-1]``.
    order       : (N,) int64 — indices that sort ``keys``
    """
    keys = np.asarray(keys)
    n = keys.shape[0]
    if n == 0:
        return (np.empty(0, dtype=keys.dtype),
                np.zeros(1, dtype=np.int64),
                np.empty(0, dtype=np.int64))

    # Introsort, not the mergesort that kind="stable" selects for int64.
    # Grouping only needs the keys adjacent, not ties kept in input order, and
    # introsort is markedly faster and allocates no scratch buffer.
    order = np.argsort(keys, kind="quicksort")
    sorted_keys = keys[order]
    # Run boundaries — the one permitted grouping idiom.
    first = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    unique_keys = sorted_keys[first]
    group_starts = np.r_[first, n].astype(np.int64)
    return unique_keys, group_starts, order.astype(np.int64)


def split_groups(order: np.ndarray, group_starts: np.ndarray) -> list:
    """Materialise the groups as a list of index arrays.

    Provided for the rare consumer that genuinely wants Python-level objects
    (e.g. building the ``Tile.point_indices`` list once per frame).  Never call
    this inside a per-point hot path — it allocates one object per group.
    """
    return np.split(order, group_starts[1:-1])


def group_sizes(group_starts: np.ndarray) -> np.ndarray:
    """Number of elements in each group."""
    return np.diff(group_starts).astype(np.int64)


def group_ids(group_starts: np.ndarray, n: int) -> np.ndarray:
    """Dense group index for every element of the *sorted* array.

    ``group_ids(...)[i]`` is the group that sorted element *i* belongs to.
    Useful for scatter-style ops (``np.add.at``) and for mapping a per-group
    result back onto points.
    """
    out = np.zeros(n, dtype=np.int64)
    if len(group_starts) > 2:
        # Mark each group start (except the first) then cumulative-sum.
        out[group_starts[1:-1]] = 1
        np.cumsum(out, out=out)
    return out


def scatter_to_points(
    per_group: np.ndarray,
    group_starts: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    """Broadcast a per-group value back to every point, in original order."""
    n = order.shape[0]
    gid = group_ids(group_starts, n)
    out = np.empty((n,) + per_group.shape[1:], dtype=per_group.dtype)
    out[order] = per_group[gid]
    return out


def merge_sorted_unique(existing: np.ndarray, incoming: np.ndarray):
    """Insert ``incoming`` keys into the sorted unique array ``existing``.

    Returns ``(merged, insert_positions, n_new)``.

    ``np.union1d`` would re-sort the concatenation of both arrays every frame,
    which for a map that keeps hundreds of thousands of cells is the single
    most expensive thing the insert path does — and almost all of that work is
    re-sorting keys that were already sorted last frame.  Here ``searchsorted``
    finds the few genuinely new keys and ``np.insert`` splices them in, which
    is one linear copy and no sort at all.
    """
    existing = np.asarray(existing, dtype=np.int64)
    incoming = np.asarray(incoming, dtype=np.int64)
    if len(existing) == 0:
        # Splice slots are computed as insert_at + arange(n_new), so an empty
        # base must report every insertion at position 0.
        return incoming, np.zeros(len(incoming), dtype=np.int64), len(incoming)
    if len(incoming) == 0:
        return existing, np.empty(0, np.int64), 0
    pos = np.searchsorted(existing, incoming)
    posc = np.clip(pos, 0, len(existing) - 1)
    present = existing[posc] == incoming
    absent = ~present
    if not np.any(absent):
        return existing, np.empty(0, np.int64), 0
    merged = np.insert(existing, pos[absent], incoming[absent])
    return merged, pos[absent], int(absent.sum())


# ────────────────────────────────────────────────────────────────
# Segment reductions
# ────────────────────────────────────────────────────────────────
_UFUNCS = {
    "sum": np.add,
    "max": np.maximum,
    "min": np.minimum,
}


def segment_reduce(
    values: np.ndarray,
    group_starts: np.ndarray,
    op: str = "sum",
    percentile: float | None = None,
) -> np.ndarray:
    """Reduce ``values`` within each group.

    ``values`` must already be in sorted (grouped) order — i.e. ``values[order]``
    from :func:`group_by_key`.  Works on 1-D arrays and on (N, K) arrays, in
    which case the reduction is applied column-wise and the result is (M, K).

    op : "sum" | "mean" | "min" | "max" | "count" | "var" | "percentile" | "any"

    ``percentile`` selects the q-th percentile within each group (0-100).  It
    is implemented as a sort-within-group plus a gather, never a Python loop:
    a lexicographic sort by (group, value) makes each group's k-th order
    statistic a single indexed read.
    """
    values = np.asarray(values)
    n = values.shape[0]
    m = len(group_starts) - 1
    if m <= 0:
        tail = values.shape[1:]
        return np.zeros((0,) + tail, dtype=np.float32)

    starts = group_starts[:-1]
    sizes = group_sizes(group_starts)

    if op == "count":
        return sizes

    if op == "any":
        s = np.add.reduceat(values.astype(np.int64), starts, axis=0)
        return s > 0

    if op in ("sum", "min", "max"):
        return _UFUNCS[op].reduceat(values, starts, axis=0)

    if op == "mean":
        s = np.add.reduceat(values.astype(np.float64), starts, axis=0)
        shape = (m,) + (1,) * (values.ndim - 1)
        return (s / np.maximum(sizes, 1).reshape(shape)).astype(np.float32)

    if op == "var":
        f = values.astype(np.float64)
        s = np.add.reduceat(f, starts, axis=0)
        s2 = np.add.reduceat(f * f, starts, axis=0)
        shape = (m,) + (1,) * (values.ndim - 1)
        cnt = np.maximum(sizes, 1).reshape(shape)
        mean = s / cnt
        return np.maximum(s2 / cnt - mean * mean, 0.0).astype(np.float32)

    if op == "percentile":
        if percentile is None:
            raise ValueError("op='percentile' requires percentile=")
        if values.ndim != 1:
            raise ValueError("percentile reduction expects a 1-D value array")
        return _segment_percentile(values, group_starts, percentile)

    raise ValueError(f"unknown reduction op: {op!r}")


def segment_sort(values: np.ndarray, group_starts: np.ndarray) -> np.ndarray:
    """Sort ``values`` ascending WITHIN each group, leaving groups in place.

    One ``np.argsort`` on a composite key rather than ``np.lexsort``: the group
    ids are already sorted and contiguous, so shifting each group's values into
    a disjoint numeric band makes a single sort equivalent to a lexicographic
    one, at roughly half the cost.  Percentiles, medians and order statistics
    then all come from gathers into this array — sort once, reduce many times.
    """
    n = values.shape[0]
    m = len(group_starts) - 1
    if n == 0 or m == 0:
        return values.copy()
    gid = group_ids(group_starts, n)
    v = values.astype(np.float64, copy=False)
    lo = float(np.min(v)) if n else 0.0
    hi = float(np.max(v)) if n else 1.0
    span = max(hi - lo, 1e-9)
    # Normalise into [0, 1) then offset by the group index: comparisons between
    # groups are then decided by the integer part alone.
    key = gid.astype(np.float64) + (v - lo) / (span * (1.0 + 1e-9))
    return values[np.argsort(key, kind="quicksort")]


def segment_percentile_sorted(
    sorted_vals: np.ndarray,
    group_starts: np.ndarray,
    q,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    """Nearest-rank percentile(s) from an already within-group-sorted array.

    ``q`` may be a scalar or a sequence, in which case the result is (M, len(q))
    — several order statistics for the price of the one sort that produced
    ``sorted_vals``.

    ``counts`` overrides the group sizes used to compute the rank.  This is how
    a percentile over a *prefix* of each group is taken: when the values are
    sorted ascending, "the elements below a threshold" are exactly a prefix, so
    passing the per-group count of that prefix yields the percentile within it
    without a second sort or a second array.
    """
    n = sorted_vals.shape[0]
    m = len(group_starts) - 1
    scalar = np.isscalar(q)
    qs = np.atleast_1d(np.asarray(q, np.float64))
    if n == 0 or m == 0:
        out = np.zeros((m, len(qs)), np.float32)
        return out[:, 0] if scalar else out

    starts = group_starts[:-1]
    sizes = group_sizes(group_starts) if counts is None \
        else np.minimum(np.asarray(counts, np.int64), group_sizes(group_starts))
    sizes = np.maximum(sizes, 0)

    out = np.zeros((m, len(qs)), np.float32)
    for k, qq in enumerate(qs):
        rank = np.floor(np.clip(qq, 0.0, 100.0) / 100.0 * (sizes - 1) + 0.5)
        rank = np.clip(rank, 0, np.maximum(sizes - 1, 0)).astype(np.int64)
        take = np.clip(starts + rank, 0, n - 1)
        out[:, k] = np.where(sizes > 0, sorted_vals[take], 0.0)
    return out[:, 0] if scalar else out


def _segment_percentile(values, group_starts, q):
    """Convenience: sort within groups, then take one percentile."""
    return segment_percentile_sorted(
        segment_sort(values, group_starts), group_starts, q)


def segment_argmax(
    values: np.ndarray,
    group_starts: np.ndarray,
) -> np.ndarray:
    """Index (into the sorted array) of the maximum element of each group."""
    n = values.shape[0]
    m = len(group_starts) - 1
    if n == 0 or m == 0:
        return np.zeros(m, dtype=np.int64)
    gid = group_ids(group_starts, n)
    order = np.lexsort((values, gid))
    # Last element of each group in this ordering is its maximum.
    ends = group_starts[1:] - 1
    return order[np.clip(ends, 0, n - 1)]


def segment_bincount(
    labels: np.ndarray,
    group_starts: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """Per-group histogram of small non-negative integer labels → (M, n_labels).

    One ``np.bincount`` over a combined (group, label) key.  No loop.
    """
    n = labels.shape[0]
    m = len(group_starts) - 1
    if n == 0 or m == 0:
        return np.zeros((m, n_labels), dtype=np.int32)
    gid = group_ids(group_starts, n)
    lab = np.clip(labels.astype(np.int64), 0, n_labels - 1)
    flat = np.bincount(gid * n_labels + lab, minlength=m * n_labels)
    return flat[: m * n_labels].reshape(m, n_labels).astype(np.int32)
