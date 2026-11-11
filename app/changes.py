"""Streaming change-patch detection between two versions of a cloud mask.

Two single-band GeoTIFF cloud masks (``0`` = cloudy, ``1`` = cloud-free) are
compared on their shared pixel grid.  Only pixels that are valid in *both*
rasters and whose values differ are considered changes.  Pixels with the same
change direction that are 4-connected (up/down/left/right) form one patch,
including patches that continue across read bands.

The rasters are read in horizontal row bands so memory use stays independent
of the raster height.  After each band, only the patches touching the band's
bottom row are kept for merging with the next band; every other patch is
finalized.  The minimum-size filter is applied once, after finalization.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from app.analysis import (
    IncompatibleFile,
    RasterMeta,
    UploadedRaster,
    _display_crs,
    _fmt_num,
    _inspect,
    _pixel_offset,
    _same_resolution,
    _valid_data,
)

# Direction codes stored in the per-band change array.
CLEAR_TO_CLOUD = 1  # old mask 1 (cloud-free) -> new mask 0 (cloudy)
CLOUD_TO_CLEAR = 2  # old mask 0 (cloudy)     -> new mask 1 (cloud-free)

DEFAULT_BAND_HEIGHT = 1024

DIRECTION_LABELS = {
    CLEAR_TO_CLOUD: "无云变有云（1→0）",
    CLOUD_TO_CLEAR: "有云变无云（0→1）",
}


@dataclass(frozen=True)
class ChangeRegion:
    """A finalized connected patch, in raster A's pixel/coordinate space."""

    direction: int
    pixel_count: int
    min_row: int
    max_row: int  # inclusive
    min_col: int
    max_col: int  # inclusive


@dataclass(frozen=True)
class ChangeDetectionResult:
    regions: list[ChangeRegion]
    overlap_pixels: int  # pixels valid in both rasters inside the overlap
    crs: str | None
    transform_a: Affine  # old-mask affine transform, for coordinate bounds


@dataclass(frozen=True)
class _Overlap:
    """Intersection geometry, expressed as windows into each raster."""

    col_off_a: int
    row_off_a: int
    col_off_b: int
    row_off_b: int
    width: int
    height: int


def _overlap(meta_a: RasterMeta, meta_b: RasterMeta) -> _Overlap | None:
    """Return the common window of two grid-aligned rasters, or ``None``."""

    ta = meta_a.transform
    tb = meta_b.transform
    # Pixel offset of raster B's origin relative to raster A's origin.  The
    # grid validation guarantees these are integral numbers of pixels.  With
    # north-up transforms e is negative, so the row offset divides by e.
    col_offset = int(round((tb.c - ta.c) / ta.a))
    row_offset = int(round((tb.f - ta.f) / ta.e))

    col0 = max(0, col_offset)
    row0 = max(0, row_offset)
    col1 = min(meta_a.width, col_offset + meta_b.width)
    row1 = min(meta_a.height, row_offset + meta_b.height)
    if col1 <= col0 or row1 <= row0:
        return None
    return _Overlap(
        col_off_a=col0,
        row_off_a=row0,
        col_off_b=col0 - col_offset,
        row_off_b=row0 - row_offset,
        width=col1 - col0,
        height=row1 - row0,
    )


def validate_mask_pair(
    mask_a: UploadedRaster, mask_b: UploadedRaster
) -> tuple[RasterMeta | None, RasterMeta | None, list[IncompatibleFile]]:
    """Validate structure and grid compatibility of the two mask versions.

    Reuses the parcel-service rules: single-band north-up GeoTIFF, identical
    CRS and resolution, origins aligned on an integral pixel grid.
    """

    errors: list[IncompatibleFile] = []
    meta_a, error_a = _inspect(mask_a)
    meta_b, error_b = _inspect(mask_b)
    if error_a is not None:
        errors.append(error_a)
    if error_b is not None:
        errors.append(error_b)
    if meta_a is None or meta_b is None or errors:
        return meta_a, meta_b, errors

    reasons: list[str] = []
    if meta_a.crs != meta_b.crs:
        reasons.append(
            "两景掩膜坐标系不一致"
            f"（{_display_crs(meta_a.crs)} vs {_display_crs(meta_b.crs)}）"
        )
    if not _same_resolution(meta_a.transform.a, meta_b.transform.a) or not (
        _same_resolution(meta_a.transform.e, meta_b.transform.e)
    ):
        reasons.append(
            "两景掩膜分辨率不一致"
            f"（{_fmt_num(abs(meta_a.transform.a))}×{_fmt_num(abs(meta_a.transform.e))}"
            f" vs {_fmt_num(abs(meta_b.transform.a))}×{_fmt_num(abs(meta_b.transform.e))}）"
        )
    else:
        ox = _pixel_offset(meta_b.transform.c - meta_a.transform.c, meta_a.transform.a)
        oy = _pixel_offset(meta_b.transform.f - meta_a.transform.f, meta_a.transform.e)
        if ox is None or oy is None:
            qx = (meta_b.transform.c - meta_a.transform.c) / meta_a.transform.a
            qy = (meta_b.transform.f - meta_a.transform.f) / meta_a.transform.e
            reasons.append(
                "两景掩膜原点未对齐到整数像元"
                f"（偏移 {qx:.6f}, {qy:.6f} 像元）"
            )
    if reasons:
        errors.append(
            IncompatibleFile(
                filename=mask_b.filename,
                role=mask_b.role,
                reason="；".join(reasons),
            )
        )
    return meta_a, meta_b, errors


def _runs(row: np.ndarray, code: int) -> list[tuple[int, int]]:
    """Return inclusive (start_col, end_col) runs of ``code`` in one row."""

    idx = np.flatnonzero(row == code)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate((idx[:1], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], idx[-1:]))
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


class _UnionFind:
    """Union-find with per-root pixel counts, bounding boxes and direction."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.direction: list[int] = []
        self.size: list[int] = []
        self.min_row: list[int] = []
        self.max_row: list[int] = []
        self.min_col: list[int] = []
        self.max_col: list[int] = []

    def new_label(
        self, direction: int, row: int, col0: int, col1: int
    ) -> int:
        label = len(self.parent)
        self.parent.append(label)
        self.direction.append(direction)
        self.size.append(col1 - col0 + 1)
        self.min_row.append(row)
        self.max_row.append(row)
        self.min_col.append(col0)
        self.max_col.append(col1)
        return label

    def find(self, label: int) -> int:
        root = label
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[label] != label:
            nxt = self.parent[label]
            self.parent[label] = root
            label = nxt
        return root

    def add_run(self, root: int, row: int, col0: int, col1: int) -> None:
        self.size[root] += col1 - col0 + 1
        if row < self.min_row[root]:
            self.min_row[root] = row
        if row > self.max_row[root]:
            self.max_row[root] = row
        if col0 < self.min_col[root]:
            self.min_col[root] = col0
        if col1 > self.max_col[root]:
            self.max_col[root] = col1

    def union(self, label_x: int, label_y: int) -> int:
        root_x = self.find(label_x)
        root_y = self.find(label_y)
        if root_x == root_y:
            return root_x
        # Attach the smaller tree under the larger one.
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        if self.min_row[root_y] < self.min_row[root_x]:
            self.min_row[root_x] = self.min_row[root_y]
        if self.max_row[root_y] > self.max_row[root_x]:
            self.max_row[root_x] = self.max_row[root_y]
        if self.min_col[root_y] < self.min_col[root_x]:
            self.min_col[root_x] = self.min_col[root_y]
        if self.max_col[root_y] > self.max_col[root_x]:
            self.max_col[root_x] = self.max_col[root_y]
        return root_x

    def snapshot(self, root: int) -> tuple[int, int, int, int, int, int]:
        return (
            self.direction[root],
            self.size[root],
            self.min_row[root],
            self.max_row[root],
            self.min_col[root],
            self.max_col[root],
        )


class _PatchMerger:
    """Run-length connected components with cross-band carry-over."""

    def __init__(self) -> None:
        self.uf = _UnionFind()
        # Previous row's runs, grouped by direction:
        # {direction: [(col0, col1, label), ...]}
        self.previous: dict[int, list[tuple[int, int, int]]] = {}
        self.finalized: list[ChangeRegion] = []

    def add_row(
        self,
        row: np.ndarray,
        global_row: int,
        col_offset: int = 0,
    ) -> None:
        current: dict[int, list[tuple[int, int, int]]] = {}
        for direction in (CLEAR_TO_CLOUD, CLOUD_TO_CLEAR):
            above = self.previous.get(direction, [])
            runs_here: list[tuple[int, int, int]] = []
            cursor = 0
            for local_col0, local_col1 in _runs(row, direction):
                col0 = local_col0 + col_offset
                col1 = local_col1 + col_offset
                root: int | None = None
                # Every previous-row run overlapping this column interval is
                # 4-connected and therefore belongs to the same patch.
                while cursor < len(above) and above[cursor][1] < col0:
                    cursor += 1
                scan = cursor
                while scan < len(above) and above[scan][0] <= col1:
                    neighbour = self.uf.find(above[scan][2])
                    if root is None:
                        root = neighbour
                    else:
                        root = self.uf.union(root, neighbour)
                    scan += 1
                if root is None:
                    root = self.uf.new_label(direction, global_row, col0, col1)
                else:
                    self.uf.add_run(root, global_row, col0, col1)
                runs_here.append((col0, col1, root))
            if runs_here:
                current[direction] = runs_here
        self.previous = current

    def finish_band(self) -> None:
        """Finalize patches that cannot reach the next band; compact the rest.

        A patch can continue into the next band only if one of its runs lies
        on the band's last processed row.  Everything else is settled now and
        removed from the union-find so its memory does not grow band after
        band.  The minimum-pixel-size filter is deliberately NOT applied yet:
        a finalized region is kept intact until the whole raster is done.
        """

        frontier_roots = {
            self.uf.find(label)
            for runs in self.previous.values()
            for _, _, label in runs
        }
        old_uf = self.uf
        old_previous = self.previous
        all_roots = [i for i, parent in enumerate(old_uf.parent) if parent == i]

        settled: list[ChangeRegion] = []
        for root in all_roots:
            if root in frontier_roots:
                continue
            direction, size, min_r, max_r, min_c, max_c = self.uf.snapshot(root)
            settled.append(
                ChangeRegion(
                    direction=direction,
                    pixel_count=size,
                    min_row=min_r,
                    max_row=max_r,
                    min_col=min_c,
                    max_col=max_c,
                )
            )
        if settled:
            self.finalized.extend(settled)

        if not frontier_roots:
            self.uf = _UnionFind()
            self.previous = {}
            return

        # Rebuild a compact union-find holding only carry-over roots.
        compact = _UnionFind()
        remap: dict[int, int] = {}
        for old_root in sorted(frontier_roots):
            direction, size, min_r, max_r, min_c, max_c = old_uf.snapshot(old_root)
            new_label = compact.new_label(direction, min_r, min_c, min_c)
            compact.size[new_label] = size
            compact.max_row[new_label] = max_r
            compact.max_col[new_label] = max_c
            remap[old_root] = new_label
        self.uf = compact
        self.previous = {
            direction: [
                (col0, col1, remap[old_root])
                for col0, col1, label in runs
                for old_root in (old_uf.find(label),)
                if old_root in remap
            ]
            for direction, runs in old_previous.items()
        }

    def finish_all(self) -> list[ChangeRegion]:
        for root in [
            i for i, parent in enumerate(self.uf.parent) if parent == i
        ]:
            direction, size, min_r, max_r, min_c, max_c = self.uf.snapshot(root)
            self.finalized.append(
                ChangeRegion(
                    direction=direction,
                    pixel_count=size,
                    min_row=min_r,
                    max_row=max_r,
                    min_col=min_c,
                    max_col=max_c,
                )
            )
        self.uf = _UnionFind()
        self.previous = {}
        return self.finalized


def _plan_bands(
    ds_a: rasterio.DatasetReader,
    ds_b: rasterio.DatasetReader,
    overlap: _Overlap,
    max_band_height: int,
) -> list[tuple[int, int]]:
    """Plan (relative_top, height) row bands aligned to file block rows.

    Block boundaries of both files become band boundaries so reads follow the
    rasters' natural strip/tile layout; bands taller than ``max_band_height``
    are subdivided to cap resident memory.
    """

    boundaries = {0}
    for ds, row_off in (
        (ds_a, overlap.row_off_a),
        (ds_b, overlap.row_off_b),
    ):
        for _, window in ds.block_windows(1):
            top = int(window.row_off) - row_off
            bottom = top + int(window.height)
            if bottom > 0 and top < overlap.height:
                boundaries.add(max(0, top))
    bounds = sorted(b for b in boundaries if 0 <= b < overlap.height)
    bounds.append(overlap.height)

    bands: list[tuple[int, int]] = []
    for top, bottom in zip(bounds, bounds[1:]):
        top, bottom = int(top), int(bottom)
        while bottom - top > max_band_height:
            bands.append((top, max_band_height))
            top += max_band_height
        if bottom > top:
            bands.append((top, bottom - top))
    return bands


def _change_band(
    values_a: np.ndarray,
    nodata_a: float | int | None,
    values_b: np.ndarray,
    nodata_b: float | int | None,
) -> tuple[np.ndarray, int]:
    """Direction-coded change array plus the count of mutually valid pixels."""

    valid_a = _valid_data(values_a, nodata_a)
    valid_b = _valid_data(values_b, nodata_b)
    if np.issubdtype(values_a.dtype, np.floating):
        valid_a &= np.isfinite(values_a)
    if np.issubdtype(values_b.dtype, np.floating):
        valid_b &= np.isfinite(values_b)
    valid = valid_a & valid_b

    change = np.zeros(values_a.shape, dtype=np.uint8)
    change[valid & (values_a == 1) & (values_b == 0)] = CLEAR_TO_CLOUD
    change[valid & (values_a == 0) & (values_b == 1)] = CLOUD_TO_CLEAR
    # Values outside {0, 1} never match either rule, so they are ignored.
    return change, int(valid.sum())


def detect_changes(
    meta_a: RasterMeta,
    meta_b: RasterMeta,
    min_pixels: int,
    band_height: int = DEFAULT_BAND_HEIGHT,
) -> ChangeDetectionResult:
    """Detect connected change patches without loading whole rasters."""

    overlap = _overlap(meta_a, meta_b)
    merger = _PatchMerger()
    overlap_pixels = 0

    if overlap is not None:
        max_band_height = max(1, int(band_height))
        with ExitStack() as stack:
            ds_a = stack.enter_context(rasterio.open(meta_a.upload.path))
            ds_b = stack.enter_context(rasterio.open(meta_b.upload.path))
            bands = _plan_bands(ds_a, ds_b, overlap, max_band_height)

            for relative_top, relative_height in bands:
                window_a = Window(
                    overlap.col_off_a,
                    overlap.row_off_a + relative_top,
                    overlap.width,
                    relative_height,
                )
                window_b = Window(
                    overlap.col_off_b,
                    overlap.row_off_b + relative_top,
                    overlap.width,
                    relative_height,
                )
                # Each read goes through the file's own blocks; windows larger
                # than a block are served by rasterio without loading the
                # whole raster.
                band_a = ds_a.read(1, window=window_a)
                band_b = ds_b.read(1, window=window_b)
                change, valid_count = _change_band(
                    band_a, meta_a.nodata, band_b, meta_b.nodata
                )
                overlap_pixels += valid_count

                for local_row in range(relative_height):
                    # Pixel coordinates are kept in raster A's global index
                    # space (both rows and columns), so finalized bounding
                    # boxes map straight onto A's affine transform.
                    global_row = overlap.row_off_a + relative_top + local_row
                    merger.add_row(
                        change[local_row], global_row, col_offset=overlap.col_off_a
                    )
                merger.finish_band()

    regions = merger.finish_all()
    qualifying = [r for r in regions if r.pixel_count >= min_pixels]
    qualifying.sort(
        key=lambda r: (r.min_row, r.min_col, r.max_row, r.max_col, r.direction)
    )
    return ChangeDetectionResult(
        regions=qualifying,
        overlap_pixels=overlap_pixels,
        crs=str(meta_a.crs) if meta_a.crs is not None else None,
        transform_a=meta_a.transform,
    )
