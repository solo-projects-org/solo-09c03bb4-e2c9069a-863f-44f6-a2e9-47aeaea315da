"""End-to-end tests for POST /changes (two-version cloud-mask comparison).

Run with ``pytest -q``.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app.main import app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
ORIGIN = (500_000.0, 3_000_000.0)
PIXEL = 10.0
NODATA = 255


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _write_mask(
    path: Path,
    values: np.ndarray,
    origin: tuple[float, float],
    *,
    nodata=NODATA,
    stripe_rows: int | None = None,
    crs: str = "EPSG:32650",
):
    height, width = values.shape
    kwargs = {}
    if stripe_rows is not None:
        kwargs.update(
            tiled=False, blockxsize=width, blockysize=stripe_rows
        )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=from_origin(origin[0], origin[1], PIXEL, PIXEL),
        nodata=nodata,
        compress="deflate",
        **kwargs,
    ) as dst:
        dst.write(values, 1)


def _post(client: TestClient, before: Path, after: Path, min_pixels: int):
    with before.open("rb") as f1, after.open("rb") as f2:
        return client.post(
            "/changes",
            files={
                "mask_before": (before.name, f1, "image/tiff"),
                "mask_after": (after.name, f2, "image/tiff"),
            },
            data={"min_pixels": str(min_pixels)},
        )


def _reference_components(
    before: np.ndarray,
    after: np.ndarray,
    nodata,
) -> list[tuple[int, int, int, int, int, int]]:
    """Whole-array 4-connected components of mutually-valid change pixels."""

    valid = (before != nodata) & (after != nodata)
    change = np.zeros(before.shape, dtype=np.uint8)
    change[valid & (before == 1) & (after == 0)] = 1
    change[valid & (before == 0) & (after == 1)] = 2
    seen = np.zeros(change.shape, dtype=bool)
    components = []
    for r in range(change.shape[0]):
        for c in range(change.shape[1]):
            if change[r, c] == 0 or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            direction = int(change[r, c])
            count = 0
            r0 = r1 = r
            c0 = c1 = c
            while queue:
                y, x = queue.popleft()
                count += 1
                r0, r1 = min(r0, y), max(r1, y)
                c0, c1 = min(c0, x), max(c1, x)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < change.shape[0]
                        and 0 <= nx < change.shape[1]
                        and not seen[ny, nx]
                        and change[ny, nx] == direction
                    ):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            components.append((direction, count, r0, r1, c0, c1))
    components.sort(key=lambda z: (z[2], z[4]))
    return components


def test_sample_pair_demonstrates_cross_band_merge(client: TestClient):
    response = _post(
        client,
        SAMPLES / "cloud_mask_v1.tif",
        SAMPLES / "cloud_mask_v2.tif",
        10,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["min_pixels"] == 10
    assert body["region_count"] == 2

    with rasterio.open(SAMPLES / "cloud_mask_v1.tif") as ds:
        before = ds.read(1)
        transform = ds.transform
        nodata = ds.nodata
        # The sample is stored as 8-row strips.
        block_heights = {int(w.height) for _, w in ds.block_windows(1)}
        assert block_heights == {8}
    with rasterio.open(SAMPLES / "cloud_mask_v2.tif") as ds:
        after = ds.read(1)
        assert ds.nodata == nodata

    overlap = (slice(2, 46), slice(3, 63))
    reference = _reference_components(
        before[overlap], after, nodata
    )
    # Translate overlap-local component boxes back into version-1 pixel space.
    reference = [
        (direction, count, r0 + 2, r1 + 2, c0 + 3, c1 + 3)
        for direction, count, r0, r1, c0, c1 in reference
    ]
    qualifying = [z for z in reference if z[1] >= 10]
    assert len(qualifying) == body["region_count"]

    for (direction, count, r0, r1, c0, c1), region in zip(
        qualifying, body["regions"], strict=True
    ):
        assert region["pixel_count"] == count
        assert region["direction"].startswith(
            "无云变有云" if direction == 1 else "有云变无云"
        )
        west = transform.c + c0 * transform.a
        east = transform.c + (c1 + 1) * transform.a
        north = transform.f + r0 * transform.e
        south = transform.f + (r1 + 1) * transform.e
        assert region["bounds"] == {
            "min_x": west,
            "min_y": south,
            "max_x": east,
            "max_y": north,
        }

    # Both qualifying regions span multiple 8-row strips.
    for region, (_, _, r0, r1, _, _) in zip(
        body["regions"], qualifying, strict=True
    ):
        assert r1 - r0 + 1 > 8

    # Region ids are 1-based in scan order.
    assert [r["region_id"] for r in body["regions"]] == [1, 2]


def test_min_pixels_threshold_filters_after_finalization(
    client: TestClient, tmp_path: Path
):
    # A 4-pixel 0->1 patch must disappear once min_pixels exceeds 4.
    before = np.zeros((20, 20), dtype=np.uint8)
    after = np.zeros((20, 20), dtype=np.uint8)
    after[5:7, 5:7] = 1
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN, stripe_rows=4)
    _write_mask(p2, after, ORIGIN, stripe_rows=4)

    small = _post(client, p1, p2, 1).json()
    assert small["region_count"] == 1
    assert small["regions"][0]["pixel_count"] == 4

    filtered = _post(client, p1, p2, 5).json()
    assert filtered["region_count"] == 0
    assert filtered["regions"] == []
    assert filtered["change_pixel_count"] == 0


def test_only_mutually_valid_pixels_are_compared(
    client: TestClient, tmp_path: Path
):
    before = np.ones((8, 8), dtype=np.uint8)
    after = np.zeros((8, 8), dtype=np.uint8)
    # nodata in either raster removes that pixel from the comparison.
    before[2, 2] = NODATA
    after[3, 3] = NODATA
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN)
    _write_mask(p2, after, ORIGIN)

    body = _post(client, p1, p2, 1).json()
    assert body["overlap_pixels"] == 62  # 64 minus two invalid pixels
    assert body["region_count"] == 1
    assert body["regions"][0]["pixel_count"] == 62
    assert body["regions"][0]["direction"].startswith("无云变有云")


def test_values_outside_0_1_are_ignored(client: TestClient, tmp_path: Path):
    before = np.zeros((6, 6), dtype=np.uint8)
    after = np.zeros((6, 6), dtype=np.uint8)
    before[:] = 1
    after[:] = 0
    after[0, 0] = 7  # not a valid mask code, no nodata marker needed
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN, nodata=None)
    _write_mask(p2, after, ORIGIN, nodata=None)

    body = _post(client, p1, p2, 1).json()
    assert body["overlap_pixels"] == 36
    assert body["regions"][0]["pixel_count"] == 35


def test_directions_are_distinguished(client: TestClient, tmp_path: Path):
    before = np.zeros((12, 12), dtype=np.uint8)
    after = np.zeros((12, 12), dtype=np.uint8)
    before[1:4, 1:4] = 1   # 1 -> 0 patch
    after[8:11, 8:11] = 1  # 0 -> 1 patch
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN, stripe_rows=4)
    _write_mask(p2, after, ORIGIN, stripe_rows=4)

    body = _post(client, p1, p2, 1).json()
    assert body["region_count"] == 2
    labels = {r["direction"] for r in body["regions"]}
    assert labels == {"无云变有云（1→0）", "有云变无云（0→1）"}


def test_cross_band_patch_merges_into_one_region(
    client: TestClient, tmp_path: Path
):
    # One 1->0 patch crossing every 4-row strip boundary.
    before = np.zeros((16, 8), dtype=np.uint8)
    after = np.zeros((16, 8), dtype=np.uint8)
    before[:, 3] = 1
    before[7, 0:6] = 1
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN, stripe_rows=4)
    _write_mask(p2, after, ORIGIN, stripe_rows=4)

    body = _post(client, p1, p2, 1).json()
    assert body["region_count"] == 1
    region = body["regions"][0]
    assert region["pixel_count"] == 16 + 5  # trunk 16 + five arm extras
    assert region["bounds"] == {
        "min_x": ORIGIN[0],
        "min_y": ORIGIN[1] - 16 * PIXEL,
        "max_x": ORIGIN[0] + 6 * PIXEL,
        "max_y": ORIGIN[1],
    }


def test_diagonal_touch_does_not_connect(client: TestClient, tmp_path: Path):
    before = np.zeros((8, 8), dtype=np.uint8)
    after = np.zeros((8, 8), dtype=np.uint8)
    before[0:2, 0:2] = 1
    before[2:4, 2:4] = 1  # corner contact only
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN, stripe_rows=3)
    _write_mask(p2, after, ORIGIN, stripe_rows=3)

    body = _post(client, p1, p2, 1).json()
    assert body["region_count"] == 2
    assert all(r["pixel_count"] == 4 for r in body["regions"])


def test_offset_grids_use_overlap_only(client: TestClient, tmp_path: Path):
    # Version 2 is shifted by (+3, +2) pixels and is smaller.
    before = np.ones((12, 12), dtype=np.uint8)
    after = np.zeros((8, 8), dtype=np.uint8)
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN)
    shifted_origin = (ORIGIN[0] + 3 * PIXEL, ORIGIN[1] - 2 * PIXEL)
    _write_mask(p2, after, shifted_origin)

    body = _post(client, p1, p2, 1).json()
    assert body["overlap_pixels"] == 64  # 8x8 overlap
    assert body["regions"][0]["pixel_count"] == 64
    # Bounds use version-1 pixel space: overlap starts at row 2, col 3.
    assert body["regions"][0]["bounds"] == {
        "min_x": ORIGIN[0] + 3 * PIXEL,
        "min_y": ORIGIN[1] - 10 * PIXEL,
        "max_x": ORIGIN[0] + 11 * PIXEL,
        "max_y": ORIGIN[1] - 2 * PIXEL,
    }


def test_no_overlap_returns_400(client: TestClient, tmp_path: Path):
    before = np.ones((4, 4), dtype=np.uint8)
    after = np.ones((4, 4), dtype=np.uint8)
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN)
    far_origin = (ORIGIN[0] + 100 * PIXEL, ORIGIN[1] - 100 * PIXEL)
    _write_mask(p2, after, far_origin)

    response = _post(client, p1, p2, 1)
    assert response.status_code == 400
    assert "没有可比较的像元" in response.json()["detail"]["message"]


def test_incompatible_crs_returns_400(client: TestClient, tmp_path: Path):
    before = np.ones((6, 6), dtype=np.uint8)
    after = np.ones((6, 6), dtype=np.uint8)
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN, crs="EPSG:32650")
    _write_mask(
        p2,
        after,
        (110.0, 30.0),
        crs="EPSG:4326",
    )
    response = _post(client, p1, p2, 1)
    assert response.status_code == 400
    errors = response.json()["detail"]["incompatible_files"]
    assert any("坐标系" in e["reason"] for e in errors)


def test_non_integer_origin_offset_returns_400(client: TestClient, tmp_path: Path):
    before = np.ones((6, 6), dtype=np.uint8)
    after = np.ones((6, 6), dtype=np.uint8)
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN)
    _write_mask(p2, after, (ORIGIN[0] + 5, ORIGIN[1]))  # half-pixel shift

    response = _post(client, p1, p2, 1)
    assert response.status_code == 400
    errors = response.json()["detail"]["incompatible_files"]
    assert any("整数像元" in e["reason"] for e in errors)


def test_multiband_file_returns_400(client: TestClient, tmp_path: Path):
    path = tmp_path / "two_band.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=2,
        dtype="uint8",
        crs="EPSG:32650",
        transform=from_origin(*ORIGIN, PIXEL, PIXEL),
    ) as dst:
        dst.write(np.ones((4, 4), dtype=np.uint8), 1)
        dst.write(np.ones((4, 4), dtype=np.uint8), 2)
    good = tmp_path / "good.tif"
    _write_mask(good, np.ones((4, 4), dtype=np.uint8), ORIGIN)

    response = _post(client, path, good, 1)
    assert response.status_code == 400
    assert any("单波段" in e["reason"] for e in response.json()["detail"]["incompatible_files"])


def test_band_height_does_not_change_results(tmp_path: Path):
    """Streaming output must be identical for every band height."""

    from app.analysis import UploadedRaster, _inspect
    from app.changes import detect_changes

    rng = np.random.default_rng(0)
    height, width = 120, 40
    before = rng.integers(0, 2, (height, width)).astype(np.uint8)
    after = rng.integers(0, 2, (height, width)).astype(np.uint8)
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN, stripe_rows=8)
    _write_mask(p2, after, ORIGIN, stripe_rows=8)

    meta_a, err_a = _inspect(UploadedRaster(str(p1), "before", "mask_before"))
    meta_b, err_b = _inspect(UploadedRaster(str(p2), "after", "mask_after"))
    assert err_a is None and err_b is None

    def key(result):
        return sorted(
            (r.direction, r.pixel_count, r.min_row, r.max_row, r.min_col, r.max_col)
            for r in result.regions
        )

    reference = key(detect_changes(meta_a, meta_b, 1, band_height=1024))
    for band_height in (1, 3, 7, 16, 64):
        assert key(detect_changes(meta_a, meta_b, 1, band_height)) == reference


def test_frontier_state_is_compacted_between_bands():
    from app.changes import _PatchMerger

    merger = _PatchMerger()
    grid = np.zeros((200, 40), dtype=np.uint8)
    for row in range(0, 200, 10):
        grid[row, 5] = 1  # isolated pixels settled within their own band

    for top in range(0, 200, 8):
        for row in range(top, min(top + 8, 200)):
            merger.add_row(grid[row], row)
        merger.finish_band()
        # Only runs on the last processed row may survive compaction.
        assert len(merger.uf.parent) <= 2

    regions = merger.finish_all()
    assert len(regions) == 20
    assert all(region.pixel_count == 1 for region in regions)


@pytest.mark.parametrize("bad_value", ["0", "-3", "abc"])
def test_invalid_min_pixels_returns_400(
    client: TestClient, tmp_path: Path, bad_value: str
):
    before = np.ones((4, 4), dtype=np.uint8)
    after = np.zeros((4, 4), dtype=np.uint8)
    p1 = tmp_path / "before.tif"
    p2 = tmp_path / "after.tif"
    _write_mask(p1, before, ORIGIN)
    _write_mask(p2, after, ORIGIN)
    with p1.open("rb") as f1, p2.open("rb") as f2:
        response = client.post(
            "/changes",
            files={
                "mask_before": (p1.name, f1, "image/tiff"),
                "mask_after": (p2.name, f2, "image/tiff"),
            },
            data={"min_pixels": bad_value},
        )
    assert response.status_code in (400, 422)
