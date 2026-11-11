#!/usr/bin/env python3
"""Generate the demonstration GeoTIFFs in samples/.

Run from the repository root:

    python scripts/generate_samples.py

The two mask scenes have different, non-zero tile origins.  Their grids are
offset by integral pixels from the parcel map and overlap each other, which is
exactly the condition the service must handle without resampling.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

OUT = Path(__file__).resolve().parent.parent / "samples"

PIXEL_SIZE = 10.0
PARCEL_ORIGIN_X = 500_000.0
PARCEL_ORIGIN_Y = 3_000_000.0
WIDTH = 120
HEIGHT = 100

# (scene origin in parcels' pixel space, width, height)
SCENE_A = ((-7, -5), 100, 90)
SCENE_B = ((15, 10), 95, 85)

# --- Two versions of one cloud mask, used by POST /changes -----------------
# Both versions share the parcel grid but have different origins/sizes.  They
# are stored in 8-row strips so the change patches cross multiple strip
# boundaries, demonstrating cross-band component merging in Swagger.
CHANGE_STRIPE_HEIGHT = 8
CHANGE_NODATA = 255
V1_SIZE = (64, 48)  # (width, height)
V2_OFFSET = (3, 2)  # version 2 origin in version-1 pixel space
V2_SIZE = (60, 44)  # overlap is therefore 60x44


def write_raster(path: Path, data: np.ndarray, origin, nodata=0, crs="EPSG:32650"):
    height, width = data.shape
    transform = from_origin(
        origin[0], origin[1], xsize=PIXEL_SIZE, ysize=PIXEL_SIZE
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=64,
        blockysize=64,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)


def write_strip_raster(
    path: Path, data: np.ndarray, origin, nodata, crs="EPSG:32650"
):
    """Write a strip-organized GeoTIFF (one block per CHANGE_STRIPE_HEIGHT rows)."""

    height, width = data.shape
    transform = from_origin(
        origin[0], origin[1], xsize=PIXEL_SIZE, ysize=PIXEL_SIZE
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        tiled=False,
        blockxsize=width,
        blockysize=CHANGE_STRIPE_HEIGHT,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)


def global_grid(offset, width, height):
    """Coordinates in the parcel map's pixel index space."""
    rows = np.arange(offset[1], offset[1] + height)[:, None]
    cols = np.arange(offset[0], offset[0] + width)[None, :]
    return cols, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    parcels = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    parcels[0:40, 0:50] = 101
    parcels[0:40, 55:110] = 102
    parcels[45:100, 0:50] = 103
    parcels[45:100, 55:110] = 104
    write_raster(
        OUT / "parcel_map.tif",
        parcels,
        (PARCEL_ORIGIN_X, PARCEL_ORIGIN_Y),
        nodata=0,
    )

    offset, width, height = SCENE_A
    cols, rows = global_grid(offset, width, height)
    mask_a = ((cols < 70) & (rows < 55)).astype(np.uint8)
    write_raster(
        OUT / "scene_01_cloud_free.tif",
        mask_a,
        (
            PARCEL_ORIGIN_X + offset[0] * PIXEL_SIZE,
            PARCEL_ORIGIN_Y - offset[1] * PIXEL_SIZE,
        ),
        nodata=0,
    )

    offset, width, height = SCENE_B
    cols, rows = global_grid(offset, width, height)
    mask_b = ((cols >= 50) & (rows >= 40)).astype(np.uint8)
    write_raster(
        OUT / "scene_02_cloud_free.tif",
        mask_b,
        (
            PARCEL_ORIGIN_X + offset[0] * PIXEL_SIZE,
            PARCEL_ORIGIN_Y - offset[1] * PIXEL_SIZE,
        ),
        nodata=0,
    )

    _write_change_pair()

    print("Generated:")
    for path in sorted(OUT.glob("*.tif")):
        print(f"  {path.relative_to(OUT.parent)} ({path.stat().st_size} bytes)")


def _write_change_pair() -> None:
    """Build the old/new cloud-mask pair for the /changes endpoint.

    All geometry below is expressed in version-1 pixel space.  Version 2
    starts at V2_OFFSET = (3, 2), so a version-1 pixel (r, c) inside the
    60x44 overlap (v1 rows 2..45, cols 3..62) maps to version-2 array index
    (r - 2, c - 3).

    Two qualifying patches each cross several 8-row strip boundaries:

    * region A, 无云变有云 (1->0), 28 pixels, bbox rows 5..14, cols 6..10
    * region B, 有云变无云 (0->1), 21 pixels, bbox rows 30..39, cols 43..51
      (one version-2 nodata hole at an arm tip removes one pixel)

    A 4-pixel patch (rows 41..42, cols 55..56) demonstrates the min_pixels
    filter, and the nodata blocks demonstrate "both valid" masking.
    """

    w1, h1 = V1_SIZE
    w2, h2 = V2_SIZE
    dr, dc = V2_OFFSET[1], V2_OFFSET[0]
    v1 = np.zeros((h1, w1), dtype=np.uint8)
    v2 = np.zeros((h2, w2), dtype=np.uint8)

    def in_v1(r: int, c: int) -> bool:
        return 0 <= r < h1 and 0 <= c < w1

    def in_v2(r: int, c: int) -> bool:
        return 0 <= r - dr < h2 and 0 <= c - dc < w2

    def set_v1(r: int, c: int, value: int) -> None:
        if in_v1(r, c):
            v1[r, c] = value

    def set_v2(r: int, c: int, value: int) -> None:
        if in_v2(r, c):
            v2[r - dr, c - dc] = value

    def rect_v1(r0, r1, c0, c1, value):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                set_v1(r, c, value)

    def rect_v2(r0, r1, c0, c1, value):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                set_v2(r, c, value)

    # Persistent cloud-free area in both versions (no change), kept away from
    # the change patches.
    rect_v1(0, 2, 50, 62, 1)
    rect_v2(2, 2, 50, 62, 1)

    # Region A (1->0), 28 pixels, one 4-connected patch crossing the strip
    # edge at overlap row 8 (v1 row 10): trunk col 8 rows 5..14 (10 pixels),
    # row-9 arm cols 6..10 (4), row-12 arm cols 6..10 (4), row-14 arm
    # cols 7..9 (2), and rows 6..7 cols 6..10 (8 new pixels) => 28 total.
    for r in range(5, 15):
        set_v1(r, 8, 1)
    rect_v1(9, 9, 6, 10, 1)
    rect_v1(12, 12, 6, 10, 1)
    rect_v1(14, 14, 7, 9, 1)
    rect_v1(6, 7, 6, 10, 1)

    # Region B (0->1): one 4-connected patch crossing strip edges at rows 32
    # and 40.  Trunk col 47 rows 30..39 (10 pixels); arms on rows 34, 37 and
    # 39 add 3 + 4 + 5 = 12 pixels, total 22.  A version-2 nodata hole at the
    # end of one arm (not on the trunk) removes one pixel without disconnecting
    # the patch, leaving 21 change pixels.
    for r in range(30, 40):
        set_v2(r, 47, 1)
    rect_v2(34, 34, 47, 50, 1)  # right arm, three new pixels (48,49,50)
    rect_v2(37, 37, 43, 46, 1)  # left arm, four new pixels (43..46)
    rect_v2(39, 39, 47, 52, 1)  # right arm, five new pixels (48..52)
    set_v2(39, 52, CHANGE_NODATA)  # hole at the arm tip: 22 - 1 = 21 pixels

    # Small 0->1 patch (4 pixels): filtered out when min_pixels >= 5.
    rect_v2(41, 42, 55, 56, 1)

    # Version-2 nodata block (v1 rows 2..5, cols 14..16): overlap pixels
    # there are not valid in both rasters and must be ignored.  It sits well
    # away from region A (cols 6..10) so no change pixel is removed.
    rect_v2(2, 5, 14, 16, CHANGE_NODATA)
    # A version-1 nodata pixel near region A exercises the same rule on the
    # old mask side.
    set_v1(10, 4, CHANGE_NODATA)

    origin1 = (PARCEL_ORIGIN_X, PARCEL_ORIGIN_Y)
    # North-up: a positive row offset moves the origin south, i.e. origin-y
    # decreases by dr * pixel size (same convention as the parcel samples).
    origin2 = (
        PARCEL_ORIGIN_X + V2_OFFSET[0] * PIXEL_SIZE,
        PARCEL_ORIGIN_Y - V2_OFFSET[1] * PIXEL_SIZE,
    )
    write_strip_raster(
        OUT / "cloud_mask_v1.tif", v1, origin1, nodata=CHANGE_NODATA
    )
    write_strip_raster(
        OUT / "cloud_mask_v2.tif", v2, origin2, nodata=CHANGE_NODATA
    )


if __name__ == "__main__":
    main()
