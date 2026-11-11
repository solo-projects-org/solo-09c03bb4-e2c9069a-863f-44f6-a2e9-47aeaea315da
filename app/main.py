"""FastAPI application for finding parcels with insufficient cloud-free coverage."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.analysis import (
    IncompatibleFile,
    UploadedRaster,
    analyze_coverage,
    validate_uploads,
)
from app.changes import (
    DIRECTION_LABELS,
    detect_changes,
    validate_mask_pair,
)

app = FastAPI(
    title="农情遥感补采分析 API",
    version="1.1.0",
    description=(
        "能力一：上传地块编号图和各景单波段无云掩膜，统计每个地块的总像元数"
        "与至少一景无云覆盖的像元数，返回覆盖率低于目标值的地块。\n\n"
        "能力二：上传同一景影像的两版单波段云掩膜（0 有云、1 无云），"
        "按行分带流式比较重叠处双方都有效的像元，把变化方向相同、四连通的"
        "像元连成一片（跨条带合并），结算后按最小像元数筛选并返回变化方向、"
        "像元数与原坐标系外接范围。"
    ),
)


class ParcelCoverage(BaseModel):
    parcel_id: int = Field(description="地块编号（地块图中的正整数）")
    total_pixels: int = Field(description="该地块在地块图中的总有效像元数")
    covered_pixels: int = Field(
        description="至少一景无云掩膜值为 1 的像元数；多景重叠只计一次"
    )
    coverage_ratio: float = Field(description="covered_pixels / total_pixels")
    below_target: bool = Field(description="覆盖率是否低于请求中的目标覆盖率")


class AnalysisResponse(BaseModel):
    target_coverage: float
    total_parcels_analyzed: int
    below_target_count: int
    parcels: list[ParcelCoverage] = Field(
        description="低于目标覆盖率的地块，按地块编号升序排列"
    )


class Bounds(BaseModel):
    min_x: float = Field(description="外接矩形西边界（原坐标系坐标）")
    min_y: float = Field(description="外接矩形南边界（原坐标系坐标）")
    max_x: float = Field(description="外接矩形东边界（原坐标系坐标）")
    max_y: float = Field(description="外接矩形北边界（原坐标系坐标）")


class ChangeRegionOut(BaseModel):
    region_id: int = Field(description="变化区域编号，按扫描顺序（先行后列）从 1 开始")
    direction: str = Field(
        description="变化方向：`无云变有云（1→0）` 或 `有云变无云（0→1）`"
    )
    pixel_count: int = Field(description="该区域包含的变化像元数")
    bounds: Bounds = Field(description="原坐标系下的外接范围（像元边缘对齐）")


class ChangeResponse(BaseModel):
    min_pixels: int
    crs: str | None = Field(
        description="两景掩膜共用的坐标系（CRS 标识）；外接范围即该坐标系下的坐标"
    )
    overlap_pixels: int = Field(
        description="两景重叠范围内，两图都有效的像元总数（仅这些像元参与比较）"
    )
    change_pixel_count: int = Field(description="达标区域合计的变化像元数")
    region_count: int
    regions: list[ChangeRegionOut] = Field(
        description="像元数达到 min_pixels 的成片变化区域，按扫描顺序排列"
    )


def _safe_upload_name(upload: UploadFile, index: int) -> str:
    raw = upload.filename or f"上传文件_{index}"
    # Keep a readable label for error messages but never use it as a path.
    name = raw.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name or f"上传文件_{index}"


def _save_upload(upload: UploadFile, destination: Path) -> None:
    upload.file.seek(0)
    with destination.open("wb") as out:
        shutil.copyfileobj(upload.file, out, length=1024 * 1024)


def _incompatible_response(errors: list[IncompatibleFile]) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "message": "输入影像不兼容，未执行分析。",
            "incompatible_files": [e.as_dict() for e in errors],
        },
    )


@app.get("/health", tags=["系统"], summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": "农情遥感补采分析 API",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["分析"],
    summary="找出无云观测覆盖不足的地块",
    responses={
        400: {
            "description": "存在不兼容的 GeoTIFF，响应中列出具体文件与原因",
        }
    },
)
def analyze(
    parcel: Annotated[
        UploadFile,
        File(
            description=(
                "地块编号图：单波段 GeoTIFF，北向上；有效地块编号为正整数"
            )
        ),
    ],
    masks: Annotated[
        list[UploadFile],
        File(
            description=(
                "各景无云掩膜：单波段 GeoTIFF，北向上，坐标系和分辨率与地块图"
                "相同；掩膜值 1 表示无云。可一次上传多景，景幅可错开重叠。"
            )
        ),
    ],
    target_coverage: Annotated[
        float,
        Form(
            description="目标覆盖率，范围 0 到 1；覆盖率严格低于该值的地块被返回"
        ),
    ],
) -> AnalysisResponse:
    if not math.isfinite(target_coverage) or not (0.0 <= target_coverage <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="target_coverage 必须是 0 到 1 之间的有限数值。",
        )
    if not masks:
        raise HTTPException(status_code=400, detail="至少上传一景无云掩膜。")

    with tempfile.TemporaryDirectory(prefix="agri-coverage-") as tmp:
        tmp_dir = Path(tmp)
        parcels_dir = tmp_dir / "parcels"
        masks_dir = tmp_dir / "masks"
        parcels_dir.mkdir()
        masks_dir.mkdir()

        parcel_name = _safe_upload_name(parcel, 0)
        parcel_dest = parcels_dir / "parcel_0000.tif"
        _save_upload(parcel, parcel_dest)
        parcel_upload = UploadedRaster(
            path=str(parcel_dest), filename=parcel_name, role="parcel"
        )

        mask_uploads: list[UploadedRaster] = []
        for index, mask in enumerate(masks, start=1):
            mask_name = _safe_upload_name(mask, index)
            mask_dest = masks_dir / f"mask_{index:04d}.tif"
            _save_upload(mask, mask_dest)
            mask_uploads.append(
                UploadedRaster(
                    path=str(mask_dest), filename=mask_name, role="mask"
                )
            )

        parcel_meta, mask_metas, errors = validate_uploads(
            parcel_upload, mask_uploads
        )
        if errors:
            raise _incompatible_response(errors)

        assert parcel_meta is not None
        assert len(mask_metas) == len(mask_uploads)
        results = analyze_coverage(parcel_meta, mask_metas)

    below = [r for r in results if r.coverage_ratio < target_coverage]
    return AnalysisResponse(
        target_coverage=target_coverage,
        total_parcels_analyzed=len(results),
        below_target_count=len(below),
        parcels=[
            ParcelCoverage(
                parcel_id=r.parcel_id,
                total_pixels=r.total_pixels,
                covered_pixels=r.covered_pixels,
                coverage_ratio=r.coverage_ratio,
                below_target=True,
            )
            for r in below
        ],
    )


@app.post(
    "/changes",
    response_model=ChangeResponse,
    tags=["分析"],
    summary="找出两版云掩膜之间值得回看原影像的成片变化",
    responses={
        400: {
            "description": "两景影像不兼容、无重叠，或 min_pixels 不合法",
        }
    },
)
def detect_mask_changes(
    mask_before: Annotated[
        UploadFile,
        File(description="旧版单波段 GeoTIFF 云掩膜：0 为有云，1 为无云"),
    ],
    mask_after: Annotated[
        UploadFile,
        File(description="新版单波段 GeoTIFF 云掩膜：0 为有云，1 为无云"),
    ],
    min_pixels: Annotated[
        int,
        Form(
            description=(
                "每片变化区域至少包含的像元数；区域结算后才按该阈值筛选，"
                "小于该值的成片变化不返回"
            )
        ),
    ],
) -> ChangeResponse:
    if not isinstance(min_pixels, int) or min_pixels < 1:
        raise HTTPException(
            status_code=400,
            detail="min_pixels 必须是不小于 1 的整数。",
        )

    with tempfile.TemporaryDirectory(prefix="mask-changes-") as tmp:
        tmp_dir = Path(tmp)
        before_name = _safe_upload_name(mask_before, 1)
        after_name = _safe_upload_name(mask_after, 2)
        before_dest = tmp_dir / "mask_before.tif"
        after_dest = tmp_dir / "mask_after.tif"
        _save_upload(mask_before, before_dest)
        _save_upload(mask_after, after_dest)

        before_upload = UploadedRaster(
            path=str(before_dest), filename=before_name, role="mask_before"
        )
        after_upload = UploadedRaster(
            path=str(after_dest), filename=after_name, role="mask_after"
        )
        meta_before, meta_after, errors = validate_mask_pair(
            before_upload, after_upload
        )
        if errors:
            raise _incompatible_response(errors)

        assert meta_before is not None and meta_after is not None
        result = detect_changes(meta_before, meta_after, min_pixels)

    if result.overlap_pixels == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "两景掩膜没有可比较的像元：空间范围不重叠，或重叠处无双方都有效的像元。",
                "incompatible_files": [
                    {
                        "filename": before_name,
                        "role": "mask_before",
                        "reason": "与新版掩膜无有效重叠",
                    },
                    {
                        "filename": after_name,
                        "role": "mask_after",
                        "reason": "与旧版掩膜无有效重叠",
                    },
                ],
            },
        )

    transform = result.transform_a
    regions_out: list[ChangeRegionOut] = []
    for index, region in enumerate(result.regions, start=1):
        # Pixel edges: column c spans [c, c+1) pixels; row r spans [r, r+1).
        west = transform.c + region.min_col * transform.a
        east = transform.c + (region.max_col + 1) * transform.a
        north = transform.f + region.min_row * transform.e
        south = transform.f + (region.max_row + 1) * transform.e
        regions_out.append(
            ChangeRegionOut(
                region_id=index,
                direction=DIRECTION_LABELS[region.direction],
                pixel_count=region.pixel_count,
                bounds=Bounds(
                    min_x=west,
                    min_y=south,
                    max_x=east,
                    max_y=north,
                ),
            )
        )

    return ChangeResponse(
        min_pixels=min_pixels,
        crs=result.crs,
        overlap_pixels=result.overlap_pixels,
        change_pixel_count=sum(r.pixel_count for r in regions_out),
        region_count=len(regions_out),
        regions=regions_out,
    )
