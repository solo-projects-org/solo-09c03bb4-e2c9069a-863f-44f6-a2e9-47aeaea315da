# 农情遥感补采分析 API

用于在安排补采前，从地块编号图和各景无云掩膜中找出覆盖率不足的地块；
并在云检测算法升级后，对比同一景影像的两版云掩膜，找出值得回看原影像的
成片变化。服务提供 FastAPI + Swagger UI，可直接上传 GeoTIFF 试算。

## 启动

```bash
docker compose up --build -d
```

打开：

- Swagger：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

可用 `APP_PORT` 修改宿主机端口，例如：

```bash
APP_PORT=8080 docker compose up --build -d
```

## 接口

### `POST /analyze`

以 `multipart/form-data` 上传：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `parcel` | 文件，单个 | 地块编号图，单波段 GeoTIFF |
| `masks` | 文件，可重复 | 各景无云掩膜，单波段 GeoTIFF，至少一景 |
| `target_coverage` | 数值 | 目标覆盖率，范围 0 到 1 |

输入约束：

- 全部文件必须是单波段 GeoTIFF，且北向上（无旋转、无剪切）。
- 全部影像使用相同坐标系、相同分辨率。
- 掩膜原点可与地块图原点不同，但偏移必须是整数个像元，保证像元格网重合。
- 地块图中只有有效的正整数是地块编号；0、负数、小数和 nodata 会被忽略。
- 掩膜中有效值 `1` 表示该像元无云；同一像元被多景覆盖时只计一次。

不兼容时返回 HTTP 400，并在 `detail.incompatible_files` 中逐个列出文件名、
角色（`parcel` / `mask`）和原因，例如：

```json
{
  "detail": {
    "message": "输入影像不兼容，未执行分析。",
    "incompatible_files": [
      {
        "filename": "bad_scene.tif",
        "role": "mask",
        "reason": "坐标系与地块图不一致（EPSG:4326 vs 地块图 EPSG:32650）"
      }
    ]
  }
}
```

成功时返回覆盖率**严格低于** `target_coverage` 的地块，按编号升序排列：

```json
{
  "target_coverage": 0.9,
  "total_parcels_analyzed": 4,
  "below_target_count": 2,
  "parcels": [
    {
      "parcel_id": 102,
      "total_pixels": 2200,
      "covered_pixels": 600,
      "coverage_ratio": 0.2727272727272727,
      "below_target": true
    },
    {
      "parcel_id": 103,
      "total_pixels": 2750,
      "covered_pixels": 500,
      "coverage_ratio": 0.18181818181818182,
      "below_target": true
    }
  ]
}
```

### `POST /changes`

对比同一景影像的两版单波段 GeoTIFF 云掩膜（`0` 为有云，`1` 为无云），
找出变化方向相同、四连通（上下左右相邻）的成片变化区域。以
`multipart/form-data` 上传：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `mask_before` | 文件，单个 | 旧版云掩膜，单波段 GeoTIFF |
| `mask_after` | 文件，单个 | 新版云掩膜，单波段 GeoTIFF |
| `min_pixels` | 整数 | 每片变化区域至少包含的像元数，必须 ≥ 1 |

输入约束沿用 `/analyze` 的格网校验：

- 两景均为单波段、北向上（无旋转、无剪切）的 GeoTIFF。
- 两景坐标系、分辨率相同；原点可不同，但偏移必须是整数个像元，保证
  像元格网重合。
- 只比较两景重叠处**双方都有效**的像元：任一景为 nodata（或浮点 NaN）
  的像元不参与比较；取值不在 `{0, 1}` 内的像元也不会被当作变化。
- 变化方向分两种：`无云变有云（1→0）` 与 `有云变无云（0→1）`；方向不同
  的相邻像元不会连成同一片。

两景没有空间重叠（或重叠处无双方都有效的像元）时返回 HTTP 400。成功时，
区域**全部结算之后**才按 `min_pixels` 筛选（保证跨条带生长的小区域不会被
提前丢弃），按扫描顺序（先行后列）返回变化方向、像元数和原坐标系下的
外接范围：

```json
{
  "min_pixels": 10,
  "crs": "EPSG:32650",
  "overlap_pixels": 2626,
  "change_pixel_count": 49,
  "region_count": 2,
  "regions": [
    {
      "region_id": 1,
      "direction": "无云变有云（1→0）",
      "pixel_count": 28,
      "bounds": {
        "min_x": 500060.0,
        "min_y": 2999850.0,
        "max_x": 500110.0,
        "max_y": 2999950.0
      }
    },
    {
      "region_id": 2,
      "direction": "有云变无云（0→1）",
      "pixel_count": 21,
      "bounds": {
        "min_x": 500430.0,
        "min_y": 2999600.0,
        "max_x": 500520.0,
        "max_y": 2999700.0
      }
    }
  ]
}
```

curl 示例：

```bash
curl -X POST http://localhost:8000/changes \
  -F "mask_before=@samples/cloud_mask_v1.tif" \
  -F "mask_after=@samples/cloud_mask_v2.tif" \
  -F "min_pixels=10"
```

## 样例

仓库自带两组可直接在 Swagger 中上传的样例：

- `samples/parcel_map.tif`：120×100，EPSG:32650，像元 10 m，地块编号 101–104
- `samples/scene_01_cloud_free.tif`：相对地块图原点偏移 `(-7, -5)` 像元
- `samples/scene_02_cloud_free.tif`：相对地块图原点偏移 `(+15, +10)` 像元

两景的幅面尺寸不同、原点错开且空间范围互相重叠，像元格网与地块图对齐，
用于验证窗口映射和多景覆盖合并。

### 云掩膜两版对比样例（`/changes`）

- `samples/cloud_mask_v1.tif`：64×48，EPSG:32650，像元 10 m，旧版掩膜
- `samples/cloud_mask_v2.tif`：60×44，原点相对旧版偏移 `(+3, +2)` 像元，
  新版掩膜；两景重叠区为 60×44

两景均按 **8 行一个条带**存储（`blockysize=8`），其中：

- 区域 A（`无云变有云（1→0）`，28 像元）的主干跨越第 8 行条带边界；
- 区域 B（`有云变无云（0→1）`，21 像元）的主干跨越第 32、40 行条带边界，
  并在一条横臂末端带一个新版 nodata 像元（该像元不参与比较，但不会切断
  连通关系）；
- 另有一个 4 像元的小区域，用于演示 `min_pixels` 筛选；
- 新版左上角（旧版坐标行 2–5、列 14–16）为 nodata 块，演示“双方都有效”
  的比较规则。

因此在 Swagger 中以 `min_pixels=10` 上传时，可直接看到两个跨条带合并后
的成片变化区域；把 `min_pixels` 调成 `1` 则能看到全部连通分量。

同样可以用 curl 试算：

```bash
curl -X POST http://localhost:8000/analyze \
  -F "parcel=@samples/parcel_map.tif" \
  -F "masks=@samples/scene_01_cloud_free.tif" \
  -F "masks=@samples/scene_02_cloud_free.tif" \
  -F "target_coverage=0.9"
```

样例中地块 101 完全覆盖，地块 104 覆盖率为 2750/3025 ≈ 90.91%，
不低于 0.9；因此只返回地块 102 和 103。

重新生成样例：

```bash
python scripts/generate_samples.py
```

本地运行与测试（可选）：

```bash
python -m pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest -q
```

## 计算与内存

### `/analyze` 地块覆盖率

- 以地块图为基准，按其自身分块遍历；异常大的条带/分块会再切到
  ≤1024×1024 像元的窗口（`app.analysis`）。不会整幅载入地块图或掩膜。
- 每个窗口只读取与地块窗口相交的掩膜窗口。越界部分按未覆盖处理。
- 每个地块的总像元数和覆盖像元数按窗口增量累计，多景掩膜用布尔 OR 合并，
  重叠不重复计数。
- 上传文件先保存到私有临时目录，请求结束后自动删除；不使用上传文件名作为
  本地路径。

### `/changes` 两版掩膜成片变化

- 先按格网校验计算两景重叠窗口，再**按行分带（水平条带）读取**，带边界
  对齐两景文件自身的块/条带行，异常高的条带会再细分（默认带高上限
  1024 行，`app.changes`）。任何时刻内存中只保留当前带，不整图载入。
- 带内逐行用**游程编码**提取变化像元段，配合并查集维护四连通区域；
  同方向且上下左右相邻的游程合并为同一区域，对角接触不合并。
- 读完一带后，只有贴在该带最后一行、还可能接到下一带的区域被压缩保留；
  其余区域立即结算并移出并查集，因此并查集规模只与前沿行数有关，不随
  总带数增长。下一带开头的游程继续与保留区域合并，实现跨条带连片。
- 区域在**全部结算之后**才按 `min_pixels` 筛选，避免跨带生长的区域被
  提前丢弃；外接范围基于旧版掩膜的仿射变换换算回原坐标系（像元边缘
  对齐），并随响应返回坐标系可读标识。
- 单元测试用整图 flood-fill 作为参考实现，并对多种带高（含 1 行一带）
  断言结果完全一致，验证流式合并与带高无关。
