# MS-ReID Web 系统增量开发记录

## 本次完成范围

本次按照 `MS-ReID Web 系统开发任务书.md` 在 `app/` 目录下完成 Web 原型系统骨架，不改动 `core/` 算法主体代码。

## 后端

已新增 FastAPI 后端：

```text
app/backend/main.py
app/backend/api/
app/backend/services/
app/backend/schemas/
app/backend/scripts/build_market1501_vector_db.py
app/backend/static/
app/backend/vector_store/qdrant/
```

已完成接口：

```text
GET  /api/health
GET  /api/visual/images
GET  /api/visual/result?image_name=xxx.jpg
POST /api/visual/result
GET  /api/experiments/results?dataset=market1501
GET  /api/experiments/results?dataset=dukemtmc
GET  /api/experiments/curves?dataset=market1501
POST /api/retrieval/search
```

实现说明：

- 静态资源挂载到 `/static`。
- 可视化页面使用 `POST /api/visual/result` 上传本地图片并实时生成 `original`、`LH`、`HL`、`HH`、`attention_overlay`，返回 base64 data URL，不保存到本地目录。`attention_overlay` 由 `MultiScale + Attention` 模型前向推理得到 `fusion.activation_heatmap` 后上采样、归一化、上色并叠加到原图，同时返回 `1x1 / 3x3 / 5x5` 三个分支权重。旧的 `GET /api/visual/images` 和 `GET /api/visual/result?image_name=...` 保留为兼容接口。
- 实验结果接口从 `core/storage/outputs/<experiment>/test/<dataset>/*.log` 解析最终 Rank-1 和 mAP。
- 训练曲线接口从 `core/storage/outputs/<experiment>/train/market1501/*.log` 解析 epoch、loss、acc。
- 检索接口接收 `multipart/form-data`，参数为 `file` 和 `top_k`，其中 `top_k` 仅允许 `1 / 5 / 10`。
- `reid_service.py` 复用 `core.config`、`core.data.transforms.build_transforms` 和 `core.modeling.build_model`，默认优先加载 Market1501 非 baseline 中 Rank-1 最优的 `multiscale` 权重，也支持 `MS_REID_WEIGHT` 环境变量。
- `qdrant_service.py` 使用 `QdrantClient(path="app/backend/vector_store/qdrant")` 的本地持久化模式，collection 名称为 `market1501_gallery`，相似度为 Cosine。

## 前端

已新增 React + TypeScript + Vite 前端：

```text
app/frontend/package.json
app/frontend/vite.config.ts
app/frontend/src/
```

已完成页面：

- 数据展示页面
  - 本地图片选择
  - 原图、LH、HL、HH、Heatmap 叠加图展示
  - MultiScale + Attention 的 1x1、3x3、5x5 分支权重展示
  - Market1501 / DukeMTMC 实验结果切换
  - Rank-1、mAP 指标与相对 Baseline 变化展示
  - Market1501 acc/loss SVG 曲线图
- 行人检索页面
  - 本地 Query 图片选择和预览
  - top_k 下拉选择
  - `multipart/form-data` 上传
  - Top-K Gallery 原图、rank、score 展示

## 设计规范

界面参考 `ui-ux-pro-max` 的极简数据仪表盘建议：

- 字体：Inter 优先，系统 sans-serif 兜底。
- 配色：`#F8FAFC` 背景、`#1E293B` 正文、`#2563EB` 主操作、`#F97316` 用于热度图强调。
- 布局：左侧导航、右侧工作区，卡片半径控制在 8px 内，表格和图表优先保证信息密度与可读性。
- 交互：按钮、下拉和上传区域均有固定尺寸与明确状态，移动端降为单列布局。

## 运行方式

后端：

```bash
uv sync
uv run uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

前端：

```bash
cd app/frontend
npm install
npm run dev
```

建库脚本：

```bash
uv run python app/backend/scripts/build_market1501_vector_db.py --recreate
```

如果默认权重不可用，可以指定：

```bash
uv run python app/backend/scripts/build_market1501_vector_db.py --weight core/storage/outputs/full_msreid/train/market1501/resnet50_model_60.pth --recreate
```

## 后续建议

- 若需要答辩演示稳定性，可先用 `--limit 500` 建一个小型 gallery 库验证端到端链路，再全量建库。
- 如果实验日志格式后续变化，应同步调整 `experiment_service.py` 的正则解析。
