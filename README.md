# Kaggle Inference Hub

把 Kaggle 双 T4 Notebook 作为可远程调度的 GPU Worker。本地 FastAPI 负责 Prompt、可选 AI Prompt Pipeline、模型路由、队列、Worker 状态和生成结果；模型推理留在 Kaggle。

当前内置三种 Worker：

- **SANA Sprint 1.6B** — Diffusers，2×T4 各自常驻一份 Pipeline。
- **Z-Image-Turbo GGUF** — `stable-diffusion.cpp`，使用 `kaggle-build` Release 的 T4 `sm_75` 预编译 runtime，2×T4 各自常驻一个 `sd-server`。
- **TripoSR** — 官方单图 3D 重建模型，2×T4 各自常驻一份模型，生成 GLB 或 OBJ。

## 架构

```text
Browser / Local API
        │
        ▼
Kaggle Inference Hub (FastAPI)
        │
        ├── AI Prompt Pipeline ─────► OpenAI-compatible API
        │         │
        │         └── 优化结果先回填 UI，由用户确认后再提交
        │
        ├── queue: sana-sprint-1.6b ─────► Kaggle SANA Worker (2×T4)
        ├── queue: z-image-turbo-gguf ───► Kaggle Z-Image Worker (2×T4)
        └── queue: triposr ◄── upload / Gallery image
                    │
                    └────────────────────► Kaggle TripoSR Worker (2×T4)
                                              │
                    encrypted WebP/GLB/OBJ ◄─┘
        │
        ▼
outputs/<model>/ + Live Gallery
```

三个模型共享本地控制面，但**任务队列按模型隔离**，不同 Notebook 不会互相抢任务。

## 本地启动

```powershell
uv sync
$env:KAGGLE_HUB_TOKEN="wangran"
uv run python recv.py
```

打开：<http://127.0.0.1:30100>

也可以：

```powershell
uv run uvicorn recv:app --host 0.0.0.0 --port 30100 --reload
```

## Cloudflare Tunnel

```powershell
cloudflared tunnel --url http://localhost:30100
```

把得到的公网 HTTPS 地址填到 Notebook 的 `BASE`/`BASE_URL`。如果你使用固定域名 Tunnel，也只需要指向该域名。

## Notebook

- `notebooks/001-sana-sprint-1.6b.ipynb`
- `notebooks/002-z-image-turbo-gguf.ipynb`
- `notebooks/003-triposr-image-to-3d.ipynb`

所有 Notebook 都会：

1. 注册自己的 `worker_id + model`。
2. 对对应模型执行 25 秒 long polling。
3. 双 GPU 并行处理。
4. 将 WebP 或 3D 文件通过 AES-GCM 加密上传。
5. 周期性发送心跳。
6. 失败时通知本地服务进行有限次数重试。

TripoSR Notebook 需要在 Kaggle 启用 **GPU T4 x2** 与 **Internet**。推荐在 Add-ons → Secrets 中添加 `BASE_URL` 和 `KAGGLE_HUB_TOKEN`。本地 UI 可以直接上传 PNG/JPEG/WebP，也可以在 SANA、Z-Image 或其他接入 Hub 的生成图卡片上点击“转为 3D”。默认输出带顶点色的 GLB，也可选择 OBJ。

其他本地模型也可以把刚生成的文件直接送入队列：

```powershell
curl.exe -X POST http://127.0.0.1:30100/task/triposr `
  -H "Authorization: Bearer $env:KAGGLE_HUB_TOKEN" `
  -F "file=@D:\path\to\generated.png" `
  -F "output_format=glb" -F "mc_resolution=256"
```

## 本地 UI

UI 中先选择模型，然后提交 Prompt。单 Prompt 输入框支持任意换行；批量输入框才采用“一行一个任务”。切换模型时 Steps 自动切换到模型默认值：

- SANA：2 steps
- Z-Image-Turbo：8 steps

页面同时展示模型队列、inflight 数量、在线 Worker、图片 Gallery 和 TripoSR 3D 下载结果。

## AI Prompt Pipeline

Pipeline 完全运行在本地 Hub 控制面，**不会改 Kaggle Notebook**。流程：

```text
原始 Prompt
    │
    ▼
AI Prompt Pipeline
    ├── enhance   增强
    ├── creative  创意扩写
    ├── translate 忠实英译
    └── clean     整理
    │
    ├── SANA adapter
    └── Z-Image adapter
    │
    ▼
回填输入框 → 用户确认 → /task → Kaggle Worker
```

默认关闭。配置任意 OpenAI-compatible `/v1` 接口：

```env
PROMPT_AI_ENABLED=true
PROMPT_AI_BASE_URL=https://api.openai.com/v1
PROMPT_AI_API_KEY=your-key
PROMPT_AI_MODEL=your-model
PROMPT_AI_CONCURRENCY=4
```

`PROMPT_AI_API_KEY` 可以留空，方便连接不需要鉴权的本地 OpenAI-compatible 服务。批量 AI 优化最多一次处理 200 条，实际并发由 `PROMPT_AI_CONCURRENCY` 限制。

AI 优化后的 Prompt 不会自动发送到 GPU。UI 会保存原始 Prompt；生成完成后 Gallery 可以展开查看 AI 前原文。若 AI 后又手工编辑，或优化后切换了目标模型，这些状态也会写入任务的 Prompt 元数据。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/models` | 模型清单 |
| `GET` | `/api/prompt/pipeline` | AI Pipeline 状态 / 模式 |
| `POST` | `/api/prompt/process` | 单条 Prompt AI 处理 |
| `POST` | `/api/prompt/process/batch` | 批量 Prompt AI 处理 |
| `POST` | `/task` | 单任务入队 |
| `POST` | `/task/batch` | 批量入队 |
| `POST` | `/task/triposr` | 上传或引用图片，加入 TripoSR 队列 |
| `GET` | `/task/next?model=...&worker_id=...` | Worker 长轮询领取指定模型任务 |
| `GET` | `/task/input/{id}` | TripoSR Worker 鉴权下载输入图片 |
| `POST` | `/task/fail` | Worker 报告失败并按策略回队列 |
| `POST` | `/worker/register` | Worker 注册 |
| `POST` | `/worker/heartbeat` | Worker 心跳 |
| `POST` | `/upload` | 上传 AES-GCM 加密结果 |
| `POST` | `/upload/artifact` | 上传 AES-GCM 加密的 GLB/OBJ |
| `GET` | `/api/status` | 队列 / inflight / Worker 状态 |
| `GET` | `/api/history` | 生成历史 |
| `WS` | `/ws` | 结果实时推送 |

完整协议见 `docs/protocol.md`。

## 兼容性

旧 SANA 客户端如果仍然请求 `/task/next` 且不带 `model`，服务端默认路由到 `sana-sprint-1.6b`。旧 `/task`、`/task/batch`、`/upload` 也保留默认 SANA 行为，因此可以逐步迁移。

## 自检

```powershell
uv run python -m compileall -q recv.py hub scripts
uv run python scripts/self_test.py
```

运行时生成的图片保存在 `outputs/`，默认不进入 Git。
