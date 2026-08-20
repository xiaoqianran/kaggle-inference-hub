# Kaggle Inference Hub

把 Kaggle 双 T4 Notebook 作为可远程调度的 GPU Worker。本地 FastAPI 负责 Prompt、可选 AI Prompt Pipeline、模型路由、队列、Worker 状态和生成结果；模型推理留在 Kaggle。

当前内置三种 Worker：

- **SANA Sprint 1.6B** — Diffusers，2×T4 各自常驻一份 Pipeline。
- **Z-Image-Turbo GGUF** — `stable-diffusion.cpp`，使用 `kaggle-build` Release 的 T4 `sm_75` 预编译 runtime，2×T4 各自常驻一个 `sd-server`。
- **TripoSR** — 官方单图 3D 重建模型，Kaggle Python 3.10 独立 venv；2×T4 各自常驻一份 TripoSR + GPU rembg Session，生成 GLB 或 OBJ。

## 架构

```text
Browser / Local API
        │
        ▼
Kaggle Inference Hub (FastAPI)
        │
        ├── SQLite durable state ── tasks / leases / workers / history
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

三个模型共享本地控制面，但**任务队列按模型隔离**，不同 Notebook 不会互相抢任务。任务、租约、Worker 和历史状态持久化到 `outputs/hub-state.sqlite3`；多个 Uvicorn 进程共享同一状态，Hub 重启后未完成任务仍然存在。

## 本地启动

仓库已经包含前端构建产物 `hub/web/`，所以日常直接启动只需要：

```powershell
uv sync
$env:KAGGLE_HUB_TOKEN="wangran"
uv run recv.py
```

然后打开：<http://127.0.0.1:30100>。`FastAPI` 会自动读取 `hub/web/index.html` 和其中的静态资源，不需要启动 Node.js。

只有在修改了 `frontend/` 源码后，才需要重新构建：

```powershell
pnpm --dir frontend install
pnpm --dir frontend build
uv run recv.py
```

日常前端开发使用两个终端。Vite 提供热更新，并把 REST、WebSocket、图片和产物路径代理到 FastAPI：

```powershell
# Terminal 1
uv run uvicorn recv:app --host 127.0.0.1 --port 30100 --reload

# Terminal 2
pnpm --dir frontend dev
```

开发页面：<http://127.0.0.1:5173>。生产构建输出到 `hub/web/`，运行时不需要 Node.js。

也可以：

```powershell
uv run uvicorn recv:app --host 0.0.0.0 --port 30100 --reload
```

也支持多进程：

```powershell
uv run uvicorn recv:app --host 0.0.0.0 --port 30100 --workers 4
```

## 打包源码

不要直接使用资源管理器的“压缩为 ZIP”，因为它不会读取 `.gitignore`。先构建前端，再生成 ZIP；脚本会自动排除 `.venv/`、`node_modules/`、`.git/`、缓存、`outputs/`、`sana_received/` 和本地配置：

```powershell
pnpm --dir frontend build
uv run python scripts/package.py
```

也可以指定输出路径：

```powershell
uv run python scripts/package.py --output D:\release\kaggle-image-inference-hub.zip
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
2. 对对应模型执行 25 秒 long polling；003 使用不可缓存的 `POST /task/claim`，001/002 的旧 `GET /task/next` 继续兼容。
3. 双 GPU 并行处理。
4. 将 WebP 或 3D 文件通过 AES-GCM 加密上传。
5. 周期性发送心跳；003 同时续租正在推理的任务。
6. 失败时通知本地服务进行有限次数重试。

TripoSR 的 003 Notebook 采用专门的常驻架构：Notebook 内核仍可为 Python 3.12，但通过 Kaggle 自带 `uv` 创建 `/kaggle/working/TripoSR/.venv`（Python 3.10）。Notebook 只负责写入并启动 `kaggle_worker.py`；真正的 TripoSR、PyTorch、rembg 都只在这个 Python 3.10 进程体系中运行。父进程先缓存模型，然后为 GPU0/GPU1 各启动一个独立 Worker 进程，因此模型初始化只发生一次。

TripoSR Notebook 需要在 Kaggle 启用 **GPU T4 x2** 与 **Internet**。当前 003 明确固定 Hub 为 `https://ranran-sana.202820.xyz`，Token 为 `wangran`；如需迁移，在启动 Cell 中同步修改这两项。本地 UI 可以直接上传 PNG/JPEG/WebP，也可以在 SANA、Z-Image 或其他接入 Hub 的生成图卡片上点击“转为 3D”。默认输出带顶点色的 GLB，也可选择 OBJ。

TripoSR Runtime 固定为 PyTorch `2.7.1+cu128`，`torchmcubes` 直接从 `xiaoqianran/kaggle-build` 的 `triposr-py310-torch2.7.1-cu128-sm75` Release 安装预编译 wheel，不再在 Kaggle 上执行 CMake/NVCC。`rembg[gpu]` 使用 `onnxruntime-gpu`，并将两个 ONNX Session 分别绑定到 `device_id=0/1`；默认 `u2net`，如更重视速度可在启动 Cell 改为 `u2netp`。

其他本地模型也可以把刚生成的文件直接送入队列：

```powershell
curl.exe -X POST http://127.0.0.1:30100/task/triposr `
  -H "Authorization: Bearer $env:KAGGLE_HUB_TOKEN" `
  -F "file=@D:\path\to\generated.png" `
  -F "output_format=glb" -F "mc_resolution=256"
```

## 本地 UI

前端采用 Vite + React + TypeScript + shadcn/ui + Tailwind CSS，视觉主题为 Catppuccin Frappé。TanStack Query 管理 Hub 服务端状态，React Hook Form + Zod 管理生成参数，WebSocket 事件直接更新 Query Cache，并保留定时同步兜底。FastAPI 继续作为唯一生产服务端。

界面按两级导航组织：高 Header 切换图片、3D、视频工作区；其下 Topbar 切换当前工作区的模型。图片 Gallery 按模型隔离展示，3D 上传与资产库只在 3D 工作区出现；Access Token、Worker 和 Hub 诊断信息集中在右上角控制中心。

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
| `POST` | `/task/claim` | 新 Worker 不可缓存的原子长轮询领取接口 |
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

旧 SANA 客户端如果仍然请求 `/task/next` 且不带 `model`，服务端默认路由到 `sana-sprint-1.6b`。旧 `/task`、`/task/batch`、`/upload` 也保留默认 SANA 行为，因此可以逐步迁移。动态 HTTP 响应均带 `Cache-Control: no-store` 和稳定的 `X-Hub-Instance`，便于识别 Tunnel 是否错误地连接到不同 Hub 数据库；带内容哈希的前端 `/assets/` 使用长期缓存。

## 自检

```powershell
pnpm --dir frontend typecheck
pnpm --dir frontend lint
pnpm --dir frontend build
uv run python -m compileall -q recv.py hub scripts
uv run python scripts/self_test.py
```

运行时生成的图片保存在 `outputs/`，默认不进入 Git。
