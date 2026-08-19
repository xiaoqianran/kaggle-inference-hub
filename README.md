# Kaggle Inference Hub

把 Kaggle 双 T4 Notebook 作为可远程调度的 GPU Worker。本地 FastAPI 只负责 Prompt、模型路由、队列、Worker 状态和生成结果；模型推理留在 Kaggle。

当前内置两种 Worker：

- **SANA Sprint 1.6B** — Diffusers，2×T4 各自常驻一份 Pipeline。
- **Z-Image-Turbo GGUF** — `stable-diffusion.cpp`，使用 `kaggle-build` Release 的 T4 `sm_75` 预编译 runtime，2×T4 各自常驻一个 `sd-server`。

## 架构

```text
Browser / Local API
        │
        ▼
Kaggle Inference Hub (FastAPI)
        │
        ├── queue: sana-sprint-1.6b ─────► Kaggle SANA Worker (2×T4)
        │
        └── queue: z-image-turbo-gguf ───► Kaggle Z-Image Worker (2×T4)
                                              │
                         encrypted WebP ◄─────┘
        │
        ▼
outputs/<model>/ + Live Gallery
```

两个模型共享本地控制面，但**任务队列按模型隔离**，不会发生 SANA Notebook 抢到 Z-Image 任务。

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

把得到的公网 HTTPS 地址填到两个 Notebook 的 `BASE`。如果你使用固定域名 Tunnel，也只需要把 `BASE` 指向该域名。

## Notebook

- `notebooks/001-sana-sprint-1.6b.ipynb`
- `notebooks/002-z-image-turbo-gguf.ipynb`

两个 Notebook 都会：

1. 注册自己的 `worker_id + model`。
2. 对对应模型执行 25 秒 long polling。
3. 双 GPU 并行处理。
4. 将 WebP 通过 AES-GCM 加密上传。
5. 周期性发送心跳。
6. 失败时通知本地服务进行有限次数重试。

## 本地 UI

UI 中先选择模型，然后提交 Prompt。单 Prompt 输入框支持任意换行；批量输入框才采用“一行一个任务”。切换模型时 Steps 自动切换到模型默认值：

- SANA：2 steps
- Z-Image-Turbo：8 steps

页面同时展示模型队列、inflight 数量、在线 Worker 和统一图片 Gallery。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/models` | 模型清单 |
| `POST` | `/task` | 单任务入队 |
| `POST` | `/task/batch` | 批量入队 |
| `GET` | `/task/next?model=...&worker_id=...` | Worker 长轮询领取指定模型任务 |
| `POST` | `/task/fail` | Worker 报告失败并按策略回队列 |
| `POST` | `/worker/register` | Worker 注册 |
| `POST` | `/worker/heartbeat` | Worker 心跳 |
| `POST` | `/upload` | 上传 AES-GCM 加密结果 |
| `GET` | `/api/status` | 队列 / inflight / Worker 状态 |
| `GET` | `/api/history` | 生成历史 |
| `WS` | `/ws` | 结果实时推送 |

完整协议见 `docs/protocol.md`。

## 兼容性

旧 SANA 客户端如果仍然请求 `/task/next` 且不带 `model`，服务端默认路由到 `sana-sprint-1.6b`。旧 `/task`、`/task/batch`、`/upload` 也保留默认 SANA 行为，因此可以逐步迁移。

## 自检

```powershell
uv run python -m py_compile recv.py main.py hub/*.py
uv run python scripts/self_test.py
```

运行时生成的图片保存在 `outputs/`，默认不进入 Git。
