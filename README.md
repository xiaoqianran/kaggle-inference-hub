# SANA Sprint 1.6B

基于 FastAPI 的 SANA Sprint 1.6B 双 GPU 远程生成控制服务：本地负责提交 Prompt 和接收图片，Kaggle GPU 负责模型推理。

## 环境要求

- Python `>=3.12`
- [uv](https://docs.astral.sh/uv/)

## 启动服务

```powershell
git clone https://github.com/xiaoqianran/kaggle-001-SANA-Sprint-1.6B.git
cd kaggle-001-SANA-Sprint-1.6B
uv sync
uv run python recv.py
```

服务启动后访问：<http://127.0.0.1:30100>

启动参数也可以直接使用 Uvicorn：

```powershell
uv run uvicorn recv:app --host 0.0.0.0 --port 30100
```

开发时自动重载：

```powershell
uv run uvicorn recv:app --host 0.0.0.0 --port 30100 --reload
```

## 控制台使用

1. 打开 <http://127.0.0.1:30100>。
2. 在 `ACCESS TOKEN` 中输入：`wangran`。
3. 在 Prompt 输入框中一行填写一个提示词。
4. 设置图片尺寸、Steps 和可选 Seed，点击“加入生成队列”。
5. 生成完成后，图片会自动显示在页面中，并保存到 `sana_received/`。

## Kaggle 双 GPU 推理

Notebook 位于：`noteboook/001-sana-sprint-1-6b.ipynb`。

在 Kaggle 中打开 Notebook 后，按顺序运行单元格：

1. 下载 SANA Sprint 1.6B 模型。
2. 加载 GPU0 和 GPU1 上的推理管线。
3. 启动 Dispatcher，轮询任务并将生成结果加密上传。

Notebook 默认连接已有的服务地址。如果更换服务地址，请修改 Notebook 中的 `BASE`、`TASK_URL` 或 `UPLOAD_URL`，并确保 Kaggle 能访问该地址。

## 常用 uv 命令

```powershell
# 安装或同步依赖
uv sync

# 添加依赖
uv add <package-name>

# 检查锁文件
uv lock --check

# 检查 Python 文件语法
uv run python -m py_compile main.py recv.py
```

## 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/status` | 查看队列和图片数量 |
| `POST` | `/task` | 提交单个 Prompt，需要 Bearer Token |
| `POST` | `/task/batch` | 批量提交 Prompt，需要 Bearer Token |
| `GET` | `/task/next` | Kaggle 长轮询领取任务 |
| `POST` | `/upload` | 接收加密生成结果 |
| `GET` | `/api/history` | 获取历史生成记录 |
| `WS` | `/ws` | 实时接收生成结果 |

需要鉴权的请求使用请求头：

```text
Authorization: Bearer wangran
```
