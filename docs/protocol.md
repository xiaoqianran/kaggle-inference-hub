# Worker Protocol

所有需要鉴权的请求都使用：

```text
Authorization: Bearer <KAGGLE_HUB_TOKEN>
```

## 模型 ID

- `sana-sprint-1.6b`
- `z-image-turbo-gguf`

## 1. Worker 注册

`POST /worker/register`

```json
{
  "worker_id": "sana-a1b2c3d4",
  "model": "sana-sprint-1.6b",
  "gpus": ["Tesla T4", "Tesla T4"],
  "runtime": "diffusers",
  "concurrency": 2
}
```

## 2. Worker 心跳

`POST /worker/heartbeat`，建议每 10 秒一次。

## 3. 长轮询领取任务

```text
GET /task/next?model=sana-sprint-1.6b&worker_id=sana-a1b2c3d4
GET /task/next?model=z-image-turbo-gguf&worker_id=zimage-a1b2c3d4
```

服务端最多等待 25 秒。无任务返回 `204`。领取成功后任务进入 `inflight`。Worker 异常消失且租约超时后，任务自动回到对应模型队列。

## 4. 上传结果

`POST /upload`，multipart 字段：`file`、`id`、`model`、`worker_id`、`gpu`、`seed`、`prompt`、`seconds`、`steps`。

`file` 为：

```text
WebP -> AES-GCM -> nonce(12 bytes) + ciphertext
```

密钥：`SHA256(KAGGLE_HUB_TOKEN)`。

## 5. 任务失败

`POST /task/fail`：

```json
{"id": 42, "error": "CUDA OOM", "requeue": true}
```

在 `KAGGLE_HUB_MAX_ATTEMPTS` 内自动回队列，超过后进入失败记录。
