import type {
  ArtifactSource,
  ArtifactSubmission,
  ActiveTask,
  CancelTaskResult,
  BatchTaskRequest,
  HistoryItem,
  HubStatus,
  MaskTaskStatus,
  ModelSpec,
  PromptBatchResult,
  PromptPipelineConfig,
  PromptProcessRequest,
  PromptProcessResult,
  QueueResponse,
  SingleTaskRequest,
} from '@/shared/api/types'

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  const text = await response.text()
  let payload: unknown = null

  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = text
    }
  }

  if (!response.ok) {
    const body = payload as { detail?: string; error?: string } | null
    throw new ApiError(body?.detail ?? body?.error ?? `HTTP ${response.status}`, response.status)
  }

  return payload as T
}

function authorizedJson(token: string): HeadersInit {
  return {
    Authorization: `Bearer ${token}`,
    'Content-Type': 'application/json',
  }
}

export function getModels(): Promise<ModelSpec[]> {
  return requestJson('/api/models')
}

export function getPromptPipeline(): Promise<PromptPipelineConfig> {
  return requestJson('/api/prompt/pipeline')
}

export function getHubStatus(): Promise<HubStatus> {
  return requestJson('/api/status')
}

export function getHistory(): Promise<HistoryItem[]> {
  return requestJson('/api/history?limit=300')
}

export function getActiveTasks(token: string, model?: string): Promise<ActiveTask[]> {
  const query = model ? `?model=${encodeURIComponent(model)}&limit=300` : '?limit=300'
  return requestJson(`/api/tasks${query}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
}

export function cancelTask(token: string, taskId: number): Promise<CancelTaskResult> {
  return requestJson(`/task/${taskId}/cancel`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
  })
}

export function processPrompt(token: string, input: PromptProcessRequest): Promise<PromptProcessResult> {
  return requestJson('/api/prompt/process', {
    method: 'POST',
    headers: authorizedJson(token),
    body: JSON.stringify(input),
  })
}

export function processPromptBatch(
  token: string,
  input: Omit<PromptProcessRequest, 'prompt'> & { prompts: string[] },
): Promise<PromptBatchResult> {
  return requestJson('/api/prompt/process/batch', {
    method: 'POST',
    headers: authorizedJson(token),
    body: JSON.stringify(input),
  })
}

export function queueSingleTask(token: string, input: SingleTaskRequest): Promise<QueueResponse> {
  return requestJson('/task', {
    method: 'POST',
    headers: authorizedJson(token),
    body: JSON.stringify(input),
  })
}

export function queueBatchTasks(token: string, input: BatchTaskRequest): Promise<QueueResponse> {
  return requestJson('/task/batch', {
    method: 'POST',
    headers: authorizedJson(token),
    body: JSON.stringify(input),
  })
}

export function queueArtifactTask(token: string, submission: ArtifactSubmission): Promise<QueueResponse> {
  const form = new FormData()
  form.append('options', JSON.stringify(submission.options))
  if (submission.source.kind === 'file') {
    form.append('file', submission.source.file, submission.source.file.name)
  } else {
    form.append('source_url', submission.source.sourceUrl)
  }
  for (const [name, file] of Object.entries(submission.auxiliaryFiles ?? {})) {
    form.append(name, file, file.name)
  }

  return requestJson(`/task/artifact/${encodeURIComponent(submission.model)}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  })
}

export function queueAutoMask(token: string, source: ArtifactSource): Promise<QueueResponse> {
  const form = new FormData()
  if (source.kind === 'file') {
    form.append('file', source.file, source.file.name)
  } else {
    form.append('source_url', source.sourceUrl)
  }
  return requestJson('/mask/auto', {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  })
}

export function getMaskStatus(token: string, taskId: number): Promise<MaskTaskStatus> {
  return requestJson(`/mask/${taskId}?_ts=${Date.now()}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
}

export async function getMaskCandidate(token: string, path: string): Promise<Blob> {
  const response = await fetch(path, {
    headers: { Authorization: `Bearer ${token}` },
    cache: 'no-store',
  })
  if (!response.ok) {
    const text = await response.text()
    throw new ApiError(text || `HTTP ${response.status}`, response.status)
  }
  return response.blob()
}


export async function createAutoMaskFile(token: string, source: ArtifactSource): Promise<File> {
  const queued = await queueAutoMask(token, source)
  const taskId = queued.task.id
  for (let attempt = 0; attempt < 180; attempt += 1) {
    const status = await getMaskStatus(token, taskId)
    if (status.status === 'ready') {
      const candidate = status.candidates?.[0]
      if (!candidate) throw new Error(`Mask #${taskId} 没有候选结果`)
      const blob = await getMaskCandidate(token, candidate.url)
      return new File([blob], `fast-sam3d-mask-${taskId}.png`, { type: 'image/png' })
    }
    if (status.status === 'failed') throw new Error(status.error ?? `Mask #${taskId} 失败`)
    await new Promise((resolve) => setTimeout(resolve, 1000))
  }
  throw new Error(`Mask #${taskId} 等待超时`)
}
