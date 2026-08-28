import type {
  ArtifactSubmission,
  BatchTaskRequest,
  HistoryItem,
  HubStatus,
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
