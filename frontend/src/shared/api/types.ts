import type { TripoResolution } from '@/shared/tripo-resolution'

export type ModelSpec = {
  id: string
  label: string
  default_steps: number
  description: string
  input_kind: 'prompt' | 'image'
  output_kind: 'image' | 'artifact'
}

export type PromptMode = {
  id: string
  label: string
}

export type PromptPipelineConfig = {
  enabled: boolean
  configured: boolean
  provider_model: string | null
  concurrency: number
  modes: PromptMode[]
}

export type PromptProcessRequest = {
  prompt: string
  model: string
  mode: string
  translate_to_english: boolean
}

export type PromptProcessResult = {
  original: string
  processed: string
  target_model: string
  mode: string
  translate_to_english: boolean
  provider_model: string
  elapsed_ms: number
}

export type PromptBatchItem = PromptProcessResult & {
  ok: boolean
  error?: string
}

export type PromptBatchResult = {
  total: number
  succeeded: number
  failed: number
  items: PromptBatchItem[]
}

export type PromptMeta = {
  mode?: string
  provider_model?: string
  elapsed_ms?: number
  translate_to_english?: boolean
  target_model?: string
  edited_after_ai?: boolean
  stale_model_adapter?: boolean
}

export type GenerationParams = {
  model: string
  width: number
  height: number
  steps: number
  seed?: number
}

export type SingleTaskRequest = GenerationParams & {
  prompt: string
  source_prompt?: string
  prompt_meta?: PromptMeta
}

export type BatchTaskRequest = GenerationParams & {
  prompts: string[]
  source_prompts?: string[]
  prompt_metas?: PromptMeta[]
}

export type QueuedTask = {
  id: number
  model: string
}

export type QueueResponse = {
  queued: number
  queue_size: number
  task: QueuedTask
  tasks?: QueuedTask[]
}

export type TripoSettings = {
  outputFormat: 'glb' | 'obj'
  resolution: TripoResolution
  removeBackground: boolean
}

export type TripoSubmission =
  | { kind: 'file'; file: File; settings: TripoSettings }
  | { kind: 'source'; sourceUrl: string; settings: TripoSettings }

export type Worker = {
  worker_id: string
  model: string
  gpus?: string[]
  runtime?: string
  concurrency?: number
  local_queue?: number
  upload_queue?: number
  active_task_id?: number | null
  online: boolean
  last_seen?: number
}

export type HubStatus = {
  hub_instance_id: string
  process_id: number
  storage: string
  queued: number
  queued_by_model: Record<string, number>
  inflight: number
  inflight_by_model: Record<string, number>
  results: number
  images: number
  artifacts: number
  failed: number
  workers: Worker[]
}

type ResultBase = {
  event_id: number
  id: number
  model: string
  worker_id?: string
  gpu: number
  seconds: number
  time: number
}

export type ImageResult = ResultBase & {
  kind: 'image'
  seed: number
  steps?: number
  prompt: string
  source_prompt?: string
  prompt_meta?: PromptMeta
  url: string
}

export type ArtifactResult = ResultBase & {
  kind: 'artifact'
  source_url?: string
  source_label?: string
  output_format: 'glb' | 'obj'
  download_url: string
  vertices?: number
  faces?: number
  mc_resolution?: number
  remove_background?: boolean
}

export type HistoryItem = ImageResult | ArtifactResult
