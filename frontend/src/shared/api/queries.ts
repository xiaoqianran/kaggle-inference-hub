import { queryOptions } from '@tanstack/react-query'

import { getHistory, getHubStatus, getModels, getPromptPipeline } from '@/shared/api/client'

export const queryKeys = {
  models: ['models'] as const,
  promptPipeline: ['prompt-pipeline'] as const,
  status: ['hub-status'] as const,
  history: ['history'] as const,
  activeTasks: ['active-tasks'] as const,
}

export const modelsQuery = queryOptions({
  queryKey: queryKeys.models,
  queryFn: getModels,
  staleTime: Number.POSITIVE_INFINITY,
})

export const promptPipelineQuery = queryOptions({
  queryKey: queryKeys.promptPipeline,
  queryFn: getPromptPipeline,
  staleTime: 30_000,
})

export const statusQuery = queryOptions({
  queryKey: queryKeys.status,
  queryFn: getHubStatus,
  refetchInterval: 1_500,
})

export const historyQuery = queryOptions({
  queryKey: queryKeys.history,
  queryFn: getHistory,
  refetchInterval: 5_000,
})
