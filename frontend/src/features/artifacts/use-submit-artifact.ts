import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { queueArtifactTask } from '@/shared/api/client'
import { queryKeys } from '@/shared/api/queries'
import type { ArtifactBatchResult, ArtifactSubmission, ModelSpec } from '@/shared/api/types'

function labelOf(submission: ArtifactSubmission, index: number): string {
  return submission.source.kind === 'file' ? submission.source.file.name : submission.source.sourceUrl.split('/').at(-1) || `#${index + 1}`
}

async function queueWithConcurrency(
  token: string,
  submissions: ArtifactSubmission[],
  concurrency = 3,
): Promise<ArtifactBatchResult> {
  const tasks: ArtifactBatchResult['tasks'] = []
  const successes: ArtifactBatchResult['successes'] = []
  const failures: ArtifactBatchResult['failures'] = []
  let cursor = 0

  async function worker() {
    while (true) {
      const index = cursor
      cursor += 1
      if (index >= submissions.length) return
      const submission = submissions[index]
      try {
        const result = await queueArtifactTask(token, submission)
        tasks.push(result.task)
        successes.push({ index, task: result.task })
      } catch (error) {
        failures.push({
          index,
          label: labelOf(submission, index),
          error: error instanceof Error ? error.message : '未知错误',
        })
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, submissions.length) }, () => worker()))
  return {
    total: submissions.length,
    queued: tasks.length,
    tasks,
    successes: successes.toSorted((a, b) => a.index - b.index),
    failures: failures.toSorted((a, b) => a.index - b.index),
  }
}

function invalidateTaskViews(queryClient: ReturnType<typeof useQueryClient>) {
  void queryClient.invalidateQueries({ queryKey: queryKeys.status })
  void queryClient.invalidateQueries({ queryKey: queryKeys.activeTasks })
}

export function useSubmitArtifact(token: string, model: ModelSpec) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (submission: ArtifactSubmission) => queueArtifactTask(token, submission),
    onSuccess: (result) => {
      toast.success(`#${result.task.id} → ${model.label}`)
      invalidateTaskViews(queryClient)
    },
    onError: (error) => {
      toast.error(`${model.label} 提交失败：${error instanceof Error ? error.message : '未知错误'}`)
    },
  })
}

export function useSubmitArtifactBatch(token: string, model: ModelSpec) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (submissions: ArtifactSubmission[]) => queueWithConcurrency(token, submissions),
    onSuccess: (result) => {
      invalidateTaskViews(queryClient)
      if (!result.failures.length) toast.success(`已加入 ${result.queued} 个 ${model.label} 任务`)
      else if (result.queued) toast.warning(`已加入 ${result.queued}/${result.total} 个任务，${result.failures.length} 个失败`)
      else toast.error(`${model.label} 批量提交全部失败`)
    },
    onError: (error) => {
      toast.error(`${model.label} 批量提交失败：${error instanceof Error ? error.message : '未知错误'}`)
    },
  })
}
