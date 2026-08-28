import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { queueArtifactTask } from '@/shared/api/client'
import { queryKeys } from '@/shared/api/queries'
import type { ArtifactSubmission, ModelSpec } from '@/shared/api/types'

export function useSubmitArtifact(token: string, model: ModelSpec) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (submission: ArtifactSubmission) => queueArtifactTask(token, submission),
    onSuccess: (result) => {
      toast.success(`#${result.task.id} → ${model.label}`)
      void queryClient.invalidateQueries({ queryKey: queryKeys.status })
    },
    onError: (error) => {
      toast.error(`${model.label} 提交失败：${error instanceof Error ? error.message : '未知错误'}`)
    },
  })
}
