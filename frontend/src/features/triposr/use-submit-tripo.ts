import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { queueTripoTask } from '@/shared/api/client'
import { queryKeys } from '@/shared/api/queries'
import type { TripoSubmission } from '@/shared/api/types'

export function useSubmitTripo(token: string, options?: { notify?: boolean }) {
  const queryClient = useQueryClient()
  const notify = options?.notify ?? true

  return useMutation({
    mutationFn: (submission: TripoSubmission) => queueTripoTask(token, submission),
    onSuccess: (result) => {
      if (notify) toast.success(`#${result.task.id} → TripoSR`)
      void queryClient.invalidateQueries({ queryKey: queryKeys.status })
    },
    onError: (error) => {
      if (notify) toast.error(`TripoSR 提交失败：${error instanceof Error ? error.message : '未知错误'}`)
    },
  })
}
