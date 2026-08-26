import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { queueFastSam3DTask } from '@/shared/api/client'
import { queryKeys } from '@/shared/api/queries'
import type { FastSam3DSubmission } from '@/shared/api/types'

export function useSubmitFastSam3D(token: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (submission: FastSam3DSubmission) => queueFastSam3DTask(token, submission),
    onSuccess: (result) => {
      toast.success(`#${result.task.id} → Fast-SAM3D`)
      void queryClient.invalidateQueries({ queryKey: queryKeys.status })
    },
    onError: (error) => {
      toast.error(`Fast-SAM3D 提交失败：${error instanceof Error ? error.message : '未知错误'}`)
    },
  })
}
