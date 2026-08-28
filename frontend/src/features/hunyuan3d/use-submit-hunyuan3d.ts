import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { queueHunyuan3DTask } from '@/shared/api/client'
import { queryKeys } from '@/shared/api/queries'
import type { Hunyuan3DSubmission } from '@/shared/api/types'

export function useSubmitHunyuan3D(token: string) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (submission: Hunyuan3DSubmission) => queueHunyuan3DTask(token, submission),
    onSuccess: (result) => {
      toast.success(`#${result.task.id} → Hunyuan3D 2.1`)
      void queryClient.invalidateQueries({ queryKey: queryKeys.status })
    },
    onError: (error) => {
      toast.error(`Hunyuan3D 2.1 提交失败：${error instanceof Error ? error.message : '未知错误'}`)
    },
  })
}
