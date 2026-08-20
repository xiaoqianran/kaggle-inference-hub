import { Box, Image, Video } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { WorkspaceKind } from '@/features/status/workspace-header'
import type { HubStatus, ModelSpec } from '@/shared/api/types'

type ModelTopbarProps = {
  workspace: WorkspaceKind
  models: ModelSpec[]
  selectedModel: string
  onSelectedModelChange: (model: string) => void
  status?: HubStatus
  resultCounts: Record<string, number>
}

export function ModelTopbar({
  workspace,
  models,
  selectedModel,
  onSelectedModelChange,
  status,
  resultCounts,
}: ModelTopbarProps) {
  const visibleModels = models.filter((model) => {
    if (workspace === 'image') return model.output_kind === 'image'
    if (workspace === '3d') return model.output_kind === 'artifact'
    return false
  })
  const WorkspaceIcon = workspace === 'image' ? Image : workspace === '3d' ? Box : Video

  return (
    <div className="sticky top-0 z-30 border-b border-border bg-[var(--ctp-base)]/92 backdrop-blur-xl">
      <div className="mx-auto flex h-14 w-full max-w-[1880px] items-center gap-3 px-4 sm:px-6 lg:px-8">
        <div className="flex shrink-0 items-center gap-2 border-r border-border pr-3 text-xs font-medium text-muted-foreground sm:pr-5">
          <WorkspaceIcon className="size-4 text-primary" />
          <span className="hidden sm:inline">模型</span>
        </div>

        <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {visibleModels.map((model) => {
            const active = selectedModel === model.id || visibleModels.length === 1
            return (
              <button
                key={model.id}
                type="button"
                onClick={() => onSelectedModelChange(model.id)}
                className={cn(
                  'flex h-9 shrink-0 items-center gap-2 rounded-lg px-3 text-xs font-medium transition',
                  active ? 'bg-[var(--ctp-surface0)] text-foreground' : 'text-muted-foreground hover:bg-[var(--ctp-surface0)]/55 hover:text-foreground',
                )}
              >
                <span className={cn('size-1.5 rounded-full', active ? 'bg-primary' : 'bg-[var(--ctp-overlay0)]')} />
                {model.label}
                <span className="font-mono text-[9px] text-[var(--ctp-overlay0)]">{resultCounts[model.id] ?? 0}</span>
              </button>
            )
          })}
          {workspace === 'video' ? (
            <span className="flex h-9 items-center px-3 text-xs text-muted-foreground">视频模型接入后会显示在这里</span>
          ) : null}
        </div>

        <div className="hidden shrink-0 items-center gap-2 md:flex">
          <Badge variant="outline" className="font-mono text-[9px] uppercase">Queue {status?.queued ?? 0}</Badge>
          <Badge variant="outline" className="font-mono text-[9px] uppercase">Running {status?.inflight ?? 0}</Badge>
        </div>
      </div>
    </div>
  )
}
