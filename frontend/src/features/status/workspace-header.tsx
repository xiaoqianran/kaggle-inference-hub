import { Box, Image, Radio, Settings2, Video } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { ConnectionState } from '@/hooks/use-hub-socket'
import type { HubStatus } from '@/shared/api/types'

export type WorkspaceKind = 'image' | '3d' | 'video'

const workspaces = [
  { id: 'image' as const, label: '图片', hint: 'Text to Image', icon: Image },
  { id: '3d' as const, label: '3D', hint: 'Image to Mesh', icon: Box },
  { id: 'video' as const, label: '视频', hint: 'Coming soon', icon: Video },
]

type WorkspaceHeaderProps = {
  workspace: WorkspaceKind
  onWorkspaceChange: (workspace: WorkspaceKind) => void
  connection: ConnectionState
  status?: HubStatus
  onOpenControlCenter: () => void
}

export function WorkspaceHeader({
  workspace,
  onWorkspaceChange,
  connection,
  status,
  onOpenControlCenter,
}: WorkspaceHeaderProps) {
  return (
    <header className="border-b border-border bg-[var(--ctp-mantle)]">
      <div className="mx-auto w-full max-w-[1880px] px-4 py-5 sm:px-6 lg:px-8 lg:py-7">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="mb-1.5 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.22em] text-primary">
              <span className="size-1.5 rounded-full bg-[var(--ctp-green)] shadow-[0_0_12px_var(--ctp-green)]" />
              Kaggle GPU Control Plane
            </div>
            <h1 className="truncate text-xl font-semibold tracking-tight text-foreground sm:text-2xl">
              Inference Hub
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <Badge variant="outline" className="hidden h-9 gap-2 bg-[var(--ctp-crust)]/35 px-3 font-mono text-[10px] uppercase sm:flex">
              <Radio className={cn('size-3', connection === 'online' ? 'text-[var(--ctp-green)]' : 'text-[var(--ctp-yellow)]')} />
              {connection}
            </Badge>
            <Button type="button" variant="outline" size="sm" className="h-9" onClick={onOpenControlCenter}>
              <Settings2 className="size-4" />
              <span className="hidden sm:inline">控制中心</span>
              {(status?.failed ?? 0) > 0 ? (
                <span className="flex min-w-5 items-center justify-center rounded-full bg-destructive px-1.5 font-mono text-[9px] text-white">
                  {status?.failed}
                </span>
              ) : null}
            </Button>
          </div>
        </div>

        <nav aria-label="媒体工作区" className="mt-6 grid grid-cols-3 gap-2 sm:max-w-xl sm:gap-3">
          {workspaces.map((item) => {
            const Icon = item.icon
            const active = workspace === item.id
            return (
              <button
                key={item.id}
                type="button"
                aria-current={active ? 'page' : undefined}
                onClick={() => onWorkspaceChange(item.id)}
                className={cn(
                  'group flex min-w-0 items-center gap-3 rounded-xl border px-3 py-3 text-left transition duration-200 sm:px-4',
                  active
                    ? 'border-primary/50 bg-primary/12 shadow-[inset_0_0_0_1px_color-mix(in_oklab,var(--primary)_12%,transparent)]'
                    : 'border-border bg-[var(--ctp-base)]/40 hover:border-[var(--ctp-surface1)] hover:bg-[var(--ctp-base)]/70',
                )}
              >
                <span className={cn('flex size-9 shrink-0 items-center justify-center rounded-lg transition', active ? 'bg-primary text-primary-foreground' : 'bg-[var(--ctp-surface0)] text-muted-foreground group-hover:text-foreground')}>
                  <Icon className="size-4" />
                </span>
                <span className="min-w-0">
                  <span className={cn('block text-sm font-medium', active ? 'text-foreground' : 'text-muted-foreground')}>{item.label}</span>
                  <span className="hidden truncate font-mono text-[9px] uppercase tracking-wider text-[var(--ctp-overlay0)] sm:block">{item.hint}</span>
                </span>
              </button>
            )
          })}
        </nav>
      </div>
    </header>
  )
}
