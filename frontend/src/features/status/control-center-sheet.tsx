import { Activity, Database, KeyRound, Server } from 'lucide-react'

import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { WorkersPanel } from '@/features/workers/workers-panel'
import type { HubStatus, ModelSpec } from '@/shared/api/types'

type ControlCenterSheetProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  token: string
  onTokenChange: (token: string) => void
  status?: HubStatus
  models: ModelSpec[]
}

export function ControlCenterSheet({
  open,
  onOpenChange,
  token,
  onTokenChange,
  status,
  models,
}: ControlCenterSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full overflow-y-auto border-l-border bg-[var(--ctp-mantle)] p-0 sm:max-w-md">
        <SheetHeader className="border-b border-border px-6 py-6 text-left">
          <SheetTitle>控制中心</SheetTitle>
          <SheetDescription>访问凭据、运行状态与 Worker 健康信息</SheetDescription>
        </SheetHeader>

        <div className="space-y-6 px-6 py-6">
          <section className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium"><KeyRound className="size-4 text-primary" /> Access token</div>
            <div className="space-y-2">
              <Label htmlFor="control-token" className="sr-only">Access token</Label>
              <Input
                id="control-token"
                type="password"
                autoComplete="off"
                value={token}
                onChange={(event) => onTokenChange(event.target.value)}
                placeholder="KAGGLE_HUB_TOKEN"
                className="bg-[var(--ctp-base)] font-mono"
              />
              <p className="text-[11px] leading-relaxed text-muted-foreground">只保存在当前浏览器的 localStorage，用于需要鉴权的任务操作。</p>
            </div>
          </section>

          <Separator />

          <section className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium"><Activity className="size-4 text-primary" /> Hub 状态</div>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: 'Queued', value: status?.queued ?? 0, icon: Server },
                { label: 'Inflight', value: status?.inflight ?? 0, icon: Activity },
                { label: 'Results', value: status?.results ?? 0, icon: Database },
                { label: 'Failed', value: status?.failed ?? 0, icon: Activity },
              ].map((item) => (
                <div key={item.label} className="rounded-lg border border-border bg-[var(--ctp-base)] p-3">
                  <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-muted-foreground">
                    {item.label}<item.icon className="size-3" />
                  </div>
                  <div className="mt-2 font-mono text-xl font-semibold tabular-nums">{item.value}</div>
                </div>
              ))}
            </div>
          </section>

          <WorkersPanel workers={status?.workers ?? []} models={models} embedded />
        </div>
      </SheetContent>
    </Sheet>
  )
}
