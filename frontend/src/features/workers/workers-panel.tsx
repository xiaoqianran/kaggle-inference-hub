import { Cpu } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { ModelSpec, Worker } from '@/shared/api/types'

type WorkersPanelProps = {
  workers: Worker[]
  models: ModelSpec[]
  embedded?: boolean
}

export function WorkersPanel({ workers, models, embedded = false }: WorkersPanelProps) {
  const labels = new Map(models.map((model) => [model.id, model.label]))
  const ordered = workers.toSorted((left, right) => Number(right.online) - Number(left.online))

  const content = (
    <div className="space-y-2">
      {ordered.length ? (
        ordered.map((worker) => (
          <div key={worker.worker_id} className="rounded-lg border border-border bg-[var(--ctp-base)] p-3">
            <div className="flex items-center justify-between gap-3">
              <span className="truncate font-mono text-xs font-medium">{worker.worker_id}</span>
              <Badge variant={worker.online ? 'default' : 'secondary'} className="shrink-0 text-[9px]">
                <span className={`mr-1.5 size-1.5 rounded-full ${worker.online ? 'bg-[var(--ctp-green)]' : 'bg-[var(--ctp-overlay0)]'}`} />
                {worker.online ? 'ONLINE' : 'OFFLINE'}
              </Badge>
            </div>
            <p className="mt-2 truncate text-[11px] text-muted-foreground">
              {labels.get(worker.model) ?? worker.model} · {worker.gpus?.join(' + ') || 'GPU'}
            </p>
            <div className="mt-2 flex gap-3 font-mono text-[9px] uppercase tracking-wider text-[var(--ctp-overlay0)]">
              <span>Local {worker.local_queue ?? 0}</span>
              <span>Upload {worker.upload_queue ?? 0}</span>
              {worker.active_task_id ? <span>Task #{worker.active_task_id}</span> : null}
            </div>
          </div>
        ))
      ) : (
        <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-sm text-muted-foreground">
          暂无 Worker
        </div>
      )}
    </div>
  )

  if (embedded) {
    return (
      <section className="space-y-3">
        <div>
          <h3 className="flex items-center gap-2 text-sm font-medium"><Cpu className="size-4 text-primary" /> Workers</h3>
          <p className="mt-1 text-[11px] text-muted-foreground">45 秒内收到心跳视为在线</p>
        </div>
        {content}
      </section>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Cpu className="size-4 text-primary" /> Workers
        </CardTitle>
        <CardDescription>45 秒内收到心跳视为在线</CardDescription>
      </CardHeader>
      <CardContent>{content}</CardContent>
    </Card>
  )
}
