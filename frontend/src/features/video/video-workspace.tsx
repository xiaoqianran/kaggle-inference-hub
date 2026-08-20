import { Clapperboard, Video } from 'lucide-react'

import { Badge } from '@/components/ui/badge'

export function VideoWorkspace() {
  return (
    <section className="col-span-full flex min-h-[56svh] items-center justify-center rounded-2xl border border-dashed border-border bg-[var(--ctp-mantle)]/40 px-6 text-center">
      <div className="max-w-md">
        <span className="mx-auto flex size-16 items-center justify-center rounded-2xl border border-[var(--ctp-mauve)]/30 bg-[var(--ctp-mauve)]/10 text-[var(--ctp-mauve)]">
          <Video className="size-7" />
        </span>
        <Badge variant="secondary" className="mt-5 gap-1.5"><Clapperboard className="size-3" /> ROADMAP</Badge>
        <h2 className="mt-4 text-xl font-semibold">视频工作区已预留</h2>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          视频 Worker 接入后，会复用同一套模型 Topbar、任务队列和按模型隔离的资产库，不需要再次重做页面架构。
        </p>
      </div>
    </section>
  )
}
