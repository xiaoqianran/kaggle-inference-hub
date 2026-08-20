import { Box, Clock3, Download, ExternalLink, Sparkles } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import type { HistoryItem } from '@/shared/api/types'

type ResultCardProps = {
  item: HistoryItem
  modelLabel: string
  onConvertTo3d?: (sourceUrl: string) => void
  isConverting?: boolean
}

const resultDateFormatter = new Intl.DateTimeFormat('zh-CN', {
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
})

export function ResultCard({ item, modelLabel, onConvertTo3d, isConverting = false }: ResultCardProps) {
  const createdAt = resultDateFormatter.format(new Date(item.time * 1000))

  if (item.kind === 'artifact') {
    const format = item.output_format.toUpperCase()
    return (
      <Card className="result-card group overflow-hidden border-border bg-card py-0 shadow-sm transition hover:-translate-y-0.5 hover:border-primary/45 hover:shadow-lg">
        <div className="relative aspect-square overflow-hidden bg-[var(--ctp-mantle)]">
          {item.source_url ? (
            <img
              src={`${item.source_url}?t=${item.time}`}
              alt={item.source_label ?? 'TripoSR input'}
              loading="lazy"
              decoding="async"
              className="size-full object-cover transition duration-500 group-hover:scale-[1.02]"
            />
          ) : (
            <div className="flex size-full items-center justify-center"><Box className="size-10 text-muted-foreground" /></div>
          )}
          <Badge className="absolute left-3 top-3 gap-1.5 shadow-md"><Box className="size-3" /> 3D READY</Badge>
          <span className="absolute right-3 top-3 rounded-md bg-[var(--ctp-crust)]/85 px-2 py-1 font-mono text-[9px] text-[var(--ctp-subtext0)] backdrop-blur">#{item.id}</span>
        </div>
        <CardContent className="flex min-h-44 flex-col p-4">
          <div className="min-w-0">
            <p className="truncate text-sm font-medium">{item.source_label ?? 'Input image'}</p>
            <p className="mt-1 font-mono text-[9px] uppercase tracking-wide text-muted-foreground">{modelLabel} · GPU{item.gpu} · {item.seconds}s</p>
          </div>
          <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[9px] uppercase text-[var(--ctp-overlay1)]">
            <span>{format}</span>
            {item.vertices ? <span>{item.vertices.toLocaleString()} vertices</span> : null}
            {item.faces ? <span>{item.faces.toLocaleString()} faces</span> : null}
          </div>
          <Button asChild className="mt-auto w-full">
            <a href={item.download_url} download><Download className="size-4" /> 下载 {format}</a>
          </Button>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="result-card group overflow-hidden border-border bg-card py-0 shadow-sm transition hover:-translate-y-0.5 hover:border-primary/45 hover:shadow-lg">
      <a href={item.url} target="_blank" rel="noreferrer" className="relative block aspect-square overflow-hidden bg-[var(--ctp-mantle)]">
        <img
          src={`${item.url}?t=${item.time}`}
          alt={item.prompt || `Generated image ${item.id}`}
          loading="lazy"
          decoding="async"
          className="size-full object-cover transition duration-500 group-hover:scale-[1.02]"
        />
        <span className="absolute right-3 top-3 flex size-8 translate-y-1 items-center justify-center rounded-full bg-[var(--ctp-crust)]/80 text-foreground opacity-0 backdrop-blur transition group-hover:translate-y-0 group-hover:opacity-100">
          <ExternalLink className="size-3.5" />
        </span>
        {item.prompt_meta?.mode ? (
          <Badge variant="secondary" className="absolute left-3 top-3 gap-1 bg-[var(--ctp-mantle)]/90 text-[9px] backdrop-blur">
            <Sparkles className="size-3 text-[var(--ctp-yellow)]" /> {item.prompt_meta.mode}
          </Badge>
        ) : null}
      </a>

      <CardContent className="flex min-h-48 flex-col p-4">
        <div className="flex items-center justify-between gap-3 font-mono text-[9px] uppercase tracking-wide text-muted-foreground">
          <span className="flex items-center gap-1.5"><Clock3 className="size-3" /> {createdAt}</span>
          <span>#{item.id} · GPU{item.gpu}</span>
        </div>
        <p className="mt-3 line-clamp-3 whitespace-pre-wrap text-xs leading-relaxed text-foreground/85">{item.prompt}</p>
        <div className="mt-3 font-mono text-[9px] uppercase text-[var(--ctp-overlay1)]">
          {item.seconds}s · seed {item.seed}{item.steps ? ` · ${item.steps} steps` : ''}
        </div>

        {item.source_prompt ? (
          <details className="mt-3 rounded-md border border-border bg-[var(--ctp-mantle)] px-3 py-2 text-[11px] text-muted-foreground">
            <summary className="cursor-pointer select-none font-medium text-foreground/75">AI 前原始 Prompt</summary>
            <p className="mt-2 line-clamp-4 whitespace-pre-wrap leading-relaxed">{item.source_prompt}</p>
          </details>
        ) : null}

        <div className="mt-auto grid grid-cols-2 gap-2 pt-4">
          <Button asChild variant="outline" size="sm">
            <a href={item.url} target="_blank" rel="noreferrer"><ExternalLink className="size-3.5" /> 原图</a>
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => onConvertTo3d?.(item.url)}
            disabled={!onConvertTo3d || isConverting}
          >
            <Box className="size-3.5" /> {isConverting ? '提交中…' : '转 3D'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

export function ResultCardSkeleton() {
  return (
    <Card className="overflow-hidden border-border bg-card py-0">
      <div className="aspect-square animate-pulse bg-[var(--ctp-surface0)]" />
      <CardContent className="space-y-3 p-4">
        <div className="h-3 w-2/3 animate-pulse rounded bg-[var(--ctp-surface0)]" />
        <div className="h-12 animate-pulse rounded bg-[var(--ctp-surface0)]" />
        <div className="h-8 animate-pulse rounded bg-[var(--ctp-surface0)]" />
      </CardContent>
    </Card>
  )
}
