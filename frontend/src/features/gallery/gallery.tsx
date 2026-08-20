import { Box, ImageIcon, RefreshCw } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ResultCard, ResultCardSkeleton } from '@/features/gallery/result-card'
import { useSubmitTripo } from '@/features/triposr/use-submit-tripo'
import type { WorkspaceKind } from '@/features/status/workspace-header'
import type { HistoryItem, ModelSpec, TripoSettings } from '@/shared/api/types'

type GalleryProps = {
  workspace: Extract<WorkspaceKind, 'image' | '3d'>
  items: HistoryItem[]
  model?: ModelSpec
  token: string
  tripoSettings: TripoSettings
  isLoading: boolean
  isRefreshing: boolean
  onRefresh: () => void
}

export function Gallery({
  workspace,
  items,
  model,
  token,
  tripoSettings,
  isLoading,
  isRefreshing,
  onRefresh,
}: GalleryProps) {
  const submitTripo = useSubmitTripo(token)
  const newestFirst = items.toReversed()
  const isImage = workspace === 'image'
  const GalleryIcon = isImage ? ImageIcon : Box

  return (
    <section className="min-w-0">
      <div className="mb-5 flex items-end justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <GalleryIcon className="size-4 text-primary" />
            <h2 className="truncate text-base font-semibold">{isImage ? '生成图库' : '3D 资产'}</h2>
            <Badge variant="secondary" className="font-mono text-[9px]">{items.length}</Badge>
          </div>
          <p className="mt-1 truncate text-xs text-muted-foreground">
            {model?.label ?? (isImage ? '当前模型' : 'TripoSR')} · {isImage ? '仅显示这个模型的结果' : '可下载的 GLB / OBJ 重建结果'}
          </p>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={onRefresh} disabled={isRefreshing}>
          <RefreshCw className={`size-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span className="hidden sm:inline">刷新</span>
        </Button>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {Array.from({ length: 8 }, (_, index) => <ResultCardSkeleton key={index} />)}
        </div>
      ) : newestFirst.length ? (
        <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {newestFirst.map((item) => (
            <ResultCard
              key={item.event_id}
              item={item}
              modelLabel={model?.label ?? item.model}
              onConvertTo3d={
                item.kind === 'image'
                  ? (sourceUrl) => submitTripo.mutate({ kind: 'source', sourceUrl, settings: tripoSettings })
                  : undefined
              }
              isConverting={
                submitTripo.isPending &&
                submitTripo.variables?.kind === 'source' &&
                submitTripo.variables.sourceUrl === (item.kind === 'image' ? item.url : '')
              }
            />
          ))}
        </div>
      ) : (
        <div className="flex min-h-96 flex-col items-center justify-center rounded-xl border border-dashed border-border bg-[var(--ctp-mantle)]/45 px-6 text-center">
          <span className="mb-4 flex size-12 items-center justify-center rounded-full bg-[var(--ctp-surface0)]">
            <GalleryIcon className="size-5 text-muted-foreground" />
          </span>
          <p className="text-sm font-medium">{isImage ? '这个模型还没有作品' : '还没有 3D 资产'}</p>
          <p className="mt-1 max-w-xs text-xs leading-relaxed text-muted-foreground">
            {isImage ? '在左侧提交 Prompt，生成完成后会自动出现在当前模型标签页。' : '从图片卡片发送到 3D，或在左侧上传本地图片。'}
          </p>
        </div>
      )}
    </section>
  )
}
