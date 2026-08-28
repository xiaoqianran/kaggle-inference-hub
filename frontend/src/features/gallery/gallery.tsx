import { Box, ChevronLeft, ChevronRight, ImageIcon, RefreshCw } from 'lucide-react'
import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ResultCard, ResultCardSkeleton } from '@/features/gallery/result-card'
import type { WorkspaceKind } from '@/features/status/workspace-header'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { HistoryItem, ModelSpec } from '@/shared/api/types'

type GalleryProps = {
  workspace: Extract<WorkspaceKind, 'image' | '3d'>
  items: HistoryItem[]
  model?: ModelSpec
  conversionModels?: ModelSpec[]
  conversionModel?: ModelSpec
  onConversionModelChange?: (model: string) => void
  onConvertTo3d?: (sourceUrl: string) => void
  isLoading: boolean
  isRefreshing: boolean
  onRefresh: () => void
}

const PAGE_SIZE = 12
type PageItem = number | 'ellipsis-left' | 'ellipsis-right'

function getPageItems(currentPage: number, pageCount: number): PageItem[] {
  if (pageCount <= 7) return Array.from({ length: pageCount }, (_, index) => index + 1)
  if (currentPage <= 4) return [1, 2, 3, 4, 5, 'ellipsis-right', pageCount]
  if (currentPage >= pageCount - 3) {
    return [1, 'ellipsis-left', pageCount - 4, pageCount - 3, pageCount - 2, pageCount - 1, pageCount]
  }
  return [1, 'ellipsis-left', currentPage - 1, currentPage, currentPage + 1, 'ellipsis-right', pageCount]
}

export function Gallery({
  workspace,
  items,
  model,
  conversionModels = [],
  conversionModel,
  onConversionModelChange,
  onConvertTo3d,
  isLoading,
  isRefreshing,
  onRefresh,
}: GalleryProps) {
  const newestFirst = items.toReversed()
  const pageCount = Math.max(1, Math.ceil(newestFirst.length / PAGE_SIZE))
  const [currentPage, setCurrentPage] = useState(1)
  const isImage = workspace === 'image'
  const GalleryIcon = isImage ? ImageIcon : Box
  const activePage = Math.min(currentPage, pageCount)
  const visibleItems = newestFirst.slice((activePage - 1) * PAGE_SIZE, activePage * PAGE_SIZE)

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
            {model?.label ?? (isImage ? '当前模型' : '3D 模型')} · {isImage ? '仅显示这个模型的结果' : '可下载的 GLB / OBJ 重建结果'}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {isImage && conversionModels.length ? (
            <Select value={conversionModel?.id ?? ''} onValueChange={onConversionModelChange}>
              <SelectTrigger className="h-8 w-[168px]" aria-label="选择图片转 3D 的目标模型">
                <Box className="size-3.5 text-primary" />
                <SelectValue placeholder="选择 3D 模型" />
              </SelectTrigger>
              <SelectContent>
                {conversionModels.map((target) => (
                  <SelectItem key={target.id} value={target.id}>{target.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : null}
          <Button type="button" variant="outline" size="sm" onClick={onRefresh} disabled={isRefreshing}>
            <RefreshCw className={`size-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span className="hidden sm:inline">刷新</span>
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {Array.from({ length: 8 }, (_, index) => <ResultCardSkeleton key={index} />)}
        </div>
      ) : newestFirst.length ? (
        <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {visibleItems.map((item) => (
            <ResultCard
              key={item.event_id}
              item={item}
              modelLabel={model?.label ?? item.model}
              onConvertTo3d={item.kind === 'image' ? onConvertTo3d : undefined}
              convertModelLabel={item.kind === 'image' ? conversionModel?.label : undefined}
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

      {!isLoading && newestFirst.length > 0 && pageCount > 1 ? (
        <nav className="mt-6 flex flex-wrap items-center justify-between gap-3" aria-label="图库分页">
          <p className="font-mono text-[10px] text-muted-foreground">
            第 {activePage} / {pageCount} 页 · 共 {items.length} 项
          </p>
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="outline"
              size="icon-sm"
              aria-label="上一页"
              onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
              disabled={activePage === 1}
            >
              <ChevronLeft />
            </Button>
            {getPageItems(activePage, pageCount).map((page) =>
              typeof page === 'number' ? (
                <Button
                  key={page}
                  type="button"
                  variant={page === activePage ? 'default' : 'outline'}
                  size="icon-sm"
                  aria-label={`第 ${page} 页`}
                  aria-current={page === activePage ? 'page' : undefined}
                  onClick={() => setCurrentPage(page)}
                >
                  {page}
                </Button>
              ) : (
                <span key={page} className="flex size-6 items-center justify-center text-xs text-muted-foreground" aria-hidden="true">
                  …
                </span>
              ),
            )}
            <Button
              type="button"
              variant="outline"
              size="icon-sm"
              aria-label="下一页"
              onClick={() => setCurrentPage((page) => Math.min(pageCount, page + 1))}
              disabled={activePage === pageCount}
            >
              <ChevronRight />
            </Button>
          </div>
        </nav>
      ) : null}
    </section>
  )
}
