import { Box, ChevronLeft, ChevronRight, ImageIcon, RefreshCw } from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { toast } from 'sonner'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ResultCard, ResultCardSkeleton } from '@/features/gallery/result-card'
import { useSubmitTripo } from '@/features/triposr/use-submit-tripo'
import { deleteHistoryBatch } from '@/shared/api/client'
import { queryKeys } from '@/shared/api/queries'
import { getTripoResolutionOption } from '@/shared/tripo-resolution'
import type { WorkspaceKind } from '@/features/status/workspace-header'
import type { ArtifactResult, HistoryItem, ModelSpec, TripoSettings } from '@/shared/api/types'

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
  token,
  tripoSettings,
  isLoading,
  isRefreshing,
  onRefresh,
}: GalleryProps) {
  const queryClient = useQueryClient()
  const submitTripo = useSubmitTripo(token)
  const regenerateTripo = useSubmitTripo(token, { notify: false })
  const deleteMutation = useMutation({
    mutationFn: (eventIds: number[]) => deleteHistoryBatch(token, eventIds),
    onSuccess: (result) => {
      toast.success(`已删除 ${result.deleted} 个 3D 资产`)
      void queryClient.invalidateQueries({ queryKey: queryKeys.history })
      void queryClient.invalidateQueries({ queryKey: queryKeys.status })
    },
    onError: (error) => {
      toast.error(`批量删除失败：${error instanceof Error ? error.message : '未知错误'}`)
    },
  })
  const newestFirst = items.toReversed()
  const pageCount = Math.max(1, Math.ceil(newestFirst.length / PAGE_SIZE))
  const [currentPage, setCurrentPage] = useState(1)
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())
  const [regeneratingIds, setRegeneratingIds] = useState<Set<number>>(new Set())
  const [batchRegenerating, setBatchRegenerating] = useState(false)
  const isImage = workspace === 'image'
  const is3d = workspace === '3d'
  const GalleryIcon = isImage ? ImageIcon : Box
  const activePage = Math.min(currentPage, pageCount)
  const visibleItems = newestFirst.slice((activePage - 1) * PAGE_SIZE, activePage * PAGE_SIZE)
  const selectedResolution = getTripoResolutionOption(tripoSettings.resolution)
  const selectedItems = newestFirst.filter(
    (item): item is ArtifactResult => item.kind === 'artifact' && selectedIds.has(item.event_id),
  )
  const allVisibleSelected = !isImage && visibleItems.length > 0 && visibleItems.every((item) => selectedIds.has(item.event_id))

  const setSelected = (eventId: number, selected: boolean) => {
    setSelectedIds((current) => {
      const next = new Set(current)
      if (selected) next.add(eventId)
      else next.delete(eventId)
      return next
    })
  }

  const toggleVisibleSelection = () => {
    setSelectedIds((current) => {
      const next = new Set(current)
      for (const item of visibleItems) {
        if (allVisibleSelected) next.delete(item.event_id)
        else next.add(item.event_id)
      }
      return next
    })
  }

  const regenerateOne = async (item: ArtifactResult) => {
    if (!item.source_url) {
      toast.error('这个资产没有保存原图，无法重新生成')
      return
    }
    setRegeneratingIds((current) => new Set(current).add(item.event_id))
    try {
      const result = await regenerateTripo.mutateAsync({ kind: 'source', sourceUrl: item.source_url, settings: tripoSettings })
      toast.success(`#${result.task.id} 已重新加入队列 · ${selectedResolution.shortLabel}`)
    } catch (error) {
      toast.error(`重新生成失败：${error instanceof Error ? error.message : '未知错误'}`)
    } finally {
      setRegeneratingIds((current) => {
        const next = new Set(current)
        next.delete(item.event_id)
        return next
      })
    }
  }

  const regenerateSelected = async () => {
    const targets = selectedItems.filter((item) => item.source_url)
    if (!targets.length) {
      toast.error('所选资产没有可用原图')
      return
    }
    setBatchRegenerating(true)
    setRegeneratingIds(new Set(targets.map((item) => item.event_id)))
    const results = await Promise.allSettled(
      targets.map((item) => regenerateTripo.mutateAsync({ kind: 'source', sourceUrl: item.source_url!, settings: tripoSettings })),
    )
    const succeeded = results.filter((result) => result.status === 'fulfilled').length
    const failed = results.length - succeeded
    setBatchRegenerating(false)
    setRegeneratingIds(new Set())
    setSelectedIds(new Set())
    if (failed) toast.error(`${succeeded} 个已提交，${failed} 个失败`)
    else toast.success(`已提交 ${succeeded} 个重新生成任务 · ${selectedResolution.shortLabel}`)
  }

  const deleteSelected = () => {
    if (!selectedItems.length || deleteMutation.isPending) return
    if (!window.confirm(`确定删除选中的 ${selectedItems.length} 个 3D 资产吗？下载文件也会被删除。`)) return
    deleteMutation.mutate(selectedItems.map((item) => item.event_id), {
      onSuccess: () => setSelectedIds(new Set()),
    })
  }

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

      {is3d && !isLoading && newestFirst.length ? (
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-[var(--ctp-mantle)]/55 px-3 py-2.5">
          <label className="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              aria-label="选择当前页全部 3D 资产"
              checked={allVisibleSelected}
              onChange={toggleVisibleSelection}
              className="size-4 accent-[var(--ctp-blue)]"
            />
            <span>选择当前页</span>
            {selectedItems.length ? <Badge variant="secondary" className="font-mono text-[9px]">已选 {selectedItems.length}</Badge> : null}
          </label>
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[10px] text-muted-foreground">重新生成使用：{selectedResolution.shortLabel}</span>
            <Button type="button" variant="outline" size="sm" onClick={regenerateSelected} disabled={!selectedItems.length || batchRegenerating || deleteMutation.isPending}>
              <RefreshCw className={`size-3.5 ${batchRegenerating ? 'animate-spin' : ''}`} /> 批量重新生成
            </Button>
            <Button type="button" variant="destructive" size="sm" onClick={deleteSelected} disabled={!selectedItems.length || deleteMutation.isPending || batchRegenerating}>
              删除 {selectedItems.length ? `(${selectedItems.length})` : ''}
            </Button>
          </div>
        </div>
      ) : null}

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
              selected={!isImage && selectedIds.has(item.event_id)}
              onSelectedChange={!isImage ? (selected) => setSelected(item.event_id, selected) : undefined}
              onRegenerate={!isImage && item.kind === 'artifact' ? () => void regenerateOne(item) : undefined}
              isRegenerating={!isImage && item.kind === 'artifact' && (regeneratingIds.has(item.event_id) || batchRegenerating)}
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
