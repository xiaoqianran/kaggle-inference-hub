import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Box, ImageIcon, LoaderCircle, Trash2, Upload, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { useSubmitArtifact, useSubmitArtifactBatch } from '@/features/artifacts/use-submit-artifact'
import { FastSam3DMaskEditor } from '@/features/fast-sam3d/fast-sam3d-mask-editor'
import { cancelTask, createAutoMaskFile, getActiveTasks } from '@/shared/api/client'
import { queryKeys } from '@/shared/api/queries'
import type {
  ArtifactOptionSpec,
  ArtifactOptionValue,
  ArtifactSource,
  ArtifactSubmission,
  HubStatus,
  ModelSpec,
} from '@/shared/api/types'

type ImageToArtifactPanelProps = {
  model: ModelSpec
  token: string
  status?: HubStatus
  sourceUrls?: string[]
  onSourceUrlsChange?: (urls: string[]) => void
}

function optionDefaults(model: ModelSpec): Record<string, ArtifactOptionValue> {
  return Object.fromEntries((model.artifact?.options ?? []).map((option) => [option.id, option.default]))
}

function optionChoiceLabel(option: ArtifactOptionSpec, value: ArtifactOptionValue): string {
  if (option.id === 'output_format') return String(value).toUpperCase()
  return String(value)
}

function parseNumberOption(option: ArtifactOptionSpec, raw: string): number {
  const parsed = option.kind === 'integer' ? Number.parseInt(raw, 10) : Number.parseFloat(raw)
  if (!Number.isFinite(parsed)) return Number(option.default)
  if (option.minimum !== null && parsed < option.minimum) return option.minimum
  if (option.maximum !== null && parsed > option.maximum) return option.maximum
  return parsed
}

function sourceLabel(source: ArtifactSource, index: number): string {
  return source.kind === 'file' ? source.file.name : source.sourceUrl.split('/').at(-1) || `图片 ${index + 1}`
}

export function ImageToArtifactPanel({ model, token, status, sourceUrls = [], onSourceUrlsChange }: ImageToArtifactPanelProps) {
  const artifact = model.artifact
  const defaults = useMemo(() => optionDefaults(model), [model])
  const [options, setOptions] = useState<Record<string, ArtifactOptionValue>>(defaults)
  const [files, setFiles] = useState<File[]>([])
  const [auxiliaryFiles, setAuxiliaryFiles] = useState<Record<string, File>>({})
  const [resetKey, setResetKey] = useState(0)
  const [batchProgress, setBatchProgress] = useState<string>()
  const submit = useSubmitArtifact(token, model)
  const submitBatch = useSubmitArtifactBatch(token, model)
  const queryClient = useQueryClient()

  const activeTasks = useQuery({
    queryKey: [...queryKeys.activeTasks, model.id],
    queryFn: () => getActiveTasks(token, model.id),
    enabled: Boolean(token),
    refetchInterval: 1_500,
  })
  const cancel = useMutation({
    mutationFn: (taskId: number) => cancelTask(token, taskId),
    onSuccess: (result) => {
      toast.success(result.status === 'cancelled' ? `#${result.id} 已取消` : `#${result.id} 已请求取消，结果将被丢弃`)
      void queryClient.invalidateQueries({ queryKey: queryKeys.activeTasks })
      void queryClient.invalidateQueries({ queryKey: queryKeys.status })
    },
    onError: (error) => toast.error(`取消失败：${error instanceof Error ? error.message : '未知错误'}`),
  })

  if (!artifact) {
    return (
      <Card className="border-border bg-card shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base"><Box className="size-4 text-primary" /> {model.label}</CardTitle>
          <CardDescription>这个模型没有声明 Image → 3D 参数 schema，前端无法安全提交。</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  const visibleOptions = artifact.options.filter((option) => option.visible)
  const scalarOptions = visibleOptions.filter((option) => option.kind !== 'boolean')
  const booleanOptions = visibleOptions.filter((option) => option.kind === 'boolean')
  const sources: ArtifactSource[] = sourceUrls.length
    ? sourceUrls.map((sourceUrl) => ({ kind: 'source' as const, sourceUrl }))
    : files.map((file) => ({ kind: 'file' as const, file }))
  const isBatch = sources.length > 1
  const singleSource = sources[0]
  const isBusy = submit.isPending || submitBatch.isPending || Boolean(batchProgress)
  const tasks = activeTasks.data ?? []
  const queuedTasks = tasks.filter((task) => task.status === 'queued' && !task.cancel_requested)

  const clearInputs = () => {
    setFiles([])
    setAuxiliaryFiles({})
    setBatchProgress(undefined)
    setResetKey((value) => value + 1)
    onSourceUrlsChange?.([])
  }

  const removeSourceUrl = (url: string) => onSourceUrlsChange?.(sourceUrls.filter((item) => item !== url))
  const removeFile = (target: File) => setFiles((current) => current.filter((file) => file !== target))

  const buildSubmission = (source: ArtifactSource, aux = auxiliaryFiles): ArtifactSubmission => ({
    model: model.id,
    source,
    options,
    auxiliaryFiles: aux,
  })

  const prepareFastSam3dBatch = async (): Promise<ArtifactSubmission[]> => {
    const prepared: Array<ArtifactSubmission | undefined> = Array(sources.length)
    let cursor = 0
    let completed = 0
    const workers = Array.from({ length: Math.min(3, sources.length) }, async () => {
      while (true) {
        const index = cursor++
        if (index >= sources.length) return
        const source = sources[index]
        setBatchProgress(`SAM2 Mask ${completed + 1}/${sources.length} · ${sourceLabel(source, index)}`)
        const mask = await createAutoMaskFile(token, source)
        prepared[index] = buildSubmission(source, { mask })
        completed += 1
        setBatchProgress(`SAM2 Mask ${completed}/${sources.length} 已完成`)
      }
    })
    await Promise.all(workers)
    return prepared.filter((item): item is ArtifactSubmission => Boolean(item))
  }

  const submitTask = async () => {
    if (!sources.length) return toast.error('请选择至少一张 PNG、JPEG 或 WebP 图片')

    if (!isBatch) {
      for (const input of artifact.auxiliary_inputs) {
        if (input.required && !auxiliaryFiles[input.id]) return toast.error(`请选择 ${input.label}`)
      }
      submit.mutate(buildSubmission(singleSource), { onSuccess: clearInputs })
      return
    }

    const unsupportedRequiredAux = artifact.auxiliary_inputs.some((input) => input.required) && model.id !== 'fast-sam3d'
    if (unsupportedRequiredAux) return toast.error(`${model.label} 的必需辅助输入目前只能单图配置`)

    try {
      const submissions = model.id === 'fast-sam3d'
        ? await prepareFastSam3dBatch()
        : sources.map((source) => buildSubmission(source, {}))
      setBatchProgress(undefined)
      submitBatch.mutate(submissions, {
        onSuccess: (result) => {
          if (result.queued === result.total) {
            clearInputs()
            return
          }
          const failedIndexes = new Set(result.failures.map((item) => item.index))
          if (sourceUrls.length) onSourceUrlsChange?.(sourceUrls.filter((_, index) => failedIndexes.has(index)))
          else setFiles(files.filter((_, index) => failedIndexes.has(index)))
          if (result.failures.length) {
            toast.error(result.failures.slice(0, 3).map((item) => `${item.label}: ${item.error}`).join('\n'))
          }
        },
      })
    } catch (error) {
      setBatchProgress(undefined)
      toast.error(`批量准备失败：${error instanceof Error ? error.message : '未知错误'}`)
    }
  }

  const cancelAllQueued = async () => {
    const ids = queuedTasks.map((task) => task.id)
    if (!ids.length) return
    const results = await Promise.allSettled(ids.map((id) => cancelTask(token, id)))
    const cancelled = results.filter((item) => item.status === 'fulfilled').length
    toast.success(`已取消 ${cancelled}/${ids.length} 个排队任务`)
    void queryClient.invalidateQueries({ queryKey: queryKeys.activeTasks })
    void queryClient.invalidateQueries({ queryKey: queryKeys.status })
  }

  return (
    <Card className="border-border bg-card shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base"><Box className="size-4 text-primary" /> {model.label} · Image to 3D</CardTitle>
        <CardDescription>{model.description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {sourceUrls.length ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between"><Label>来自生成图库 · {sourceUrls.length} 张</Label><Button type="button" variant="ghost" size="sm" onClick={clearInputs}><X className="size-3.5" />清空</Button></div>
            <div className="grid grid-cols-4 gap-2">
              {sourceUrls.slice(0, 8).map((url) => (
                <div key={url} className="group relative aspect-square overflow-hidden rounded-md border bg-[var(--ctp-mantle)]">
                  <img src={url} alt="待转换图片" className="size-full object-cover" />
                  <button type="button" aria-label="移除图片" onClick={() => removeSourceUrl(url)} className="absolute right-1 top-1 flex size-6 items-center justify-center rounded-full bg-black/65 text-white opacity-0 transition group-hover:opacity-100"><X className="size-3" /></button>
                </div>
              ))}
            </div>
            {sourceUrls.length > 8 ? <p className="text-[11px] text-muted-foreground">另有 {sourceUrls.length - 8} 张已选择</p> : null}
          </div>
        ) : (
          <div className="space-y-2">
            <Label htmlFor={`${model.id}-source`}>PNG / JPEG / WebP · 可多选 · 单张最大 20 MB</Label>
            <Input
              key={`${model.id}-source-${resetKey}`}
              id={`${model.id}-source`}
              type="file"
              multiple
              accept="image/png,image/jpeg,image/webp"
              onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
              className="cursor-pointer file:mr-3 file:text-foreground"
            />
            {files.length ? (
              <div className="space-y-1 rounded-lg border border-border p-2">
                {files.slice(0, 8).map((file) => <div key={`${file.name}-${file.lastModified}`} className="flex items-center gap-2 text-xs"><ImageIcon className="size-3.5 text-primary" /><span className="min-w-0 flex-1 truncate">{file.name}</span><Button type="button" variant="ghost" size="icon-sm" onClick={() => removeFile(file)}><X /></Button></div>)}
                {files.length > 8 ? <p className="text-[11px] text-muted-foreground">另有 {files.length - 8} 个文件</p> : null}
              </div>
            ) : null}
          </div>
        )}

        {artifact.auxiliary_inputs.map((input) => {
          if (model.id === 'fast-sam3d' && input.id === 'mask') {
            if (isBatch) return <div key={input.id} className="rounded-lg border border-primary/20 bg-primary/5 p-3 text-xs leading-relaxed text-muted-foreground"><span className="font-medium text-foreground">Fast-SAM3D 批量 Mask</span><br />将为每张图片自动运行 SAM2，并采用 Top-1 候选。需要手工修 Mask 时请改为单图模式。</div>
            return (
              <FastSam3DMaskEditor
                key={`fast-sam3d-mask-${resetKey}`}
                token={token}
                sourceUrl={singleSource?.kind === 'source' ? singleSource.sourceUrl : undefined}
                file={singleSource?.kind === 'file' ? singleSource.file : null}
                resetKey={resetKey}
                onMaskChange={(mask) => setAuxiliaryFiles((current) => {
                  const next = { ...current }
                  if (mask) next.mask = mask
                  else delete next.mask
                  return next
                })}
              />
            )
          }
          return (
            <div key={input.id} className="space-y-2">
              <Label htmlFor={`${model.id}-${input.id}`}>{input.label}{input.required ? ' · 必需' : ''}</Label>
              <Input id={`${model.id}-${input.id}`} type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => {
                const selected = event.target.files?.[0]
                setAuxiliaryFiles((current) => ({ ...current, ...(selected ? { [input.id]: selected } : {}) }))
              }} />
              {input.help ? <p className="text-[11px] text-muted-foreground">{input.help}</p> : null}
            </div>
          )
        })}

        {scalarOptions.length ? <div className="grid grid-cols-2 gap-3">{scalarOptions.map((option) => (
          <div key={option.id} className="space-y-2"><Label htmlFor={`${model.id}-${option.id}`}>{option.label}</Label>{option.kind === 'select' ? (
            <Select value={String(options[option.id] ?? option.default)} onValueChange={(raw) => { const choice = option.choices.find((value) => String(value) === raw) ?? option.default; setOptions((current) => ({ ...current, [option.id]: choice })) }}><SelectTrigger id={`${model.id}-${option.id}`} className="w-full"><SelectValue /></SelectTrigger><SelectContent>{option.choices.map((choice) => <SelectItem key={String(choice)} value={String(choice)}>{optionChoiceLabel(option, choice)}</SelectItem>)}</SelectContent></Select>
          ) : <Input id={`${model.id}-${option.id}`} type="number" min={option.minimum ?? undefined} max={option.maximum ?? undefined} step={option.kind === 'integer' ? 1 : 'any'} value={Number(options[option.id] ?? option.default)} onChange={(event) => setOptions((current) => ({ ...current, [option.id]: parseNumberOption(option, event.target.value) }))} />}{option.help ? <p className="text-[11px] text-muted-foreground">{option.help}</p> : null}</div>
        ))}</div> : null}

        {booleanOptions.map((option) => <div key={option.id} className="flex items-center justify-between rounded-lg border border-border bg-[var(--ctp-mantle)] px-3 py-2.5"><div><Label htmlFor={`${model.id}-${option.id}`}>{option.label}</Label>{option.help ? <p className="mt-0.5 text-[11px] text-muted-foreground">{option.help}</p> : null}</div><Switch id={`${model.id}-${option.id}`} checked={Boolean(options[option.id] ?? option.default)} onCheckedChange={(checked) => setOptions((current) => ({ ...current, [option.id]: checked }))} /></div>)}

        <Button type="button" className="w-full" onClick={() => void submitTask()} disabled={isBusy || !sources.length}>
          {isBusy ? <LoaderCircle className="size-4 animate-spin" /> : <Upload className="size-4" />}
          {batchProgress ?? (isBusy ? '提交中…' : `加入 ${sources.length || ''} ${model.label} 任务`)}
        </Button>

        <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-muted-foreground"><span>Queued {status?.queued_by_model[model.id] ?? 0}</span><span>Inflight {status?.inflight_by_model[model.id] ?? 0}</span></div>

        <div className="space-y-2 border-t border-border pt-4">
          <div className="flex items-center justify-between"><div><p className="text-xs font-medium">当前 3D 任务</p><p className="text-[10px] text-muted-foreground">Queued 可立即撤销；Inflight 取消后结果会被 Hub 丢弃。</p></div>{queuedTasks.length ? <Button type="button" variant="ghost" size="sm" onClick={() => void cancelAllQueued()}><Trash2 className="size-3.5" />取消全部排队</Button> : null}</div>
          {tasks.length ? <div className="max-h-56 space-y-1.5 overflow-y-auto">{tasks.map((task) => (
            <div key={task.id} className="flex items-center gap-2 rounded-md border border-border bg-[var(--ctp-mantle)] px-2.5 py-2">
              {task.source_url ? <img src={task.source_url} alt="任务来源" className="size-8 rounded object-cover" /> : <Box className="size-4 text-muted-foreground" />}
              <div className="min-w-0 flex-1"><p className="truncate text-[11px] font-medium">#{task.id} · {task.source_label || 'input image'}</p><p className="font-mono text-[9px] uppercase text-muted-foreground">{task.cancel_requested ? 'CANCELLING' : task.status}{task.worker_id ? ` · ${task.worker_id}` : ''}</p></div>
              <Button type="button" variant="ghost" size="icon-sm" aria-label={`取消任务 ${task.id}`} disabled={task.cancel_requested || cancel.isPending} onClick={() => cancel.mutate(task.id)}><X className="size-3.5" /></Button>
            </div>
          ))}</div> : <p className="rounded-md border border-dashed border-border px-3 py-4 text-center text-[11px] text-muted-foreground">当前没有活动任务</p>}
        </div>
      </CardContent>
    </Card>
  )
}
