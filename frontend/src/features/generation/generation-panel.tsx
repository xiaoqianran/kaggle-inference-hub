import { zodResolver } from '@hookform/resolvers/zod'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { BrainCircuit, RotateCcw, Send, Sparkles, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useForm, useWatch } from 'react-hook-form'
import { toast } from 'sonner'
import { z } from 'zod'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import {
  processPrompt,
  processPromptBatch,
  queueBatchTasks,
  queueSingleTask,
} from '@/shared/api/client'
import { queryKeys } from '@/shared/api/queries'
import type {
  HubStatus,
  ModelSpec,
  PromptBatchItem,
  PromptMeta,
  PromptPipelineConfig,
  PromptProcessResult,
  SingleTaskRequest,
  BatchTaskRequest,
} from '@/shared/api/types'

const generationSchema = z.object({
  width: z.number().int().min(64).max(4096),
  height: z.number().int().min(64).max(4096),
  steps: z.number().int().min(1).max(200),
})

type GenerationForm = z.infer<typeof generationSchema>
type EditorMode = 'single' | 'batch'

type SingleAiState = {
  source: string
  processed: string
  meta: PromptMeta
  edited: boolean
}

type BatchAiState = {
  sourcePrompts: string[]
  processedPrompts: string[]
  items: PromptBatchItem[]
  edited: boolean
}

type GenerationPanelProps = {
  model: ModelSpec
  pipeline?: PromptPipelineConfig
  status?: HubStatus
  token: string
  modelResultCount: number
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : '未知错误'
}

function splitBatch(value: string): string[] {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
}

function createPromptMeta(result: PromptProcessResult, extra: PromptMeta = {}): PromptMeta {
  return {
    mode: result.mode,
    provider_model: result.provider_model,
    elapsed_ms: result.elapsed_ms,
    translate_to_english: result.translate_to_english,
    target_model: result.target_model,
    ...extra,
  }
}

export function GenerationPanel({
  model: currentModel,
  pipeline,
  status,
  token,
  modelResultCount,
}: GenerationPanelProps) {
  const queryClient = useQueryClient()
  const [editorMode, setEditorMode] = useState<EditorMode>('single')
  const [singlePrompt, setSinglePrompt] = useState('')
  const [batchPrompt, setBatchPrompt] = useState('')
  const [seed, setSeed] = useState('')
  const [singleAi, setSingleAi] = useState<SingleAiState | null>(null)
  const [batchAi, setBatchAi] = useState<BatchAiState | null>(null)
  const [aiMode, setAiMode] = useState(() => localStorage.getItem('prompt_ai_mode') ?? 'enhance')
  const [translate, setTranslate] = useState(() => localStorage.getItem('prompt_ai_translate') !== '0')

  const form = useForm<GenerationForm>({
    resolver: zodResolver(generationSchema),
    defaultValues: { width: 1024, height: 1024, steps: 2 },
  })
  const steps = useWatch({ control: form.control, name: 'steps' })

  useEffect(() => {
    const stored = localStorage.getItem(`steps_${currentModel.id}`)
    form.setValue('steps', stored ? Number(stored) : currentModel.default_steps)
  }, [currentModel.default_steps, currentModel.id, form])

  const selectedAiMode = pipeline?.modes.some((mode) => mode.id === aiMode)
    ? aiMode
    : pipeline?.modes[0]?.id ?? 'enhance'

  const refreshStatus = () => void queryClient.invalidateQueries({ queryKey: queryKeys.status })

  const optimizeSingle = useMutation({
    mutationFn: () =>
      processPrompt(token, {
        prompt: singlePrompt.trim(),
        model: currentModel.id,
        mode: selectedAiMode,
        translate_to_english: translate,
      }),
    onSuccess: (result) => {
      const rootSource = singleAi?.source ?? singlePrompt.trim()
      setSinglePrompt(result.processed)
      setSingleAi({
        source: rootSource,
        processed: result.processed,
        meta: createPromptMeta(result),
        edited: false,
      })
      toast.success(`AI 优化完成 · ${result.elapsed_ms} ms`)
    },
    onError: (error) => toast.error(`AI 优化失败：${messageOf(error)}`),
  })

  const optimizeBatch = useMutation({
    mutationFn: (prompts: string[]) =>
      processPromptBatch(token, {
        prompts,
        model: currentModel.id,
        mode: selectedAiMode,
        translate_to_english: translate,
      }),
    onSuccess: (result, submittedPrompts) => {
      const rootSources =
        batchAi?.sourcePrompts.length === submittedPrompts.length
          ? batchAi.sourcePrompts
          : submittedPrompts
      const processed = result.items.map((item) => item.processed)
      setBatchPrompt(processed.join('\n'))
      setBatchAi({
        sourcePrompts: rootSources,
        processedPrompts: processed,
        items: result.items,
        edited: false,
      })
      const suffix = result.failed ? ` · ${result.failed} 条回退原文` : ''
      if (result.failed === result.total) {
        toast.error(`AI 批量处理失败 ${result.succeeded}/${result.total}${suffix}`)
      } else {
        toast.success(`AI 批量完成 ${result.succeeded}/${result.total}${suffix}`)
      }
    },
    onError: (error) => toast.error(`AI 批量优化失败：${messageOf(error)}`),
  })

  const submitSingle = useMutation({
    mutationFn: ([accessToken, input]: [string, SingleTaskRequest]) => queueSingleTask(accessToken, input),
    onSuccess: (result) => {
      toast.success(`#${result.task.id} → ${currentModel.label}`)
      setSinglePrompt('')
      setSingleAi(null)
      refreshStatus()
    },
    onError: (error) => toast.error(`提交失败：${messageOf(error)}`),
  })

  const submitBatch = useMutation({
    mutationFn: ([accessToken, input]: [string, BatchTaskRequest]) => queueBatchTasks(accessToken, input),
    onSuccess: (result) => {
      toast.success(`已加入 ${result.queued} 个任务`)
      setBatchPrompt('')
      setBatchAi(null)
      refreshStatus()
    },
    onError: (error) => toast.error(`批量提交失败：${messageOf(error)}`),
  })

  const batchLines = splitBatch(batchPrompt)
  const pipelineReady = Boolean(pipeline?.configured)
  const isSubmitting = submitSingle.isPending || submitBatch.isPending

  const commonParams = (values: GenerationForm) => ({
    model: currentModel.id,
    width: values.width,
    height: values.height,
    steps: values.steps,
    ...(seed.trim() ? { seed: Number(seed) } : {}),
  })

  const handleSubmit = form.handleSubmit((values) => {
    if (editorMode === 'single') {
      const prompt = singlePrompt.trim()
      if (!prompt) {
        toast.error('请输入 Prompt')
        return
      }
      submitSingle.mutate([
        token,
        {
          ...commonParams(values),
          prompt,
          ...(singleAi
            ? {
                source_prompt: singleAi.source,
                prompt_meta: {
                  ...singleAi.meta,
                  edited_after_ai: singleAi.edited,
                  stale_model_adapter: singleAi.meta.target_model !== currentModel.id,
                },
              }
            : {}),
        },
      ])
      return
    }

    if (!batchLines.length) {
      toast.error('请输入批量 Prompt')
      return
    }
    submitBatch.mutate([
      token,
      {
        ...commonParams(values),
        prompts: batchLines,
        ...(batchAi?.sourcePrompts.length === batchLines.length
          ? {
              source_prompts: batchAi.sourcePrompts,
              prompt_metas: batchLines.map((_, index) => {
                const item = batchAi.items[index]
                if (!item?.ok) return {}
                return createPromptMeta(item, {
                  edited_after_ai: batchAi.edited,
                  stale_model_adapter: item.target_model !== currentModel.id,
                })
              }),
            }
          : {}),
      },
    ])
  })

  const changeSteps = (value: number) => {
    form.setValue('steps', value, { shouldValidate: true })
    localStorage.setItem(`steps_${currentModel.id}`, String(value))
  }

  const runSingleAi = () => {
    if (!singlePrompt.trim()) return toast.error('请输入 Prompt')
    if (!pipelineReady) return toast.error('AI Prompt Pipeline 未配置')
    optimizeSingle.mutate()
  }

  const runBatchAi = () => {
    if (!batchLines.length) return toast.error('请输入批量 Prompt')
    if (!pipelineReady) return toast.error('AI Prompt Pipeline 未配置')
    optimizeBatch.mutate(batchLines)
  }

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      <Card className="border-border bg-card shadow-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <BrainCircuit className="size-4 text-primary" /> Prompt Studio
              </CardTitle>
              <CardDescription className="mt-1 line-clamp-1">
                {pipeline?.configured
                  ? `${pipeline.provider_model} · 并发 ${pipeline.concurrency}`
                  : pipeline?.enabled
                    ? 'Pipeline 尚未完整配置'
                    : 'AI Prompt Pipeline 当前关闭'}
              </CardDescription>
            </div>
            <Badge variant={pipelineReady ? 'default' : 'secondary'}>
              {pipelineReady ? 'AI READY' : 'AI OFF'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 grid grid-cols-[1fr_auto] gap-3">
            <Select
              value={selectedAiMode}
              onValueChange={(value) => {
                setAiMode(value)
                localStorage.setItem('prompt_ai_mode', value)
              }}
              disabled={!pipelineReady}
            >
              <SelectTrigger aria-label="AI 处理模式" className="w-full">
                <SelectValue placeholder="处理模式" />
              </SelectTrigger>
              <SelectContent>
                {(pipeline?.modes ?? []).map((mode) => (
                  <SelectItem key={mode.id} value={mode.id}>
                    {mode.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="flex items-center gap-2 rounded-md border border-input px-3">
              <Label htmlFor="translate" className="whitespace-nowrap text-xs text-muted-foreground">
                英文输出
              </Label>
              <Switch
                id="translate"
                checked={translate}
                onCheckedChange={(checked) => {
                  setTranslate(checked)
                  localStorage.setItem('prompt_ai_translate', checked ? '1' : '0')
                }}
                disabled={!pipelineReady}
              />
            </div>
          </div>

          <Tabs value={editorMode} onValueChange={(value) => setEditorMode(value as EditorMode)}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="single">单个 Prompt</TabsTrigger>
              <TabsTrigger value="batch">
                批量任务
                {batchLines.length ? <Badge className="ml-2 h-5 px-1.5">{batchLines.length}</Badge> : null}
              </TabsTrigger>
            </TabsList>
            <TabsContent value="single" className="mt-3 space-y-3">
              <Textarea
                aria-label="单个 Prompt"
                value={singlePrompt}
                onChange={(event) => {
                  const value = event.target.value
                  setSinglePrompt(value)
                  if (singleAi && value.trim() !== singleAi.processed.trim()) {
                    setSingleAi({ ...singleAi, edited: true })
                  }
                }}
                onKeyDown={(event) => {
                  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                    event.preventDefault()
                    void handleSubmit()
                  }
                }}
                placeholder="描述你想生成的画面…"
                className="min-h-52 resize-y leading-relaxed"
              />
              <div className="flex flex-wrap gap-2">
                <Button type="button" variant="secondary" onClick={runSingleAi} disabled={!pipelineReady || optimizeSingle.isPending}>
                  <Sparkles className="size-4" />
                  {optimizeSingle.isPending ? '优化中…' : 'AI 优化'}
                </Button>
                {singleAi ? (
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => {
                      setSinglePrompt(singleAi.source)
                      setSingleAi(null)
                    }}
                  >
                    <RotateCcw className="size-4" /> 恢复原文
                  </Button>
                ) : null}
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => {
                    setSinglePrompt('')
                    setSingleAi(null)
                  }}
                >
                  <Trash2 className="size-4" /> 清空
                </Button>
              </div>
            </TabsContent>
            <TabsContent value="batch" className="mt-3 space-y-3">
              <Textarea
                aria-label="批量 Prompt"
                value={batchPrompt}
                onChange={(event) => {
                  const value = event.target.value
                  setBatchPrompt(value)
                  if (batchAi && splitBatch(value).join('\n') !== batchAi.processedPrompts.join('\n')) {
                    setBatchAi({ ...batchAi, edited: true })
                  }
                }}
                onKeyDown={(event) => {
                  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                    event.preventDefault()
                    void handleSubmit()
                  }
                }}
                placeholder={'A mountain lake at sunrise\nA futuristic Tokyo street\nA forest covered in mist'}
                className="min-h-52 resize-y font-mono text-xs leading-relaxed"
              />
              <div className="flex flex-wrap gap-2">
                <Button type="button" variant="secondary" onClick={runBatchAi} disabled={!pipelineReady || optimizeBatch.isPending}>
                  <Sparkles className="size-4" />
                  {optimizeBatch.isPending ? '批量优化中…' : 'AI 批量优化'}
                </Button>
                {batchAi ? (
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => {
                      setBatchPrompt(batchAi.sourcePrompts.join('\n'))
                      setBatchAi(null)
                    }}
                  >
                    <RotateCcw className="size-4" /> 恢复原文
                  </Button>
                ) : null}
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => {
                    setBatchPrompt('')
                    setBatchAi(null)
                  }}
                >
                  <Trash2 className="size-4" /> 清空
                </Button>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <Card className="border-border bg-card shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="text-base">生成参数</CardTitle>
          <CardDescription>切换模型会自动恢复对应的 Steps</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label htmlFor="width">Width</Label>
              <Input id="width" type="number" {...form.register('width', { valueAsNumber: true })} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="height">Height</Label>
              <Input id="height" type="number" {...form.register('height', { valueAsNumber: true })} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="steps">Steps</Label>
              <Input
                id="steps"
                type="number"
                value={steps}
                onChange={(event) => changeSteps(Number(event.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="seed">Base seed</Label>
              <Input id="seed" type="number" value={seed} onChange={(event) => setSeed(event.target.value)} placeholder="随机" />
            </div>
          </div>

          <div className="grid grid-cols-4 gap-2 rounded-lg border border-border bg-[var(--ctp-mantle)] p-3">
            {[
              ['BATCH', batchLines.length],
              ['QUEUE', status?.queued_by_model[currentModel.id] ?? 0],
              ['RESULTS', modelResultCount],
              ['ONLINE', status?.workers.filter((worker) => worker.online && worker.model === currentModel.id).length ?? 0],
            ].map(([label, value]) => (
              <div key={String(label)} className="min-w-0 text-center">
                <div className="font-mono text-base font-semibold tabular-nums">{value}</div>
                <div className="truncate text-[9px] tracking-widest text-muted-foreground">{label}</div>
              </div>
            ))}
          </div>

          <Button type="submit" size="lg" className="w-full" disabled={isSubmitting}>
            <Send className="size-4" />
            {isSubmitting
              ? '正在提交…'
              : editorMode === 'single'
                ? '提交 1 个任务'
                : `提交 ${batchLines.length} 个任务`}
            <span className="ml-auto hidden font-mono text-[10px] opacity-60 sm:inline">CTRL ↵</span>
          </Button>
        </CardContent>
      </Card>
    </form>
  )
}
