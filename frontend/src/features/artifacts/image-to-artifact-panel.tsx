import { Box, ImageIcon, Upload, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { useSubmitArtifact } from '@/features/artifacts/use-submit-artifact'
import type { ArtifactOptionSpec, ArtifactOptionValue, HubStatus, ModelSpec } from '@/shared/api/types'

type ImageToArtifactPanelProps = {
  model: ModelSpec
  token: string
  status?: HubStatus
  sourceUrl?: string
  onSourceUrlClear?: () => void
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

export function ImageToArtifactPanel({ model, token, status, sourceUrl, onSourceUrlClear }: ImageToArtifactPanelProps) {
  const artifact = model.artifact
  const defaults = useMemo(() => optionDefaults(model), [model])
  const [options, setOptions] = useState<Record<string, ArtifactOptionValue>>(defaults)
  const [file, setFile] = useState<File | null>(null)
  const [auxiliaryFiles, setAuxiliaryFiles] = useState<Record<string, File>>({})
  const [resetKey, setResetKey] = useState(0)
  const submit = useSubmitArtifact(token, model)

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

  const clearInputs = () => {
    setFile(null)
    setAuxiliaryFiles({})
    setResetKey((value) => value + 1)
    onSourceUrlClear?.()
  }

  const submitTask = () => {
    if (!sourceUrl && !file) return toast.error('请选择一张 PNG、JPEG 或 WebP 图片')
    for (const input of artifact.auxiliary_inputs) {
      if (input.required && !auxiliaryFiles[input.id]) return toast.error(`请选择 ${input.label}`)
    }
    submit.mutate(
      {
        model: model.id,
        source: sourceUrl ? { kind: 'source', sourceUrl } : { kind: 'file', file: file! },
        options,
        auxiliaryFiles,
      },
      { onSuccess: clearInputs },
    )
  }

  return (
    <Card className="border-border bg-card shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Box className="size-4 text-primary" /> {model.label} · Image to 3D
        </CardTitle>
        <CardDescription>{model.description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {sourceUrl ? (
          <div className="flex items-center gap-3 rounded-lg border border-primary/25 bg-primary/5 p-3">
            <div className="flex size-12 shrink-0 items-center justify-center overflow-hidden rounded-md bg-[var(--ctp-mantle)]">
              <img src={sourceUrl} alt="待转换的图库图片" className="size-full object-cover" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="flex items-center gap-1.5 text-xs font-medium"><ImageIcon className="size-3.5 text-primary" /> 来自生成图库</p>
              <p className="mt-1 truncate font-mono text-[10px] text-muted-foreground">{sourceUrl}</p>
            </div>
            <Button type="button" variant="ghost" size="icon-sm" aria-label="改用本地图片" onClick={onSourceUrlClear}>
              <X className="size-3.5" />
            </Button>
          </div>
        ) : (
          <div className="space-y-2">
            <Label htmlFor={`${model.id}-source`}>PNG / JPEG / WebP · 最大 20 MB</Label>
            <Input
              key={`${model.id}-source-${resetKey}`}
              id={`${model.id}-source`}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              className="cursor-pointer file:mr-3 file:text-foreground"
            />
          </div>
        )}

        {artifact.auxiliary_inputs.map((input) => (
          <div key={input.id} className="space-y-2">
            <Label htmlFor={`${model.id}-${input.id}`}>{input.label}{input.required ? ' · 必需' : ''}</Label>
            <Input
              key={`${model.id}-${input.id}-${resetKey}`}
              id={`${model.id}-${input.id}`}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(event) => {
                const selected = event.target.files?.[0]
                setAuxiliaryFiles((current) => {
                  const next = { ...current }
                  if (selected) next[input.id] = selected
                  else delete next[input.id]
                  return next
                })
              }}
              className="cursor-pointer file:mr-3 file:text-foreground"
            />
            {input.help ? <p className="text-[11px] text-muted-foreground">{input.help}</p> : null}
          </div>
        ))}

        {scalarOptions.length ? (
          <div className="grid grid-cols-2 gap-3">
            {scalarOptions.map((option) => (
              <div key={option.id} className="space-y-2">
                <Label htmlFor={`${model.id}-${option.id}`}>{option.label}</Label>
                {option.kind === 'select' ? (
                  <Select
                    value={String(options[option.id] ?? option.default)}
                    onValueChange={(raw) => {
                      const choice = option.choices.find((value) => String(value) === raw) ?? option.default
                      setOptions((current) => ({ ...current, [option.id]: choice }))
                    }}
                  >
                    <SelectTrigger id={`${model.id}-${option.id}`} className="w-full"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {option.choices.map((choice) => (
                        <SelectItem key={String(choice)} value={String(choice)}>{optionChoiceLabel(option, choice)}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id={`${model.id}-${option.id}`}
                    type="number"
                    min={option.minimum ?? undefined}
                    max={option.maximum ?? undefined}
                    step={option.kind === 'integer' ? 1 : 'any'}
                    value={Number(options[option.id] ?? option.default)}
                    onChange={(event) =>
                      setOptions((current) => ({ ...current, [option.id]: parseNumberOption(option, event.target.value) }))
                    }
                  />
                )}
                {option.help ? <p className="text-[11px] text-muted-foreground">{option.help}</p> : null}
              </div>
            ))}
          </div>
        ) : null}

        {booleanOptions.map((option) => (
          <div key={option.id} className="flex items-center justify-between rounded-lg border border-border bg-[var(--ctp-mantle)] px-3 py-2.5">
            <div>
              <Label htmlFor={`${model.id}-${option.id}`}>{option.label}</Label>
              {option.help ? <p className="mt-0.5 text-[11px] text-muted-foreground">{option.help}</p> : null}
            </div>
            <Switch
              id={`${model.id}-${option.id}`}
              checked={Boolean(options[option.id] ?? option.default)}
              onCheckedChange={(checked) => setOptions((current) => ({ ...current, [option.id]: checked }))}
            />
          </div>
        ))}

        <Button type="button" className="w-full" onClick={submitTask} disabled={submit.isPending}>
          <Upload className="size-4" /> {submit.isPending ? '提交中…' : `加入 ${model.label} 队列`}
        </Button>

        <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          <span>Queued {status?.queued_by_model[model.id] ?? 0}</span>
          <span>Inflight {status?.inflight_by_model[model.id] ?? 0}</span>
        </div>
      </CardContent>
    </Card>
  )
}
