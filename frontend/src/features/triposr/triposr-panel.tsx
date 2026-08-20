import { Box, Info, Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { useSubmitTripo } from '@/features/triposr/use-submit-tripo'
import type { HubStatus, TripoSettings } from '@/shared/api/types'
import { TRIPO_RESOLUTION_OPTIONS, getTripoResolutionOption } from '@/shared/tripo-resolution'

type TripoPanelProps = {
  token: string
  settings: TripoSettings
  onSettingsChange: (settings: TripoSettings) => void
  status?: HubStatus
}

export function TripoPanel({ token, settings, onSettingsChange, status }: TripoPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const submit = useSubmitTripo(token)
  const selectedResolution = getTripoResolutionOption(settings.resolution)

  const submitFile = () => {
    if (!file) return toast.error('请选择一张 PNG、JPEG 或 WebP 图片')
    submit.mutate(
      { kind: 'file', file, settings },
      {
        onSuccess: () => {
          setFile(null)
          if (inputRef.current) inputRef.current.value = ''
        },
      },
    )
  }

  return (
    <Card className="border-border bg-card shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Box className="size-4 text-primary" /> TripoSR · Image to 3D
        </CardTitle>
        <CardDescription>上传本地图片，或从结果卡片直接创建 3D 模型</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="tripo-file">PNG / JPEG / WebP · 最大 20 MB</Label>
          <Input
            ref={inputRef}
            id="tripo-file"
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="cursor-pointer file:mr-3 file:text-foreground"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <Label htmlFor="tripo-output">Output</Label>
            <Select
              value={settings.outputFormat}
              onValueChange={(value) =>
                onSettingsChange({ ...settings, outputFormat: value as TripoSettings['outputFormat'] })
              }
            >
              <SelectTrigger id="tripo-output" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="glb">GLB（推荐）</SelectItem>
                <SelectItem value="obj">OBJ</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="tripo-resolution">模型细节（MC 网格）</Label>
            <Select
              value={String(settings.resolution)}
              onValueChange={(value) =>
                onSettingsChange({ ...settings, resolution: Number(value) as TripoSettings['resolution'] })
              }
            >
              <SelectTrigger id="tripo-resolution" className="w-full">
                <SelectValue>{selectedResolution.shortLabel}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {TRIPO_RESOLUTION_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={String(option.value)} textValue={option.shortLabel}>
                    <span className="flex flex-col items-start gap-0.5 py-0.5">
                      <span className="font-medium">{option.shortLabel}</span>
                      <span className="text-[10px] leading-tight text-muted-foreground">{option.description}</span>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="flex items-start gap-1 text-[10px] leading-relaxed text-muted-foreground">
              <Info className="mt-0.5 size-3 shrink-0" />
              数字越大，3D 表面网格越细；不是输入图片尺寸，会增加生成时间和显存占用。
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between rounded-lg border border-border bg-[var(--ctp-mantle)] px-3 py-2.5">
          <div>
            <Label htmlFor="remove-bg">自动移除背景</Label>
            <p className="mt-0.5 text-[11px] text-muted-foreground">缩放并居中单个主体</p>
          </div>
          <Switch
            id="remove-bg"
            checked={settings.removeBackground}
            onCheckedChange={(checked) => onSettingsChange({ ...settings, removeBackground: checked })}
          />
        </div>

        <Button type="button" className="w-full" onClick={submitFile} disabled={submit.isPending}>
          <Upload className="size-4" /> {submit.isPending ? '上传中…' : '上传并加入 3D 队列'}
        </Button>

        <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          <span>Queued {status?.queued_by_model.triposr ?? 0}</span>
          <span>Completed {status?.artifacts ?? 0}</span>
        </div>
      </CardContent>
    </Card>
  )
}
