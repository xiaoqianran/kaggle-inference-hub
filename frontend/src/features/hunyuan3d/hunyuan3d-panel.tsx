import { Box, Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useSubmitHunyuan3D } from '@/features/hunyuan3d/use-submit-hunyuan3d'
import type { HubStatus, Hunyuan3DSettings } from '@/shared/api/types'

type Hunyuan3DPanelProps = {
  token: string
  settings: Hunyuan3DSettings
  onSettingsChange: (settings: Hunyuan3DSettings) => void
  status?: HubStatus
}

export function Hunyuan3DPanel({ token, settings, onSettingsChange, status }: Hunyuan3DPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const submit = useSubmitHunyuan3D(token)

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
          <Box className="size-4 text-primary" /> Hunyuan3D 2.1 · Image to PBR 3D
        </CardTitle>
        <CardDescription>双 T4 Shape + Paint 流程，输出带 PBR 材质的 GLB</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="hunyuan3d-file">PNG / JPEG / WebP · 最大 20 MB</Label>
          <Input
            ref={inputRef}
            id="hunyuan3d-file"
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="cursor-pointer file:mr-3 file:text-foreground"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <Label htmlFor="hunyuan-shape-steps">Shape Steps</Label>
            <Input
              id="hunyuan-shape-steps"
              type="number"
              min={1}
              max={50}
              value={settings.shapeSteps}
              onChange={(event) =>
                onSettingsChange({ ...settings, shapeSteps: Math.max(1, Math.min(50, Number(event.target.value) || 1)) })
              }
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="hunyuan-octree">Octree</Label>
            <Select
              value={String(settings.octreeResolution)}
              onValueChange={(value) =>
                onSettingsChange({ ...settings, octreeResolution: Number(value) as Hunyuan3DSettings['octreeResolution'] })
              }
            >
              <SelectTrigger id="hunyuan-octree" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="128">128 · 快速</SelectItem>
                <SelectItem value="256">256 · 标准</SelectItem>
                <SelectItem value="384">384 · 精细</SelectItem>
                <SelectItem value="512">512 · 高显存</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <Label htmlFor="hunyuan-paint-views">Paint Views</Label>
            <Select
              value={String(settings.paintViews)}
              onValueChange={(value) => onSettingsChange({ ...settings, paintViews: Number(value) })}
            >
              <SelectTrigger id="hunyuan-paint-views" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="2">2</SelectItem>
                <SelectItem value="4">4 · 标准</SelectItem>
                <SelectItem value="6">6</SelectItem>
                <SelectItem value="8">8</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="hunyuan-texture">Texture</Label>
            <Select
              value={String(settings.textureSize)}
              onValueChange={(value) =>
                onSettingsChange({ ...settings, textureSize: Number(value) as Hunyuan3DSettings['textureSize'] })
              }
            >
              <SelectTrigger id="hunyuan-texture" className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="1024">1024</SelectItem>
                <SelectItem value="2048">2048 · 标准</SelectItem>
                <SelectItem value="4096">4096</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-[var(--ctp-mantle)] px-3 py-2.5">
          <p className="text-xs font-medium">Paint Resolution {settings.paintResolution}</p>
          <p className="mt-0.5 text-[11px] text-muted-foreground">当前 Kaggle 适配默认固定为 256；Hub 会把参数随任务下发</p>
        </div>

        <Button type="button" className="w-full" onClick={submitFile} disabled={submit.isPending}>
          <Upload className="size-4" /> {submit.isPending ? '上传中…' : '上传并加入 Hunyuan3D 队列'}
        </Button>

        <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          <span>Queued {status?.queued_by_model['hunyuan3d-2.1'] ?? 0}</span>
          <span>Inflight {status?.inflight_by_model['hunyuan3d-2.1'] ?? 0}</span>
        </div>
      </CardContent>
    </Card>
  )
}
