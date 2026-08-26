import { Box, Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useSubmitFastSam3D } from '@/features/fast-sam3d/use-submit-fast-sam3d'
import type { FastSam3DSettings, HubStatus } from '@/shared/api/types'

type FastSam3DPanelProps = {
  token: string
  settings: FastSam3DSettings
  onSettingsChange: (settings: FastSam3DSettings) => void
  status?: HubStatus
}

export function FastSam3DPanel({ token, settings, onSettingsChange, status }: FastSam3DPanelProps) {
  const imageRef = useRef<HTMLInputElement>(null)
  const maskRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [mask, setMask] = useState<File | null>(null)
  const submit = useSubmitFastSam3D(token)

  const submitFiles = () => {
    if (!file) return toast.error('请选择 RGB 图片')
    if (!mask) return toast.error('请选择与 RGB 同尺寸的 mask')
    submit.mutate(
      { kind: 'file', file, mask, settings },
      {
        onSuccess: () => {
          setFile(null)
          setMask(null)
          if (imageRef.current) imageRef.current.value = ''
          if (maskRef.current) maskRef.current.value = ''
        },
      },
    )
  }

  return (
    <Card className="border-border bg-card shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Box className="size-4 text-primary" /> Fast-SAM3D · Masked Image to 3D
        </CardTitle>
        <CardDescription>双 T4 常驻 Worker；需要 RGB 图片和同尺寸非空 mask</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="fast-sam3d-image">RGB · PNG / JPEG / WebP</Label>
          <Input
            ref={imageRef}
            id="fast-sam3d-image"
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="cursor-pointer file:mr-3 file:text-foreground"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="fast-sam3d-mask">Mask · 建议 PNG 黑白图</Label>
          <Input
            ref={maskRef}
            id="fast-sam3d-mask"
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(event) => setMask(event.target.files?.[0] ?? null)}
            className="cursor-pointer file:mr-3 file:text-foreground"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="fast-sam3d-seed">Seed</Label>
          <Input
            id="fast-sam3d-seed"
            type="number"
            value={settings.seed}
            onChange={(event) => onSettingsChange({ ...settings, seed: Number(event.target.value) || 0 })}
          />
        </div>
        <div className="rounded-lg border border-border bg-[var(--ctp-mantle)] px-3 py-2.5">
          <p className="text-xs font-medium">Acceleration 固定开启</p>
          <p className="mt-0.5 text-[11px] text-muted-foreground">常驻加载 SS + SLaT + mesh aggregation 配置</p>
        </div>
        <Button type="button" className="w-full" onClick={submitFiles} disabled={submit.isPending}>
          <Upload className="size-4" /> {submit.isPending ? '上传中…' : '上传 image + mask 并加入队列'}
        </Button>
        <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          <span>Queued {status?.queued_by_model['fast-sam3d'] ?? 0}</span>
          <span>Inflight {status?.inflight_by_model['fast-sam3d'] ?? 0}</span>
        </div>
      </CardContent>
    </Card>
  )
}
