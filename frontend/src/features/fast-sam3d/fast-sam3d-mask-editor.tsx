import { Brush, Eraser, LoaderCircle, RotateCcw, Sparkles } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { getMaskCandidate, getMaskStatus, queueAutoMask } from '@/shared/api/client'
import type { ArtifactSource, MaskCandidate } from '@/shared/api/types'

type LoadedCandidate = MaskCandidate & {
  blob: Blob
  objectUrl: string
}

type FastSam3DMaskEditorProps = {
  token: string
  sourceUrl?: string
  file: File | null
  resetKey: number
  onMaskChange: (file: File | null) => void
}

const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

export function FastSam3DMaskEditor({ token, sourceUrl, file, resetKey, onMaskChange }: FastSam3DMaskEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const drawingRef = useRef(false)
  const candidateUrlsRef = useRef<string[]>([])
  const autoStartedRef = useRef<string | null>(null)
  const [candidates, setCandidates] = useState<LoadedCandidate[]>([])
  const [selectedCandidate, setSelectedCandidate] = useState(0)
  const [maskStatus, setMaskStatus] = useState<'idle' | 'queued' | 'inflight' | 'ready' | 'failed'>('idle')
  const [maskError, setMaskError] = useState('')
  const [tool, setTool] = useState<'add' | 'erase'>('add')
  const [brushSize, setBrushSize] = useState(28)

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : sourceUrl), [file, sourceUrl])
  const sourceKey = file ? `file:${file.name}:${file.size}:${file.lastModified}` : sourceUrl ? `url:${sourceUrl}` : ''

  useEffect(() => {
    return () => {
      if (file && previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [file, previewUrl])

  const clearCandidateUrls = () => {
    for (const url of candidateUrlsRef.current) URL.revokeObjectURL(url)
    candidateUrlsRef.current = []
  }

  const publishMask = async () => {
    const canvas = canvasRef.current
    if (!canvas || !canvas.width || !canvas.height) return
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
    if (!blob) return
    onMaskChange(new File([blob], 'fast-sam3d-mask.png', { type: 'image/png' }))
  }

  const loadMaskBlob = async (blob: Blob) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const bitmap = await createImageBitmap(blob)
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const context = canvas.getContext('2d')
    if (!context) return
    context.globalCompositeOperation = 'source-over'
    context.globalAlpha = 1
    context.fillStyle = '#000000'
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.drawImage(bitmap, 0, 0)
    bitmap.close()
    await publishMask()
  }

  const selectCandidate = async (index: number, items = candidates) => {
    const candidate = items[index]
    if (!candidate) return
    setSelectedCandidate(index)
    await loadMaskBlob(candidate.blob)
  }

  const requestAutoMask = async (source: ArtifactSource) => {
    clearCandidateUrls()
    setCandidates([])
    onMaskChange(null)
    setMaskError('')
    setMaskStatus('queued')
    try {
      const queued = await queueAutoMask(token, source)
      const taskId = queued.task.id
      for (let attempt = 0; attempt < 180; attempt += 1) {
        const status = await getMaskStatus(token, taskId)
        setMaskStatus(status.status)
        if (status.status === 'failed') throw new Error(status.error || 'SAM2 自动遮罩失败')
        if (status.status === 'ready') {
          const metadata = status.candidates ?? []
          if (!metadata.length) throw new Error('SAM2 没有返回可用遮罩')
          const loaded = await Promise.all(
            metadata.map(async (candidate) => {
              const blob = await getMaskCandidate(token, candidate.url)
              const objectUrl = URL.createObjectURL(blob)
              candidateUrlsRef.current.push(objectUrl)
              return { ...candidate, blob, objectUrl }
            }),
          )
          setCandidates(loaded)
          setMaskStatus('ready')
          await selectCandidate(0, loaded)
          return
        }
        await sleep(1000)
      }
      throw new Error('SAM2 自动遮罩等待超时')
    } catch (error) {
      const message = error instanceof Error ? error.message : '未知错误'
      setMaskStatus('failed')
      setMaskError(message)
      toast.error(`自动遮罩失败：${message}`)
    }
  }

  useEffect(() => {
    clearCandidateUrls()
    setCandidates([])
    setSelectedCandidate(0)
    setMaskStatus('idle')
    setMaskError('')
    onMaskChange(null)
    const canvas = canvasRef.current
    if (canvas) {
      canvas.width = 0
      canvas.height = 0
    }
    if (!sourceKey) {
      autoStartedRef.current = null
      return
    }
    if (autoStartedRef.current === sourceKey) return
    autoStartedRef.current = sourceKey
    const source: ArtifactSource = sourceUrl ? { kind: 'source', sourceUrl } : { kind: 'file', file: file! }
    void requestAutoMask(source)
    // sourceKey intentionally represents the selected image identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey, resetKey, token])

  useEffect(() => () => clearCandidateUrls(), [])

  const drawAtPointer = (event: React.PointerEvent<HTMLCanvasElement>, start: boolean) => {
    const canvas = canvasRef.current
    if (!canvas || !canvas.width || !canvas.height) return
    const rect = canvas.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * canvas.width
    const y = ((event.clientY - rect.top) / rect.height) * canvas.height
    const scale = canvas.width / Math.max(1, rect.width)
    const context = canvas.getContext('2d')
    if (!context) return
    context.globalCompositeOperation = 'source-over'
    context.globalAlpha = 1
    context.strokeStyle = tool === 'add' ? '#ffffff' : '#000000'
    context.fillStyle = context.strokeStyle
    context.lineCap = 'round'
    context.lineJoin = 'round'
    context.lineWidth = brushSize * scale
    if (start) {
      context.beginPath()
      context.arc(x, y, Math.max(1, context.lineWidth / 2), 0, Math.PI * 2)
      context.fill()
      context.beginPath()
      context.moveTo(x, y)
    } else {
      context.lineTo(x, y)
      context.stroke()
    }
  }

  const retryAuto = () => {
    if (!sourceKey) return
    const source: ArtifactSource = sourceUrl ? { kind: 'source', sourceUrl } : { kind: 'file', file: file! }
    void requestAutoMask(source)
  }

  if (!sourceKey || !previewUrl) {
    return (
      <div className="rounded-lg border border-dashed border-border px-3 py-4 text-center text-xs text-muted-foreground">
        先选择 RGB 图片；Fast-SAM3D 会自动提交到 Kaggle SAM2 生成遮罩。
      </div>
    )
  }

  const busy = maskStatus === 'queued' || maskStatus === 'inflight'

  return (
    <div className="space-y-3 rounded-lg border border-border bg-[var(--ctp-mantle)] p-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <Label>智能遮罩 · Kaggle SAM2.1 Small</Label>
          <p className="mt-0.5 text-[11px] text-muted-foreground">
            自动选主体；不使用浏览器 WebGPU。白色区域会送给 Fast-SAM3D。
          </p>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={retryAuto} disabled={busy}>
          {busy ? <LoaderCircle className="size-3.5 animate-spin" /> : <Sparkles className="size-3.5" />}
          {busy ? 'SAM2 处理中' : '重新识别'}
        </Button>
      </div>

      <div className="relative overflow-hidden rounded-lg border border-border bg-black">
        <img src={previewUrl} alt="Fast-SAM3D 原图" className="block h-auto w-full select-none" draggable={false} />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 size-full touch-none cursor-crosshair"
          style={{ mixBlendMode: 'screen', opacity: 0.58 }}
          onPointerDown={(event) => {
            if (maskStatus !== 'ready') return
            drawingRef.current = true
            event.currentTarget.setPointerCapture(event.pointerId)
            drawAtPointer(event, true)
          }}
          onPointerMove={(event) => {
            if (drawingRef.current) drawAtPointer(event, false)
          }}
          onPointerUp={(event) => {
            if (!drawingRef.current) return
            drawingRef.current = false
            if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
            void publishMask()
          }}
          onPointerCancel={() => {
            drawingRef.current = false
          }}
        />
        {busy ? (
          <div className="absolute inset-0 flex items-center justify-center bg-black/55 text-xs text-white">
            <LoaderCircle className="mr-2 size-4 animate-spin" /> Kaggle T4 正在生成候选遮罩…
          </div>
        ) : null}
      </div>

      {maskError ? <p className="text-xs text-destructive">{maskError}；仍可在下方手动上传 Mask。</p> : null}

      {candidates.length ? (
        <div className="space-y-2">
          <p className="text-[11px] text-muted-foreground">SAM2 候选 · 默认已选择评分最高的主体</p>
          <div className="grid grid-cols-3 gap-2">
            {candidates.map((candidate, index) => (
              <button
                key={candidate.index}
                type="button"
                className={`overflow-hidden rounded-md border p-1 text-left transition ${selectedCandidate === index ? 'border-primary bg-primary/10' : 'border-border'}`}
                onClick={() => void selectCandidate(index)}
              >
                <img src={candidate.objectUrl} alt={`Mask 候选 ${index + 1}`} className="aspect-square w-full bg-black object-contain" />
                <span className="mt-1 block font-mono text-[9px] text-muted-foreground">
                  #{index + 1} score {candidate.score.toFixed(3)} · {(candidate.area_ratio * 100).toFixed(1)}%
                </span>
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-2">
        <Button type="button" variant={tool === 'add' ? 'default' : 'outline'} size="sm" onClick={() => setTool('add')} disabled={maskStatus !== 'ready'}>
          <Brush className="size-3.5" /> 补选
        </Button>
        <Button type="button" variant={tool === 'erase' ? 'default' : 'outline'} size="sm" onClick={() => setTool('erase')} disabled={maskStatus !== 'ready'}>
          <Eraser className="size-3.5" /> 擦除
        </Button>
        <Button type="button" variant="outline" size="sm" onClick={() => void selectCandidate(selectedCandidate)} disabled={!candidates.length}>
          <RotateCcw className="size-3.5" /> 重置候选
        </Button>
        <label className="ml-auto flex items-center gap-2 text-[11px] text-muted-foreground">
          画笔 {brushSize}px
          <input type="range" min="6" max="96" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} />
        </label>
      </div>

      <div className="space-y-1.5">
        <Label htmlFor={`fast-sam3d-manual-mask-${resetKey}`} className="text-[11px] text-muted-foreground">手动 Mask 兜底</Label>
        <Input
          id={`fast-sam3d-manual-mask-${resetKey}`}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(event) => {
            const selected = event.target.files?.[0]
            if (selected) void loadMaskBlob(selected)
          }}
          className="h-8 cursor-pointer text-xs file:mr-2 file:text-foreground"
        />
      </div>
    </div>
  )
}
