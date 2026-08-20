import { Download, X } from 'lucide-react'
import { Dialog as DialogPrimitive } from 'radix-ui'

import { Button } from '@/components/ui/button'

type ImagePreviewDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  src: string
  alt: string
}

export function ImagePreviewDialog({ open, onOpenChange, src, alt }: ImagePreviewDialogProps) {
  if (!open) return null

  return (
    <DialogPrimitive.Root open onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm" />
        <DialogPrimitive.Content className="fixed inset-0 z-50 flex size-full flex-col bg-[var(--ctp-crust)]/95 p-3 outline-none sm:inset-4 sm:size-auto sm:rounded-2xl sm:border sm:border-border">
          <DialogPrimitive.Title className="sr-only">放大查看图片</DialogPrimitive.Title>
          <DialogPrimitive.Description className="sr-only">
            图片预览，按 Escape 或关闭按钮退出
          </DialogPrimitive.Description>

          <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-xl bg-black/20">
            <img src={src} alt={alt} className="max-h-full max-w-full object-contain" />
          </div>

          <div className="flex items-center justify-between gap-3 px-1 pt-3">
            <p className="truncate text-xs text-muted-foreground">放大预览 · 按 Escape 关闭</p>
            <div className="flex shrink-0 items-center gap-2">
              <Button asChild type="button" variant="outline" size="sm">
                <a href={src} download>
                  <Download className="size-3.5" /> 下载原图
                </a>
              </Button>
              <DialogPrimitive.Close asChild>
                <Button type="button" variant="outline" size="sm">
                  <X className="size-3.5" /> 关闭
                </Button>
              </DialogPrimitive.Close>
            </div>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
