import { AlertTriangle, Box, LoaderCircle, RotateCcw } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import {
  ACESFilmicToneMapping,
  AnimationMixer,
  Box3,
  Clock,
  Color,
  DirectionalLight,
  GridHelper,
  HemisphereLight,
  PerspectiveCamera,
  Scene,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
} from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import type { Object3D } from 'three'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import type { ArtifactResult } from '@/shared/api/types'

type GlbPreviewSheetProps = {
  item: ArtifactResult
  open: boolean
  onOpenChange: (open: boolean) => void
}

type PreviewStatus = 'idle' | 'loading' | 'ready' | 'error'

type PreviewStats = {
  meshes: number
  triangles: number
  animations: number
}

function disposeMaterial(material: unknown) {
  if (!material || typeof material !== 'object') return

  for (const value of Object.values(material)) {
    if (value && typeof value === 'object') {
      const texture = value as { isTexture?: boolean; dispose?: () => void }
      if (texture.isTexture) texture.dispose?.()
    }
  }

  ;(material as { dispose?: () => void }).dispose?.()
}

function disposeModel(root: Object3D | null) {
  root?.traverse((object) => {
    const node = object as Object3D & {
      geometry?: { dispose?: () => void }
      material?: unknown | unknown[]
    }
    node.geometry?.dispose?.()

    if (Array.isArray(node.material)) {
      node.material.forEach(disposeMaterial)
    } else {
      disposeMaterial(node.material)
    }
  })
}

function formatTriangleCount(value: number) {
  return value >= 1000 ? `${(value / 1000).toFixed(value >= 100_000 ? 0 : 1)}k` : String(value)
}

function GlbCanvas({ url }: { url: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState<PreviewStatus>('idle')
  const [error, setError] = useState('')
  const [stats, setStats] = useState<PreviewStats | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let disposed = false
    let animationFrame = 0
    let modelRoot: Object3D | null = null
    let renderer: WebGLRenderer | null = null
    let controls: OrbitControls | null = null
    let resizeObserver: ResizeObserver | null = null
    const abortController = new AbortController()

    setStatus('loading')
    setError('')
    setStats(null)
    container.replaceChildren()

    const load = async () => {
      try {
        const response = await fetch(url, {
          signal: abortController.signal,
          cache: 'force-cache',
        })
        if (!response.ok) throw new Error(`模型请求失败（HTTP ${response.status}）`)
        const buffer = await response.arrayBuffer()
        if (disposed) return

        const scene = new Scene()
        scene.background = new Color('#292c3c')

        const camera = new PerspectiveCamera(35, 1, 0.01, 100)
        camera.position.set(2.8, 1.8, 2.8)

        renderer = new WebGLRenderer({
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance',
        })
        renderer.outputColorSpace = SRGBColorSpace
        renderer.toneMapping = ACESFilmicToneMapping
        renderer.toneMappingExposure = 1.15
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75))
        renderer.domElement.className = 'size-full block'
        renderer.domElement.setAttribute('aria-label', 'GLB 3D 模型预览画布')
        container.appendChild(renderer.domElement)

        scene.add(new HemisphereLight('#c6d0f5', '#232634', 2.1))
        const keyLight = new DirectionalLight('#ffffff', 2.8)
        keyLight.position.set(4, 6, 5)
        scene.add(keyLight)
        const fillLight = new DirectionalLight('#8caaee', 0.9)
        fillLight.position.set(-4, 2, -3)
        scene.add(fillLight)
        scene.add(new GridHelper(8, 16, '#626880', '#414559'))

        const loader = new GLTFLoader()
        const gltf = await loader.parseAsync(buffer, new URL('.', window.location.href).href)
        if (disposed) {
          disposeModel(gltf.scene)
          return
        }

        modelRoot = gltf.scene
        let meshes = 0
        let triangles = 0
        modelRoot.traverse((object) => {
          const node = object as {
            isMesh?: boolean
            geometry?: {
              getIndex?: () => { count: number } | null
              getAttribute?: (name: string) => { count: number } | undefined
            }
            frustumCulled?: boolean
          }
          if (!node.isMesh) return
          meshes += 1
          node.frustumCulled = true
          const index = node.geometry?.getIndex?.()
          const position = node.geometry?.getAttribute?.('position')
          triangles += Math.floor((index?.count ?? position?.count ?? 0) / 3)
        })

        const bounds = new Box3().setFromObject(modelRoot)
        const center = bounds.getCenter(new Vector3())
        const size = bounds.getSize(new Vector3())
        const maxDimension = Math.max(size.x, size.y, size.z, 0.001)
        modelRoot.position.sub(center)
        modelRoot.scale.setScalar(2 / maxDimension)
        scene.add(modelRoot)

        const controlsInstance = new OrbitControls(camera, renderer.domElement)
        controlsInstance.enableDamping = false
        controlsInstance.enablePan = true
        controlsInstance.minDistance = 1.4
        controlsInstance.maxDistance = 8
        controlsInstance.rotateSpeed = 0.7
        controlsInstance.zoomSpeed = 0.8
        controlsInstance.target.set(0, 0, 0)
        controlsInstance.update()
        controls = controlsInstance

        const render = () => {
          if (disposed || !renderer) return
          renderer.render(scene, camera)
        }
        const resize = () => {
          if (disposed || !renderer) return
          const width = container.clientWidth
          const height = container.clientHeight
          if (!width || !height) return
          renderer.setSize(width, height, false)
          camera.aspect = width / height
          camera.updateProjectionMatrix()
          render()
        }

        const animations = gltf.animations.length
        if (animations) {
          const mixer = new AnimationMixer(modelRoot)
          gltf.animations.forEach((clip) => mixer.clipAction(clip).play())
          const clock = new Clock()
          const animate = () => {
            if (disposed) return
            animationFrame = window.requestAnimationFrame(animate)
            mixer.update(clock.getDelta())
            render()
          }
          animate()
        } else {
          controlsInstance.addEventListener('change', render)
          render()
        }

        resizeObserver = new ResizeObserver(resize)
        resizeObserver.observe(container)
        resize()
        setStats({ meshes, triangles, animations })
        setStatus('ready')
      } catch (caught) {
        if (disposed || (caught instanceof DOMException && caught.name === 'AbortError')) return
        setError(caught instanceof Error ? caught.message : 'GLB 加载失败，请确认文件有效。')
        setStatus('error')
      }
    }

    void load()

    return () => {
      disposed = true
      abortController.abort()
      window.cancelAnimationFrame(animationFrame)
      resizeObserver?.disconnect()
      controls?.dispose()
      disposeModel(modelRoot)
      renderer?.dispose()
      renderer?.forceContextLoss?.()
      renderer?.domElement.remove()
    }
  }, [url])

  return (
    <div ref={containerRef} className="relative size-full min-h-80 overflow-hidden rounded-xl bg-[var(--ctp-mantle)]">
      {status === 'loading' ? (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-[var(--ctp-mantle)]/90 text-sm text-muted-foreground backdrop-blur-sm">
          <LoaderCircle className="size-6 animate-spin text-primary" />
          <span>正在加载 GLB…</span>
        </div>
      ) : null}
      {status === 'error' ? (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 px-8 text-center">
          <AlertTriangle className="size-7 text-destructive" />
          <p className="text-sm font-medium">无法预览这个 GLB</p>
          <p className="max-w-md text-xs leading-relaxed text-muted-foreground">{error}</p>
        </div>
      ) : null}
      {status === 'ready' && stats ? (
        <div className="pointer-events-none absolute bottom-3 left-3 z-10 flex flex-wrap items-center gap-2">
          <Badge variant="secondary" className="bg-[var(--ctp-crust)]/80 font-mono text-[9px] backdrop-blur">
            {stats.meshes} mesh · {formatTriangleCount(stats.triangles)} tris
          </Badge>
          {stats.animations ? (
            <Badge variant="secondary" className="bg-[var(--ctp-crust)]/80 font-mono text-[9px] backdrop-blur">
              {stats.animations} animation
            </Badge>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

export function GlbPreviewSheet({ item, open, onOpenChange }: GlbPreviewSheetProps) {
  const [reloadNonce, setReloadNonce] = useState(0)
  const url = item.download_url

  if (!open) return null

  return (
    <Sheet open onOpenChange={onOpenChange}>
      <SheetContent className="w-full border-l-border bg-[var(--ctp-crust)] p-0 sm:max-w-3xl" side="right">
        <div className="flex min-h-0 flex-1 flex-col">
          <SheetHeader className="border-b border-border px-5 py-5 pr-14 text-left">
            <div className="flex items-center gap-2">
              <Box className="size-4 text-primary" />
              <SheetTitle className="truncate">{item.source_label ?? 'GLB 资产预览'}</SheetTitle>
            </div>
            <SheetDescription>
              拖动旋转 · 滚轮缩放 · 右键平移
            </SheetDescription>
          </SheetHeader>

          <div className="min-h-0 flex-1 p-4">
            {open ? <GlbCanvas key={`${url}-${reloadNonce}`} url={url} /> : null}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border px-4 py-3">
            <p className="font-mono text-[9px] uppercase tracking-wide text-muted-foreground">
              GLB · {item.vertices?.toLocaleString() ?? '—'} vertices · {item.faces?.toLocaleString() ?? '—'} faces
            </p>
            <div className="flex items-center gap-2">
              <Button type="button" size="sm" variant="outline" onClick={() => setReloadNonce((value) => value + 1)}>
                <RotateCcw className="size-3.5" /> 重载
              </Button>
              <Button asChild size="sm">
                <a href={item.download_url} download>
                  下载 GLB
                </a>
              </Button>
            </div>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
