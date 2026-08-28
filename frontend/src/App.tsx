import { useQuery } from '@tanstack/react-query'
import { AlertCircle } from 'lucide-react'
import { lazy, Suspense, useEffect, useMemo, useState } from 'react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Gallery } from '@/features/gallery/gallery'
import { ModelTopbar } from '@/features/status/model-topbar'
import { WorkspaceHeader, type WorkspaceKind } from '@/features/status/workspace-header'
import { useHubSocket } from '@/hooks/use-hub-socket'
import { historyQuery, modelsQuery, promptPipelineQuery, statusQuery } from '@/shared/api/queries'

const DEFAULT_MODEL = 'sana-sprint-1.6b'
const GenerationPanel = lazy(async () => {
  const module = await import('@/features/generation/generation-panel')
  return { default: module.GenerationPanel }
})
const ControlCenterSheet = lazy(async () => {
  const module = await import('@/features/status/control-center-sheet')
  return { default: module.ControlCenterSheet }
})
const ImageToArtifactPanel = lazy(async () => {
  const module = await import('@/features/artifacts/image-to-artifact-panel')
  return { default: module.ImageToArtifactPanel }
})
const VideoWorkspace = lazy(async () => {
  const module = await import('@/features/video/video-workspace')
  return { default: module.VideoWorkspace }
})

function WorkspaceFallback({ className = 'min-h-52' }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-[var(--ctp-mantle)]/60 ${className}`} aria-hidden="true" />
}

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem('kaggle_hub_token') ?? '')
  const [workspace, setWorkspace] = useState<WorkspaceKind>('image')
  const [controlCenterOpen, setControlCenterOpen] = useState(false)
  const [selectedImageModel, setSelectedImageModel] = useState(
    () => localStorage.getItem('kaggle_hub_image_model') ?? DEFAULT_MODEL,
  )
  const [selectedArtifactModel, setSelectedArtifactModel] = useState(
    () => localStorage.getItem('kaggle_hub_artifact_model') ?? 'triposr',
  )
  const [artifactSourceUrl, setArtifactSourceUrl] = useState<string>()

  const models = useQuery(modelsQuery)
  const pipeline = useQuery(promptPipelineQuery)
  const status = useQuery(statusQuery)
  const history = useQuery(historyQuery)
  const connection = useHubSocket()

  useEffect(() => {
    localStorage.setItem('kaggle_hub_token', token)
  }, [token])

  const imageModels = models.data?.filter((model) => model.output_kind === 'image') ?? []
  const artifactModels = models.data?.filter((model) => model.output_kind === 'artifact') ?? []
  const currentModel = imageModels.find((model) => model.id === selectedImageModel) ?? imageModels[0]
  const currentArtifactModel = artifactModels.find((model) => model.id === selectedArtifactModel) ?? artifactModels[0]
  const resultCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const item of history.data ?? []) counts[item.model] = (counts[item.model] ?? 0) + 1
    return counts
  }, [history.data])
  const historyItems = history.data ?? []

  const updateImageModel = (model: string) => {
    setSelectedImageModel(model)
    localStorage.setItem('kaggle_hub_image_model', model)
  }

  const updateArtifactModel = (model: string) => {
    setSelectedArtifactModel(model)
    localStorage.setItem('kaggle_hub_artifact_model', model)
  }

  const updateWorkspaceModel = (model: string) => {
    if (workspace === '3d') updateArtifactModel(model)
    else if (workspace === 'image') updateImageModel(model)
  }

  const prepareImageTo3d = (sourceUrl: string) => {
    setArtifactSourceUrl(sourceUrl)
    setWorkspace('3d')
  }

  const startupError = models.error ?? pipeline.error ?? status.error ?? history.error
  const imageItems = currentModel
    ? historyItems.filter((item) => item.kind === 'image' && item.model === currentModel.id)
    : []
  const artifactItems = currentArtifactModel
    ? historyItems.filter((item) => item.kind === 'artifact' && item.model === currentArtifactModel.id)
    : []

  return (
    <div className="min-h-svh bg-background">
      <WorkspaceHeader
        workspace={workspace}
        onWorkspaceChange={setWorkspace}
        connection={connection}
        status={status.data}
        onOpenControlCenter={() => setControlCenterOpen(true)}
      />
      <ModelTopbar
        workspace={workspace}
        models={models.data ?? []}
        selectedModel={workspace === '3d' ? currentArtifactModel?.id ?? selectedArtifactModel : currentModel?.id ?? selectedImageModel}
        onSelectedModelChange={updateWorkspaceModel}
        status={status.data}
        resultCounts={resultCounts}
      />

      {startupError ? (
        <div className="mx-auto mt-5 w-full max-w-[1880px] px-4 sm:px-6 lg:px-8">
          <Alert variant="destructive">
            <AlertCircle className="size-4" />
            <AlertTitle>控制面数据读取失败</AlertTitle>
            <AlertDescription>{startupError instanceof Error ? startupError.message : '请确认 FastAPI 服务正在运行'}</AlertDescription>
          </Alert>
        </div>
      ) : null}

      <main className="mx-auto grid w-full max-w-[1880px] grid-cols-1 gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[360px_minmax(0,1fr)] lg:px-8 lg:py-8">
        {workspace === 'image' && currentModel ? (
          <aside className={workspace === 'image' ? 'lg:sticky lg:top-20 lg:self-start' : 'hidden'}>
            <Suspense fallback={<WorkspaceFallback />}>
              <GenerationPanel
                model={currentModel}
                pipeline={pipeline.data}
                status={status.data}
                token={token}
                modelResultCount={resultCounts[currentModel.id] ?? 0}
              />
            </Suspense>
          </aside>
        ) : null}

        {workspace === 'image' && currentModel ? (
            <Gallery
              key={currentModel.id}
              workspace="image"
              items={imageItems}
              model={currentModel}
              conversionModels={artifactModels}
              conversionModel={currentArtifactModel}
              onConversionModelChange={updateArtifactModel}
              onConvertTo3d={prepareImageTo3d}
              isLoading={history.isLoading}
              isRefreshing={history.isFetching}
              onRefresh={() => void history.refetch()}
            />
        ) : null}

        {workspace === '3d' && currentArtifactModel ? (
          <aside className="lg:sticky lg:top-20 lg:self-start">
            <Suspense fallback={<WorkspaceFallback />}>
              <ImageToArtifactPanel
                key={currentArtifactModel.id}
                model={currentArtifactModel}
                token={token}
                status={status.data}
                sourceUrl={artifactSourceUrl}
                onSourceUrlClear={() => setArtifactSourceUrl(undefined)}
              />
            </Suspense>
          </aside>
        ) : null}

        {workspace === '3d' && currentArtifactModel ? (
            <Gallery
              key={currentArtifactModel.id}
              workspace="3d"
              items={artifactItems}
              model={currentArtifactModel}
              isLoading={history.isLoading}
              isRefreshing={history.isFetching}
              onRefresh={() => void history.refetch()}
            />
        ) : null}

        {workspace === 'video' ? (
          <Suspense fallback={<WorkspaceFallback className="col-span-full min-h-96" />}>
            <VideoWorkspace />
          </Suspense>
        ) : null}
      </main>

      {controlCenterOpen ? (
        <Suspense fallback={null}>
          <ControlCenterSheet
            open
            onOpenChange={setControlCenterOpen}
            token={token}
            onTokenChange={setToken}
            status={status.data}
            models={models.data ?? []}
          />
        </Suspense>
      ) : null}
    </div>
  )
}
