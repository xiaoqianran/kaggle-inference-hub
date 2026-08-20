import { useQuery } from '@tanstack/react-query'
import { AlertCircle } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Gallery } from '@/features/gallery/gallery'
import { GenerationPanel } from '@/features/generation/generation-panel'
import { ControlCenterSheet } from '@/features/status/control-center-sheet'
import { ModelTopbar } from '@/features/status/model-topbar'
import { WorkspaceHeader, type WorkspaceKind } from '@/features/status/workspace-header'
import { TripoPanel } from '@/features/triposr/triposr-panel'
import { VideoWorkspace } from '@/features/video/video-workspace'
import { useHubSocket } from '@/hooks/use-hub-socket'
import { historyQuery, modelsQuery, promptPipelineQuery, statusQuery } from '@/shared/api/queries'
import type { TripoSettings } from '@/shared/api/types'

const DEFAULT_MODEL = 'sana-sprint-1.6b'

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem('kaggle_hub_token') ?? '')
  const [workspace, setWorkspace] = useState<WorkspaceKind>('image')
  const [controlCenterOpen, setControlCenterOpen] = useState(false)
  const [selectedModel, setSelectedModel] = useState(
    () => localStorage.getItem('kaggle_hub_model') ?? DEFAULT_MODEL,
  )
  const [tripoSettings, setTripoSettings] = useState<TripoSettings>({
    outputFormat: 'glb',
    resolution: 256,
    removeBackground: true,
  })

  const models = useQuery(modelsQuery)
  const pipeline = useQuery(promptPipelineQuery)
  const status = useQuery(statusQuery)
  const history = useQuery(historyQuery)
  const connection = useHubSocket()

  useEffect(() => {
    localStorage.setItem('kaggle_hub_token', token)
  }, [token])

  const imageModels = models.data?.filter((model) => model.output_kind === 'image') ?? []
  const currentModel = imageModels.find((model) => model.id === selectedModel) ?? imageModels[0]
  const resultCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const item of history.data ?? []) counts[item.model] = (counts[item.model] ?? 0) + 1
    return counts
  }, [history.data])
  const historyItems = history.data ?? []

  const updateModel = (model: string) => {
    setSelectedModel(model)
    localStorage.setItem('kaggle_hub_model', model)
  }

  const startupError = models.error ?? pipeline.error ?? status.error ?? history.error
  const imageItems = currentModel
    ? historyItems.filter((item) => item.kind === 'image' && item.model === currentModel.id)
    : []
  const artifactItems = historyItems.filter((item) => item.kind === 'artifact')
  const tripoModel = models.data?.find((model) => model.output_kind === 'artifact')

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
        selectedModel={workspace === '3d' ? tripoModel?.id ?? 'triposr' : currentModel?.id ?? selectedModel}
        onSelectedModelChange={updateModel}
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
        {currentModel ? (
          <aside className={workspace === 'image' ? 'lg:sticky lg:top-20 lg:self-start' : 'hidden'}>
            <GenerationPanel
              model={currentModel}
              pipeline={pipeline.data}
              status={status.data}
              token={token}
              modelResultCount={resultCounts[currentModel.id] ?? 0}
            />
          </aside>
        ) : null}

        {workspace === 'image' && currentModel ? (
            <Gallery
              workspace="image"
              items={imageItems}
              model={currentModel}
              token={token}
              tripoSettings={tripoSettings}
              isLoading={history.isLoading}
              isRefreshing={history.isFetching}
              onRefresh={() => void history.refetch()}
            />
        ) : null}

        <aside className={workspace === '3d' ? 'lg:sticky lg:top-20 lg:self-start' : 'hidden'}>
          <TripoPanel token={token} settings={tripoSettings} onSettingsChange={setTripoSettings} status={status.data} />
        </aside>

        {workspace === '3d' ? (
            <Gallery
              workspace="3d"
              items={artifactItems}
              model={tripoModel}
              token={token}
              tripoSettings={tripoSettings}
              isLoading={history.isLoading}
              isRefreshing={history.isFetching}
              onRefresh={() => void history.refetch()}
            />
        ) : null}

        {workspace === 'video' ? <VideoWorkspace /> : null}
      </main>

      <ControlCenterSheet
        open={controlCenterOpen}
        onOpenChange={setControlCenterOpen}
        token={token}
        onTokenChange={setToken}
        status={status.data}
        models={models.data ?? []}
      />
    </div>
  )
}
