import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { queryKeys } from '@/shared/api/queries'
import type { HistoryItem } from '@/shared/api/types'

export type ConnectionState = 'connecting' | 'online' | 'offline'

export function useHubSocket(): ConnectionState {
  const queryClient = useQueryClient()
  const [connection, setConnection] = useState<ConnectionState>('connecting')

  useEffect(() => {
    let socket: WebSocket | null = null
    let reconnectTimer: number | null = null
    let stopped = false

    const connect = () => {
      if (stopped) return
      setConnection('connecting')
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${window.location.host}/ws`)

      socket.addEventListener('open', () => {
        setConnection('online')
        void queryClient.invalidateQueries({ queryKey: queryKeys.history })
        void queryClient.invalidateQueries({ queryKey: queryKeys.status })
        void queryClient.invalidateQueries({ queryKey: queryKeys.activeTasks })
      })

      socket.addEventListener('message', (event) => {
        const item = JSON.parse(String(event.data)) as HistoryItem
        queryClient.setQueryData<HistoryItem[]>(queryKeys.history, (current = []) => {
          if (current.some((entry) => entry.event_id === item.event_id)) return current
          return [...current, item]
        })
        void queryClient.invalidateQueries({ queryKey: queryKeys.status })
        void queryClient.invalidateQueries({ queryKey: queryKeys.activeTasks })
      })

      socket.addEventListener('close', () => {
        if (stopped) return
        setConnection('offline')
        reconnectTimer = window.setTimeout(connect, 1_500)
      })

      socket.addEventListener('error', () => socket?.close())
    }

    connect()
    return () => {
      stopped = true
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [queryClient])

  return connection
}
