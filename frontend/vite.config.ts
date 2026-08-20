import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    outDir: '../hub/web',
    emptyOutDir: true,
    // The GLB viewer is an intentional lazy boundary; keep its chunk budget tight.
    chunkSizeWarningLimit: 650,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:30100',
      '/task': 'http://127.0.0.1:30100',
      '/worker': 'http://127.0.0.1:30100',
      '/upload': 'http://127.0.0.1:30100',
      '/images': 'http://127.0.0.1:30100',
      '/outputs': 'http://127.0.0.1:30100',
      '/ws': {
        target: 'ws://127.0.0.1:30100',
        ws: true,
      },
    },
  },
})
