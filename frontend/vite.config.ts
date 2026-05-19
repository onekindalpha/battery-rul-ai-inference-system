import { defineConfig } from 'vite'
const target = process.env.VITE_API_TARGET || 'http://127.0.0.1:8000'
export default defineConfig({
  server: {
    host: true,
    port: 5173,
    proxy: { '/api': { target, changeOrigin: true }, '/precomputed': { target, changeOrigin: true } }
  }
})
