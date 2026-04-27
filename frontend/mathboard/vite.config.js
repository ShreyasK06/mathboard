import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// `base: '/mathboard/'` so assets resolve under https://<user>.github.io/mathboard/.
// Override with VITE_BASE=/ for local dev or root deployment.
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE ?? '/mathboard/',
})
