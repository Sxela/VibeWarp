import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { resolve } from 'node:path';

export default defineConfig({
  plugins: [svelte()],
  build: { outDir: resolve('../vibewarp/web_static'), emptyOutDir: true },
  // Svelte ships a server build and a browser build. Under vitest the server one gets
  // picked by default and `mount()` throws — the components need the DOM one.
  resolve: { conditions: ['browser'] },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.js'],
    include: ['src/**/*.test.js'],
  },
});
