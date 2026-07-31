import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxying keeps the browser talking to a single origin, so the websocket and
// the REST calls need no CORS negotiation during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
