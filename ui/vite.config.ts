import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The Thor API lives on the LAN. Browsers enforce CORS, so for development we also expose it
// same-origin under /thor (set the API base URL to "/thor" in Settings to use the proxy).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const thor = env.VITE_THOR_URL || "http://192.168.178.182:8080";
  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy: {
        "/thor": {
          target: thor,
          changeOrigin: true,
          ws: true,
          rewrite: (p: string) => p.replace(/^\/thor/, ""),
          configure: (proxy) => {
            // Keep the dev console quiet while the Thor is unreachable: answer 502 instead of logging.
            proxy.on("error", (_err, _req, res) => {
              if ("writeHead" in res && !res.headersSent) {
                res.writeHead(502, { "content-type": "application/json" });
                res.end('{"error":"thor unreachable"}');
              }
            });
          },
        },
      },
    },
    preview: { port: 4173 },
    build: { chunkSizeWarningLimit: 1800, target: "es2022" },
  };
});
