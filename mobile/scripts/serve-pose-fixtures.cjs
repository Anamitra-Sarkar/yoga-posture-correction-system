const http = require("http");
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "../test-assets/pose-pipeline");
const mime = { ".html": "text/html; charset=utf-8", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".png": "image/png" };

http.createServer((request, response) => {
  const pathname = new URL(request.url, "http://127.0.0.1").pathname;
  if (request.method === "POST" && pathname === "/results") {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      fs.writeFileSync("/tmp/asana-pose-fixture-results.json", Buffer.concat(chunks));
      response.writeHead(204, { "Access-Control-Allow-Origin": "*", "Cache-Control": "no-store" });
      response.end();
    });
    return;
  }
  const requested = pathname === "/" ? "/mediapipe-harness.html" : pathname;
  const filePath = path.resolve(root, `.${requested}`);
  if (!filePath.startsWith(root)) { response.writeHead(403); response.end(); return; }
  fs.readFile(filePath, (error, data) => {
    if (error) { response.writeHead(404); response.end("Not found"); return; }
    response.writeHead(200, { "Content-Type": mime[path.extname(filePath).toLowerCase()] || "application/octet-stream", "Access-Control-Allow-Origin": "*", "Cache-Control": "no-store" });
    response.end(data);
  });
}).listen(8765, "0.0.0.0", () => console.log("Pose fixture server ready"));
