// Run: cd frontend && node_modules/.bin/tsc src/utils/sequenceBuffer.ts --outDir /tmp/sb --target es2019 --module commonjs && cp ../backup/test_sequence_buffer.js /tmp/sb/test.js && node /tmp/sb/test.js
const { SequenceBuffer, SEQ_FRAMES, SEQ_SPAN_MS } = require("./sequenceBuffer.js");
const assert = (c, m) => { if (!c) { console.log("FAIL:", m); process.exitCode = 1; } else console.log("ok  :", m); };
const lm = (x, vis = 1) => Array.from({ length: 33 }, (_, j) => [x + j, 0.5, 0, vis]);
// 1) a body moving at constant velocity, camera at 30 fps for 3 s -> window must be a LINEAR ramp in time at 25 fps
let b = new SequenceBuffer(); const v = 0.001; // x units per ms
for (let t = 0; t <= 3000; t += 1000 / 30) b.push(lm(v * t), t);
let w = b.window();
assert(w && w.length === SEQ_FRAMES && w[0].length === 99, "shape 60x99");
const dx = w[1][0] - w[0][0]; assert(Math.abs(dx - v * 40) < 1e-9, `consecutive frames are 40 ms apart (dx=${dx.toFixed(6)})`);
assert(Math.abs((w[59][0] - w[0][0]) - v * SEQ_SPAN_MS) < 1e-9, "window spans 2.36 s");
// 2) the same motion at ~2 fps (the OLD app cadence) must NOT yield a window (too sparse)
b = new SequenceBuffer(); for (let t = 0; t <= 5000; t += 500) b.push(lm(v * t), t); assert(b.window() === null, "2 fps input -> null (falls back to per-frame path)");
// 3) not enough history yet
b = new SequenceBuffer(); for (let t = 0; t <= 1500; t += 33) b.push(lm(0), t); assert(b.window() === null, "1.5 s of history -> null");
// 4) camera stall inside the window
b = new SequenceBuffer(); for (let t = 0; t <= 2000; t += 33) b.push(lm(0), t); for (let t = 2600; t <= 4000; t += 33) b.push(lm(0), t); assert(b.window() === null, "600 ms stall inside the window -> null");
// 5) low-visibility joint holds its last good position
b = new SequenceBuffer(); for (let t = 0; t <= 3000; t += 40) { const l = lm(0); if (t >= 1000) l[5] = [99, 99, 99, 0.1]; b.push(l, t); }
w = b.window(); assert(w && w.every(f => f[15] === 5 && f[16] === 0.5), "invisible joint keeps last good coords, not the garbage ones");
// 6) 15 fps camera still works
b = new SequenceBuffer(); for (let t = 0; t <= 3000; t += 1000 / 15) b.push(lm(v * t), t); w = b.window(); assert(w && Math.abs((w[59][0] - w[0][0]) - v * SEQ_SPAN_MS) < 1e-9, "15 fps camera resampled to the same 25 fps window");
// 7) wrong landmark count is ignored; clear() resets
b = new SequenceBuffer(); b.push([[0, 0, 0, 1]], 0); assert(b.window() === null, "bad frame ignored"); 
