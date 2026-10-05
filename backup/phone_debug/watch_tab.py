"""Read-only watcher for the app's OWN tab on the phone while its owner tests: logs state changes (camera, engine, HUD text, errors,
results/s, skeleton strokes drawn). Installs only counters on the canvas API and listens to console errors. Other tabs are never touched.
Usage: python3 watch_tab.py [minutes]"""
import asyncio, json, sys, time, urllib.request, websockets
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 5
SITE = "https://yoga-posture-correction-system.vercel.app"
HOOK = r"""(() => { if (window.__w) return; window.__w = { clr: 0, str: 0, errs: [] };
  const cr = CanvasRenderingContext2D.prototype.clearRect, st = CanvasRenderingContext2D.prototype.stroke;
  CanvasRenderingContext2D.prototype.clearRect = function (...a) { if (this.canvas && this.canvas.className === 'ap-canvas') window.__w.clr++; return cr.apply(this, a); };
  CanvasRenderingContext2D.prototype.stroke = function (...a) { if (this.canvas && this.canvas.className === 'ap-canvas') window.__w.str++; return st.apply(this, a); };
  window.addEventListener('error', e => window.__w.errs.push(String(e.message).slice(0, 160))); window.addEventListener('unhandledrejection', e => window.__w.errs.push('rej: ' + String(e.reason && e.reason.message || e.reason).slice(0, 160))); })()"""
STATE = r"""(() => { const v = document.querySelector('video'); const w = window.__w || {};
  return JSON.stringify({ vis: document.visibilityState, cam: !!(v && v.srcObject), play: !!(v && !v.paused && v.readyState >= 2), eng: localStorage.getItem('asana.engine') ? 'cpu' : 'gpu', lang: document.documentElement.lang,
   hud: [...document.querySelectorAll('.ap-hud span, .ap-hud button')].map(x => x.textContent.trim()).filter(Boolean).slice(0, 4), err: (document.querySelector('.ap-idle.err h2') || {}).textContent || null,
   clr: w.clr || 0, str: w.str || 0, errs: (w.errs || []).slice(-3) }); })()"""
def tabs(): return [t for t in json.loads(urllib.request.urlopen("http://localhost:9222/json", timeout=10).read()) if t["type"] == "page" and t["url"].startswith(SITE) and "-" not in t["url"].split("//")[1].split(".")[0].replace("yoga-posture-correction-system", "")]
async def main():
    end = time.time() + MINUTES * 60; last = None; prev = (0, 0, time.time()); logs = []
    print(f"watching for {MINUTES} min; start the camera in the app tab and stand in view", flush=True)
    while time.time() < end:
        try:
            cands = tabs()
            vis = None
            for t in cands:
                try:
                    async with websockets.connect(t["webSocketDebuggerUrl"], max_size=10_000_000, open_timeout=3) as ws:
                        async def ev(js):
                            await ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": js, "returnByValue": True}}))
                            return json.loads(await asyncio.wait_for(ws.recv(), 4))["result"].get("result", {}).get("value")
                        await ev(HOOK); s = json.loads(await ev(STATE))
                        if s["vis"] == "visible": vis = s; break
                except Exception: continue
            if vis:
                now = time.time(); dt = now - prev[2]; rps = round((vis["clr"] - prev[0]) / dt, 1) if dt else 0; sps = round((vis["str"] - prev[1]) / dt, 1) if dt else 0
                prev = (vis["clr"], vis["str"], now)
                key = (vis["cam"], vis["play"], vis["eng"], vis["lang"], tuple(vis["hud"]), vis["err"], vis["errs"][-1:] and vis["errs"][-1], sps > 0)
                if key != last:
                    print(time.strftime("%H:%M:%S"), json.dumps({**vis, "results_per_s": rps, "strokes_per_s": sps}, ensure_ascii=False), flush=True); last = key
        except Exception as e:
            print("watch error", type(e).__name__, flush=True)
        await asyncio.sleep(3)
    print("done", flush=True)
asyncio.run(main())
