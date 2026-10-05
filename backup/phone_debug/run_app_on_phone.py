"""Run the deployed app in a NEW Chrome tab on the USB-connected phone, grant the camera to that tab, press Start, and log every
WebGL context request/failure plus the app's own state. Usage: python3 run_app_on_phone.py <url> [seconds]"""
import asyncio, json, sys, time, urllib.request, websockets
URL = sys.argv[1] if len(sys.argv) > 1 else "https://yoga-posture-correction-system.vercel.app/"
SECS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
BASE = "http://localhost:9222"
HOOK = r"""
(() => {
  window.__ctx = []; window.__ev = [];
  const orig = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (t, a) { const r = orig.call(this, t, a); if (/webgl/i.test(t)) window.__ctx.push({ t, attrs: a ? JSON.stringify(a) : '', ok: !!r, off: false, at: Math.round(performance.now()) }); return r; };
  if (window.OffscreenCanvas) { const o2 = OffscreenCanvas.prototype.getContext; OffscreenCanvas.prototype.getContext = function (t, a) { const r = o2.call(this, t, a); if (/webgl/i.test(t)) window.__ctx.push({ t, attrs: a ? JSON.stringify(a) : '', ok: !!r, off: true, at: Math.round(performance.now()) }); return r; }; }
  addEventListener('webglcontextcreationerror', e => window.__ev.push('creationerror: ' + (e.statusMessage || '').slice(0, 300)), true);
  addEventListener('webglcontextlost', e => window.__ev.push('contextlost'), true);
  window.__clr = 0; const cr = CanvasRenderingContext2D.prototype.clearRect;
  CanvasRenderingContext2D.prototype.clearRect = function (...a) { if (this.canvas && this.canvas.className === 'ap-canvas') window.__clr++; return cr.apply(this, a); };
  window.__alerts = []; const al = window.alert; window.__rawAlert = al;
  window.addEventListener('error', e => window.__ev.push('error: ' + String(e.message).slice(0, 200)));
  window.addEventListener('unhandledrejection', e => window.__ev.push('rejection: ' + String(e.reason && e.reason.message || e.reason).slice(0, 200)));
})();
"""
async def main():
    ver = json.loads(urllib.request.urlopen(f"{BASE}/json/version", timeout=10).read())
    async with websockets.connect(ver["webSocketDebuggerUrl"], max_size=50_000_000) as bws:
        async def bsend(i, m, p): await bws.send(json.dumps({"id": i, "method": m, "params": p})); return json.loads(await bws.recv())
        tid = (await bsend(1, "Target.createTarget", {"url": "about:blank"}))["result"]["targetId"]
        origin = "/".join(URL.split("/")[:3])
        await bsend(2, "Browser.grantPermissions", {"origin": origin, "permissions": ["videoCapture"]})
        async with websockets.connect(f"ws://localhost:9222/devtools/page/{tid}", max_size=50_000_000) as ws:
            n = [10]; logs = []
            async def send(m, p=None):
                n[0] += 1; i = n[0]; await ws.send(json.dumps({"id": i, "method": m, "params": p or {}}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == i: return msg.get("result", {})
                    if msg.get("method") == "Runtime.consoleAPICalled": logs.append(msg["params"]["type"] + ": " + " ".join(str(a.get("value", a.get("description", "")))[:160] for a in msg["params"]["args"]))
            async def ev(js):
                r = await send("Runtime.evaluate", {"expression": js, "awaitPromise": True, "returnByValue": True})
                return r.get("result", {}).get("value", r.get("exceptionDetails", {}).get("text"))
            await send("Page.enable"); await send("Runtime.enable")
            await send("Page.addScriptToEvaluateOnNewDocument", {"source": HOOK})
            await send("Page.navigate", {"url": URL})
            await asyncio.sleep(8)
            print("loaded:", await ev("document.title + ' | vis=' + document.visibilityState"))
            print("start button:", await ev("(() => { const b = [...document.querySelectorAll('button')].find(b => /start|शुरू|চালু/i.test(b.textContent) && b.offsetParent); if (b) { b.click(); return b.textContent.trim(); } return 'NOT FOUND'; })()"))
            t0 = time.time()
            while time.time() - t0 < SECS:
                await asyncio.sleep(5)
                st = await ev("JSON.stringify({ctx: window.__ctx.length, err: !!document.querySelector('.ap-idle.err'), errTitle: (document.querySelector('.ap-idle.err h2')||{}).textContent||'', compat: !!document.querySelector('.compat'), hud: [...document.querySelectorAll('.ap-hud span, .ap-hud button')].map(x=>x.textContent.trim()).filter(Boolean).slice(0,4), results: window.__clr, mode: localStorage.getItem('asana.engine') ? 'cpu' : 'gpu'})")
                print(f"t+{int(time.time()-t0)}s {st}")
            print("WebGL context requests:", await ev("JSON.stringify(window.__ctx)"))
            print("events:", await ev("JSON.stringify(window.__ev)"))
            print("console:", logs[-12:])
            print("fallback TF.js loaded:", await ev("typeof window.tf"), "| tfjs flag:", await ev("localStorage.getItem('asana.engine.tfjs')"))
            print("stored engine mode:", await ev("localStorage.getItem('asana.engine')"))
            print("details panel:", await ev("(document.querySelector('.ap-gfx pre')||{}).textContent||'none'"))
            await ev("document.querySelector('.ap-dock-main.stop, .ap-btn.dark') && document.querySelector('.ap-dock-main.stop, .ap-btn.dark').click()")
        await bsend(3, "Target.closeTarget", {"targetId": tid})
asyncio.run(main())
