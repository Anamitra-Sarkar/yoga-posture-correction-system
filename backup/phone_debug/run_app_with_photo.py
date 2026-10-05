"""Run the deployed app on the phone with a PHOTO of a person as the 'camera' (stubbed getUserMedia), to verify the whole pipeline
(engine -> landmarks -> skeleton drawn -> pose recognised) on that hardware regardless of who stands in front of the phone.
Usage: python3 run_app_with_photo.py <app url> <photo path on the app origin, e.g. /pose-images/warrior_2.jpg> [seconds]"""
import asyncio, json, sys, time, urllib.request, websockets
URL = sys.argv[1]; PHOTO = sys.argv[2]; SECS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
HOOK = r"""
(() => {
  window.__strokes = 0; window.__arcs = 0; window.__clr = 0; window.__bodies = 0;
  const st = CanvasRenderingContext2D.prototype.stroke, ar = CanvasRenderingContext2D.prototype.arc, cr = CanvasRenderingContext2D.prototype.clearRect;
  const mine = (c) => c && c.className === 'ap-canvas';
  CanvasRenderingContext2D.prototype.stroke = function (...a) { if (mine(this.canvas)) window.__strokes++; return st.apply(this, a); };
  CanvasRenderingContext2D.prototype.arc = function (...a) { if (mine(this.canvas)) window.__arcs++; return ar.apply(this, a); };
  CanvasRenderingContext2D.prototype.clearRect = function (...a) { if (mine(this.canvas)) window.__clr++; return cr.apply(this, a); };
  const of = window.fetch; window.fetch = function (i, init) { try { const u = typeof i === 'string' ? i : i.url; if (/analyse_frame/.test(u)) window.__bodies++; } catch (e) {} return of.apply(this, arguments); };
  navigator.mediaDevices.enumerateDevices = async () => [{ kind: 'videoinput', deviceId: 'c1', label: 'Cam' }];
  navigator.mediaDevices.getUserMedia = async () => {
    const img = new Image(); img.src = '__PHOTO__'; await new Promise(r => img.onload = r);
    const cv = document.createElement('canvas'); cv.width = 720; cv.height = 960; const g = cv.getContext('2d');
    const draw = () => { g.fillStyle = '#000'; g.fillRect(0, 0, 720, 960); const s = Math.min(720 / img.width, 960 / img.height); g.drawImage(img, (720 - img.width * s) / 2, (960 - img.height * s) / 2, img.width * s, img.height * s); };
    draw(); setInterval(draw, 66); return cv.captureStream(15);
  };
})();
""".replace("__PHOTO__", PHOTO)
async def main():
    ver = json.loads(urllib.request.urlopen("http://localhost:9222/json/version", timeout=10).read())
    async with websockets.connect(ver["webSocketDebuggerUrl"], max_size=50_000_000) as bws:
        await bws.send(json.dumps({"id": 1, "method": "Target.createTarget", "params": {"url": "about:blank"}})); tid = json.loads(await bws.recv())["result"]["targetId"]
        async with websockets.connect(f"ws://localhost:9222/devtools/page/{tid}", max_size=50_000_000) as ws:
            n = [10]; logs = []
            async def send(m, p=None):
                n[0] += 1; i = n[0]; await ws.send(json.dumps({"id": i, "method": m, "params": p or {}}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == i: return msg.get("result", {})
                    if msg.get("method") == "Runtime.consoleAPICalled": logs.append(msg["params"]["type"] + ": " + " ".join(str(a.get("value", a.get("description", "")))[:140] for a in msg["params"]["args"]))
            async def ev(js):
                r = await send("Runtime.evaluate", {"expression": js, "awaitPromise": True, "returnByValue": True})
                return r.get("result", {}).get("value", r.get("exceptionDetails", {}).get("text"))
            await send("Page.enable"); await send("Runtime.enable")
            await send("Page.addScriptToEvaluateOnNewDocument", {"source": HOOK})
            await send("Page.navigate", {"url": URL}); await asyncio.sleep(8)
            print("start:", await ev("(() => { const b = [...document.querySelectorAll('button')].find(b => /start|शुरू|চালু/i.test(b.textContent) && b.offsetParent); if (b) { b.click(); return b.textContent.trim(); } return 'NOT FOUND'; })()"))
            t0 = time.time(); switched = False
            while time.time() - t0 < SECS:
                await asyncio.sleep(5)
                if len(sys.argv) > 4 and not switched and time.time() - t0 > 10:
                    switched = True
                    print("switch language ->", sys.argv[4], await ev("(async () => { const open = [...document.querySelectorAll('button')].find(b => /^(EN|HI|BN)\\b/.test(b.textContent.trim()) && b.offsetParent); if (!open) return 'no language button'; open.click(); await new Promise(r => setTimeout(r, 300)); const names = {en: 'English', hi: 'हिन्दी', bn: 'বাংলা'}; const item = [...document.querySelectorAll('[role=option]')].find(b => b.textContent.trim() === names['" + sys.argv[4] + "']); if (!item) return 'no item'; item.click(); return 'clicked ' + item.textContent.trim(); })()"))
                print(f"t+{int(time.time()-t0)}s", await ev("JSON.stringify({results: window.__clr, strokes: window.__strokes, joints: window.__arcs, serverCalls: window.__bodies, err: !!document.querySelector('.ap-idle.err'), mode: localStorage.getItem('asana.engine') ? 'cpu' : 'gpu', hud: [...document.querySelectorAll('.ap-hud span, .ap-hud button')].map(x => x.textContent.trim()).filter(Boolean).slice(0, 4)})"))
            if len(sys.argv) > 5:
                shot = await send("Page.captureScreenshot", {"format": "png"})
                import base64; open(sys.argv[5], "wb").write(base64.b64decode(shot["data"])); print("screenshot saved:", sys.argv[5])
            print("console:", logs[-6:])
            await ev("[...document.querySelectorAll('button')].find(b => /^\\s*stop/i.test(b.textContent))?.click()")
        await bws.send(json.dumps({"id": 3, "method": "Target.closeTarget", "params": {"targetId": tid}})); await bws.recv()
asyncio.run(main())
