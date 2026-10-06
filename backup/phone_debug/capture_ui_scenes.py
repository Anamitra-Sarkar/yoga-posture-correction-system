"""Capture screenshots of the DEPLOYED app (phone or desktop browser) for the paper, with a credited reference photo as the 'camera'.

Usage: DEVTOOLS_PORT=9222 python3 capture_ui_scenes.py <app origin> <out dir> [scene names...]
  phone:   adb forward tcp:9222 localabstract:chrome_devtools_remote   (Chrome open on the phone)
  desktop: brave-browser --headless=new --remote-debugging-port=9333 ...   then DEVTOOLS_PORT=9333 EMU=1440x900x1
Only tabs this script creates are touched. The app's own saved preferences (language, theme, mode, voice, target pose) are read first and put
back at the end, so the owner's settings are unchanged. Voice is forced off for the scenes (no sound on the phone).
Each scene = a fresh tab: stub camera -> press Start -> wait -> optional taps -> PNG + the visible text of the page (for the record).
"""
import asyncio, base64, json, os, sys, time, urllib.request, websockets

PORT = os.environ.get("DEVTOOLS_PORT", "9222")
EMU = os.environ.get("EMU")            # "WxHxDPR" -> Emulation.setDeviceMetricsOverride (desktop runs)
ORIGIN = sys.argv[1].rstrip("/"); OUT = sys.argv[2]; ONLY = set(sys.argv[3:])
KEYS = ["asana.lang", "asana.theme", "asana.practiceMode", "asana.targetPose", "asana.voice", "asana.range.v1"]   # range = the saved calibration; restored at the end

# name, photo (path on the app origin, or a full https URL: Wikimedia Commons sends CORS headers), prefs written before load, path, wait seconds after Start
# Every scene sets lang/theme/mode itself (BASE) so one scene's settings cannot leak into the next.
BASE = {"asana.lang": "en", "asana.theme": "light", "asana.practiceMode": "free"}
TRI = "https://upload.wikimedia.org/wikipedia/commons/9/9d/Trikonasana_Yoga-Asana_Nina-Mel.jpg"                      # Kennguru, CC BY 3.0
CHAIR = "https://upload.wikimedia.org/wikipedia/commons/5/59/Utkatasana_Yoga-Asana_Nina-Mel.jpg"                      # Kennguru, CC BY 3.0
DOG = "https://thumb.wikimedia.org/wikipedia/commons/thumb/5/57/Downward-Facing-Dog.JPG/960px-Downward-Facing-Dog.JPG"  # Iveto, CC BY 3.0
OPEN_RANGE = """(() => { const o = document.querySelector('button.ap-iconbtn[aria-label="Open sidebar"]'); if (o) o.click();
  return new Promise(r => setTimeout(() => { const h = [...document.querySelectorAll('.ap-acc-head')].find(b => /range|सीमा|সীমা/i.test(b.textContent));
    if (h) { if (h.getAttribute('aria-expanded') !== 'true') h.click(); h.scrollIntoView({block: 'start'}); } r(h ? 'range card found' : 'range card NOT found'); }, 900)); })()"""
SCENES = [
    dict(name="free_warrior2_en", photo="/pose-images/warrior_2.jpg", prefs={}, wait=40),
    dict(name="free_tree_hi", photo="/pose-images/tree_pose.jpg", prefs={"asana.lang": "hi"}, wait=40),
    dict(name="free_warrior2_bn_dark", photo="/pose-images/warrior_2.jpg", prefs={"asana.lang": "bn", "asana.theme": "dark"}, wait=40),
    dict(name="guided_tree_wrong", photo="/pose-images/warrior_2.jpg", prefs={"asana.practiceMode": "guided", "asana.targetPose": "tree_pose"}, wait=40),
    dict(name="guided_tree_right", photo="/pose-images/tree_pose.jpg", prefs={"asana.practiceMode": "guided", "asana.targetPose": "tree_pose"}, wait=40),
    dict(name="landing", photo=None, prefs={}, path="/landing", wait=6, start=False),
    # second batch: different poses and photos
    dict(name="free_triangle_en", photo=TRI, prefs={}, wait=40),
    dict(name="free_chair_hi", photo=CHAIR, prefs={"asana.lang": "hi"}, wait=40),
    dict(name="free_dog_bn_dark", photo=DOG, prefs={"asana.lang": "bn", "asana.theme": "dark"}, wait=40),
    dict(name="guided_triangle_right", photo=TRI, prefs={"asana.practiceMode": "guided", "asana.targetPose": "triangle"}, wait=40),
    dict(name="guided_chair_wrong", photo=DOG, prefs={"asana.practiceMode": "guided", "asana.targetPose": "chair_pose"}, wait=40),
    dict(name="guided_plank_en", photo="/pose-images/plank.jpg", prefs={"asana.practiceMode": "guided", "asana.targetPose": "plank"}, wait=40),
    # calibration ("your range of motion"): start with no saved range, show the 15 s countdown, then open the sidebar card with the learned ranges
    dict(name="calib_progress", photo="/pose-images/mountain_pose.jpg", prefs={"asana.range.v1": None}, wait=8,
         then=dict(name="calib_range", wait=17, js=OPEN_RANGE)),
    dict(name="landing_dark", photo=None, prefs={"asana.theme": "dark"}, path="/landing", wait=6, start=False),
]

HOOK = r"""
(() => {
  try { const P = __PREFS__; for (const k in P) { if (P[k] === null) localStorage.removeItem(k); else localStorage.setItem(k, P[k]); } localStorage.setItem('asana.voice', 'off'); } catch (e) {}
  const PHOTO = '__PHOTO__';
  if (PHOTO) {
    navigator.mediaDevices.enumerateDevices = async () => [{ kind: 'videoinput', deviceId: 'c1', label: 'Cam' }];
    navigator.mediaDevices.getUserMedia = async () => {
      const img = new Image(); img.crossOrigin = 'anonymous'; img.src = PHOTO; await new Promise(r => img.onload = r);
      const cv = document.createElement('canvas'); cv.width = 720; cv.height = 960; const g = cv.getContext('2d');
      const draw = () => { g.fillStyle = '#000'; g.fillRect(0, 0, 720, 960); const s = Math.min(720 / img.width, 960 / img.height); g.drawImage(img, (720 - img.width * s) / 2, (960 - img.height * s) / 2, img.width * s, img.height * s); };
      draw(); setInterval(draw, 66); return cv.captureStream(15);
    };
  }
})();
"""

async def with_tab(fn, url="about:blank"):
    ver = json.loads(urllib.request.urlopen(f"http://localhost:{PORT}/json/version", timeout=10).read())
    async with websockets.connect(ver["webSocketDebuggerUrl"], max_size=80_000_000) as bws:
        await bws.send(json.dumps({"id": 1, "method": "Target.createTarget", "params": {"url": url}}))
        tid = json.loads(await bws.recv())["result"]["targetId"]
        try:
            async with websockets.connect(f"ws://localhost:{PORT}/devtools/page/{tid}", max_size=80_000_000) as ws:
                n = [10]
                async def send(m, p=None):
                    n[0] += 1; i = n[0]; await ws.send(json.dumps({"id": i, "method": m, "params": p or {}}))
                    while True:
                        msg = json.loads(await ws.recv())
                        if msg.get("id") == i: return msg.get("result", {})
                async def ev(js):
                    r = await send("Runtime.evaluate", {"expression": js, "awaitPromise": True, "returnByValue": True})
                    return r.get("result", {}).get("value", r.get("exceptionDetails", {}).get("text"))
                await send("Page.enable"); await send("Runtime.enable")
                if EMU:
                    w, h, d = EMU.split("x"); await send("Emulation.setDeviceMetricsOverride", {"width": int(w), "height": int(h), "deviceScaleFactor": float(d), "mobile": int(w) < 800})
                return await fn(send, ev)
        finally:
            await bws.send(json.dumps({"id": 3, "method": "Target.closeTarget", "params": {"targetId": tid}})); await bws.recv()

async def read_prefs():
    async def fn(send, ev):
        await send("Page.navigate", {"url": ORIGIN + "/landing"}); await asyncio.sleep(4)
        return json.loads(await ev("JSON.stringify(" + json.dumps(KEYS) + ".reduce((o,k)=>{o[k]=localStorage.getItem(k);return o},{}))"))
    return await with_tab(fn)

async def write_prefs(prefs):
    async def fn(send, ev):
        await send("Page.navigate", {"url": ORIGIN + "/landing"}); await asyncio.sleep(4)
        return await ev("(()=>{const P=" + json.dumps(prefs) + ";for(const k in P){if(P[k]===null)localStorage.removeItem(k);else localStorage.setItem(k,P[k]);}return 'restored'})()")
    return await with_tab(fn)

async def scene(s):
    prefs = json.dumps({**BASE, **s["prefs"]}); photo = (s["photo"] if (s["photo"] or "").startswith("http") else ORIGIN + s["photo"]) if s["photo"] else ""; hook = HOOK.replace("__PREFS__", prefs).replace("__PHOTO__", photo)
    async def fn(send, ev):
        await send("Page.addScriptToEvaluateOnNewDocument", {"source": hook})
        await send("Page.navigate", {"url": ORIGIN + s.get("path", "/")}); await asyncio.sleep(8)
        info = {}
        if s.get("start", True):
            info["start"] = await ev("(() => { const b = [...document.querySelectorAll('button')].find(b => /start|शुरू|চালু/i.test(b.textContent) && b.offsetParent); if (b) { b.click(); return b.textContent.trim(); } return 'NOT FOUND'; })()")
        await asyncio.sleep(s["wait"])
        info["text"] = (await ev("document.body.innerText.slice(0, 1800)")) or ""
        info["engine"] = await ev("localStorage.getItem('asana.engine')")
        shot = await send("Page.captureScreenshot", {"format": "png"})
        open(os.path.join(OUT, s["name"] + ".png"), "wb").write(base64.b64decode(shot["data"]))
        if s.get("then"):
            t2 = s["then"]; await asyncio.sleep(t2["wait"]); info["then_js"] = await ev(t2["js"]); await asyncio.sleep(1.5)
            info["then_text"] = (await ev("document.body.innerText.slice(0, 1800)")) or ""
            shot2 = await send("Page.captureScreenshot", {"format": "png"})
            open(os.path.join(OUT, t2["name"] + ".png"), "wb").write(base64.b64decode(shot2["data"]))
        await ev("[...document.querySelectorAll('button')].find(b => /^\\s*(stop|रोक|থাম)/i.test(b.textContent))?.click()")
        return info
    return await with_tab(fn)

async def main():
    os.makedirs(OUT, exist_ok=True)
    orig = await read_prefs(); print("original prefs:", orig, flush=True)
    log = {}
    try:
        for s in SCENES:
            if ONLY and s["name"] not in ONLY: continue
            t0 = time.time(); log[s["name"]] = await scene(s); print(s["name"], "done in", int(time.time() - t0), "s:", (log[s["name"]].get("text") or "")[:160].replace("\n", " | "), flush=True)
    finally:
        print(await write_prefs(orig), flush=True)
    json.dump(log, open(os.path.join(OUT, "capture_log.json"), "w"), ensure_ascii=False, indent=1)
asyncio.run(main())
