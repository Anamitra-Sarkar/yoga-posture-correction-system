"""Pose-prediction test on the phone with PHOTOS instead of a person: the app tab on the phone gets a stubbed camera that shows Wikimedia Commons
yoga photos (looked up by the tab itself), and the app's REAL predictions (the server's pose_id for each /analyse_frame call) are logged per photo.
Usage: python3 photo_pose_test.py [url] [photos_per_pose]    Results: photo_pose_test_last.json + a printed table."""
import asyncio, json, sys, time, urllib.request, websockets
URL = sys.argv[1] if len(sys.argv) > 1 else "https://yoga-posture-correction-system.vercel.app/"
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SPEC = {  # expected pose_id: (Commons search queries, title must match, title must not match)
  "mountain_pose": (["Tadasana", "mountain pose yoga"], r"tadasana|mountain-pose-1|mountain-pose-2|mountain pose", r"prayer|bound|reverse"),
  "tree_pose": (["Vrikshasana", "tree pose yoga"], r"vriksh|vrksa|tree pose", r""),
  "warrior_2": (["Virabhadrasana II", "warrior 2 yoga pose"], r"virabhadrasana.?(ii|2)\b|warrior.?(ii|2)? ?(pose|yoga)", r"rainbow|greenpeace|ship"),
  "chair_pose": (["Utkatasana", "chair pose yoga utkatasana"], r"utkat", r""),
  "triangle": (["Trikonasana", "triangle pose yoga trikonasana"], r"trikon", r"parivrtta|revolved"),
  "downward_dog": (["Adho Mukha Svanasana", "downward facing dog yoga"], r"adho.?mukha|downward|shvanasana|svanasana", r"surya|namaskar|bondi"),
  "seated_easy_pose": (["Sukhasana", "easy pose yoga cross-legged sitting"], r"sukhasana|sukkasana|cross-legged sitting", r""),
  "child_pose": (["Balasana", "child pose yoga balasana"], r"balasana|child pose|childs pose|child's pose", r"mudra|symbol"),
  "corpse": (["Savasana", "corpse pose yoga shavasana"], r"savasana|shavasana|corpse", r""),
  "plank": (["Phalakasana", "plank pose yoga"], r"phalak|plank", r""),
  "cobra_pose": (["Bhujangasana", "cobra pose yoga"], r"bhujang|cobra", r"snake|king|ah-1|hmla|helicopter|super cobra|marine|aircraft"),
}
LOCAL = {"mountain_pose": "/pose-images/mountain_pose.jpg", "tree_pose": "/pose-images/tree_pose.jpg", "warrior_2": "/pose-images/warrior_2.jpg", "plank": "/pose-images/plank.jpg", "cobra_pose": "/pose-images/cobra_pose.jpg"}
HOOK = r"""(() => {
  window.__srv = []; window.__t0 = performance.now(); window.__img = null;
  const of = window.fetch;
  window.fetch = function (i, init) { const u = typeof i === 'string' ? i : i.url; const p = of.apply(this, arguments);
    if (/analyse_frame/.test(u)) p.then(r => r.clone().json()).then(d => window.__srv.push({ t: Math.round(performance.now() - window.__t0), pose: d.pose_id, motion: d.motion_state, score: d.correctness_score, cands: (d.candidates || []).slice(0, 3).map(c => c.pose_id + ':' + Math.round(c.probability * 100)), gated: d.cascade && d.cascade.gated })).catch(() => {});
    return p; };
  SpeechSynthesis.prototype.speak = function () {};
  navigator.mediaDevices.enumerateDevices = async () => [{ kind: 'videoinput', deviceId: 'c1', label: 'Cam' }];
  navigator.mediaDevices.getUserMedia = async () => {
    const cv = document.createElement('canvas'); cv.width = 1080; cv.height = 1080; const g = cv.getContext('2d');
    const draw = () => { g.fillStyle = '#222'; g.fillRect(0, 0, 1080, 1080); const im = window.__img; if (im) { const s = Math.min(1080 / im.naturalWidth, 1080 / im.naturalHeight); g.drawImage(im, (1080 - im.naturalWidth * s) / 2, (1080 - im.naturalHeight * s) / 2, im.naturalWidth * s, im.naturalHeight * s); } };
    draw(); setInterval(draw, 66); return cv.captureStream(15);
  };
})();"""
FIND = r"""(async (spec, per) => {
  const sleep = (ms) => new Promise(r => setTimeout(r, ms)); const sets = {};
  for (const [pose, [qs, inc, exc]] of Object.entries(spec)) {
    const seen = new Map();
    for (const q of qs) {
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const url = 'https://commons.wikimedia.org/w/api.php?action=query&format=json&origin=*&generator=search&gsrnamespace=6&gsrlimit=40&gsrsearch=' + encodeURIComponent(q + ' filemime:image/jpeg') + '&prop=imageinfo&iiprop=url|mime|size&iiurlwidth=640';
          const r = await fetch(url); if (r.status === 429) { await sleep(2500); continue; }
          const j = await r.json();
          for (const p of Object.values((j.query && j.query.pages) || {})) {
            const i = p.imageinfo && p.imageinfo[0]; const t = p.title.replace(/^File:/, '');
            if (!i || i.mime !== 'image/jpeg' || i.width < 500 || i.height < 450 || !new RegExp(inc, 'i').test(t) || (exc && new RegExp(exc, 'i').test(t))) continue;
            if (!seen.has(i.thumburl)) seen.set(i.thumburl, t.slice(0, 60));
          }
          break;
        } catch (e) { await sleep(1500); }
      }
      await sleep(900);
    }
    sets[pose] = [...seen.entries()].slice(0, per).map(([url, title]) => ({ url, title }));
  }
  window.__sets = sets; return Object.fromEntries(Object.entries(sets).map(([k, v]) => [k, v.length]));
})"""
RUN = r"""(async (local) => {
  const sleep = (ms) => new Promise(r => setTimeout(r, ms)); window.__rows = []; window.__done = false;
  const load = (u) => new Promise((ok) => { const im = new Image(); im.crossOrigin = 'anonymous'; im.onload = () => ok(im); im.onerror = () => ok(null); im.src = u; });
  for (const [pose, items] of Object.entries(window.__sets)) {
    const all = (local[pose] ? [{ url: local[pose], title: 'app reference photo' }] : []).concat(items);
    for (const it of all) {
      const im = await load(it.url); if (!im) { window.__rows.push({ pose, title: it.title, url: it.url, error: 'load' }); continue; }
      window.__img = im; const n0 = window.__srv.length; const t0 = performance.now();
      while (performance.now() - t0 < 11000 && window.__srv.length < n0 + 5) await sleep(250);
      const got = window.__srv.slice(n0).slice(2);        // drop the first two answers: they may still belong to the previous photo
      window.__rows.push({ pose, title: it.title, url: it.url, preds: got.map(g => g.pose), cands: got.map(g => g.cands[0] || ''), n: got.length });
    }
  }
  window.__done = true;
})"""
async def main():
    ver = json.loads(urllib.request.urlopen("http://localhost:9222/json/version", timeout=10).read())
    async with websockets.connect(ver["webSocketDebuggerUrl"], max_size=50_000_000) as bws:
        await bws.send(json.dumps({"id": 1, "method": "Target.createTarget", "params": {"url": "about:blank"}})); tid = json.loads(await bws.recv())["result"]["targetId"]
        async with websockets.connect(f"ws://localhost:9222/devtools/page/{tid}", max_size=50_000_000) as ws:
            n = [10]
            async def send(m, p=None):
                n[0] += 1; i = n[0]; await ws.send(json.dumps({"id": i, "method": m, "params": p or {}}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == i: return msg.get("result", {})
            async def ev(js, wait=True):
                r = await send("Runtime.evaluate", {"expression": js, "awaitPromise": wait, "returnByValue": True}); return r.get("result", {}).get("value", r.get("exceptionDetails", {}).get("text"))
            await send("Page.enable"); await send("Runtime.enable")
            await send("Page.addScriptToEvaluateOnNewDocument", {"source": HOOK})
            await send("Page.navigate", {"url": URL}); await asyncio.sleep(8)
            spec = {k: [v[0], v[1], v[2]] for k, v in SPEC.items()}
            print("photo sets found:", await ev(f"({FIND})({json.dumps(spec)}, {PER})"), flush=True)
            await ev("(() => { const b = [...document.querySelectorAll('button')].find(b => /start|शुरू|চালু/i.test(b.textContent) && b.offsetParent); b && b.click(); })()", wait=False)
            await asyncio.sleep(6)
            await send("Runtime.evaluate", {"expression": f"({RUN})({json.dumps(LOCAL)})", "awaitPromise": False})
            t0 = time.time()
            while time.time() - t0 < 1500:
                await asyncio.sleep(15)
                prog = await ev("JSON.stringify({done: window.__done, rows: window.__rows.length})")
                if json.loads(prog)["done"]: break
            rows = json.loads(await ev("JSON.stringify(window.__rows)"))
            await ev("[...document.querySelectorAll('button')].find(b => /^\\s*stop/i.test(b.textContent))?.click()")
        await bws.send(json.dumps({"id": 3, "method": "Target.closeTarget", "params": {"targetId": tid}})); await bws.recv()
    json.dump(rows, open("photo_pose_test_last.json", "w"), indent=1)
    by = {}
    for r in rows: by.setdefault(r["pose"], []).append(r)
    print("\n=== PHOTO POSE TEST: the app's real predictions on the phone ===")
    ok_poses = 0
    for pose, rs in by.items():
        good = 0; used = 0; lines = []
        for r in rs:
            if r.get("error") or not r.get("preds"): lines.append(f"    ? {r['title'][:46]:46s} no prediction"); continue
            used += 1; mode = max(set(r["preds"]), key=r["preds"].count); hit = mode == pose; good += hit
            lines.append(f"    {'OK ' if hit else 'XX '} {r['title'][:46]:46s} -> {mode}  {r['preds']}")
        verdict = "PASS" if used and good / used >= 0.5 else "miss"; ok_poses += verdict == "PASS"
        print(f"{pose:18s} {good}/{used} photos correct -> {verdict}"); print("\n".join(lines))
    print(f"\nposes recognised (>= 50% of their photos): {ok_poses} of {len(by)}")
asyncio.run(main())
