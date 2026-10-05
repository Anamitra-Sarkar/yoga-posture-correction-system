"""Guided live pose test on the USB-connected phone: shows big on-screen prompts, the person performs each pose in front of the front camera,
and the app's REAL predictions (the server's pose_id for every /analyse_frame call, plus the pose shown in the UI) are logged per pose.
The test tab is the app itself (production) with speech silenced for the test only. Usage: python3 live_pose_test.py [url] [get_ready_s]"""
import asyncio, json, sys, time, urllib.request, websockets
URL = sys.argv[1] if len(sys.argv) > 1 else "https://yoga-posture-correction-system.vercel.app/"
READY = int(sys.argv[2]) if len(sys.argv) > 2 else 40
POSES = [  # (expected pose_id, title, how to do it)
    ("mountain_pose", "MOUNTAIN", "Stand tall, feet together, arms down by your sides"),
    ("tree_pose", "TREE", "Stand on one leg, other foot on the inner thigh, hands together overhead"),
    ("warrior_2", "WARRIOR II", "Wide stance, front knee bent, arms straight out to the sides"),
    ("chair_pose", "CHAIR", "Bend your knees as if sitting back, arms up"),
    ("triangle", "TRIANGLE", "Wide legs, bend sideways, one hand to your shin, other hand up"),
    ("seated_easy_pose", "SEATED (cross-legged)", "Sit cross-legged on the floor, back straight"),
    ("child_pose", "CHILD'S POSE", "Kneel, sit back on your heels, fold forward, arms forward"),
    ("downward_dog", "DOWNWARD DOG", "Hands and feet on the floor, hips high (an upside-down V)"),
]
SETTLE, HOLD = 8, 16
HOOK = r"""(() => {
  window.__srv = []; window.__t0 = performance.now();
  const of = window.fetch;
  window.fetch = function (i, init) { const u = typeof i === 'string' ? i : i.url; const p = of.apply(this, arguments);
    if (/analyse_frame/.test(u)) p.then(r => r.clone().json()).then(d => window.__srv.push({ t: Math.round(performance.now() - window.__t0), pose: d.pose_id, motion: d.motion_state, score: d.correctness_score, cands: (d.candidates || []).slice(0, 3).map(c => c.pose_id + ':' + Math.round(c.probability * 100)) })).catch(() => {});
    return p; };
  window.__str = 0; const st = CanvasRenderingContext2D.prototype.stroke;
  CanvasRenderingContext2D.prototype.stroke = function (...a) { if (this.canvas && this.canvas.className === 'ap-canvas') window.__str++; return st.apply(this, a); };
  SpeechSynthesis.prototype.speak = function () {};          // silence the coach for the test only
  navigator.wakeLock && (navigator.wakeLock.request = () => Promise.reject(new Error('n/a')));
})();"""
OVERLAY = r"""(() => { let d = document.getElementById('__prompt'); if (!d) { d = document.createElement('div'); d.id = '__prompt'; d.style.cssText = 'transition:background .3s;position:fixed;left:0;right:0;top:0;z-index:2147483647;background:rgba(0,0,0,.82);color:#fff;font:700 40px/1.15 sans-serif;padding:14px 16px;text-align:center'; document.body.appendChild(d); }
  d.style.background = '__BG__'; d.innerHTML = '__HTML__'; })()"""
SEEN = {}
async def main():
    ver = json.loads(urllib.request.urlopen("http://localhost:9222/json/version", timeout=10).read())
    async with websockets.connect(ver["webSocketDebuggerUrl"], max_size=50_000_000) as bws:
        await bws.send(json.dumps({"id": 1, "method": "Target.createTarget", "params": {"url": "about:blank"}})); tid = json.loads(await bws.recv())["result"]["targetId"]
        origin = "/".join(URL.split("/")[:3])
        await bws.send(json.dumps({"id": 2, "method": "Browser.grantPermissions", "params": {"origin": origin, "permissions": ["videoCapture"]}})); await bws.recv()
        async with websockets.connect(f"ws://localhost:9222/devtools/page/{tid}", max_size=50_000_000) as ws:
            n = [10]
            async def send(m, p=None):
                n[0] += 1; i = n[0]; await ws.send(json.dumps({"id": i, "method": m, "params": p or {}}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == i: return msg.get("result", {})
            async def ev(js):
                r = await send("Runtime.evaluate", {"expression": js, "awaitPromise": True, "returnByValue": True}); return r.get("result", {}).get("value")
            async def say(html, bg='rgba(0,0,0,.82)'): await ev(OVERLAY.replace("__HTML__", html.replace("'", "\\'")).replace("__BG__", bg))
            await send("Page.enable"); await send("Runtime.enable")
            await send("Page.addScriptToEvaluateOnNewDocument", {"source": HOOK})
            await send("Page.navigate", {"url": URL}); await asyncio.sleep(8)
            await ev("(() => { const b = [...document.querySelectorAll('button')].find(b => /start|शुरू|চালু/i.test(b.textContent) && b.offsetParent); b && b.click(); })()")
            # wait until the app actually sees a body (it only sends frames for analysis when it does), then count down
            seen = 0; waited = 0
            while seen < 3 and waited < 300:
                count = int(await ev("window.__srv.length"))
                seen = seen + 1 if count > 0 and int(await ev("(window.__srv.filter(r => performance.now() - window.__t0 - r.t < 4000)).length")) > 0 else 0
                await say("WAITING TO SEE YOU<br><span style='font-size:26px;font-weight:400'>Prop the phone upright ~2 m away, front camera toward you, and step back until your whole body is in view</span>")
                await asyncio.sleep(2); waited += 2
            if seen < 3: print("never saw a person within 5 minutes"); return
            for s in range(READY, 0, -1):
                await say(f"I SEE YOU - GET READY<br><span style='font-size:26px;font-weight:400'>First pose in</span><br>{s}")
                await asyncio.sleep(1)
            results = []; seen_stats = {}
            last_str = int(await ev("window.__str"))
            async def tick():
                nonlocal last_str
                cur = int(await ev("window.__str")); seen = cur - last_str > 5; last_str = cur
                hud = await ev("[...document.querySelectorAll('.ap-hud span')].map(x => x.textContent.trim()).filter(Boolean).join(' | ')")
                return seen, hud or ""
            for pid, title, how in POSES:
                t_hold = None; seen_s = 0; total_s = 0; huds = {}
                for s_ in range(SETTLE):
                    seen, hud = await tick()
                    await say(f"{title}<br><span style='font-size:24px;font-weight:400'>{how}</span><br><span style='font-size:28px'>GET INTO POSITION · {SETTLE - s_}</span><br><span style='font-size:26px'>{'✓ I SEE YOU' if seen else '✗ NOT SEEING YOU — whole body in view, please'}</span>", "rgba(20,110,60,.9)" if seen else "rgba(150,30,30,.9)")
                    await asyncio.sleep(1)
                t_hold = int(await ev("Math.round(performance.now() - window.__t0)"))
                while seen_s < HOLD and total_s < 40:
                    seen, hud = await tick(); total_s += 1; seen_s += 1 if seen else 0
                    for k in ("Step into view", "Step back so your feet", "Looking for a pose", "In Transition"):
                        if k in hud: huds[k] = huds.get(k, 0) + 1
                    await say(f"{title}<br><span style='font-size:24px;font-weight:400'>{how}</span><br><span style='font-size:28px'>HOLD STILL · {max(0, HOLD - seen_s)} s</span><br><span style='font-size:26px'>{'✓ I SEE YOU' if seen else '✗ NOT SEEING YOU — whole body in view, please'}</span>", "rgba(20,110,60,.9)" if seen else "rgba(150,30,30,.9)")
                    await asyncio.sleep(1)
                t_end = int(await ev("Math.round(performance.now() - window.__t0)"))
                results.append((pid, title, t_hold, t_end)); seen_stats[pid] = (seen_s, total_s, huds)
            SEEN.update(seen_stats)
            await say("DONE — thank you!")
            srv = json.loads(await ev("JSON.stringify(window.__srv)"))
            await ev("[...document.querySelectorAll('button')].find(b => /^\\s*stop/i.test(b.textContent))?.click()")
        await bws.send(json.dumps({"id": 3, "method": "Target.closeTarget", "params": {"targetId": tid}})); await bws.recv()
    seen_stats = SEEN
    print("\n=== LIVE POSE TEST (app predictions during the HOLD phase) ===")
    passed = 0
    for pid, title, a, b in results:
        rows = [r for r in srv if a <= r["t"] <= b]
        n_ = len(rows)
        if not n_: print(f"{title:22s} expected {pid:18s} NO PREDICTIONS [seen {seen_stats[pid][0]}/{seen_stats[pid][1]} s; framing: {seen_stats[pid][2]}]"); continue
        hit = sum(1 for r in rows if r["pose"] == pid); top3 = sum(1 for r in rows if any(c.startswith(pid + ":") for c in r["cands"]))
        dist = {}
        for r in rows: dist[r["pose"]] = dist.get(r["pose"], 0) + 1
        top = sorted(dist.items(), key=lambda kv: -kv[1])[:3]
        ok = hit / n_ >= 0.5; passed += ok; st_ = seen_stats.get(pid)
        print(f"{title:22s} expected {pid:18s} correct {hit}/{n_} ({round(100*hit/n_)}%)  in top-3 {top3}/{n_}  predicted: {top}  -> {'PASS' if ok else 'miss'}  [seen {st_[0]}/{st_[1]} s; framing: {st_[2]}]")
    print(f"\nposes recognised (>= 50% of predictions during the hold): {passed} of {len(results)}")
    json.dump({"results": results, "server": srv}, open("live_pose_test_last.json", "w"))
asyncio.run(main())
