"""Tiny Chrome-DevTools-Protocol client for a USB-connected Android phone (adb forward tcp:9222 localabstract:chrome_devtools_remote).

Only ever touches a tab IT creates; never reads or changes the user's other tabs.
Usage (python3):  from phone_cdp import Phone;  p = Phone();  t = p.new_tab("about:blank");  print(t.eval("navigator.userAgent"));  t.close()
"""
import asyncio, json, urllib.request, websockets

BASE = "http://localhost:9222"

class Tab:
    def __init__(self, phone, target_id):
        self.phone, self.id = phone, target_id
        self.ws_url = f"ws://localhost:9222/devtools/page/{target_id}"
    def eval(self, js, await_promise=True, timeout=60):
        async def run():
            async with websockets.connect(self.ws_url, max_size=50_000_000) as ws:
                await ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": js, "awaitPromise": await_promise, "returnByValue": True, "timeout": int(timeout * 1000)}}))
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout + 5))
                    if msg.get("id") == 1:
                        r = msg.get("result", {})
                        if "exceptionDetails" in r: return {"error": r["exceptionDetails"].get("exception", {}).get("description", str(r["exceptionDetails"]))[:600]}
                        return r.get("result", {}).get("value")
        return asyncio.run(run())
    def close(self):
        try: urllib.request.urlopen(f"{BASE}/json/close/{self.id}", timeout=5).read()
        except Exception: pass

class Phone:
    def new_tab(self, url="about:blank"):
        async def run():
            ver = json.loads(urllib.request.urlopen(f"{BASE}/json/version", timeout=10).read())
            async with websockets.connect(ver["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
                await ws.send(json.dumps({"id": 1, "method": "Target.createTarget", "params": {"url": url}}))
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 20))
                    if msg.get("id") == 1: return msg["result"]["targetId"]
        return Tab(self, asyncio.run(run()))
