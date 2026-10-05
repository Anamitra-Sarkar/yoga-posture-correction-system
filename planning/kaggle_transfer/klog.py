import json, base64, urllib.request, os, sys
slug = sys.argv[1]; tail = int(sys.argv[2]) if len(sys.argv) > 2 else 25
k = json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))
tok = base64.b64encode(f"{k['username']}:{k['key']}".encode()).decode()
req = urllib.request.Request(f"https://www.kaggle.com/api/v1/kernels/output?userName=anamitrasarkar007&kernelSlug={slug}", headers={"Authorization": "Basic " + tok})
j = json.load(urllib.request.urlopen(req, timeout=90))
lines = []
try:
    for e in json.loads(j.get("log") or "[]"):
        d = e.get("data", "").rstrip()
        if d and not d.startswith(("/usr/local", "  cells", "  text", "[NbConvert", "   ━")): lines += d.splitlines()
except Exception as ex: lines = [f"log parse: {ex}"]
print("\n".join(l[:220] for l in lines[-tail:]))
