"""Print the log of a kernel of the arkosarkarhehe account via the REST `log` field (never `kaggle kernels output`, which downloads the output files).
Usage: python3 klog_arko.py <kernel-slug> [max line width]"""
import json, base64, urllib.request, sys
slug = sys.argv[1]; k = json.load(open('/home/anamitra/kaggle.json'))
tok = base64.b64encode(f"{k['username']}:{k['key']}".encode()).decode()
req = urllib.request.Request(f"https://www.kaggle.com/api/v1/kernels/output?userName=arkosarkarhehe&kernelSlug={slug}", headers={"Authorization": "Basic " + tok})
j = json.load(urllib.request.urlopen(req, timeout=90))
for e in json.loads(j.get("log") or "[]"):
    for l in e.get("data", "").splitlines():
        if l.strip() and not l.startswith(("Fetching", "/usr", "  warn", "[NbConvert", "  cells", "  text")): print(l[:int(sys.argv[2]) if len(sys.argv) > 2 else 1500])
