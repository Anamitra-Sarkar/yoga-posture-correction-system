"""Print a file from a (private) HF repo to stdout without saving it: python3 hfcat.py <repo> <path> [datasets]"""
import sys, requests
tok = open("/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token").read().strip()
repo, path = sys.argv[1], sys.argv[2]; kind = sys.argv[3] if len(sys.argv) > 3 else "models"
r = requests.get(f"https://huggingface.co/{'' if kind == 'models' else kind + '/'}{repo}/resolve/main/{path}", headers={"Authorization": f"Bearer {tok}"}, timeout=60)
print(r.status_code, file=sys.stderr); sys.stdout.write(r.text if r.ok else "")
