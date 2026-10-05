import subprocess, sys
print("python", sys.version.split()[0], flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe", "opencv-python-headless"], capture_output=True, text=True)
print("pip rc", r.returncode, (r.stderr or r.stdout)[-400:].replace("\n", " | "), flush=True)
try:
    import mediapipe as mp
    print("mediapipe", mp.__version__, "| legacy solutions.pose:", hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"), "| tasks:", hasattr(mp, "tasks"), flush=True)
except Exception as e:
    print("IMPORT FAILED", type(e).__name__, e, flush=True)
print("ffmpeg:", subprocess.run("ffmpeg -version | head -1", shell=True, capture_output=True, text=True).stdout.strip(), flush=True)
