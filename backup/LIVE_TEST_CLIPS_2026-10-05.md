# Live-test clips (play these in front of the camera) -- 2026-10-05

Each link jumps to a steady, cue-verified hold of the named pose (the instructor named it AND the body shape matched AND it was held still).
Play it full-screen on a second screen or phone, ~1 m from the laptop camera, bright screen, so the whole body is in frame, and let it run **at least 20 s**:
the app only classifies once every ~10 s, so a 5 s clip can be missed entirely.

**What this is and is not:** a sanity / demo test of the live pipeline (camera -> MediaPipe -> backend -> UI, both modes, all three languages).
These videos were used to TRAIN the models, so do not quote results from them as accuracy; use unseen people for the benchmark (BENCHMARKS.md section 8).

| Pose | Clip (video, start, length) |
|---|---|
| Downward dog | https://youtu.be/SZU7Sbgu57o?t=1270 (21:10, 28 s) - https://youtu.be/HmZFwoUU3WQ?t=844 (14:04, 22 s) - https://youtu.be/8ibxmzJziHU?t=1354 (22:34, 18 s) |
| Warrior 2 | https://youtu.be/L-z1HLkS_-Y?t=1141 (19:01, 12 s; short - start 5 s early) - https://youtu.be/HmZFwoUU3WQ?t=1184 (19:44, 5 s; very short) |
| Tree | https://youtu.be/JHjV-wFTwSw?t=1304 (21:44, 19 s) |
| Triangle | https://youtu.be/QiebZSlTw_U?t=1026 (17:06, 24 s) - https://youtu.be/L-z1HLkS_-Y?t=1400 (23:20, 22 s) - https://youtu.be/7ciS93shMNQ?t=1501 (25:01, 16 s) |
| Seated easy pose | https://youtu.be/s-1vMbAgYWU?t=900 (15:00, 22 s) - https://youtu.be/4K2xTVRDJgA?t=566 (9:26, 17 s) - https://youtu.be/EvMTrP8eRvM?t=44 (0:44, 17 s) |
| Child pose | https://youtu.be/O2EY79Ys_qg?t=545 (9:05, 49 s) - https://youtu.be/8ibxmzJziHU?t=2557 (42:37, 48 s) - https://youtu.be/s-1vMbAgYWU?t=13 (0:13, 47 s) |
| Corpse | https://youtu.be/SZU7Sbgu57o?t=1904 (31:44, 58 s) - https://youtu.be/149Iac5fmoE?t=769 (12:49, 25 s) - https://youtu.be/RQMtwbhXD7A?t=3120 (52:00, 24 s) |
| Mountain | https://youtu.be/8ibxmzJziHU?t=896 (14:56, 28 s) - https://youtu.be/v7AYKMP6rOE?t=925 (15:25, 20 s) - https://youtu.be/149Iac5fmoE?t=433 (7:13, 14 s) |
| Lunge | https://youtu.be/QiebZSlTw_U?t=653 (10:53, 20 s) - https://youtu.be/4ORRiN2_aVI?t=715 (11:55, 16 s) - https://youtu.be/JHjV-wFTwSw?t=1829 (30:29, 13 s) |
| Standing forward fold | https://youtu.be/QiebZSlTw_U?t=678 (11:18, 31 s) - https://youtu.be/8ibxmzJziHU?t=1979 (32:59, 28 s) - https://youtu.be/4ORRiN2_aVI?t=774 (12:54, 14 s) |
| Plank | only one 5 s hold exists: https://youtu.be/RQMtwbhXD7A?t=1710 -- do this one yourself, or use a plank video you trust |
| Cobra | NO hold of 5 s or more exists in the training videos (cobra is always passed through quickly) -- do this one yourself; it is also one of the weakest poses on real photos |

**Moving, no pose held** (the app should stay quiet or say "transitioning", and must not name a pose with confidence):
https://youtu.be/Eml2xnoLpYE?t=815 (13:35, 123 s) - https://youtu.be/L-z1HLkS_-Y?t=23 (0:23, 114 s) - https://youtu.be/QiebZSlTw_U?t=2139 (35:39, 110 s) - https://youtu.be/8ibxmzJziHU?t=2639 (43:59, 108 s)

**Also try, to test the "not one of my poses" gate:** a pose that is NOT in the app (e.g. a crow, pigeon or wheel video), walking around, sitting at a desk, stretching an arm.
Expected: unrecognised / quiet, not a confident named pose (false alarms were ~21% on held-out photos, so an occasional wrong name is normal).

**Checklist per clip:** Free mode (detected pose + score) - Guided mode (choose the matching pose: should score; choose a different one: should say "Wrong Pose") -
switch the UI to Hindi and Bengali and check the spoken guidance (needs a device with a Hindi/Bengali voice; otherwise the new notice appears).
