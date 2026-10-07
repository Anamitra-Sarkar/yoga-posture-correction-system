"""What can the cue-verified labels say about FLOW (transitions between held poses)?  Token-free (public HF dataset), CPU, prints statistics only.
Frame labels (vol/csv/cue/cue_full.csv, column imperfect_pose_label): a pose name = cue-verified hold frame; 'transition/unknown' = moving / unknown; '__ignore__' = ambiguous.
We find HOLD segments (runs of one pose), the GAPS between consecutive different holds A -> B, and how many directional pairs have enough support across videos."""
import collections, glob, json, os
import numpy as np, pandas as pd
from huggingface_hub import snapshot_download
ROOT = "/kaggle/working"; DS = "Arko007/Yoga-1M"
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, allow_patterns=["vol/csv/cue/cue_full.csv"])
df = pd.read_csv(f"{ROOT}/vol/csv/cue/cue_full.csv", usecols=["video_id", "frame_num", "imperfect_pose_label"])
print("rows", len(df), "videos", df.video_id.nunique(), flush=True)
lab = df.imperfect_pose_label.astype(str)
print("label counts (top 40):", dict(lab.value_counts().head(40)), flush=True)
U, IGN = "transition/unknown", "__ignore__"
MIN_HOLD = 25          # frames (1 s at 25 fps) for a run to count as a hold
segs = collections.defaultdict(list)   # video -> [(pose, start, end)]
for vid, g in df.sort_values(["video_id", "frame_num"]).groupby("video_id"):
    L = g.imperfect_pose_label.astype(str).values; start = 0
    for i in range(1, len(L) + 1):
        if i == len(L) or L[i] != L[start]:
            if L[start] not in (U, IGN) and i - start >= MIN_HOLD: segs[vid].append((L[start], start, i - 1))
            start = i
n_hold = sum(len(v) for v in segs.values())
print(f"hold segments (>= {MIN_HOLD} frames): {n_hold} in {len(segs)} videos", flush=True)
hold_len = collections.defaultdict(list)
for v in segs.values():
    for p, a, b in v: hold_len[p].append(b - a + 1)
print("holds per pose (count, total frames, videos):", {p: (len(l), sum(l), sum(1 for v in segs.values() if any(x[0] == p for x in v))) for p, l in sorted(hold_len.items(), key=lambda kv: -sum(kv[1]))}, flush=True)
gaps = []   # (video, A, B, gap_frames)
for vid, v in segs.items():
    for (pa, a0, a1), (pb, b0, b1) in zip(v, v[1:]): gaps.append((vid, pa, pb, b0 - a1 - 1))
gl = np.array([g[3] for g in gaps]); diff = np.array([g[1] != g[2] for g in gaps])
print("consecutive hold pairs:", len(gaps), "different poses:", int(diff.sum()), flush=True)
for lo, hi in [(0, 25), (25, 50), (50, 100), (100, 150), (150, 250), (250, 500), (500, 1500), (1500, 10**9)]:
    m = (gl >= lo) & (gl < hi); print(f"  gap {lo:>4}-{hi:<10} frames: all {int(m.sum()):>5} | A!=B {int((m & diff).sum()):>5}", flush=True)
for G in (100, 150, 250, 500):
    pair_vids = collections.defaultdict(set); pair_n = collections.Counter()
    for vid, pa, pb, gap in gaps:
        if pa != pb and gap <= G: pair_vids[(pa, pb)].add(vid); pair_n[(pa, pb)] += 1
    multi = {k: (pair_n[k], len(v)) for k, v in pair_vids.items() if len(v) >= 3}
    print(f"\n[gap <= {G} frames] directional pairs: {len(pair_n)} | in >= 3 videos: {len(multi)} | transitions covered by those: {sum(n for n, _ in multi.values())} of {sum(pair_n.values())}", flush=True)
    print("   top pairs (count, videos):", {f'{a}->{b}': pair_n[(a, b)] and (pair_n[(a, b)], len(pair_vids[(a, b)])) for (a, b), _ in pair_n.most_common(25)}, flush=True)
    dest = collections.Counter(); orig = collections.Counter()
    for (pa, pb), n in pair_n.items(): dest[pb] += n; orig[pa] += n
    print("   destination poses:", dict(dest.most_common(20)), flush=True); print("   origin poses:", dict(orig.most_common(20)), flush=True)
    # reversible pairs: both A->B and B->A present
    rev = [k for k in pair_n if (k[1], k[0]) in pair_n]; print("   pairs with both directions present:", len(rev), flush=True)
print("DONE", flush=True)
