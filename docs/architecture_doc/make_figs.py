"""Figures for the architecture document, drawn from the repo's own code (backend/app/models/sequence.py)."""
import sys
sys.path.insert(0, "../../backend")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.size": 9})
from app.models.sequence import MEDIAPIPE_EDGES, get_normalized_adjacency

# Schematic standing pose, subject's left on the right of the picture (as MediaPipe reports it).
P = {0:(0,9.6),1:(.35,9.9),2:(.6,9.9),3:(.85,9.9),4:(-.35,9.9),5:(-.6,9.9),6:(-.85,9.9),7:(1.2,9.6),8:(-1.2,9.6),
     9:(.3,9.1),10:(-.3,9.1),11:(1.8,8.0),12:(-1.8,8.0),13:(2.5,6.2),14:(-2.5,6.2),15:(2.8,4.5),16:(-2.8,4.5),
     17:(3.15,4.0),18:(-3.15,4.0),19:(3.0,3.7),20:(-3.0,3.7),21:(2.55,4.0),22:(-2.55,4.0),
     23:(1.0,4.5),24:(-1.0,4.5),25:(1.15,2.3),26:(-1.15,2.3),27:(1.25,0.2),28:(-1.25,0.2),
     29:(1.4,-0.15),30:(-1.4,-0.15),31:(1.75,-0.45),32:(-1.75,-0.45)}
def group(i, j):
    if max(i, j) <= 10: return "head"
    if min(i, j) >= 23: return "legs"
    if {i, j} <= {11, 12, 23, 24}: return "torso"
    if min(i, j) in (11, 12, 13, 14) or (11 <= max(i, j) <= 22 and min(i, j) >= 11): return "arms"
    return "torso"
col = {"head": "#8a8a8a", "arms": "#2b7bba", "torso": "#2f8f5b", "legs": "#c0562b"}
fig, ax = plt.subplots(figsize=(3.6, 4.6))
for i, j in MEDIAPIPE_EDGES:
    g = group(i, j)
    if i in (23, 25, 27, 29, 31) and j in (24, 26, 28, 30, 32): g = "legs"
    ax.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], color=col[g], lw=1.6, zorder=1)
for k, (x, y) in P.items():
    ax.scatter([x], [y], s=26, color="white", edgecolor="black", lw=0.8, zorder=2)
for k in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
    x, y = P[k]; dx = 0.28 if x >= 0 else -0.28
    ax.text(x + dx, y + 0.12, str(k), fontsize=7, ha="left" if x >= 0 else "right", va="bottom")
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-4.3, 4.3); ax.set_ylim(-1.0, 10.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], color=col[k], lw=2, label=k) for k in ("head", "arms", "torso", "legs")], loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06), handlelength=1.2, columnspacing=1.0)
fig.savefig("fig_skeleton_graph.pdf", bbox_inches="tight"); plt.close(fig)

A = get_normalized_adjacency().numpy()
fig, ax = plt.subplots(figsize=(3.9, 3.5))
im = ax.imshow(A, cmap="Blues", vmin=0, vmax=A.max())
ax.set_xticks([0, 11, 23, 32]); ax.set_yticks([0, 11, 23, 32])
ax.set_xlabel("joint $w$ (neighbour)"); ax.set_ylabel("joint $v$")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(r"$A_{\mathrm{norm}}[v,w]$")
for b in (10.5, 22.5): ax.axhline(b, color="black", lw=0.4, ls=":"); ax.axvline(b, color="black", lw=0.4, ls=":")
fig.savefig("fig_adjacency.pdf", bbox_inches="tight"); plt.close(fig)
print("edges", len(MEDIAPIPE_EDGES), "nonzeros", int((A > 0).sum()), "diag range", float(A.diagonal().min()), float(A.diagonal().max()))
