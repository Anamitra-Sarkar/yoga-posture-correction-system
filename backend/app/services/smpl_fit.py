import numpy as np
import torch
from huggingface_hub import hf_hub_download
from app.config import settings

try:
    import spaces
    HAS_ZEROGPU = True
except Exception:
    HAS_ZEROGPU = False

    class _NoOpSpaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            # Outside a ZeroGPU Space (local dev, plain Docker Space), `spaces.GPU`
            # doesn't exist. This no-op stand-in lets the same code run everywhere,
            # just without the dynamic GPU allocation.
            if fn is not None:
                return fn
            def decorator(f):
                return f
            return decorator
    spaces = _NoOpSpaces()

SMPL_REPO = "Arko007/smpl-models"
SMPL_NEUTRAL_PATH_IN_REPO = "SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"

# MediaPipe landmark index -> standard SMPL 24-joint kinematic-tree index, for
# the joints present in both skeletons (SMPL has no eyes/ears/fingers/heel/
# foot-index equivalents, so only the shared major joints are mapped).
MP_TO_SMPL = {
    23: 1, 24: 2,    # hip l/r
    25: 4, 26: 5,    # knee l/r
    27: 7, 28: 8,    # ankle l/r
    11: 16, 12: 17,  # shoulder l/r
    13: 18, 14: 19,  # elbow l/r
    15: 20, 16: 21,  # wrist l/r
}

# The subset of the above that occlusion.py's symmetric mirror solver also
# handles (13,14,15,16,25,26,27,28) -- these are the joints this fit can
# actually improve on vs. a plain mirror-reflection heuristic.
OCCLUDABLE_JOINTS = {k: v for k, v in MP_TO_SMPL.items() if k in (13, 14, 15, 16, 25, 26, 27, 28)}

MIN_VISIBLE_ANCHORS = 6  # below this, the fit is underdetermined; let the caller fall back

_smpl_model = None


def _load_smpl_model():
    global _smpl_model
    if _smpl_model is not None:
        return _smpl_model

    # Unpickling the official SMPL .pkl needs `chumpy` (it wraps some arrays
    # as chumpy objects), but chumpy itself imports deprecated numpy aliases
    # (np.bool, np.float, ...) removed in numpy>=1.24. Patch them in first
    # rather than pinning an old numpy just for this one legacy dependency.
    for name, alias in [("bool", bool), ("int", int), ("float", float),
                         ("complex", complex), ("object", object),
                         ("str", str), ("unicode", str)]:
        if not hasattr(np, name):
            setattr(np, name, alias)

    import smplx
    model_path = hf_hub_download(repo_id=SMPL_REPO, filename=SMPL_NEUTRAL_PATH_IN_REPO, token=settings.HF_TOKEN)
    _smpl_model = smplx.SMPL(model_path=model_path, gender="neutral", batch_size=1)
    return _smpl_model


@spaces.GPU
def _fit_smpl_to_visible_joints(target_idx_list, target_xyz_np, num_iters=60):
    """
    Optimizes SMPL pose/translation/scale so its regressed joints match the
    visible MediaPipe joints, then returns ALL 24 SMPL joint positions. Runs
    under ZeroGPU when deployed on the Gradio Space; falls back to whatever
    device is available (CPU) elsewhere via the no-op `spaces.GPU` above.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_smpl_model().to(device)

    target_idx = torch.tensor(target_idx_list, dtype=torch.long, device=device)
    target_xyz = torch.tensor(np.stack(target_xyz_np), dtype=torch.float32, device=device)

    body_pose = torch.zeros((1, 23 * 3), device=device, requires_grad=True)
    global_orient = torch.zeros((1, 3), device=device, requires_grad=True)
    transl = torch.zeros((1, 3), device=device, requires_grad=True)
    scale = torch.ones((1,), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([body_pose, global_orient, transl, scale], lr=0.05)
    prev_loss = float("inf")
    for _ in range(num_iters):
        optimizer.zero_grad()
        out = model(body_pose=body_pose, global_orient=global_orient)
        joints = out.joints[0] * scale + transl
        loss = torch.nn.functional.mse_loss(joints[target_idx], target_xyz)
        loss.backward()
        optimizer.step()

        # This is a small (6-12 joint) MSE fit, not a full-scene optimization —
        # it typically converges well before num_iters on CPU, so stop early
        # once improvement stalls instead of always paying the full budget.
        loss_val = loss.item()
        if abs(prev_loss - loss_val) < 1e-6:
            break
        prev_loss = loss_val

    with torch.no_grad():
        out = model(body_pose=body_pose, global_orient=global_orient)
        joints = (out.joints[0] * scale + transl).cpu().numpy()
    return joints


def recover_occluded_joints(mp_landmarks, occluded_indices, visibility_threshold=0.5):
    """
    mp_landmarks: [33,4] array-like (x, y, z, visibility) in MediaPipe space.
    occluded_indices: iterable of MediaPipe joint indices currently occluded.
    Returns {mp_idx: [x, y, z]} for whichever occluded joints in
    OCCLUDABLE_JOINTS could be recovered by fitting the SMPL body model to the
    still-visible joints -- a genuine kinematic/body-model-constrained
    recovery, as opposed to the simpler symmetric mirror heuristic. Returns
    {} (letting the caller keep its existing fallback) if there aren't enough
    visible anchor joints to fit reliably, or if anything in this path fails.
    """
    mp_arr = np.asarray(mp_landmarks, dtype=np.float32)
    occluded_set = set(occluded_indices)

    relevant_occluded = [idx for idx in OCCLUDABLE_JOINTS if idx in occluded_set]
    if not relevant_occluded:
        return {}

    target_idx_list = []
    target_xyz_np = []
    for mp_idx, smpl_idx in MP_TO_SMPL.items():
        if mp_idx not in occluded_set and mp_arr[mp_idx, 3] >= visibility_threshold:
            target_idx_list.append(smpl_idx)
            target_xyz_np.append(mp_arr[mp_idx, :3])

    if len(target_idx_list) < MIN_VISIBLE_ANCHORS:
        return {}

    try:
        joints = _fit_smpl_to_visible_joints(target_idx_list, target_xyz_np)
    except Exception as e:
        import traceback
        print(f"SMPL fit unavailable, falling back to mirror-only recovery: {e}", flush=True)
        traceback.print_exc()
        return {}

    return {mp_idx: joints[smpl_idx].tolist() for mp_idx, smpl_idx in OCCLUDABLE_JOINTS.items() if mp_idx in occluded_set}
