import re, pathlib
W = pathlib.Path("/home/anamitra/yoga_posture_workspace"); K = W / "planning/kaggle_transfer/kernel_stgcn_cmp"
seq = (W / "backend/app/models/sequence.py").read_text()
geo = (W / "backend/app/utils/geometry.py").read_text(); m = re.search(r"def normalize_coordinate_sequence.*?return coords_normalized\.reshape\(60, 99\)", geo, re.S); assert m
t = (K / "runner_cmp.py.tpl").read_text().replace("@@BACKEND_SEQ@@", seq).replace("@@BACKEND_NORM@@", m.group(0))
assert "'''" not in seq and "'''" not in m.group(0)
(K / "runner_cmp.py").write_text(t); print("runner_cmp.py", len(t), "bytes")
