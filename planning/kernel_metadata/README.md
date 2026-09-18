# Kaggle kernel metadata

Exactly what each kernel was pushed with, so a run can be reproduced or
re-pushed without reconstructing the dataset wiring from memory.

Push a kernel by putting its script and the matching `kernel-metadata.json`
in one directory:

```bash
mkdir -p /tmp/k && cd /tmp/k
cp <script>.py .
cp <this dir>/<kernel>.json kernel-metadata.json   # must be named this
kaggle kernels push -p .
```

The `code_file` field must match the script's filename in that directory.

| file | script in `planning/` | notes |
|---|---|---|
| `asanaai-stgcn-transitions.json` | `kaggle_stgcn_transitions.py` | GPU. Transition-aware ST-GCN. Bar to beat: **63.0% macro**. |
| `asanaai-photo-corpus-v2.json` | `harvest_photo_corpus_v2.py` | CPU + internet, no datasets. Commons + Openverse harvest. |
| `asanaai-mlp-domain.json` | `kaggle_mlp_photo_domain.py` | GPU. The run that measured 35.5% vs 14.1%. |

**Not yet pushed** — `kaggle_mlp_photo_v2.py`, which retrains on the expanded
corpus. It needs the harvest kernel's output mounted, so copy
`asanaai-mlp-domain.json`, change `id`/`title`/`code_file`, and add:

```json
"kernel_sources": ["arkosarkarhehe/asanaai-photo-corpus-v2"]
```

Kaggle datasets mount NESTED (`/kaggle/input/datasets/<owner>/<slug>`), which is
why every script globs recursively for its inputs rather than hardcoding a path.

Get logs via the API `log` field. Do NOT use `kaggle kernels output` on the
large kernels — it burns the data budget.
