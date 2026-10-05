# Archive (kept for history, not part of the running system)

* `WORKSPACE_PLAN.md`, `PROJECT_STATUS.md` — the June 2026 planning/handover documents. Superseded by `backup/RESUME_HERE_2026-10-05.md` and `docs/`.
* `legacy_root/old_monolithic_app.py` — the original single-file Gradio/FastAPI app that the modular `backend/` replaced.
* `legacy_root/test_app.py`, `legacy_root/run_full_pipeline_test.py` — manual smoke scripts written for the early modular layout (they load models from the Hub / use machine-specific paths). The maintained tests are in `backend/tests/` and `backend/tools/`.
* (removed) `test_groq.py` — imported a module (`app`) that no longer exists; its job is done by `backend/test_backend_groq.py`.
