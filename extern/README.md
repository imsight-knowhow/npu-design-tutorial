# extern/

Third-party source checkouts live here (typically cloned with `--depth 1`).

- This directory is for convenience when browsing upstream code alongside this repo.
- Checked out sources are ignored by Git via `extern/.gitignore`.
- If you need to vendor or patch an external dependency, do not put it here—add it as a proper dependency or submodule instead.

## What’s currently here

These are local checkouts for reading/reference and are not automatically “wired into” the tutorial code unless explicitly referenced elsewhere.

- `extern/accel-sim-framework/`
  - Upstream: https://github.com/accel-sim/accel-sim-framework
  - What it is: Accel-Sim (and AccelWattch) GPU simulation framework (trace-driven SASS frontend + GPGPU-Sim 4.x performance model + tooling).
  - Useful entry points:
    - `extern/accel-sim-framework/README.md`
    - `extern/accel-sim-framework/gpu-simulator/README.md`
    - `extern/accel-sim-framework/util/job_launching/README.md`
- `extern/LLMCompass/`
  - Upstream: https://github.com/PrincetonUniversity/LLMCompass
  - What it is: LLMCompass (ISCA’24) framework for LLM inference hardware evaluation (operator models + mapper/heuristics + cost/area model).
  - Useful entry points:
    - `extern/LLMCompass/README.md`
    - `extern/LLMCompass/software_model/transformer.py`
    - `extern/LLMCompass/design_space_exploration/dse.py`
- `extern/vidur/`
  - Upstream: see `extern/vidur/` for project metadata (this repo includes its own docs).
  - What it is: a third-party project kept locally for reference.

## Updating a checkout

If you want to refresh any checkout, delete its folder and re-clone it. Example:

```bash
rm -rf extern/<name>
git clone --depth 1 <upstream-url> extern/<name>
```
