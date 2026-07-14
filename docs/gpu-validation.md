# Overnight GPU validation

`tools/gpu_validation.py` runs GPU cases in separate processes, continues past
missing optional assets, resumes completed jobs, compares paired render frames,
and continually updates `report.json` and `report.md`.

## Paired notebook-parity tests

The `tests` manifest section is the preferred way to compare VibeWarp against
the original notebook. Each test names ONE original WarpFusion settings file
(from `refs/examples`) that both sides share — VibeWarp loads it through
`load_warpfusion_settings`, the reference notebook receives the same file as
its settings template — so untranslated keys cannot silently diverge:

```json
"tests": [
  {
    "id": "parity-basic-no-cn",
    "settings": "refs/examples/parity-sd15-cn-basic.txt",
    "overrides": {"controlnet": {"enabled": false}}
  }
]
```

Each test expands into two jobs plus a comparison. Outputs land in
`<run_dir>/<timestamp>_<test-id>/vibewarp/` and
`<run_dir>/<timestamp>_<test-id>/notebook/`; final
frames are additionally flattened to `frame_NNNNNN.<ext>` at each side's root
for direct eyeballing, and per-test metrics are written to
`<run_dir>/<timestamp>_<test-id>/comparison.json`. A normal rerun resumes the
newest timestamped pair; `--fresh` creates a new pair without overwriting old
evidence. Run the suite with the run directory as
output root:

```powershell
python tools\gpu_validation.py `
  --manifest configs\gpu_parity.local.json --output gpu_parity
```

`--only <test-id>` selects one test (both sides plus the comparison).
The paired harness sets `VIBEWARP_PARITY_MODE=1` on the VibeWarp job so its
attention backend matches the reference's manual split/einsum path. Normal
VibeWarp renders continue to use SDPA (or xformers when available).
Optional per-test keys: `notebook: false` (vibewarp side only), `thresholds`,
`requires`, `env`, `enabled`, and `only_controlnets` (filter the shared base
settings to a named CN subset); `defaults.thresholds` supplies the comparison
default. After a reference job completes, the runner verifies the notebook's
saved settings snapshot against the requested translation and fails the job
loudly if the GUI silently ignored any key (its loader keeps the previous
widget value when a value has an unexpected type — schedules must always be
lists/dicts, never bare scalars).

Start by copying the public paired-test template. Set the reference notebook,
reference environment, input video, WarpFusion settings export, and model
paths for the local machine. Optional model-specific cases are disabled until
explicitly enabled:

```powershell
Copy-Item configs/gpu_parity.example.json configs/gpu_parity.local.json
# Edit gpu_parity.local.json, then:
python tools/gpu_validation.py `
  --manifest configs/gpu_parity.local.json --output gpu_parity
```

Files matching `configs/*.local.json` are ignored by Git so machine paths do
not accidentally become public. The `.example.json` manifests are the public,
sanitized starting points.

`base_config_kind` may be `json` for a structured `RunConfig` JSON file,
`warpfusion` for a notebook export, or `vibewarp` for a settings snapshot saved
beside an earlier VibeWarp render.

```powershell
Copy-Item configs/gpu_validation.example.json configs/gpu_validation.local.json
# Edit gpu_validation.local.json, then:
python tools\gpu_validation.py `
  --manifest configs\gpu_validation.local.json `
  --output gpu_validation
```

Re-running the command resumes the suite and skips jobs whose `result.json`
already says `pass`. Use `--fresh` to rerun passing jobs, `--clean` to discard
the whole suite first, or `--only empty-cn-stock empty-cn-patched` for a subset.
Every job gets its resolved `config.json`, combined `run.log`, output artifacts,
and result metadata under `gpu_validation/<suite>/jobs/<job-id>/`.

## Empty-ControlNet gate

The first comparison renders the same frames twice:

1. the current stock non-ControlNet forward;
2. the existing ControlNet-patched forward with an empty ControlNet dictionary.

Production always uses the unified forward. The first job sets the
validation-only `VIBEWARP_VALIDATE_STOCK_FORWARD=1` escape hatch to retain the
legacy baseline for regression checks; the second uses normal production code.
The comparison requires byte-exact PNG equality.

## Reference-notebook comparisons

Set the manifest-level `reference_notebook` source, work directory, and Python
paths plus a notebook-format `settings_template` from `refs/examples`, then
mark a job with `"reference_notebook": true`. The runner overlays that template
with the job's resolved VibeWarp config, loads it through the notebook GUI's
own `load_settings()` path, and executes original
notebook cells 18–55 in the reference environment. Repository installs/updates
are disabled, the output directory and frame range are isolated, and the
executed notebook plus translated settings are retained with the artifacts.
The translator derives the notebook's `controlnet_models_dir` from resolved
ControlNet entry paths so reference runs reuse existing weights instead of
downloading duplicate checkpoints into the notebook installation.

Reference jobs must be paired with their VibeWarp counterparts in
`comparisons`. A failed translation or notebook cell fails the job loudly.
Because GPU implementations can differ numerically, notebook comparisons use
explicit image tolerances rather than claiming byte identity.

Image comparisons report exact equality, MAE, RMSE, maximum absolute channel
error, and PSNR. Thresholds can require `exact`, upper bounds for `mae`, `rmse`,
or `max_abs`, and a lower bound with `min_psnr`.
