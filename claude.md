# VibeWarp

VibeWarp is a port of the [WarpFusion](https://github.com/Sxela/WarpFusion) Colab notebook
(v0.37) into an installable Python package with a Svelte web UI, with its dependencies
vendored or consolidated rather than git-cloned at runtime.

## The notebook is the reference implementation

Where VibeWarp and the notebook disagree, **the notebook is assumed right until proven
otherwise**. If something feels weird or counterintuitive but the notebook does it that way
and works, match the notebook first, cover it with tests, and refactor afterwards.

The reference notebook and its text dump live in `refs/` (not tracked — fetch them from the
WarpFusion repo if you need them). `refs/examples/` holds settings files known to work.

## Plan before you change things

Say what you intend to do before doing it, especially for anything touching the render
path. A wrong render is expensive to notice and expensive to bisect.

## Tests — always, both suites

```bash
python -m pytest -m "not gpu" --strict-markers   # ~1200 tests
cd webui && npm test                             # vitest + @testing-library/svelte
cd webui && npm run build                        # the FE is built static and served by the
                                                 # BE — ALWAYS rebuild after a FE change
```

The frontend suite exists because the UI shipped two bugs the Python suite could never
catch: the prompt editor writing a list into a `Dict[int, str]`, and mode buttons that
silently did nothing. **Component behaviour the backend cannot see needs a component test.**

If a test fails, fix it — even if you did not touch that part.

## Correctness is measured against the notebook

`tools/gpu_validation.py` renders **both sides** — VibeWarp and the notebook, in the
notebook's own environment — and compares them frame by frame (MAE per pixel). See
[docs/gpu-validation.md](docs/gpu-validation.md).

Two lessons that have cost real time, worth internalising:

- **A parity number is worthless until you have proven the reference actually applied the
  feature.** The notebook's settings loader swallows per-key errors and keeps the previous
  widget value, so it is entirely possible to "test" a feature that was never switched on.
- **Re-measure rather than trust a recorded number.** Results go stale, and a tidy theory
  built on stale numbers is worse than no theory at all.

## Where things are

- [docs/roadmap.md](docs/roadmap.md) — what is verified, what is not, and where to help
- [docs/architecture.md](docs/architecture.md) — render pipeline, settings flow, dependencies
- [docs/settings.md](docs/settings.md) — every WarpFusion setting mapped to its VibeWarp equivalent
- [docs/cli.md](docs/cli.md) — flags, Python API, output layout

Update the docs and README alongside a change, not after it.

## UI layout is declared in the backend

`vibewarp/ui_layout.py` assigns every config field a tab (`tier`) and a group; the schema
carries that, and the Svelte app renders whatever the schema says. A field with no
classification renders in **no tab** — and a test fails, naming it. Do not add a field list
to the frontend: it would drift the moment `config.py` changed.
