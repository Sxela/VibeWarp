# Contributing to VibeWarp

Bug reports, documentation improvements, and focused pull requests are welcome.
For substantial changes, open an issue first so the approach can be discussed before
implementation work begins.

## Where to start

**[docs/roadmap.md](docs/roadmap.md)** is the honest state of the port: what is verified
against the reference notebook, what merely runs, and what nobody has checked yet. It lists
the open work roughly by cost-to-value and marks the self-contained tasks that do not
require understanding the whole render pipeline.

VibeWarp is a port of the WarpFusion Colab notebook. **The notebook is the reference
implementation** — where the two disagree, it is assumed right until proven otherwise. If
something looks counterintuitive but the notebook does it that way, match it first and
refactor afterwards, behind a test.

## Development setup

VibeWarp requires Python 3.11 or newer and Node.js 22 for frontend development.
Create a virtual environment, install the package and its development extras, then
install the locked frontend dependencies:

```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
python -m pip install -e ".[ui,sched,dev]"

cd webui
npm ci
```

Install PyTorch for your operating system and accelerator first if the default PyPI
wheel is not appropriate. See <https://pytorch.org/get-started/locally/>.

## Tests

Run the same CPU-compatible checks as GitHub Actions:

```bash
python -m pytest -m "not gpu" --strict-markers
cd webui
npm test
npm run build
```

CUDA-only tests require a configured NVIDIA GPU and can be selected with
`python -m pytest -m gpu`. Changes to rendering behavior should also be exercised with
the validation process in [docs/gpu-validation.md](docs/gpu-validation.md).

## Pull requests

- Keep changes focused and include tests for behavior changes.
- Update user documentation when configuration or CLI behavior changes.
- Do not commit model weights, generated renders, local settings, or credentials.
- Preserve copyright and license notices when modifying vendored code.
- Describe the models, hardware, and validation command used for GPU-sensitive changes.
