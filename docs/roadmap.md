# Roadmap and known gaps

VibeWarp is a port of the [WarpFusion](https://github.com/Sxela/WarpFusion) Colab notebook
into an installable Python package with a web UI. The notebook is the reference
implementation: where the two disagree, the notebook is assumed right until proven
otherwise.

This page is the honest state of that port — what is verified, what merely *runs*, and
what nobody has checked yet. It exists so a contributor can pick something up without
first having to rediscover where the bodies are buried.

If you want to help, the **[Open work](#open-work)** section is ordered roughly by
cost-to-value. Anything marked **good first issue** is self-contained and does not require
understanding the whole render pipeline.

---

## How correctness is measured

Everything below is stated in terms of **MAE against the reference notebook** on the same
settings, seed and input video — mean absolute error per pixel on the decoded frames,
where `0.0` is byte-identical. See [gpu-validation.md](gpu-validation.md) for how to run
the suite.

Rough scale, for reading the numbers below:

| MAE | means |
|---|---|
| < 0.02 | visually indistinguishable |
| 0.02 – 0.08 | same image, small drift in detail |
| > 0.15 | a *different* image (same prompt, different result) |

The harness renders **both sides** — VibeWarp and the notebook, in the notebook's own
environment — and compares them frame by frame. A number with no paired notebook run
behind it is not evidence, and this page tries hard to say which is which.

---

## What is verified against the notebook

| Case | MAE | Notes |
|---|---|---|
| SD1.5, no ControlNet | **0.0125** | pass |
| SD1.5 + softedge | **0.0116** | pass |
| SD1.5 + tile | 0.017 | |
| SD1.5 + ip2p | 0.018 | |
| SD1.5, full 5-net ControlNet stack | 0.054 | |
| SD1.5 + inpaint | 0.066 | |
| Reconstructed-noise mode (± ControlNet) | 0.044 / 0.075 | |
| **AnimateDiff SD1.5, sliding context** | **0.0147** | notebook's own default config (32/8/16/10) |
| AnimateDiff SD1.5, seams + `reinject_stylized` | 0.005 – 0.013 | tighter than the SD1.5 base |

## What runs but is NOT verified

These are implemented and produce plausible output. Nobody has compared them to the
notebook, so "works" means "does not crash and looks right", which is exactly the standard
that hid five bugs in AnimateDiff before it was finally compared.

- **SDXL** — including SDXL ControlNets (written natively; the notebook shells out to
  ComfyUI for these, so there is no reference implementation to copy). Currently blocked,
  see below.
- **AnimateDiff SDXL / HotshotXL**
- **FreeU · IP-Adapter · background masking · captions · content-aware scheduling ·
  softcap · tiled VAE · colormatch · temporalnet CN · reference CN · `fixed_code` noise ·
  `pingpong_noise` · `batched_adiff_rec_noise` · `rec_sliding_ctx`** — no parity test
  exists for any of these.

---

## Open work

### 1. LORA without ControlNets diverges — MAE 0.221 · *good first issue*

`parity-lora-no-cn` is the only SD1.5 case that is plainly wrong. The same settings **with**
ControlNets (`parity-lora-cn`) come out at 0.077, so the ControlNet conditioning is strong
enough to mask whatever the LORA path is doing wrong.

Isolated, reproducible, and definitely on our side rather than the notebook's. This is the
cheapest real bug on the list.

```bash
python tools/gpu_validation.py --manifest configs/gpu_parity.local.json \
  --only parity-lora-no-cn --fresh
```

### 2. The SDXL blocker — MAE ~0.20

`parity-sdxl-no-cn` renders the *same composition* with a *different look*. What makes it
strange: at the first denoiser call, the latent `x`, the `sigma`, and both conditioning
tensors `c` / `uc` **all match the notebook to 1e-6**. So the divergence is downstream of
conditioning, inside the UNet evaluation.

Already ruled out (please don't re-chase):

- the CLIP early-stop / clip-skip reimplementation (bit-exact, max diff `0.0`)
- per-step sampler RNG (MAE unchanged with the deterministic `sample_euler`)
- seeding, initial noise, and the sigma schedule

Remaining suspects, in order:

1. **The pooled `y` vector is unverified.** For SDXL, `y` (pooled bigG + size embeddings)
   drives style as hard as the cross-attention does, and a mismatch would look *exactly*
   like what we see. Our probe emits it; the notebook probe does not yet.
2. Our unified `controlled_unet_forward` (used even with zero ControlNets) vs the
   notebook's `apply_model_sdxl_cn`.
3. Attention backend — the notebook environment has xformers, ours falls back to SDPA.
4. `alphas_cumprod` in float32 vs the notebook's float64.

Suggested next step: land the `y` probe on the notebook side, and compare **final latents**
rather than decoded images so the VAE is out of the loop.

This one gates a lot: SDXL+CN, AnimateDiff-SDXL and the SDXL img2img probe are all
downstream of it.

### 3. depth-only ControlNet — MAE 0.114

Well above its siblings (softedge 0.012, tile 0.017, ip2p 0.018), which points at the depth
annotator or its conditioning rather than at the ControlNet machinery.

### 4. AnimateDiff + ControlNet: drift compounds across batches

Batch 0 is at parity (0.0126 — essentially the no-CN figure, so the ControlNet conditioning
itself is right), but each subsequent batch is ~2.5× worse than the same run without
ControlNets:

| region | +CN | no-CN |
|---|---|---|
| batch 0 | 0.0126 | 0.0102 |
| batch 1 | 0.0820 | 0.0330 |
| batch 2 | 0.0983 | 0.0509 |

A clean signature: fine on a cold batch, degrading only once a batch depends on the
previous batch's output. The likely mechanism is that the ControlNet hint for overlap
frames is annotated from the **stylized** (previously rendered) frame, so batch N−1's small
error is re-annotated and amplified. **Verify what the notebook actually does before
changing anything.**

### 5. Cases nobody has compared yet

`parity-sdxl-cn`, `parity-sdxl-animatediff-hotshot-{no-cn,cn}`, and img2img with
`style_strength < 1` for both SD1.5 and SDXL. The tests are already defined in
`configs/gpu_parity.local.json`; they just need a GPU and someone to read the numbers.

### 6. Parity coverage for the untested features · *good first issue*

Fourteen features (listed above) have **no parity test at all**. Adding one is mostly
mechanical — a settings file under `refs/examples/` and a test entry in
`configs/gpu_parity.local.json` — and each one is independently useful. The cheapest are
the ones that are a single config flag: `softcap`, `tiled VAE`, `fixed_code` noise.

### 7. Harness: make both sides emit the same probe · *good first issue*

The two sides report their first-denoiser-call probe **differently**: the notebook writes
`diag_model_inputs.json`, VibeWarp prints `[diag] cfg_model_inputs=` into `run.log`. That
asymmetry makes every probe comparison manual and error-prone. Making VibeWarp write the
same JSON artifact would make the SDXL hunt (and any future one) far less painful.

---

## Things worth knowing before you start

**The notebook's AnimateDiff is not seed-reproducible.** Its initial noise
(`big_noise`) is drawn from an **unseeded** CPU generator — nothing calls `seed_everything`
before it. Two notebook runs at the *same seed* differ at MAE 0.30, which is *worse* than
the notebook differs from us. Image-level comparison is therefore meaningless unless both
sides start from the same noise, so the harness dumps the reference's noise and injects it
into VibeWarp automatically (`VIBEWARP_ADIFF_NOISE`). VibeWarp itself **is** seed-reproducible;
we deliberately do not copy this bug.

**Some divergences are the notebook's, not ours**, and are documented rather than
replicated — e.g. `do_run_adiff` runs one spurious extra batch at the tail, which
overwrites the final frame with a degraded render. Where we deliberately differ, the code
says so and explains why.

**A parity number is worthless until you have proven the reference actually applied the
feature.** The notebook's settings loader swallows per-key errors and silently keeps the
previous widget value, so it is entirely possible to run a "parity test" of a feature that
was never switched on. The harness verifies the notebook's *saved settings snapshot*
against what was requested for exactly this reason — if you add a new setting to the
translation, add it to `VERIFIED_KEYS` too.

---

## Beyond parity

Parity is the current priority, but not the only way to help.

- **Frontend tests.** The UI has 30 component tests (`cd webui && npm test`) and plenty of
  room for more. Component behaviour the Python suite cannot see needs a component test —
  the two bugs that prompted the suite (a prompt editor writing a list into a
  `Dict[int, str]`, and dead mode buttons) were both invisible to the backend.
- **Annotator performance.** Annotators run at the render resolution by default, which is
  what the notebook does. Cost is quadratic: softedge is ~470 ms/frame at 1920px versus
  ~58 ms at 512. Per-net `detect_resolution` already exists as a setting; a considered
  default (and a quality comparison to justify it) would be a real win.
- **Error surfacing.** Validation errors deep-link to the offending field, except inside
  the ControlNet editor, which has no anchors yet.
- **Documentation.** [architecture.md](architecture.md) has the pipeline diagrams;
  [settings.md](settings.md) maps every WarpFusion setting to its VibeWarp equivalent.

---

## Ground rules

They exist because ignoring them has cost real time:

1. **The notebook is the reference.** If something feels counterintuitive but the notebook
   does it, match the notebook first and refactor afterwards, behind a test.
2. **Re-measure rather than trust a recorded number.** A batch of parity results in this
   repo once sat unrefreshed through a month of fixes; a tidy theory got built on them, and
   a 20-minute re-run demolished it.
3. **Verify a contract, not a signature.** Checking that a function *accepts* a callback is
   not the same as checking what it must *return*. (This fork's samplers replace the latent
   with the callback's return value — an unusual convention that broke every render until a
   test drove the real samplers rather than a mock.)
4. **Every change gets tests**, Python and frontend. See [CONTRIBUTING.md](../CONTRIBUTING.md).
