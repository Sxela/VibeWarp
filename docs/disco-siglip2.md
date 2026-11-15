# SigLIP 2 guidance for Disco

Use **siglip2-base** for the first comparison. The UI also offers
**siglip2-large**, which costs more memory and compute. Google documents these
models for image/text representation tasks; that does not establish better
diffusion guidance or a fix for the saturation observed in progressive tests.
[Base model card](https://huggingface.co/google/siglip2-base-patch16-224).

The selector and differentiable adapter were already wired. This preparation
downloaded the real Base checkpoint and tokenizer into
`models/disco/siglip2-base`, corrected explicit lowercase/fixed-length text
preprocessing, and verified the real CUDA path. Large was not downloaded.
No default guidance encoder selection was changed.

## Use it

Choose a Disco model, open **Guidance encoders**, and select **siglip2-base**.
Uncheck the classic encoders for an isolated comparison, or retain them for
combined guidance. Each checked encoder contributes its own loss; adding
encoders changes guidance strength and runtime as well as semantics.

Alternatively import [the example settings](../examples/disco_siglip2_base.json),
then choose your video and an existing Disco checkpoint if using a custom
path. The example selects SigLIP alone, 512px output and full-resolution
sampling. It is a working setup example, not a quality-tuned preset.

Selection shows required files without transferring them. Click **Download
required models** or **Start render** to prepare missing files through the
shared download progress UI. Locally installed Base files include a 1,500,800,904-byte safetensors file
and the processor/tokenizer assets. The HF download cache used for preparation
was kept under `.cache/huggingface`.

## Preprocessing and guidance

Both SigLIP 2 sizes load in FP16, as do all classic CLIP guidance encoders.
The vision input is cast to FP16 after normalization; embeddings and guidance
losses stay FP32. Old saved encoder precision settings cannot restore FP32
model loading.

SigLIP 2 FixRes Base uses 224px image input. Differentiable image
normalization follows its processor's mean/std; cutouts preserve image
gradients. Text is lowercased and explicitly padded/truncated to the model's
configured training length (64 tokens for the downloaded Base checkpoint).
These requirements follow the
[Transformers SigLIP 2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2#text-embeddings-and-retrieval).

Guidance currently uses normalized image/text embeddings with Disco's
spherical-distance loss. It does not reinterpret SigLIP's sigmoid classifier
probability as a guidance score. Frozen encoder weights still allow gradients
with respect to the input image. Current Transformers tensor and pooled-output
feature APIs are both handled.

## Verified on 2026-11-05

RTX 5090, Torch 2.11.0+cu128, Transformers 5.5.4:

- Real Base checkpoint: finite, nonzero image gradients; no weight gradients.
- Text embeddings match the reference preprocessing exactly in this check.
- Image-embedding maximum absolute difference from reference preprocessing:
  `7.45e-8`.
- Full 512px Disco render: 30 DDIM steps, secondary denoiser, eight cutouts,
  SigLIP Base alone, seed 42. Peak allocated memory was 4.240 GiB. This was
  a single functional run without warmup, not a performance comparison.
- The test image remains strongly saturated. Switching encoders alone has
  not demonstrated a fix for the earlier image-quality problem.
- 44 relevant backend tests passed, including the preprocessing regression.

[Real-checkpoint gradient report](../.cache/siglip2-validation.json),
[render configuration/results](../.cache/disco-siglip2-base-smoke/results.json),
[test image](../.cache/disco-siglip2-base-smoke/full_0.png).
