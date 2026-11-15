"""Known Comfy weight sources, matching docs/settings.md setup recipes."""
COMFY_SOURCES = {
    'flux-2-klein-4b-fp8.safetensors': ('diffusion_models', 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors'),
    'qwen_3_4b.safetensors': ('text_encoders', 'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors'),
    'flux2-vae.safetensors': ('vae', 'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/vae/flux2-vae.safetensors'),
    'qwen_3_8b_fp8mixed.safetensors': ('text_encoders', 'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors'),
    'hidream_o1_image_fp8_scaled.safetensors': ('checkpoints', 'https://huggingface.co/Comfy-Org/HiDream-O1-Image/resolve/main/checkpoints/hidream_o1_image_fp8_scaled.safetensors'),
    'gemma4_e4b_it_fp8_scaled.safetensors': ('text_encoders', 'https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e4b_it_fp8_scaled.safetensors'),
    'qwen_image_edit_2511_int8_convrot.safetensors': ('diffusion_models', 'https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors'),
    'Qwen-Image-Edit-2511-Q5_K_M.gguf': ('unet', 'https://huggingface.co/vantagewithai/Qwen-Image-Edit-2511-GGUF/resolve/main/Qwen-Image-Edit-2511-Q5_K_M.gguf'),
    'qwen_image_vae.safetensors': ('vae', 'https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors'),
    'Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors': ('loras', 'https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors'),
    'mage_flow_edit_int8_convrot.safetensors': ('diffusion_models', 'https://huggingface.co/Comfy-Org/Mage-Flow/resolve/main/diffusion_models/mage_flow_edit_int8_convrot.safetensors'),
    'mage_flow_edit_turbo_int8_convrot.safetensors': ('diffusion_models', 'https://huggingface.co/Comfy-Org/Mage-Flow/resolve/main/diffusion_models/mage_flow_edit_turbo_int8_convrot.safetensors'),
    'qwen3vl_4b_bf16.safetensors': ('text_encoders', 'https://huggingface.co/Comfy-Org/Mage-Flow/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors'),
    'mage_flow_vae_bf16.safetensors': ('vae', 'https://huggingface.co/Comfy-Org/Mage-Flow/resolve/main/vae/mage_flow_vae_bf16.safetensors'),
    'flux-2-klein-9b-fp8.safetensors': ('diffusion_models', 'https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors'),
    'qwen_2.5_vl_7b_fp8_scaled.safetensors': ('text_encoders', 'https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors'),
}
