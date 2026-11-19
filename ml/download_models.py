"""Download the InfiniteTalk (talking avatar) model set into ComfyUI/models. ~32 GB."""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from pathlib import Path
from huggingface_hub import hf_hub_download

M = Path(__file__).parent / "ComfyUI" / "models"
KJ = "Kijai/WanVideo_comfy"
jobs = [
    (KJ, "InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors", "diffusion_models"),
    (KJ, "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors", "diffusion_models"),
    (KJ, "Wan2_1_VAE_bf16.safetensors", "vae"),
    (KJ, "umt5-xxl-enc-fp8_e4m3fn.safetensors", "text_encoders"),
    (KJ, "open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors", "clip_vision"),
    (KJ, "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras"),
]
for repo, fname, sub in jobs:
    dest = M / sub
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(fname).name
    if target.exists():
        print("have", target.name, flush=True)
        continue
    print("downloading", fname, flush=True)
    p = hf_hub_download(repo, fname, local_dir=str(dest / "_tmp"))
    Path(p).replace(target)
    print("done", target.name, f"{target.stat().st_size/1e9:.1f} GB", flush=True)
# audio encoder used by InfiniteTalk
d = M / "transformers" / "TencentGameMate" / "chinese-wav2vec2-base"
d.mkdir(parents=True, exist_ok=True)
for f in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
    hf_hub_download("TencentGameMate/chinese-wav2vec2-base", f, local_dir=str(d))
print("ALL DONE", flush=True)

# ComfyUI's own CLIP-vision encoder for Wan (the workflow's CLIPVisionLoader expects this exact name)
cv = M / "clip_vision" / "clip_vision_h.safetensors"
if not cv.exists():
    p = hf_hub_download("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", local_dir=str(M / "clip_vision" / "_tmp"))
    Path(p).replace(cv)
    print("done clip_vision_h.safetensors", flush=True)
