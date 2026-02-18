"""
LiveAvatar LipSync Training Script.

Trains LiveAvatar's CausalWanModel_S2V for lip-sync V2V inpainting using diffusion loss.
The ONLY architecture change is expanding patch_embedding Conv3d from 16->49 (or 33) input channels.
Everything else (wav2vec2 audio, causal attention, KV cache, block-wise generation) stays unchanged.

49 channels = 16 (noisy latents) + 1 (mouth mask) + 16 (masked latents) + 16 (reference latents)
33 channels = 16 (noisy latents) + 1 (mouth mask) + 16 (masked latents)  [use_ref_frames: false]

Usage:
    accelerate launch train_lipsync.py --config configs/lipsync_train.yaml
"""

import argparse
import gc
import logging
import math
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def zero_pad_to_49ch(latents_16ch):
    """Pad 16-channel latents to 49 channels with zeros.

    [B, 16, T, H, W] -> [B, 49, T, H, W]

    When channels 16-48 are zero, Conv3d output matches the original 16-ch encoding
    because: output = sum(weight[:,i] * input[:,i]), and zero input -> zero contribution.
    """
    B, _C, T, H, W = latents_16ch.shape
    pad = torch.zeros(B, 33, T, H, W, dtype=latents_16ch.dtype, device=latents_16ch.device)
    return torch.cat([latents_16ch, pad], dim=1)


def zero_pad_to_33ch(latents_16ch):
    """Pad 16-channel latents to 33 channels with zeros.
    [B, 16, T, H, W] -> [B, 33, T, H, W]
    """
    B, _C, T, H, W = latents_16ch.shape
    pad = torch.zeros(B, 17, T, H, W, dtype=latents_16ch.dtype, device=latents_16ch.device)
    return torch.cat([latents_16ch, pad], dim=1)


def construct_49ch_block(noisy_block, mask_block, masked_block, ref_block):
    """Construct 49-channel model input for one temporal block.

    Args:
        noisy_block:  [B, 16, 3, H, W] -- noisy x_t
        mask_block:   [B, 1, 3, H, W]  -- mouth mask
        masked_block: [B, 16, 3, H, W] -- masked video latents
        ref_block:    [B, 16, 3, H, W] -- reference identity latents

    Returns: [B, 49, 3, H, W]
    """
    return torch.cat([noisy_block, mask_block, masked_block, ref_block], dim=1)


def construct_33ch_block(noisy_block, mask_block, masked_block):
    """Construct 33-channel model input (no reference latents).
    Returns: [B, 33, 3, H, W]
    """
    return torch.cat([noisy_block, mask_block, masked_block], dim=1)


def vae_encode_batch(vae, videos):
    """Encode a batch of videos through VAE.

    Args:
        vae: Wan2_1_VAE instance. vae.encode() takes a list of [C, T, H, W] tensors.
        videos: [B, C, T, H, W] tensor

    Returns: [B, 16, Tzip, H/8, W/8] where Tzip = 1 + (T-1)/4
    """
    video_list = [videos[b] for b in range(videos.shape[0])]
    encoded = vae.encode(video_list)  # list of [16, Tzip, H/8, W/8]
    return torch.stack(encoded)  # [B, 16, Tzip, H/8, W/8]


def move_vae(vae, device):
    """Move Wan2_1_VAE (non-nn.Module wrapper) to a device."""
    vae.model.to(device)
    vae.mean = vae.mean.to(device)
    vae.std = vae.std.to(device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    vae.device = device


# ──────────────────────────────────────────────────────────────────────────────
# Model setup: load, expand patch_embedding, apply LoRA
# ──────────────────────────────────────────────────────────────────────────────

def load_and_prepare_model(config, device):
    """Load pretrained CausalWanModel_S2V, expand to 49ch (or 33ch), apply LoRA.

    Supports loading the LiveAvatar DMD (distillation) LoRA checkpoint in two modes:
      merge_dmd_lora=True:  merge DMD into base weights, then apply fresh LoRA
      merge_dmd_lora=False: apply LoRA, then load DMD weights into it (continue training)
    """
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file as load_safetensors

    # Add LiveAvatar to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LiveAvatar"))
    from liveavatar.models.wan.causal_model_s2v import CausalWanModel_S2V

    dmd_lora_path = config.get("dmd_lora_path")
    merge_dmd = config.get("merge_dmd_lora", False)

    # 1. Load base model with original 16-ch
    logger.info(f"Loading pretrained model from {config['checkpoint_dir']}")
    model = CausalWanModel_S2V.from_pretrained(
        config["checkpoint_dir"],
        torch_dtype=torch.bfloat16,
    )

    # 2. (Mode A only) Merge DMD LoRA into base weights for 4-step distillation
    if dmd_lora_path and merge_dmd:
        dmd_state = load_safetensors(dmd_lora_path)
        dmd_state = {f"base_model.model.{k}": v for k, v in dmd_state.items() if "lora" in k}
        dmd_targets = ["q", "k", "v", "o", "ffn.0", "ffn.2"]
        dmd_config = LoraConfig(r=128, lora_alpha=64.0, init_lora_weights=True, target_modules=dmd_targets)
        model = get_peft_model(model, dmd_config)
        model.load_state_dict(dmd_state, strict=False)
        model = model.merge_and_unload()
        logger.info(f"DMD LoRA merged into base weights from {dmd_lora_path}")

    # 3. Expand patch_embedding: 16 -> in_channels (49 with ref frames, 33 without)
    use_ref_frames = config.get("use_ref_frames", True)
    in_channels = 49 if use_ref_frames else 33

    old_conv = model.patch_embedding
    dim = model.dim
    patch_size = tuple(model.patch_size)
    new_conv = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :16] = old_conv.weight
        new_conv.bias.data.copy_(old_conv.bias.data)
    model.patch_embedding = new_conv
    model.in_dim = in_channels
    logger.info(f"Expanded patch_embedding: 16 -> {in_channels} channels")

    # 4. Apply LoRA
    lora_targets = config.get("lora_targets", "q,k,v,o,ffn.0,ffn.2").split(",")
    lora_config = LoraConfig(
        r=config.get("lora_rank", 128),
        lora_alpha=config.get("lora_alpha", 64.0),
        init_lora_weights=True,
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"Applied LoRA rank={config.get('lora_rank', 128)} to {lora_targets}")

    # 5. (Mode B only) Load DMD LoRA weights into PEFT model (continue training)
    if dmd_lora_path and not merge_dmd:
        dmd_state = load_safetensors(dmd_lora_path)
        dmd_state = {f"base_model.model.{k}": v for k, v in dmd_state.items() if "lora" in k}
        missing, unexpected = model.load_state_dict(dmd_state, strict=False)
        loaded = len(dmd_state) - len(unexpected)
        logger.info(f"DMD LoRA weights loaded (continue-training mode): {loaded} params from {dmd_lora_path}")

    # 6. Unfreeze patch_embedding (PEFT freezes everything except LoRA)
    for p in model.base_model.model.patch_embedding.parameters():
        p.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M total parameters")

    model = model.to(device)
    model.train()
    # Enable gradient checkpointing on the underlying CausalWanModel_S2V
    # (PEFT wraps it as model.base_model.model)
    model.base_model.model.gradient_checkpointing = True

    return model


def load_vae(config, device):
    """Load Wan 2.1 VAE."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LiveAvatar"))
    from liveavatar.models.wan.wan_2_2.modules.vae2_1 import Wan2_1_VAE

    vae = Wan2_1_VAE(
        vae_pth=os.path.join(config["checkpoint_dir"], config.get("vae_checkpoint", "Wan2.1_VAE.pth")),
        device=device,
        dtype=torch.bfloat16,
    )
    return vae


def load_audio_encoder(config, device):
    """Load wav2vec2 audio encoder for training."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LiveAvatar"))
    from liveavatar.models.wan.causal_audio_encoder import AudioEncoder

    model_id = os.path.join(config["checkpoint_dir"], config.get("wav2vec_model", "wav2vec2-large-xlsr-53-english"))
    audio_enc = AudioEncoder(device=str(device), model_id=model_id)
    audio_enc.model.eval()
    audio_enc.model.requires_grad_(False)
    return audio_enc


def load_t5_encoder(config, device):
    """Load T5 text encoder."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LiveAvatar"))
    from liveavatar.models.wan.wan_2_2.modules.t5 import T5EncoderModel

    t5 = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=os.path.join(config["checkpoint_dir"], config.get("t5_checkpoint", "models_t5_umt5-xxl-enc-bf16.pth")),
        tokenizer_path=os.path.join(config["checkpoint_dir"], config.get("t5_tokenizer", "google/umt5-xxl")),
    )
    return t5


# ──────────────────────────────────────────────────────────────────────────────
# KV Cache initialization (mirrors causal_s2v_pipeline.py:601-660)
# ──────────────────────────────────────────────────────────────────────────────

def initialize_kv_cache(num_layers, batch_size, kv_cache_size, dtype, device):
    """Initialize KV cache for all transformer layers."""
    kv_cache = []
    for _ in range(num_layers):
        layer_cache = {
            "k": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=device),
            "cond_k": torch.zeros([batch_size, 2800, 40, 128], dtype=dtype, device=device),
            "cond_v": torch.zeros([batch_size, 2800, 40, 128], dtype=dtype, device=device),
            "cond_end": torch.tensor([0], dtype=torch.long, device=device),
        }
        kv_cache.append(layer_cache)
    return kv_cache


def initialize_crossattn_cache(num_layers, batch_size, dtype, device):
    """Initialize cross-attention cache for all transformer layers."""
    cache = []
    for _ in range(num_layers):
        cache.append({
            "k": torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device),
            "is_init": False,
        })
    return cache


def reset_kv_cache(kv_cache):
    """Reset KV cache for a new training sample by creating fresh tensors.

    IMPORTANT: Uses torch.zeros_like() instead of .zero_() to break autograd
    graph references. In-place .zero_() preserves the tensor's grad_fn, which
    pins the entire computation graph (~75GB) in memory across iterations.
    Fresh allocation creates tensors with grad_fn=None, allowing GC to free
    the previous iteration's graph.
    """
    for layer_cache in kv_cache:
        layer_cache["k"] = torch.zeros_like(layer_cache["k"])
        layer_cache["v"] = torch.zeros_like(layer_cache["v"])
        layer_cache["cond_k"] = torch.zeros_like(layer_cache["cond_k"])
        layer_cache["cond_v"] = torch.zeros_like(layer_cache["cond_v"])
        layer_cache["cond_end"] = torch.zeros_like(layer_cache["cond_end"])


def reset_crossattn_cache(crossattn_cache, batch_size):
    """Reset cross-attention cache for a new training sample."""
    device = crossattn_cache[0]["k"].device
    dtype = crossattn_cache[0]["k"].dtype
    for layer_cache in crossattn_cache:
        layer_cache["k"] = torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device)
        layer_cache["v"] = torch.zeros([batch_size, 0, 40, 128], dtype=dtype, device=device)
        layer_cache["is_init"] = False


def move_kv_cache_to_device(kv_cache, device):
    """Move all tensors in a KV cache to the specified device (in-place).

    Used for CPU offloading: only the active per-timestep cache lives on GPU,
    matching inference's _move_kv_cache_to_working_gpu pattern.
    """
    for layer in kv_cache:
        for key in layer:
            if isinstance(layer[key], torch.Tensor):
                layer[key] = layer[key].to(device, non_blocking=True)
    return kv_cache


def clone_kv_cache(kv_cache, device):
    """Deep copy a KV cache to the specified device.

    After prefill, all 4 per-timestep caches start identical (same t=0 state).
    We prefill once on GPU and clone to CPU 4 times.
    """
    return [
        {k: v.clone().to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
         for k, v in layer.items()}
        for layer in kv_cache
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────────────────────────────────────

def training_step(
    batch,
    model,
    vae,
    audio_encoder,
    t5_encoder,
    restricted_timesteps,
    restricted_sigmas,
    kv_cache,
    crossattn_cache,
    config,
    device,
    accelerator=None,
    timestep_weights=None,
):
    """Execute one training step with block-wise gradient accumulation (Algorithm 2).

    Instead of running all 7 blocks forward with grad then a single backward
    (which OOMs at ~141GB peak), uses a two-phase approach:

    Phase 1: Forward all 7 blocks WITHOUT gradients, caching velocity predictions
             and populating KV cache. Peak memory ~68GB (no autograd graph stored).
    Phase 2: Re-run blocks in REVERSE order (6→0), one at a time WITH gradients.
             Each block computes full-sequence MSE loss + backward, then detaches
             KV cache to free the graph. Peak memory ~77GB (1 block's graph).

    Reverse order is safe because attention is causal: block N reads only KV[0..N-1],
    so re-running block N never sees stale entries from blocks N+1..6.

    Gradient math: each backward produces gradient from 3 of 21 frames. After 7
    calls, accumulated gradient = Σ per_block_grad = full ∂loss/∂θ.

    Trade-off: drops cross-block KV gradient flow (small second-order signal).

    Returns: loss value as float (backward already performed)
    """
    use_precomputed = "x_0" in batch
    use_ref_frames = config.get("use_ref_frames", True)
    batch_size = batch["x_0"].shape[0] if use_precomputed else batch["video"].shape[0]
    num_blocks = 7
    latent_frames_per_block = 3
    motion_frames_video = 73
    lat_motion_frames = 19  # 1 + 72/4

    height = config.get("height", 512)
    width = config.get("width", 512)
    H_lat = height // 8
    W_lat = width // 8

    # ── STEP 1: VAE Encode (no grad) ──────────────────────────────────────
    if use_precomputed:
        # Full precomputed path — no VAE needed
        x_0 = batch["x_0"].to(device, dtype=torch.bfloat16)
        masked_latents = batch["masked_latents"].to(device, dtype=torch.bfloat16)
        if use_ref_frames:
            ref_latents_49ch = batch["ref_latents_49ch"].to(device, dtype=torch.bfloat16)
        ref_latents_sink = batch["ref_latents_sink"].to(device, dtype=torch.bfloat16)
        motion_latents = batch["motion_latents"].to(device, dtype=torch.bfloat16)
        mask_latent = batch["mask_latent"].to(device, dtype=torch.bfloat16)
        mouth_mask = batch["mouth_mask"].to(device)
    else:
        # Original path: VAE encode from raw video
        # Move VAE to GPU for encoding, then offload to free memory for backward pass
        move_vae(vae, device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            video = batch["video"].to(device)            # [B, 3, 81, H, W]
            ref_frames = batch["ref_frames"].to(device)  # [B, 3, 81, H, W]
            mouth_mask = batch["mouth_mask"].to(device)  # [B, 1, H, W]

            # GT video -> latents (diffusion target)
            # 81 frames -> 21 latent frames (Tzip = 1 + 80/4 = 21). Train on all 21.
            x_0 = vae_encode_batch(vae, video)  # [B, 16, 21, H_lat, W_lat]

            # Masked video -> latents (mouth zeroed in pixel space BEFORE encoding)
            # mask.png convention: 1.0 = keep (upper face), 0.0 = inpaint (mouth/chin)
            mask_pixel = mouth_mask.unsqueeze(2).expand(-1, -1, 81, -1, -1)
            masked_video = video * mask_pixel
            masked_latents = vae_encode_batch(vae, masked_video)  # [B, 16, 21, H_lat, W_lat]

            # Reference -> latents for 49-ch input (per-block identity conditioning)
            if use_ref_frames:
                ref_latents_49ch = vae_encode_batch(vae, ref_frames)  # [B, 16, 21, H_lat, W_lat]

            # Reference -> single frame latent for _forward_sink (conditioning cache)
            # VAE CausalConv3d needs 5 frames for proper temporal context; [:,:,1:] selects
            # the second latent frame which has full causal context (matches pipeline line 902-903)
            ref_single = ref_frames[:, :, 0:1, :, :]  # [B, 3, 1, H, W]
            ref_5frames = ref_single.repeat(1, 1, 5, 1, 1)  # [B, 3, 5, H, W]
            ref_latents_sink = vae_encode_batch(vae, ref_5frames)[:, :, 1:]  # [B, 16, 1, H_lat, W_lat]

            # Motion latents: repeat reference image to fill 73 video frames.
            # Motion is unused in _forward_inference (see CLAUDE.md issue #2), but _forward_sink
            # caches motion tokens for KV prefill. Using repeated ref matches inference behavior
            # (pipeline line 907: ref_pixel_values.repeat(..., motion_frames, ...))
            motion_pixel = ref_single.repeat(1, 1, motion_frames_video, 1, 1)
            motion_latents = vae_encode_batch(vae, motion_pixel)  # [B, 16, 19, H_lat, W_lat]

            # Mask -> latent space (trilinear interpolation)
            mask_for_latent = mask_pixel.float()
            mask_latent = F.interpolate(
                mask_for_latent, size=(21, H_lat, W_lat), mode="trilinear", align_corners=False
            )  # [B, 1, 21, H_lat, W_lat]

        # Free pixel-space tensors and offload VAE to CPU
        del video, ref_frames, masked_video, mask_pixel, ref_5frames, motion_pixel
        move_vae(vae, "cpu")
        torch.cuda.empty_cache()

    # ── STEP 2: Audio Extraction (no grad) ────────────────────────────────
    if use_precomputed:
        # Precomputed audio embeddings
        audio_emb = batch["audio_emb"].to(device, dtype=torch.bfloat16)  # [B, 25, 1024, 84]
        # Audio dropout (training-time augmentation)
        audio_drop_prob = config.get("audio_dropout", 0.0)
        if audio_drop_prob > 0.0:
            for b_idx in range(batch_size):
                if torch.rand(1).item() < audio_drop_prob:
                    audio_emb[b_idx] = 0.0
    else:
        # Original path: wav2vec2 at 50Hz -> resample to 30Hz -> bucket to video fps (25)
        # Need 84 audio entries: 7 blocks x 12 video-frames/block = 84. The actual video has
        # only 81 frames, so entries 81-83 are zero-padded by get_audio_embed_bucket_fps
        # (functionally equivalent to silence at clip boundary).
        audio_fps = config.get("audio_fps", 25)  # Matches LiveAvatar shared_config.sample_fps
        audio_batch_frames = num_blocks * latent_frames_per_block * 4  # 7 * 3 * 4 = 84

        audio_encoder.model.to(device)
        with torch.no_grad():
            audio_embs = []
            for b_idx in range(batch_size):
                z = audio_encoder.extract_audio_feat(
                    batch["audio_path"][b_idx], return_all_layers=True
                )
                # z: [25, N_30fps, 1024] -- 25 wav2vec layers at 30Hz
                audio_bucket, _ = audio_encoder.get_audio_embed_bucket_fps(
                    z, fps=audio_fps, batch_frames=audio_batch_frames, m=0
                )
                # audio_bucket may be larger than batch_frames (covers full audio).
                # Since video always starts from frame 0, take only the first 84 entries.
                audio_bucket = audio_bucket[:audio_batch_frames]  # [84, 25, 1024]
                audio_bucket = audio_bucket.unsqueeze(0).permute(0, 2, 3, 1)  # [1, 25, 1024, 84]
                audio_embs.append(audio_bucket.squeeze(0))  # [25, 1024, 84]

            audio_emb = torch.stack(audio_embs).to(device, dtype=torch.bfloat16)  # [B, 25, 1024, 84]

            # Audio dropout: zero out audio for CFG (classifier-free guidance) training
            audio_drop_prob = config.get("audio_dropout", 0.0)
            if audio_drop_prob > 0.0:
                for b_idx in range(batch_size):
                    if torch.rand(1).item() < audio_drop_prob:
                        audio_emb[b_idx] = 0.0

        audio_encoder.model.to("cpu")
        torch.cuda.empty_cache()

    # ── STEP 3: Text Encoding (no grad) ───────────────────────────────────
    if use_precomputed:
        # Precomputed text with dropout
        text_drop_prob = config.get("text_dropout", 0.1)
        context = []
        for b_idx in range(batch_size):
            if torch.rand(1).item() < text_drop_prob:
                # Use precomputed empty-string embedding (exact match with live T5)
                if "empty_text_emb" in batch:
                    emb = batch["empty_text_emb"]
                    # Could be stacked [B, L, 4096] or list of [L, 4096]
                    if isinstance(emb, list):
                        emb = emb[b_idx]
                    elif emb.dim() == 3:
                        emb = emb[b_idx]
                    context.append(emb.to(device, dtype=torch.bfloat16))
                else:
                    # Fallback: zeros (approximate, should not happen if precomputed correctly)
                    context.append(torch.zeros(1, 4096, device=device, dtype=torch.bfloat16))
            else:
                emb = batch["text_emb"]
                if isinstance(emb, list):
                    emb = emb[b_idx]
                elif emb.dim() == 3:
                    emb = emb[b_idx]
                context.append(emb.to(device, dtype=torch.bfloat16))
    else:
        # Original path
        t5_encoder.model.to(device)
        with torch.no_grad():
            text_list = batch["text"]
            texts_for_encode = []
            for txt in text_list:
                if torch.rand(1).item() < config.get("text_dropout", 0.1):
                    txt = ""
                texts_for_encode.append(txt)
            context = t5_encoder(texts_for_encode, device)  # list of [L, 4096]

        t5_encoder.model.to("cpu")
        torch.cuda.empty_cache()

    # ── STEP 4: Noise + target (restricted 4-step) ────────────────────────
    x_0 = x_0.to(device, dtype=torch.bfloat16)
    masked_latents = masked_latents.to(device, dtype=torch.bfloat16)
    if use_ref_frames:
        ref_latents_49ch = ref_latents_49ch.to(device, dtype=torch.bfloat16)
    ref_latents_sink = ref_latents_sink.to(device, dtype=torch.bfloat16)
    motion_latents = motion_latents.to(device, dtype=torch.bfloat16)
    mask_latent = mask_latent.to(device, dtype=torch.bfloat16)

    choice = torch.randint(0, len(restricted_timesteps), (batch_size,), device=device)
    sigma = restricted_sigmas[choice].to(device).view(batch_size, 1, 1, 1, 1).to(torch.bfloat16)
    timestep_val = restricted_timesteps[choice].to(device)  # [B]

    noise = torch.randn_like(x_0)
    x_t = (1.0 - sigma) * x_0 + sigma * noise  # [B, 16, 21, H_lat, W_lat]
    velocity_target = noise - x_0  # [B, 16, 21, H_lat, W_lat]

    # ── STEP 5: Prepare padding + caches ──────────────────────────────────
    if use_ref_frames:
        ref_latents_padded = zero_pad_to_49ch(ref_latents_sink)  # [B, 49, 1, H_lat, W_lat]
    else:
        ref_latents_padded = zero_pad_to_33ch(ref_latents_sink)  # [B, 33, 1, H_lat, W_lat]
    # patch_size=(1,2,2) spatial downsampling: each latent frame -> (H_lat/2)*(W_lat/2) tokens
    frame_seq_length = (H_lat // 2) * (W_lat // 2)

    zeros_cond = torch.zeros(batch_size, 16, latent_frames_per_block, H_lat, W_lat, device=device, dtype=torch.bfloat16)

    # Reset caches
    reset_kv_cache(kv_cache)
    reset_crossattn_cache(crossattn_cache, batch_size)

    # Reset block_mask so it gets recomputed
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    base_model.block_mask = None

    # ── STEP 6: Prefill via _forward_sink (no grad) ──────────────────────
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        dummy_x = [torch.zeros(16, latent_frames_per_block, H_lat, W_lat, device=device, dtype=torch.bfloat16)] * batch_size

        # Audio for first block: 12 video frames
        audio_block0 = audio_emb[..., 0:12]

        base_model._forward_sink(
            x=dummy_x,
            t=torch.zeros(batch_size, latent_frames_per_block, device=device, dtype=torch.bfloat16),
            context=context,
            seq_len=None,
            cond_states=zeros_cond,
            motion_latents=motion_latents,
            ref_latents=ref_latents_padded,
            audio_input=audio_block0,
            motion_frames=[motion_frames_video, lat_motion_frames],
            drop_motion_frames=False,
            sink_flag=True,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=0,
            current_end=latent_frames_per_block * frame_seq_length,
        )

    # ── STEP 7+8: Block-wise gradient accumulation (Algorithm 2) ────────
    # Instead of running all 7 blocks forward with grad (OOMs at ~141GB peak),
    # use two phases:
    #   Phase 1: Forward all blocks WITHOUT grad (cache predictions + KV state)
    #   Phase 2: Re-run blocks in REVERSE order (6→0), one at a time WITH grad,
    #            computing loss + backward per block, then freeing the graph.
    # Peak memory drops from ~141GB to ~77GB at the cost of losing cross-block
    # KV gradient flow (a small second-order signal).
    W_mouth = config.get("mouth_weight", 5.0)

    # ──── Phase 1: No-grad forward pass (cache predictions) ──────────
    v_cache = []  # cached velocity predictions per block, detached

    with torch.no_grad():
        for block_idx in range(num_blocks):
            block_start_frame = block_idx * latent_frames_per_block
            block_end_frame = (block_idx + 1) * latent_frames_per_block

            # Slice per-block tensors
            x_t_block = x_t[:, :, block_start_frame:block_end_frame]
            mask_block = mask_latent[:, :, block_start_frame:block_end_frame]
            masked_block = masked_latents[:, :, block_start_frame:block_end_frame]

            # Audio for this block: 12 video frames per block
            audio_block = audio_emb[..., block_idx * 12 : (block_idx + 1) * 12]

            # Build channel input (49ch with ref frames, 33ch without)
            if use_ref_frames:
                ref_block = ref_latents_49ch[:, :, block_start_frame:block_end_frame]
                x_49 = construct_49ch_block(x_t_block, mask_block, masked_block, ref_block)
            else:
                x_49 = construct_33ch_block(x_t_block, mask_block, masked_block)

            # Timestep: [B, 3] same value across all frames in block
            t_block = timestep_val.unsqueeze(1).expand(batch_size, latent_frames_per_block).float()

            # Convert to per-sample list (LiveAvatar convention)
            x_list = [x_49[b] for b in range(batch_size)]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                velocity_list = base_model._forward_inference(
                    x=x_list,
                    t=t_block,
                    context=context,
                    seq_len=None,
                    cond_states=zeros_cond,
                    motion_latents=motion_latents,
                    ref_latents=ref_latents_padded,
                    audio_input=audio_block,
                    motion_frames=[motion_frames_video, lat_motion_frames],
                    drop_motion_frames=False,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=block_start_frame * frame_seq_length,
                    current_end=block_end_frame * frame_seq_length,
                )
            # Returns: list of [16, 3, H_lat, W_lat] per batch item
            velocity_pred = torch.stack(velocity_list)  # [B, 16, 3, H_lat, W_lat]
            v_cache.append(velocity_pred)  # already detached (under no_grad)

            # Detach audio attrs (no-op since no grad, but kept for symmetry with Phase 2)
            if hasattr(base_model, 'merged_audio_emb') and base_model.merged_audio_emb is not None:
                base_model.merged_audio_emb = base_model.merged_audio_emb.detach()
            if hasattr(base_model, 'audio_emb_global') and base_model.audio_emb_global is not None:
                base_model.audio_emb_global = base_model.audio_emb_global.detach()


    # Compute reference loss from Phase 1 predictions (for validation)
    v_output_ref = torch.cat(v_cache, dim=2)  # [B, 16, 21, H_lat, W_lat]
    # mask_latent: 1=keep (upper face), 0=inpaint (mouth). Invert so mouth gets upweighted.
    mouth_indicator = 1.0 - mask_latent.float()
    full_weight_ref = 1.0 + (W_mouth - 1.0) * mouth_indicator
    loss_ref = (
        F.mse_loss(v_output_ref.float(), velocity_target.float(), reduction="none")
        * full_weight_ref
    ).mean()
    if timestep_weights is not None:
        loss_ref = loss_ref * timestep_weights[choice].mean()
    loss_ref_val = loss_ref.item()
    del v_output_ref, full_weight_ref, loss_ref

    # ──── Phase 2: Reverse per-block backward (gradient accumulation) ─
    # Re-run each block WITH grad in reverse order (6→0), compute loss,
    # backward, then free graph. Gradients accumulate on shared LoRA params.
    #
    # Reverse order is correct because attention is causal: block N reads
    # only KV[0..N-1], so re-running block N never sees stale entries from
    # blocks N+1..6 that were already processed and detached.
    for block_idx in range(num_blocks - 1, -1, -1):
        block_start_frame = block_idx * latent_frames_per_block
        block_end_frame = (block_idx + 1) * latent_frames_per_block

        # Slice per-block tensors (same as Phase 1)
        x_t_block = x_t[:, :, block_start_frame:block_end_frame]
        mask_block = mask_latent[:, :, block_start_frame:block_end_frame]
        masked_block = masked_latents[:, :, block_start_frame:block_end_frame]
        audio_block = audio_emb[..., block_idx * 12 : (block_idx + 1) * 12]

        if use_ref_frames:
            ref_block = ref_latents_49ch[:, :, block_start_frame:block_end_frame]
            x_49 = construct_49ch_block(x_t_block, mask_block, masked_block, ref_block)
        else:
            x_49 = construct_33ch_block(x_t_block, mask_block, masked_block)
        t_block = timestep_val.unsqueeze(1).expand(batch_size, latent_frames_per_block).float()
        x_list = [x_49[b] for b in range(batch_size)]

        # Re-run this single block WITH gradients
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            velocity_list = base_model._forward_inference(
                x=x_list,
                t=t_block,
                context=context,
                seq_len=None,
                cond_states=zeros_cond,
                motion_latents=motion_latents,
                ref_latents=ref_latents_padded,
                audio_input=audio_block,
                motion_frames=[motion_frames_video, lat_motion_frames],
                drop_motion_frames=False,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=block_start_frame * frame_seq_length,
                current_end=block_end_frame * frame_seq_length,
            )
        velocity_pred_grad = torch.stack(velocity_list)  # [B, 16, 3, H_lat, W_lat]

        # Build full-sequence output: this block fresh (with grad), others cached (no grad)
        full_predictions = []
        for j in range(num_blocks):
            if j == block_idx:
                full_predictions.append(velocity_pred_grad)
            else:
                full_predictions.append(v_cache[j])
        velocity_output = torch.cat(full_predictions, dim=2)  # [B, 16, 21, H_lat, W_lat]

        # Mouth-weighted MSE loss: upweight the inpaint region (mouth/chin)
        # mask_latent: 1=keep (upper face), 0=inpaint (mouth). Invert so mouth gets W_mouth weight.
        mouth_indicator = 1.0 - mask_latent.float()
        full_weight = 1.0 + (W_mouth - 1.0) * mouth_indicator
        loss_elem = F.mse_loss(
            velocity_output.float(), velocity_target.float(), reduction="none"
        )
        loss = (loss_elem * full_weight).mean()
        if timestep_weights is not None:
            loss = loss * timestep_weights[choice].mean()

        # Backward — gradients accumulate across blocks on shared LoRA params
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        # Detach KV cache to free this block's autograd graph
        for _layer_cache in kv_cache:
            for _cache_key in ['k', 'v', 'cond_k', 'cond_v']:
                _layer_cache[_cache_key] = _layer_cache[_cache_key].detach()

        # Detach audio attrs to prevent graph retention across blocks
        if hasattr(base_model, 'merged_audio_emb') and base_model.merged_audio_emb is not None:
            base_model.merged_audio_emb = base_model.merged_audio_emb.detach()
        if hasattr(base_model, 'audio_emb_global') and base_model.audio_emb_global is not None:
            base_model.audio_emb_global = base_model.audio_emb_global.detach()

        del velocity_pred_grad, velocity_output, loss, loss_elem, full_weight, full_predictions
        torch.cuda.empty_cache()

    # # KV cache diagnostics
    # for _diag_layer in [0, 20, 39]:
    #     _k = kv_cache[_diag_layer]["k"]
    #     _v = kv_cache[_diag_layer]["v"]
    #     logger.info(
    #         f"[diag] post-bwd kv[{_diag_layer}]: "
    #         f"k.grad_fn={_k.grad_fn} k.ver={_k._version} k.ptr={_k.data_ptr()} "
    #         f"v.grad_fn={_v.grad_fn} v.ver={_v._version} v.ptr={_v.data_ptr()}"
    #     )
    #
    # # Model attribute diagnostics
    # for _attr_name in ['merged_audio_emb', 'audio_emb_global', 'pre_compute_freqs']:
    #     _attr = getattr(base_model, _attr_name, None)
    #     if _attr is not None and isinstance(_attr, torch.Tensor):
    #         logger.info(
    #             f"[diag] post-bwd {_attr_name}: grad_fn={_attr.grad_fn} "
    #             f"req_grad={_attr.requires_grad} shape={list(_attr.shape)} "
    #             f"ptr={_attr.data_ptr()}"
    #         )
    #     elif _attr is not None and isinstance(_attr, (list, tuple)):
    #         logger.info(f"[diag] post-bwd {_attr_name}: type={type(_attr).__name__} len={len(_attr)}")
    #
    # # Rope cache diagnostics
    # if hasattr(base_model, 'rope_cache'):
    #     _rc = base_model.rope_cache
    #     for _rc_key, _rc_val in _rc.items():
    #         if isinstance(_rc_val, torch.Tensor):
    #             logger.info(
    #                 f"[diag] rope_cache[{_rc_key}]: shape={list(_rc_val.shape)} "
    #                 f"grad_fn={_rc_val.grad_fn}"
    #             )
    #         elif isinstance(_rc_val, (list, tuple)):
    #             logger.info(f"[diag] rope_cache[{_rc_key}]: len={len(_rc_val)}")
    #         else:
    #             logger.info(f"[diag] rope_cache[{_rc_key}]: {_rc_val}")

    return torch.tensor(loss_ref_val, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# Saving / Loading checkpoints
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step, output_dir):
    """Save LoRA adapters + patch_embedding."""
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)

    # Save LoRA adapters via PEFT
    model.save_pretrained(os.path.join(save_dir, "lora"))

    # Save patch_embedding separately
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    torch.save(
        base_model.patch_embedding.state_dict(),
        os.path.join(save_dir, "patch_embedding.pt"),
    )

    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    logger.info(f"Saved checkpoint at step {step} to {save_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Video saving utilities
# ──────────────────────────────────────────────────────────────────────────────

def save_video_with_audio(frames_tensor, audio_source_path, output_path, fps=25):
    """Save generated video frames as MP4 with audio from source video.

    Args:
        frames_tensor: [1, 3, T, H, W] float tensor in [-1, 1]
        audio_source_path: Path to video containing the audio track
        output_path: Where to save the final video
        fps: Frame rate (default 25, matches LiveAvatar)
    """
    import imageio

    # [1, 3, T, H, W] -> [T, H, W, 3] uint8
    frames = frames_tensor[0]  # [3, T, H, W]
    frames = frames.permute(1, 2, 3, 0)  # [T, H, W, 3]
    frames = ((frames.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()  # [-1,1] -> [0,255]
    frames_np = frames.cpu().numpy()

    # Save silent video
    temp_silent = output_path.replace(".mp4", "_silent.mp4")
    writer = imageio.get_writer(temp_silent, fps=fps, quality=9, codec="libx264")
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()

    # Extract audio from source video
    temp_audio = output_path.replace(".mp4", "_audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_source_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", "-loglevel", "error", temp_audio],
        check=False,
    )

    # Merge video + audio
    if os.path.exists(temp_audio):
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_silent, "-i", temp_audio,
             "-c:v", "copy", "-c:a", "aac", "-shortest", "-loglevel", "error", output_path],
            check=False,
        )
        os.remove(temp_audio)
    else:
        # No audio track — just rename silent video
        os.rename(temp_silent, output_path)
        return

    if os.path.exists(temp_silent):
        os.remove(temp_silent)


def save_gt_video(batch, sample_idx, output_path, fps=25):
    """Save ground-truth video frames from a dataset batch as MP4 with original audio.

    Args:
        batch: Dataset batch dict with "video" [B, 3, T, H, W] and "audio_path" list
        sample_idx: Which sample in the batch
        output_path: Where to save
        fps: Frame rate
    """
    gt_frames = batch["video"][sample_idx:sample_idx+1]  # [1, 3, T, H, W]
    audio_path = batch["audio_path"][sample_idx]
    save_video_with_audio(gt_frames, audio_path, output_path, fps=fps)


# ──────────────────────────────────────────────────────────────────────────────
# Validation (LiveAvatar-style inference loop)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model,
    vae,
    audio_encoder,
    t5_encoder,
    val_dataset,
    scheduler,
    config,
    device,
    global_step,
    output_dir,
    mode="recon",
    max_samples=5,
    offload_kv_cache=False,
    wandb_run=None,
    _syncnet_state={},  # mutable default for lazy init (StableAvatar pattern)
):
    """Run validation using LiveAvatar's block-wise inference loop.

    Generates videos from pure noise using iterative denoising (4 timesteps per block),
    then saves them alongside GT videos for visual comparison.

    Args:
        model: The unwrapped PeftModel (not DDP-wrapped)
        scheduler: FlowMatchEulerDiscreteScheduler instance (reused from training)
        mode: "recon" (same video/audio) or "mixed" (different audio source)
        max_samples: Maximum number of validation samples to generate
    """
    use_ref_frames = config.get("use_ref_frames", True)
    num_blocks = 7
    latent_frames_per_block = 3
    motion_frames_video = 73
    lat_motion_frames = 19
    num_latent_frames = 21

    height = config.get("height", 512)
    width = config.get("width", 512)
    H_lat = height // 8
    W_lat = width // 8
    frame_seq_length = (H_lat // 2) * (W_lat // 2)

    # Prepare output directory
    step_dir = os.path.join(output_dir, f"step_{global_step}", mode)
    os.makedirs(step_dir, exist_ok=True)

    # Set up scheduler for 4-step inference (matches training)
    scheduler.set_timesteps(4, device=device)
    timesteps = scheduler.timesteps  # 4 timestep values

    # Access base model
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    was_training = model.training
    model.eval()
    base_model.block_mask = None

    num_layers = base_model.num_layers
    num_samples = min(max_samples, len(val_dataset))

    wandb_videos = []
    sync_d_list = []
    sync_c_list = []

    # Lazy-init SyncNet models (persists across validation calls via mutable default)
    syncnet_model = None
    syncnet_detector = None
    if config.get("compute_sync_metrics", False):
        if 'syncnet' not in _syncnet_state:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
            from syncnet import SyncNetEval, SyncNetDetector
            _syncnet_state['syncnet'] = SyncNetEval(device="cuda").float()
            _syncnet_state['syncnet'].loadParameters(config["syncnet_model_path"])
            _syncnet_state['syncnet_detector'] = SyncNetDetector(
                device="cuda",
                detect_results_dir=os.path.join(output_dir, "syncnet_detect_results"),
                s3fd_model_path=config["s3fd_model_path"],
            )
        syncnet_model = _syncnet_state['syncnet']
        syncnet_detector = _syncnet_state['syncnet_detector']

    logger.info(f"[val] Starting {mode} validation: {num_samples} samples, 4 timesteps")

    for sample_idx in range(num_samples):
        try:
            sample = val_dataset[sample_idx]
            # Wrap in batch dimension
            batch = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]
                for k, v in sample.items()
            }

            video_id = sample.get("video_id", f"sample_{sample_idx}")
            audio_id = sample.get("audio_id", video_id)

            # ── Encode inputs ─────────────────────────────────────────────
            move_vae(vae, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                video = batch["video"].to(device)           # [1, 3, 81, H, W]
                ref_frames = batch["ref_frames"].to(device) # [1, 3, 81, H, W]
                mouth_mask = batch["mouth_mask"].to(device)  # [1, 1, H, W]

                # Masked video (mouth zeroed)
                # mask.png convention: 1.0 = keep (upper face), 0.0 = inpaint (mouth/chin)
                mask_pixel = mouth_mask.unsqueeze(2).expand(-1, -1, 81, -1, -1)
                masked_video = video * mask_pixel
                masked_latents = vae_encode_batch(vae, masked_video)

                # Reference latents for 49ch
                if use_ref_frames:
                    ref_latents_49ch = vae_encode_batch(vae, ref_frames)

                # Reference latent for sink (single frame with temporal context)
                ref_single = ref_frames[:, :, 0:1, :, :]
                ref_5frames = ref_single.repeat(1, 1, 5, 1, 1)
                ref_latents_sink = vae_encode_batch(vae, ref_5frames)[:, :, 1:]

                # Motion latents (repeated reference)
                motion_pixel = ref_single.repeat(1, 1, motion_frames_video, 1, 1)
                motion_latents = vae_encode_batch(vae, motion_pixel)

                # Mask in latent space
                mask_for_latent = mask_pixel.float()
                mask_latent = F.interpolate(
                    mask_for_latent, size=(num_latent_frames, H_lat, W_lat),
                    mode="trilinear", align_corners=False,
                )

            del masked_video, mask_pixel, ref_5frames, motion_pixel
            move_vae(vae, "cpu")
            torch.cuda.empty_cache()

            # ── Audio extraction ──────────────────────────────────────────
            audio_fps = config.get("audio_fps", 25)
            audio_batch_frames = num_blocks * latent_frames_per_block * 4  # 84

            audio_encoder.model.to(device)
            z = audio_encoder.extract_audio_feat(
                batch["audio_path"][0], return_all_layers=True
            )
            audio_bucket, _ = audio_encoder.get_audio_embed_bucket_fps(
                z, fps=audio_fps, batch_frames=audio_batch_frames, m=0
            )
            audio_bucket = audio_bucket[:audio_batch_frames]  # Truncate to first 84 entries
            audio_emb = audio_bucket.unsqueeze(0).permute(0, 2, 3, 1)  # [1, 25, 1024, 84]
            audio_emb = audio_emb.to(device, dtype=torch.bfloat16)
            audio_encoder.model.to("cpu")
            torch.cuda.empty_cache()

            # ── Text encoding ─────────────────────────────────────────────
            t5_encoder.model.to(device)
            context = t5_encoder(batch["text"], device)
            t5_encoder.model.to("cpu")
            torch.cuda.empty_cache()

            # ── Prepare tensors ───────────────────────────────────────────
            masked_latents = masked_latents.to(device, dtype=torch.bfloat16)
            if use_ref_frames:
                ref_latents_49ch = ref_latents_49ch.to(device, dtype=torch.bfloat16)
            ref_latents_sink = ref_latents_sink.to(device, dtype=torch.bfloat16)
            motion_latents = motion_latents.to(device, dtype=torch.bfloat16)
            mask_latent = mask_latent.to(device, dtype=torch.bfloat16)

            if use_ref_frames:
                ref_latents_padded = zero_pad_to_49ch(ref_latents_sink)
            else:
                ref_latents_padded = zero_pad_to_33ch(ref_latents_sink)
            zeros_cond = torch.zeros(1, 16, latent_frames_per_block, H_lat, W_lat,
                                     device=device, dtype=torch.bfloat16)

            # Initialize caches for this sample
            # Inference uses 4 independent KV caches (one per denoising timestep)
            # to prevent cross-timestep contamination (causal_s2v_pipeline.py:995-1002).
            # We prefill once on GPU then clone for each timestep.
            kv_cache_size = num_latent_frames * frame_seq_length
            kv_cache_gpu = initialize_kv_cache(num_layers, 1, kv_cache_size, torch.bfloat16, device)
            crossattn_cache = initialize_crossattn_cache(num_layers, 1, torch.bfloat16, device)
            base_model.block_mask = None

            # ── Prefill via _forward_sink ──────────────────────────────────
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                dummy_x = [torch.zeros(16, latent_frames_per_block, H_lat, W_lat,
                                       device=device, dtype=torch.bfloat16)]
                base_model._forward_sink(
                    x=dummy_x,
                    t=torch.zeros(1, latent_frames_per_block, device=device, dtype=torch.bfloat16),
                    context=context,
                    seq_len=None,
                    cond_states=zeros_cond,
                    motion_latents=motion_latents,
                    ref_latents=ref_latents_padded,
                    audio_input=audio_emb[..., 0:12],
                    motion_frames=[motion_frames_video, lat_motion_frames],
                    drop_motion_frames=False,
                    sink_flag=True,
                    kv_cache=kv_cache_gpu,
                    crossattn_cache=crossattn_cache,
                    current_start=0,
                    current_end=latent_frames_per_block * frame_seq_length,
                )

            # Clone prefilled cache for all 4 timesteps, then free the original
            # (matches inference prefill loop at causal_s2v_pipeline.py:1030-1058)
            clone_device = "cpu" if offload_kv_cache else device
            kv_caches = {}
            for ts_idx in range(4):
                kv_caches[str(ts_idx + 1)] = clone_kv_cache(kv_cache_gpu, clone_device)
            del kv_cache_gpu
            torch.cuda.empty_cache()

            # ── Initialize pure noise ─────────────────────────────────────
            x_t = torch.randn(1, 16, num_latent_frames, H_lat, W_lat,
                              device=device, dtype=torch.bfloat16)
            output_latents = torch.zeros_like(x_t)

            # ── Block-wise denoising (7 blocks × 4 timesteps) ─────────────
            for block_idx in range(num_blocks):
                block_start = block_idx * latent_frames_per_block
                block_end = (block_idx + 1) * latent_frames_per_block

                # Fixed conditioning for this block (doesn't change across timesteps)
                mask_block = mask_latent[:, :, block_start:block_end]
                masked_block = masked_latents[:, :, block_start:block_end]
                if use_ref_frames:
                    ref_block = ref_latents_49ch[:, :, block_start:block_end]
                audio_block = audio_emb[..., block_idx * 12 : (block_idx + 1) * 12]

                # Start from noise for this block
                block_latents = x_t[:, :, block_start:block_end]  # [1, 16, 3, H, W]

                # Reset scheduler state for this block (matches causal_s2v_pipeline.py:1079-1080)
                scheduler._step_index = 0
                scheduler._begin_index = 0

                # Iterate scheduler timesteps (4 steps: from noisy to clean)
                # Each timestep uses its own KV cache (matches causal_s2v_pipeline.py:1103-1106)
                for i_ts, t_val in enumerate(timesteps):
                    cache_key = str(i_ts + 1)
                    if offload_kv_cache:
                        move_kv_cache_to_device(kv_caches[cache_key], device)

                    # Construct channel input with current noisy block latents
                    if use_ref_frames:
                        x_49 = construct_49ch_block(block_latents, mask_block, masked_block, ref_block)
                    else:
                        x_49 = construct_33ch_block(block_latents, mask_block, masked_block)

                    # Timestep tensor: [1, 3]
                    t_block = torch.tensor([t_val] * latent_frames_per_block,
                                           device=device).unsqueeze(0).float()

                    x_list = [x_49[0]]  # Per-sample list (batch=1)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        velocity_list = base_model._forward_inference(
                            x=x_list,
                            t=t_block,
                            context=context,
                            seq_len=None,
                            cond_states=zeros_cond,
                            motion_latents=motion_latents,
                            ref_latents=ref_latents_padded,
                            audio_input=audio_block,
                            motion_frames=[motion_frames_video, lat_motion_frames],
                            drop_motion_frames=False,
                            kv_cache=kv_caches[cache_key],
                            crossattn_cache=crossattn_cache,
                            current_start=block_start * frame_seq_length,
                            current_end=block_end * frame_seq_length,
                        )

                    if offload_kv_cache:
                        move_kv_cache_to_device(kv_caches[cache_key], "cpu")
                        torch.cuda.empty_cache()

                    velocity_pred = velocity_list[0].unsqueeze(0)  # [1, 16, 3, H, W]

                    # scheduler.step: velocity prediction → denoised latents
                    block_latents = scheduler.step(
                        velocity_pred, t_val, block_latents,
                        return_dict=False,
                    )[0]  # [1, 16, 3, H, W]

                # Store denoised block
                output_latents[:, :, block_start:block_end] = block_latents

            # ── VAE decode ────────────────────────────────────────────────
            # Concatenate motion + generated latents for temporal context,
            # then decode and crop (matches causal_s2v_pipeline.py:1145-1150)
            move_vae(vae, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                decode_input = torch.cat([motion_latents, output_latents], dim=2)
                decoded_list = vae.decode([decode_input[0]])  # list of [3, T, H, W]
                decoded = torch.stack(decoded_list)  # [1, 3, T, H, W]
                # Crop to generated frames only (drop motion history + 3-frame overlap)
                decoded = decoded[:, :, -(81):]
                decoded = decoded[:, :, 3:]  # Drop first 3 overlap frames

            move_vae(vae, "cpu")
            torch.cuda.empty_cache()

            # ── Save outputs ──────────────────────────────────────────────
            if mode == "recon":
                gen_path = os.path.join(step_dir, f"{video_id}_gen.mp4")
                gt_path = os.path.join(step_dir, f"{video_id}_gt.mp4")
            else:
                gen_path = os.path.join(step_dir, f"v{video_id}_a{audio_id}_gen.mp4")
                gt_path = None  # No GT for mixed (different audio)

            save_video_with_audio(decoded.cpu().float(), batch["audio_path"][0], gen_path)
            logger.info(f"[val] Saved {mode} generated: {gen_path}")

            # SyncNet metrics (outside autocast to avoid bf16 BatchNorm issues)
            if syncnet_model is not None and os.path.exists(gen_path):
                try:
                    from utils.syncnet import syncnet_eval as _syncnet_eval
                    temp_dir = os.path.join(output_dir, "syncnet_temp")
                    with torch.amp.autocast("cuda", enabled=False):
                        av_offset, sync_d, sync_c = _syncnet_eval(
                            syncnet_model, syncnet_detector, gen_path, temp_dir,
                            detect_results_dir=os.path.join(output_dir, "syncnet_detect_results"),
                        )
                    sync_d_list.append(sync_d)
                    sync_c_list.append(sync_c)
                    logger.info(f"  [{mode}][{video_id}] Sync-D={sync_d:.3f}, Sync-C={sync_c:.3f}")
                except Exception as e:
                    logger.warning(f"  [{mode}][{video_id}] SyncNet failed: {e}")

            # Collect video for wandb
            if wandb_run is not None:
                import wandb
                try:
                    wandb_videos.append(
                        wandb.Video(gen_path, caption=f"{mode}/{video_id}", fps=25, format="mp4")
                    )
                except Exception:
                    pass

            if gt_path is not None:
                save_gt_video(
                    {"video": batch["video"], "audio_path": batch["audio_path"]},
                    0, gt_path,
                )
                logger.info(f"[val] Saved GT: {gt_path}")

            # Free per-sample tensors
            del kv_caches, crossattn_cache, output_latents, x_t, decoded
            if use_ref_frames:
                del masked_latents, ref_latents_49ch, ref_latents_sink, motion_latents
            else:
                del masked_latents, ref_latents_sink, motion_latents
            del mask_latent, audio_emb, context
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"[val] Failed on sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Log validation metrics + videos to wandb
    if wandb_run is not None:
        import wandb
        log_dict = {}
        if wandb_videos:
            log_dict[f"val/{mode}_videos"] = wandb_videos
        if sync_d_list:
            avg_sync_d = sum(sync_d_list) / len(sync_d_list)
            avg_sync_c = sum(sync_c_list) / len(sync_c_list)
            log_dict[f"val/{mode}_sync_d"] = avg_sync_d
            log_dict[f"val/{mode}_sync_c"] = avg_sync_c
            logger.info(f"  [{mode}] Avg Sync-D={avg_sync_d:.3f}, Avg Sync-C={avg_sync_c:.3f}")
        if log_dict:
            wandb_run.log(log_dict, step=global_step)

    # Restore model state
    if was_training:
        model.train()

    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[val] {mode} validation complete. Outputs in {step_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Accelerate handles DDP, mixed precision, and device placement
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Increase NCCL timeout so non-main ranks don't die during long validation runs
    from datetime import timedelta
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.get("gradient_accumulation", 2),
        kwargs_handlers=[ddp_kwargs, pg_kwargs],
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    # ── Wandb init ───────────────────────────────────────────────────────
    wandb_run = None
    if config.get("use_wandb", False) and is_main:
        try:
            import wandb
            api_key = config.get("wandb_api_key") or os.environ.get("WANDB_API_KEY")
            if api_key:
                try:
                    wandb.login(key=api_key, relogin=True)
                except Exception as e:
                    logger.warning(f"[wandb] login failed: {e}")
            wandb_run = wandb.init(
                project=config.get("wandb_project", "LiveAvatar-LipSync"),
                entity=config.get("wandb_entity"),
                name=config.get("wandb_run_name") or os.path.basename(config.get("output_dir", "lipsync")),
                config=config,
                id=config.get("wandb_run_id"),
                resume="allow" if config.get("wandb_run_id") else None,
                dir=config.get("output_dir", "outputs"),
            )
        except Exception as e:
            logger.warning(f"[wandb] init failed: {e}. Continuing without wandb.")
            wandb_run = None

    if is_main:
        logger.info(f"Config: {config}")

    # ── Load models ───────────────────────────────────────────────────────
    model = load_and_prepare_model(config, device)

    # Log gradient status per component for memory analysis
    if is_main:
        grad_summary = {}
        for name, param in model.named_parameters():
            # Group by top-level component (e.g. "base_model.model.blocks.0.self_attn.q")
            prefix = name.split(".")[0]
            if prefix not in grad_summary:
                grad_summary[prefix] = {"trainable": 0, "frozen": 0}
            if param.requires_grad:
                grad_summary[prefix]["trainable"] += param.numel()
            else:
                grad_summary[prefix]["frozen"] += param.numel()
        logger.info("=== Gradient status by component ===")
        for comp, counts in sorted(grad_summary.items()):
            t_mb = counts["trainable"] / 1e6
            f_mb = counts["frozen"] / 1e6
            logger.info(f"  {comp}: trainable={t_mb:.1f}M, frozen={f_mb:.1f}M")
        logger.info("====================================")

        # Also log GPU memory after model load
        alloc_gb = torch.cuda.memory_allocated(device) / 1e9
        reserved_gb = torch.cuda.memory_reserved(device) / 1e9
        logger.info(f"GPU memory after model load: allocated={alloc_gb:.1f}GB, reserved={reserved_gb:.1f}GB")

    # Load VAE, audio encoder, T5 -- all frozen, loaded on CPU
    # They are moved to GPU only during preprocessing in training_step, then offloaded back
    # When using precomputed data, encoders are not needed for training (only for validation)
    use_precomputed = config.get("use_precomputed", False)
    if use_precomputed:
        if is_main:
            logger.info("Using precomputed data — skipping encoder loading for training")
            logger.info("  (Encoders will be loaded on-demand for validation only)")
        # Lazy-load encoders only when validation needs them
        vae = None
        audio_encoder = None
        t5_encoder = None
    else:
        vae = load_vae(config, "cpu")
        audio_encoder = load_audio_encoder(config, "cpu")
        t5_encoder = load_t5_encoder(config, "cpu")

    # ── Scheduler setup ───────────────────────────────────────────────────
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3)
    scheduler.set_timesteps(4, device=device)
    restricted_timesteps = scheduler.timesteps.clone()  # 4 values
    restricted_sigmas = scheduler.sigmas[:4].clone()

    if is_main:
        logger.info(f"Restricted timesteps: {restricted_timesteps}")
        logger.info(f"Restricted sigmas: {restricted_sigmas}")

    # Optional: Gaussian timestep loss weighting (Self-Forcing flow_match.py:56-61)
    # Disabled by default — uniform weighting is standard for flow matching velocity prediction.
    # When enabled, creates a 1000-step schedule and looks up Gaussian weights at our 4 positions.
    timestep_weights = None
    if config.get("timestep_loss_weight", False):
        _temp_sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3)
        _temp_sched.set_timesteps(1000, device=device)
        _full_ts = _temp_sched.timesteps.float()
        _N = 1000
        _y = torch.exp(-2 * ((_full_ts - _N / 2) / _N) ** 2)
        _y_shifted = _y - _y.min()
        _full_weights = _y_shifted * (_N / _y_shifted.sum())
        timestep_weights = torch.zeros(len(restricted_timesteps), device=device)
        for _i, _t in enumerate(restricted_timesteps):
            _idx = torch.argmin((_full_ts - _t.float()).abs())
            timestep_weights[_i] = _full_weights[_idx]
        del _temp_sched, _full_ts, _full_weights, _y, _y_shifted
        if is_main:
            logger.info(f"Timestep weights (Gaussian): {timestep_weights}")

    # ── Dataset + DataLoader ──────────────────────────────────────────────
    from lipsync_dataset import LipSyncDataset, ValLipSyncDataset, PrecomputedLipSyncDataset, lipsync_collate_fn

    if use_precomputed:
        precomputed_csv = config.get("precomputed_csv")
        if not precomputed_csv:
            raise ValueError("use_precomputed=true but precomputed_csv not set in config")
        dataset = PrecomputedLipSyncDataset(
            precomputed_csv=precomputed_csv,
            mask_path=config["mask_path"],
            height=config.get("height", 512),
            width=config.get("width", 512),
        )
        if is_main:
            logger.info(f"Loaded precomputed dataset: {len(dataset)} samples from {precomputed_csv}")
    else:
        dataset = LipSyncDataset(
            data_root=config["data_root"],
            metadata_csv=config.get("metadata_csv"),
            height=config.get("height", 512),
            width=config.get("width", 512),
            num_frames=config.get("num_frames", 81),
            mask_path=config["mask_path"],
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        collate_fn=lipsync_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ── Validation datasets (optional) ───────────────────────────────────
    val_recon_dataset = None
    val_mixed_dataset = None
    val_common_kwargs = dict(
        data_root=config["data_root"],
        height=config.get("height", 512),
        width=config.get("width", 512),
        num_frames=config.get("num_frames", 81),
        mask_path=config["mask_path"],
    )
    if config.get("val_recon_csv"):
        val_recon_dataset = ValLipSyncDataset(
            metadata_csv=config["val_recon_csv"], **val_common_kwargs
        )
        if is_main:
            logger.info(f"Loaded recon validation dataset: {len(val_recon_dataset)} samples")
    if config.get("val_mixed_csv"):
        val_mixed_dataset = ValLipSyncDataset(
            metadata_csv=config["val_mixed_csv"], **val_common_kwargs
        )
        if is_main:
            logger.info(f"Loaded mixed validation dataset: {len(val_mixed_dataset)} samples")

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.get("learning_rate", 5e-5),
        betas=(0.9, 0.999),
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Prepare model, optimizer, dataloader with Accelerate
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Access the underlying CausalWanModel_S2V (unwrap DDP + PEFT)
    def get_base_model(m):
        m = accelerator.unwrap_model(m)
        if hasattr(m, "base_model"):
            m = m.base_model.model
        return m

    # ── Initialize KV caches ─────────────────────────────────────────────
    num_layers = get_base_model(model).num_layers  # 40
    batch_size = config.get("batch_size", 1)
    height = config.get("height", 512)
    width = config.get("width", 512)
    H_lat = height // 8
    W_lat = width // 8
    frame_seq_length = (H_lat // 2) * (W_lat // 2)
    # 21 latent frames x frame_seq_length tokens/frame
    kv_cache_size = 21 * frame_seq_length
    kv_cache = initialize_kv_cache(num_layers, batch_size, kv_cache_size, torch.bfloat16, device)
    crossattn_cache = initialize_crossattn_cache(num_layers, batch_size, torch.bfloat16, device)

    # ── DEBUG: Gradient hooks (commented out — no longer needed) ─────────
    # if is_main:
    #     def _make_grad_hook(name):
    #         def hook(grad):
    #             norm = grad.norm().item()
    #             logger.info(f"[grad] {name}: norm={norm:.6f} shape={list(grad.shape)}")
    #             return grad
    #         return hook
    #
    #     base_for_hooks = get_base_model(model)
    #     for layer_idx in [0, 20, 39]:
    #         block = base_for_hooks.blocks[layer_idx]
    #         attn = block.self_attn
    #         for proj_name in ['q', 'k', 'v', 'o']:
    #             proj = getattr(attn, proj_name)
    #             if hasattr(proj, 'lora_A'):
    #                 for adapter_name, lora_mod in proj.lora_A.items():
    #                     lora_mod.weight.register_hook(
    #                         _make_grad_hook(f"layer{layer_idx}.{proj_name}.lora_A.{adapter_name}")
    #                     )
    #                 for adapter_name, lora_mod in proj.lora_B.items():
    #                     lora_mod.weight.register_hook(
    #                         _make_grad_hook(f"layer{layer_idx}.{proj_name}.lora_B.{adapter_name}")
    #                     )
    #     logger.info("[debug] Registered gradient hooks on q/k/v/o LoRA for layers [0, 20, 39]")
    #
    #     audio_inj = base_for_hooks.audio_injector
    #     for inj_idx in [0, 5, 11]:
    #         inj = audio_inj.injector[inj_idx]
    #         for proj_name in ['q', 'k', 'v', 'o']:
    #             proj = getattr(inj, proj_name)
    #             if hasattr(proj, 'lora_A'):
    #                 for adapter_name, lora_mod in proj.lora_A.items():
    #                     lora_mod.weight.register_hook(
    #                         _make_grad_hook(f"audio_inj{inj_idx}.{proj_name}.lora_A.{adapter_name}")
    #                     )
    #                 for adapter_name, lora_mod in proj.lora_B.items():
    #                     lora_mod.weight.register_hook(
    #                         _make_grad_hook(f"audio_inj{inj_idx}.{proj_name}.lora_B.{adapter_name}")
    #                     )
    #     logger.info("[debug] Registered gradient hooks on audio_injector LoRA for injectors [0, 5, 11]")

    # ── Training loop ─────────────────────────────────────────────────────
    output_dir = config.get("output_dir", "outputs/lipsync_train")
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = config.get("num_epochs", 50)
    save_steps = config.get("save_steps", 500)
    log_steps = config.get("log_steps", 10)
    val_steps = config.get("val_steps", 200)
    val_num_samples = config.get("val_num_samples", 5)
    val_output_dir = config.get("val_output_dir", os.path.join(output_dir, "val_outputs"))
    max_grad_norm = config.get("max_grad_norm", 1.0)

    global_step = 0

    # Loss tracking for wandb
    ema_loss = 0.0
    ema_beta = 0.99
    window_loss_sum = 0.0
    window_loss_count = 0
    cum_loss_sum = 0.0
    cum_loss_count = 0
    wandb_log_every = config.get("wandb_log_every", 1)

    grad_accum = config.get("gradient_accumulation", 1)
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum)

    # ── Lazy encoder loading for validation in precomputed mode ─────────
    def ensure_encoders_loaded():
        """Lazy-load VAE, audio encoder, and T5 for validation (precomputed mode only)."""
        nonlocal vae, audio_encoder, t5_encoder
        if vae is None:
            logger.info("[lazy] Loading VAE for validation...")
            vae = load_vae(config, "cpu")
        if audio_encoder is None:
            logger.info("[lazy] Loading audio encoder for validation...")
            audio_encoder = load_audio_encoder(config, "cpu")
        if t5_encoder is None:
            logger.info("[lazy] Loading T5 encoder for validation...")
            t5_encoder = load_t5_encoder(config, "cpu")

    # ── Pre-training validation (baseline checkpoint behavior) ─────────
    if config.get("validate_before_training", False):
        accelerator.wait_for_everyone()
        if is_main:
            logger.info("[ValBootstrap] Running pre-training validation at step 0...")
            ensure_encoders_loaded()
            unwrapped = accelerator.unwrap_model(model)
            if val_recon_dataset is not None:
                validate(
                    model=unwrapped, vae=vae,
                    audio_encoder=audio_encoder, t5_encoder=t5_encoder,
                    val_dataset=val_recon_dataset, scheduler=scheduler,
                    config=config, device=device,
                    global_step=0, output_dir=val_output_dir,
                    mode="recon", max_samples=val_num_samples,
                    wandb_run=wandb_run,
                    offload_kv_cache=True,
                )
            if val_mixed_dataset is not None:
                validate(
                    model=unwrapped, vae=vae,
                    audio_encoder=audio_encoder, t5_encoder=t5_encoder,
                    val_dataset=val_mixed_dataset, scheduler=scheduler,
                    config=config, device=device,
                    global_step=0, output_dir=val_output_dir,
                    mode="mixed", max_samples=val_num_samples,
                    wandb_run=wandb_run,
                    offload_kv_cache=True,
                )
            unwrapped.base_model.model.gradient_checkpointing = True
            logger.info("[ValBootstrap] Pre-training validation complete.")
        accelerator.wait_for_everyone()

    for epoch in range(num_epochs):
        pbar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch}",
            disable=not is_main,
        )
        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()

            with accelerator.accumulate(model):
                # # ── DEBUG: Pre-iteration memory (Step 2A) ────────────
                # _mem_before = torch.cuda.memory_allocated(device) / 1e9
                # logger.info(
                #     f"[mem] iter batch_idx={batch_idx} START: {_mem_before:.1f}GB alloc, "
                #     f"{torch.cuda.memory_reserved(device)/1e9:.1f}GB reserved"
                # )
                # logger.info(
                #     f"[diag] kv_cache[0]['k'] ptr={kv_cache[0]['k'].data_ptr()} "
                #     f"id={id(kv_cache[0]['k'])} ver={kv_cache[0]['k']._version}"
                # )

                optimizer.zero_grad()

                # training_step accumulates block predictions then calls backward once,
                # returning a float (backward already done)
                loss_tensor = training_step(
                    batch=batch,
                    model=accelerator.unwrap_model(model),
                    vae=vae,
                    audio_encoder=audio_encoder,
                    t5_encoder=t5_encoder,
                    restricted_timesteps=restricted_timesteps,
                    restricted_sigmas=restricted_sigmas,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    config=config,
                    device=device,
                    accelerator=accelerator,
                    timestep_weights=timestep_weights,
                )

                # Gather loss across all GPUs for consistent logging
                avg_loss = accelerator.gather(loss_tensor.unsqueeze(0)).mean().item()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)

                optimizer.step()

                gc.collect()
                torch.cuda.empty_cache()
                # # ── DEBUG: Post-iteration memory (Step 2B) ───────────
                # _mem_after = torch.cuda.memory_allocated(device) / 1e9
                # logger.info(
                #     f"[mem] iter batch_idx={batch_idx} END: {_mem_after:.1f}GB alloc, "
                #     f"{torch.cuda.memory_reserved(device)/1e9:.1f}GB reserved, "
                #     f"delta={_mem_after - _mem_before:+.1f}GB"
                # )
                # logger.info(
                #     f"[diag] post-step kv_cache[0]['k'] ptr={kv_cache[0]['k'].data_ptr()} "
                #     f"id={id(kv_cache[0]['k'])} ver={kv_cache[0]['k']._version}"
                # )

            if accelerator.sync_gradients:
                global_step += 1
                pbar.update(1)

                # # ── DEBUG: CUDA memory snapshot (commented out) ─────
                # if is_main and global_step == 1:
                #     _stats = torch.cuda.memory_stats(device)
                #     for _stat_key in [
                #         'allocated_bytes.all.current', 'allocated_bytes.all.peak',
                #         'reserved_bytes.all.current', 'active_bytes.all.current',
                #         'num_alloc_retries',
                #     ]:
                #         logger.info(f"[cuda_stats] {_stat_key}: {_stats.get(_stat_key, 'N/A')}")
                #
                #     import gc as gc_module
                #     gc_module.collect()
                #     _gpu_tensors = []
                #     for _obj in gc_module.get_objects():
                #         try:
                #             if torch.is_tensor(_obj) and _obj.is_cuda:
                #                 _gpu_tensors.append(
                #                     (list(_obj.shape), str(_obj.dtype),
                #                      _obj.element_size() * _obj.nelement(),
                #                      _obj.requires_grad, _obj.grad_fn is not None)
                #                 )
                #         except Exception:
                #             pass
                #     _gpu_tensors.sort(key=lambda x: -x[2])
                #     _total_bytes = sum(t[2] for t in _gpu_tensors)
                #     logger.info(
                #         f"[cuda_audit] {len(_gpu_tensors)} GPU tensors, "
                #         f"total={_total_bytes/1e9:.1f}GB"
                #     )
                #     for _shape, _dtype, _nbytes, _req_grad, _has_fn in _gpu_tensors[:30]:
                #         logger.info(
                #             f"[cuda_audit]   {_nbytes/1e6:.1f}MB {_dtype} {_shape} "
                #             f"req_grad={_req_grad} has_grad_fn={_has_fn}"
                #         )

                # ── Loss tracking ─────────────────────────────────────
                step_loss = avg_loss

                if global_step == 1:
                    ema_loss = step_loss
                else:
                    ema_loss = ema_beta * ema_loss + (1 - ema_beta) * step_loss

                cum_loss_sum += step_loss
                cum_loss_count += 1
                window_loss_sum += step_loss
                window_loss_count += 1

                # ── tqdm + console logging ─────────────────────────────
                elapsed = time.time() - step_start
                gpu_gb = torch.cuda.memory_allocated(device) / 1e9
                pbar.set_postfix(
                    step=global_step,
                    loss=f"{step_loss:.4f}",
                    ema=f"{ema_loss:.4f}",
                    gpu=f"{gpu_gb:.0f}G",
                    t=f"{elapsed:.0f}s",
                )

                if is_main and global_step % log_steps == 0:
                    logger.info(
                        f"Epoch {epoch} | Step {global_step} | Loss {step_loss:.4f} | "
                        f"EMA {ema_loss:.4f} | GPU {gpu_gb:.1f}GB | Time {elapsed:.1f}s"
                    )

                # ── wandb logging ──────────────────────────────────────
                if wandb_run is not None and global_step % wandb_log_every == 0:
                    wandb_log = {
                        "loss/step": step_loss,
                        "loss/ema": ema_loss,
                        "loss/window_mean": window_loss_sum / max(window_loss_count, 1),
                        "loss/cum_mean": cum_loss_sum / max(cum_loss_count, 1),
                        "train/epoch": epoch,
                        "mem/gpu_allocated_gb": gpu_gb,
                        "mem/gpu_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
                    }
                    wandb_run.log(wandb_log, step=global_step)
                    # Reset window
                    window_loss_sum = 0.0
                    window_loss_count = 0

                if is_main and global_step % save_steps == 0:
                    unwrapped = accelerator.unwrap_model(model)
                    save_checkpoint(unwrapped, optimizer, global_step, output_dir)

                # ── Validation (all ranks barrier) ───────────────────
                if global_step % val_steps == 0:
                    # All ranks pause here so non-main ranks don't race
                    # ahead into the next training step during validation
                    accelerator.wait_for_everyone()
                    if is_main:
                        ensure_encoders_loaded()
                        unwrapped = accelerator.unwrap_model(model)
                        # Free training KV cache memory before validation
                        reset_kv_cache(kv_cache)
                        reset_crossattn_cache(crossattn_cache, batch_size)
                        gc.collect()
                        torch.cuda.empty_cache()
                        if val_recon_dataset is not None:
                            validate(
                                model=unwrapped, vae=vae,
                                audio_encoder=audio_encoder, t5_encoder=t5_encoder,
                                val_dataset=val_recon_dataset, scheduler=scheduler,
                                config=config, device=device,
                                global_step=global_step, output_dir=val_output_dir,
                                mode="recon", max_samples=val_num_samples,
                                wandb_run=wandb_run,
                                offload_kv_cache=True,
                            )
                        if val_mixed_dataset is not None:
                            validate(
                                model=unwrapped, vae=vae,
                                audio_encoder=audio_encoder, t5_encoder=t5_encoder,
                                val_dataset=val_mixed_dataset, scheduler=scheduler,
                                config=config, device=device,
                                global_step=global_step, output_dir=val_output_dir,
                                mode="mixed", max_samples=val_num_samples,
                                wandb_run=wandb_run,
                                offload_kv_cache=True,
                            )
                        # Re-enable gradient checkpointing after eval mode toggle
                        unwrapped.base_model.model.gradient_checkpointing = True
                    accelerator.wait_for_everyone()

        pbar.close()

    # Final save
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint(unwrapped, optimizer, global_step, output_dir)

        if wandb_run is not None:
            wandb_run.summary["final/loss_ema"] = ema_loss
            wandb_run.summary["final/loss_cum_mean"] = cum_loss_sum / max(cum_loss_count, 1)
            wandb_run.summary["final/global_step"] = global_step
            wandb_run.finish()

        logger.info("Training complete.")


if __name__ == "__main__":
    main()
