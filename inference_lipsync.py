#!/usr/bin/env python3
"""Standalone inference for lipsync I2V model.

Two modes:
  full            — 4-step denoising from noise (port of validate())
  single_timestep — noise at one timestep, one forward pass, recover x_0
"""

import argparse
import gc
import logging
import os
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Reuse utilities from training
from train_lipsync import (
    load_vae, load_audio_encoder, load_t5_encoder,
    vae_encode_batch, move_vae,
    initialize_kv_cache, initialize_crossattn_cache,
    clone_kv_cache, move_kv_cache_to_device,
    save_video_with_audio, save_gt_video,
    zero_pad_to_49ch, zero_pad_to_33ch,
    construct_49ch_block, construct_33ch_block,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(config, checkpoint_path, device):
    """Load base model with trained LoRA weights from a checkpoint.

    Unlike load_and_prepare_model (training), this loads *trained* LoRA weights
    from a checkpoint instead of initializing fresh LoRA for training.
    """
    from peft import PeftModel, LoraConfig, get_peft_model
    from safetensors.torch import load_file as load_safetensors

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LiveAvatar"))
    from liveavatar.models.wan.causal_model_s2v import CausalWanModel_S2V

    training_mode = config.get("training_mode", "v2v")
    is_i2v = (training_mode == "i2v")
    use_ref_frames = config.get("use_ref_frames", True)
    inpaint_mode = config.get("inpaint_mode", "concat")
    dmd_lora_path = config.get("dmd_lora_path")
    merge_dmd = config.get("merge_dmd_lora", False)

    # 1. Load base model (16ch)
    logger.info(f"Loading base model from {config['checkpoint_dir']}")
    model = CausalWanModel_S2V.from_pretrained(
        config["checkpoint_dir"],
        torch_dtype=torch.bfloat16,
    )

    # 2. Merge DMD LoRA into base weights if configured (same as training setup)
    if dmd_lora_path and merge_dmd:
        dmd_state = load_safetensors(dmd_lora_path)
        dmd_state = {f"base_model.model.{k}": v for k, v in dmd_state.items() if "lora" in k}
        dmd_targets = ["q", "k", "v", "o", "ffn.0", "ffn.2"]
        dmd_config = LoraConfig(r=128, lora_alpha=64.0, init_lora_weights=True, target_modules=dmd_targets)
        model = get_peft_model(model, dmd_config)
        model.load_state_dict(dmd_state, strict=False)
        model = model.merge_and_unload()
        logger.info(f"DMD LoRA merged from {dmd_lora_path}")

    # 3. Expand patch_embedding if needed (non-I2V modes)
    if is_i2v or inpaint_mode == "direct":
        in_channels = 16
    else:
        in_channels = 49 if use_ref_frames else 33

    if in_channels != 16:
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

    # 4. Load trained LoRA from checkpoint
    lora_dir = os.path.join(checkpoint_path, "lora")
    logger.info(f"Loading trained LoRA from {lora_dir}")
    model = PeftModel.from_pretrained(model, lora_dir)

    # 5. Load trained patch_embedding (non-I2V only; I2V keeps frozen)
    if not is_i2v:
        pe_path = os.path.join(checkpoint_path, "patch_embedding.pt")
        if os.path.exists(pe_path):
            pe_state = torch.load(pe_path, map_location="cpu", weights_only=True)
            model.base_model.model.patch_embedding.load_state_dict(pe_state)
            logger.info(f"Loaded trained patch_embedding from {pe_path}")

    # 6. Set to eval mode
    model = model.to(device)
    model.eval()
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    base_model.gradient_checkpointing = False

    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total/1e6:.1f}M params")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Shared encoding
# ──────────────────────────────────────────────────────────────────────────────

def encode_sample(sample, vae, audio_encoder, t5_encoder, config, device,
                  encode_gt_latents=False):
    """Encode a single validation sample (VAE, audio, T5).

    Args:
        encode_gt_latents: If True, also VAE-encode the GT video to get x_0 latents
            (needed for single_timestep mode).

    Returns dict with all encoded tensors.
    """
    use_ref_frames = config.get("use_ref_frames", True)
    inpaint_mode = config.get("inpaint_mode", "concat")
    training_mode = config.get("training_mode", "v2v")
    is_i2v = (training_mode == "i2v")
    num_blocks = 7
    latent_frames_per_block = 3
    motion_frames_video = 73
    num_latent_frames = 21

    height = config.get("height", 512)
    width = config.get("width", 512)
    H_lat = height // 8
    W_lat = width // 8

    # Wrap in batch dimension
    batch = {
        k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]
        for k, v in sample.items()
    }

    # ── VAE encoding ──────────────────────────────────────────────────
    move_vae(vae, device)
    gt_latents = None
    masked_latents = None
    ref_latents_49ch = None
    mask_latent = None
    mouth_mask = None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        video = batch["video"].to(device)          # [1, 3, 81, H, W]
        ref_frames = batch["ref_frames"].to(device)  # [1, 3, 81, H, W]

        if encode_gt_latents:
            gt_latents = vae_encode_batch(vae, video)  # [1, 16, 21, H_lat, W_lat]

        if is_i2v:
            ref_single = video[:, :, 0:1, :, :]
        else:
            mouth_mask = batch["mouth_mask"].to(device)
            mask_pixel = mouth_mask.unsqueeze(2).expand(-1, -1, 81, -1, -1)
            masked_video = video * mask_pixel
            masked_latents = vae_encode_batch(vae, masked_video)

            if use_ref_frames and inpaint_mode != "direct":
                ref_latents_49ch = vae_encode_batch(vae, ref_frames)

            ref_single = ref_frames[:, :, 0:1, :, :]

            mask_for_latent = mask_pixel.float()
            mask_latent = F.interpolate(
                mask_for_latent, size=(num_latent_frames, H_lat, W_lat),
                mode="trilinear", align_corners=False,
            )
            del masked_video, mask_pixel

        # Common: ref_latents_sink and motion_latents
        ref_5frames = ref_single.repeat(1, 1, 5, 1, 1)
        ref_latents_sink = vae_encode_batch(vae, ref_5frames)[:, :, 1:]

        motion_pixel = ref_single.repeat(1, 1, motion_frames_video, 1, 1)
        motion_latents = vae_encode_batch(vae, motion_pixel)
        del ref_5frames, motion_pixel

    move_vae(vae, "cpu")
    torch.cuda.empty_cache()

    # ── Audio extraction ──────────────────────────────────────────────
    audio_fps = config.get("audio_fps", 25)
    audio_batch_frames = num_blocks * latent_frames_per_block * 4  # 84

    audio_encoder.model.to(device)
    z = audio_encoder.extract_audio_feat(
        batch["audio_path"][0], return_all_layers=True
    )
    audio_bucket, _ = audio_encoder.get_audio_embed_bucket_fps(
        z, fps=audio_fps, batch_frames=audio_batch_frames, m=0
    )
    audio_bucket = audio_bucket[:audio_batch_frames]
    audio_emb = audio_bucket.unsqueeze(0).permute(0, 2, 3, 1)  # [1, 25, 1024, 84]
    audio_emb = audio_emb.to(device, dtype=torch.bfloat16)
    audio_encoder.model.to("cpu")
    torch.cuda.empty_cache()

    # ── Text encoding ─────────────────────────────────────────────────
    t5_encoder.model.to(device)
    context = t5_encoder(batch["text"], device)
    t5_encoder.model.to("cpu")
    torch.cuda.empty_cache()

    # ── Prepare tensors ───────────────────────────────────────────────
    if not is_i2v:
        masked_latents = masked_latents.to(device, dtype=torch.bfloat16)
        if use_ref_frames and inpaint_mode != "direct":
            ref_latents_49ch = ref_latents_49ch.to(device, dtype=torch.bfloat16)
        mask_latent = mask_latent.to(device, dtype=torch.bfloat16)
    ref_latents_sink = ref_latents_sink.to(device, dtype=torch.bfloat16)
    motion_latents = motion_latents.to(device, dtype=torch.bfloat16)

    if is_i2v or inpaint_mode == "direct":
        ref_latents_padded = ref_latents_sink
    elif use_ref_frames:
        ref_latents_padded = zero_pad_to_49ch(ref_latents_sink)
    else:
        ref_latents_padded = zero_pad_to_33ch(ref_latents_sink)

    zeros_cond = torch.zeros(1, 16, latent_frames_per_block, H_lat, W_lat,
                             device=device, dtype=torch.bfloat16)

    if gt_latents is not None:
        gt_latents = gt_latents.to(device, dtype=torch.bfloat16)

    return {
        "batch": batch,
        "ref_latents_sink": ref_latents_sink,
        "motion_latents": motion_latents,
        "ref_latents_padded": ref_latents_padded,
        "audio_emb": audio_emb,
        "context": context,
        "masked_latents": masked_latents,
        "ref_latents_49ch": ref_latents_49ch,
        "mask_latent": mask_latent,
        "mouth_mask": mouth_mask,
        "zeros_cond": zeros_cond,
        "gt_latents": gt_latents,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Full 4-step denoising (port of validate())
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_full_inference(model, vae, audio_encoder, t5_encoder,
                       val_dataset, scheduler, config, device,
                       output_dir, max_samples, val_csv_mode="recon"):
    """Full 4-step denoising inference (port of validate())."""
    use_ref_frames = config.get("use_ref_frames", True)
    inpaint_mode = config.get("inpaint_mode", "concat")
    training_mode = config.get("training_mode", "v2v")
    is_i2v = (training_mode == "i2v")
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

    os.makedirs(output_dir, exist_ok=True)

    # Set up scheduler for 4-step inference
    scheduler.set_timesteps(4, device=device)
    timesteps = scheduler.timesteps

    # Access base model
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    base_model.block_mask = None
    num_layers = base_model.num_layers

    num_samples = min(max_samples, len(val_dataset))
    logger.info(f"[full] Starting inference: {num_samples} samples, 4 timesteps")

    for sample_idx in range(num_samples):
        try:
            sample = val_dataset[sample_idx]
            video_id = sample.get("video_id", f"sample_{sample_idx}")
            audio_id = sample.get("audio_id", video_id)

            logger.info(f"[full] Processing sample {sample_idx+1}/{num_samples}: {video_id}")

            # ── Encode inputs ─────────────────────────────────────────
            encoded = encode_sample(sample, vae, audio_encoder, t5_encoder, config, device)
            batch = encoded["batch"]

            # ── Prefill KV cache ──────────────────────────────────────
            kv_cache_size = num_latent_frames * frame_seq_length
            kv_cache_gpu = initialize_kv_cache(num_layers, 1, kv_cache_size, torch.bfloat16, device)
            crossattn_cache = initialize_crossattn_cache(num_layers, 1, torch.bfloat16, device)
            base_model.block_mask = None

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                dummy_x = [torch.zeros(16, latent_frames_per_block, H_lat, W_lat,
                                       device=device, dtype=torch.bfloat16)]
                base_model._forward_sink(
                    x=dummy_x,
                    t=torch.zeros(1, latent_frames_per_block, device=device, dtype=torch.bfloat16),
                    context=encoded["context"],
                    seq_len=None,
                    cond_states=encoded["zeros_cond"],
                    motion_latents=encoded["motion_latents"],
                    ref_latents=encoded["ref_latents_padded"],
                    audio_input=encoded["audio_emb"][..., 0:12],
                    motion_frames=[motion_frames_video, lat_motion_frames],
                    drop_motion_frames=False,
                    sink_flag=True,
                    kv_cache=kv_cache_gpu,
                    crossattn_cache=crossattn_cache,
                    current_start=0,
                    current_end=latent_frames_per_block * frame_seq_length,
                )

            # Clone prefilled cache for all 4 timesteps
            kv_caches = {}
            for ts_idx in range(4):
                kv_caches[str(ts_idx + 1)] = clone_kv_cache(kv_cache_gpu, device)
            del kv_cache_gpu
            torch.cuda.empty_cache()

            # ── Initialize noise ──────────────────────────────────────
            if inpaint_mode == "direct" and not is_i2v:
                noise = torch.randn(1, 16, num_latent_frames, H_lat, W_lat,
                                    device=device, dtype=torch.bfloat16)
                mouth_indicator = (1.0 - encoded["mask_latent"]).to(dtype=torch.bfloat16)
                x_t = encoded["masked_latents"] * (1.0 - mouth_indicator) + noise * mouth_indicator
            else:
                x_t = torch.randn(1, 16, num_latent_frames, H_lat, W_lat,
                                  device=device, dtype=torch.bfloat16)
            output_latents = torch.zeros_like(x_t)

            # ── Block-wise denoising (7 blocks x 4 timesteps) ─────────
            for block_idx in range(num_blocks):
                block_start = block_idx * latent_frames_per_block
                block_end = (block_idx + 1) * latent_frames_per_block

                if not is_i2v:
                    mask_block = encoded["mask_latent"][:, :, block_start:block_end]
                    masked_block = encoded["masked_latents"][:, :, block_start:block_end]
                    if use_ref_frames and inpaint_mode != "direct":
                        ref_block = encoded["ref_latents_49ch"][:, :, block_start:block_end]
                audio_block = encoded["audio_emb"][..., block_idx * 12 : (block_idx + 1) * 12]

                block_latents = x_t[:, :, block_start:block_end]

                # Reset scheduler state for this block
                scheduler._step_index = 0
                scheduler._begin_index = 0

                for i_ts, t_val in enumerate(timesteps):
                    cache_key = str(i_ts + 1)

                    # Construct channel input
                    if is_i2v or inpaint_mode == "direct":
                        x_input = block_latents
                    elif use_ref_frames:
                        x_input = construct_49ch_block(block_latents, mask_block, masked_block, ref_block)
                    else:
                        x_input = construct_33ch_block(block_latents, mask_block, masked_block)

                    t_block = torch.tensor([t_val] * latent_frames_per_block,
                                           device=device).unsqueeze(0).float()
                    x_list = [x_input[0]]

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        velocity_list = base_model._forward_inference(
                            x=x_list,
                            t=t_block,
                            context=encoded["context"],
                            seq_len=None,
                            cond_states=encoded["zeros_cond"],
                            motion_latents=encoded["motion_latents"],
                            ref_latents=encoded["ref_latents_padded"],
                            audio_input=audio_block,
                            motion_frames=[motion_frames_video, lat_motion_frames],
                            drop_motion_frames=False,
                            kv_cache=kv_caches[cache_key],
                            crossattn_cache=crossattn_cache,
                            current_start=block_start * frame_seq_length,
                            current_end=block_end * frame_seq_length,
                        )

                    velocity_pred = velocity_list[0].unsqueeze(0)

                    block_latents = scheduler.step(
                        velocity_pred, t_val, block_latents,
                        return_dict=False,
                    )[0]

                    if inpaint_mode == "direct" and not is_i2v:
                        block_latents = masked_block * mask_block + block_latents * (1.0 - mask_block)

                output_latents[:, :, block_start:block_end] = block_latents

            # ── VAE decode ────────────────────────────────────────────
            move_vae(vae, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                decode_input = torch.cat([encoded["motion_latents"], output_latents], dim=2)
                decoded_list = vae.decode([decode_input[0]])
                decoded = torch.stack(decoded_list)
                decoded = decoded[:, :, -(81):]
                decoded = decoded[:, :, 3:]
            move_vae(vae, "cpu")
            torch.cuda.empty_cache()

            # ── Save outputs ──────────────────────────────────────────
            if val_csv_mode == "recon":
                gen_path = os.path.join(output_dir, f"{video_id}_gen.mp4")
                gt_path = os.path.join(output_dir, f"{video_id}_gt.mp4")
            else:
                gen_path = os.path.join(output_dir, f"v{video_id}_a{audio_id}_gen.mp4")
                gt_path = None

            save_video_with_audio(decoded.cpu().float(), batch["audio_path"][0], gen_path)
            logger.info(f"[full] Saved generated: {gen_path}")

            if inpaint_mode == "direct" and not is_i2v:
                video_dev = batch["video"].to(device)
                gt_cropped = video_dev[:, :, 3:]
                mask_pixel = encoded["mouth_mask"].unsqueeze(2).expand(-1, -1, gt_cropped.shape[2], -1, -1)
                replaced = gt_cropped * mask_pixel + decoded.to(device) * (1.0 - mask_pixel)
                replaced_path = gen_path.replace("_gen.mp4", "_replaced.mp4")
                save_video_with_audio(replaced.cpu().float(), batch["audio_path"][0], replaced_path)
                logger.info(f"[full] Saved replaced: {replaced_path}")

            if gt_path is not None:
                save_gt_video(
                    {"video": batch["video"], "audio_path": batch["audio_path"]},
                    0, gt_path,
                )
                logger.info(f"[full] Saved GT: {gt_path}")

            # Cleanup
            del kv_caches, crossattn_cache, output_latents, x_t, decoded
            del encoded
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"[full] Failed on sample {sample_idx}: {e}")
            traceback.print_exc()
            continue

    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[full] Inference complete. Outputs in {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Single-timestep inference (training-style)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_single_timestep_inference(model, vae, audio_encoder, t5_encoder,
                                   val_dataset, scheduler, config, device,
                                   output_dir, max_samples, timestep_index=None,
                                   val_csv_mode="recon"):
    """Single-timestep inference: noise at one timestep, one forward pass, recover x_0.

    Mimics the training setting: add noise at one of the 4 restricted timesteps,
    run a single forward pass per block, and recover x_0 directly from the
    velocity prediction.
    """
    use_ref_frames = config.get("use_ref_frames", True)
    inpaint_mode = config.get("inpaint_mode", "concat")
    training_mode = config.get("training_mode", "v2v")
    is_i2v = (training_mode == "i2v")
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

    os.makedirs(output_dir, exist_ok=True)

    # Get restricted timesteps/sigmas (same as training)
    scheduler.set_timesteps(4, device=device)
    restricted_timesteps = scheduler.timesteps.clone()
    restricted_sigmas = scheduler.sigmas[:4].clone()
    logger.info(f"[single_ts] Restricted timesteps: {restricted_timesteps}")
    logger.info(f"[single_ts] Restricted sigmas: {restricted_sigmas}")

    # Access base model
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    base_model.block_mask = None
    num_layers = base_model.num_layers

    num_samples = min(max_samples, len(val_dataset))
    logger.info(f"[single_ts] Starting inference: {num_samples} samples, "
                f"timestep_index={'random' if timestep_index is None else timestep_index}")

    for sample_idx in range(num_samples):
        try:
            sample = val_dataset[sample_idx]
            video_id = sample.get("video_id", f"sample_{sample_idx}")
            audio_id = sample.get("audio_id", video_id)

            # Pick timestep
            if timestep_index is not None:
                choice = timestep_index
            else:
                choice = torch.randint(0, len(restricted_timesteps), (1,)).item()

            sigma = restricted_sigmas[choice].to(device=device, dtype=torch.bfloat16)
            t_val = restricted_timesteps[choice]

            logger.info(f"[single_ts] Processing sample {sample_idx+1}/{num_samples}: "
                        f"{video_id} (ts_idx={choice}, sigma={sigma:.4f})")

            # ── Encode inputs (including GT latents) ──────────────────
            encoded = encode_sample(sample, vae, audio_encoder, t5_encoder, config, device,
                                    encode_gt_latents=True)
            batch = encoded["batch"]
            gt_latents = encoded["gt_latents"]  # [1, 16, 21, H_lat, W_lat]

            # ── Apply noise at chosen timestep ────────────────────────
            noise = torch.randn_like(gt_latents)
            x_t = (1 - sigma) * gt_latents + sigma * noise

            # ── Prefill KV cache ──────────────────────────────────────
            kv_cache_size = num_latent_frames * frame_seq_length
            kv_cache = initialize_kv_cache(num_layers, 1, kv_cache_size, torch.bfloat16, device)
            crossattn_cache = initialize_crossattn_cache(num_layers, 1, torch.bfloat16, device)
            base_model.block_mask = None

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                dummy_x = [torch.zeros(16, latent_frames_per_block, H_lat, W_lat,
                                       device=device, dtype=torch.bfloat16)]
                base_model._forward_sink(
                    x=dummy_x,
                    t=torch.zeros(1, latent_frames_per_block, device=device, dtype=torch.bfloat16),
                    context=encoded["context"],
                    seq_len=None,
                    cond_states=encoded["zeros_cond"],
                    motion_latents=encoded["motion_latents"],
                    ref_latents=encoded["ref_latents_padded"],
                    audio_input=encoded["audio_emb"][..., 0:12],
                    motion_frames=[motion_frames_video, lat_motion_frames],
                    drop_motion_frames=False,
                    sink_flag=True,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=0,
                    current_end=latent_frames_per_block * frame_seq_length,
                )

            # Single KV cache — no cloning (only one timestep)
            output_latents = torch.zeros_like(x_t)

            # ── Block-wise forward pass (7 blocks, 1 timestep each) ───
            for block_idx in range(num_blocks):
                block_start = block_idx * latent_frames_per_block
                block_end = (block_idx + 1) * latent_frames_per_block

                x_t_block = x_t[:, :, block_start:block_end]
                audio_block = encoded["audio_emb"][..., block_idx * 12 : (block_idx + 1) * 12]

                # Construct channel input
                if is_i2v or inpaint_mode == "direct":
                    x_input = x_t_block
                elif use_ref_frames:
                    mask_block = encoded["mask_latent"][:, :, block_start:block_end]
                    masked_block = encoded["masked_latents"][:, :, block_start:block_end]
                    ref_block = encoded["ref_latents_49ch"][:, :, block_start:block_end]
                    x_input = construct_49ch_block(x_t_block, mask_block, masked_block, ref_block)
                else:
                    mask_block = encoded["mask_latent"][:, :, block_start:block_end]
                    masked_block = encoded["masked_latents"][:, :, block_start:block_end]
                    x_input = construct_33ch_block(x_t_block, mask_block, masked_block)

                t_block = torch.tensor([t_val] * latent_frames_per_block,
                                       device=device).unsqueeze(0).float()
                x_list = [x_input[0]]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    velocity_list = base_model._forward_inference(
                        x=x_list,
                        t=t_block,
                        context=encoded["context"],
                        seq_len=None,
                        cond_states=encoded["zeros_cond"],
                        motion_latents=encoded["motion_latents"],
                        ref_latents=encoded["ref_latents_padded"],
                        audio_input=audio_block,
                        motion_frames=[motion_frames_video, lat_motion_frames],
                        drop_motion_frames=False,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=block_start * frame_seq_length,
                        current_end=block_end * frame_seq_length,
                    )

                velocity_pred = velocity_list[0].unsqueeze(0)

                # Recover x_0 directly: x_0 = x_t - sigma * velocity
                x_0_pred = x_t_block - sigma * velocity_pred
                output_latents[:, :, block_start:block_end] = x_0_pred

            # ── VAE decode ────────────────────────────────────────────
            move_vae(vae, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                decode_input = torch.cat([encoded["motion_latents"], output_latents], dim=2)
                decoded_list = vae.decode([decode_input[0]])
                decoded = torch.stack(decoded_list)
                decoded = decoded[:, :, -(81):]
                decoded = decoded[:, :, 3:]
            move_vae(vae, "cpu")
            torch.cuda.empty_cache()

            # ── Save outputs ──────────────────────────────────────────
            ts_dir = os.path.join(output_dir, f"ts{choice}")
            os.makedirs(ts_dir, exist_ok=True)

            if val_csv_mode == "recon":
                gen_path = os.path.join(ts_dir, f"{video_id}_gen.mp4")
                gt_path = os.path.join(ts_dir, f"{video_id}_gt.mp4")
            else:
                gen_path = os.path.join(ts_dir, f"v{video_id}_a{audio_id}_gen.mp4")
                gt_path = None

            save_video_with_audio(decoded.cpu().float(), batch["audio_path"][0], gen_path)
            logger.info(f"[single_ts] Saved generated: {gen_path}")

            if gt_path is not None:
                save_gt_video(
                    {"video": batch["video"], "audio_path": batch["audio_path"]},
                    0, gt_path,
                )
                logger.info(f"[single_ts] Saved GT: {gt_path}")

            # Cleanup
            del kv_cache, crossattn_cache, output_latents, x_t, decoded
            del gt_latents, noise, encoded
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"[single_ts] Failed on sample {sample_idx}: {e}")
            traceback.print_exc()
            continue

    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[single_ts] Inference complete. Outputs in {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inference for lipsync model")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--mode", required=True, choices=["full", "single_timestep"],
                        help="full: 4-step denoising; single_timestep: training-style")
    parser.add_argument("--output_dir", default="inference_outputs",
                        help="Base output directory")
    parser.add_argument("--val_csv", default="recon", choices=["recon", "mixed"],
                        help="Which validation CSV to use")
    parser.add_argument("--max_samples", type=int, default=5,
                        help="Maximum number of validation samples")
    parser.add_argument("--timestep_index", type=int, default=None,
                        help="Which of the 4 restricted timesteps to use "
                             "(single_timestep mode only; default: random per sample)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda")

    # Load models
    logger.info("Loading model from checkpoint...")
    model = load_model_from_checkpoint(config, args.checkpoint, device)

    logger.info("Loading VAE...")
    vae = load_vae(config, "cpu")

    logger.info("Loading audio encoder...")
    audio_encoder = load_audio_encoder(config, "cpu")

    logger.info("Loading T5 encoder...")
    t5_encoder = load_t5_encoder(config, "cpu")

    # Load validation dataset
    from lipsync_dataset import ValLipSyncDataset

    training_mode = config.get("training_mode", "v2v")
    val_csv_key = "val_recon_csv" if args.val_csv == "recon" else "val_mixed_csv"
    val_csv = config.get(val_csv_key)
    if not val_csv:
        raise ValueError(f"No {val_csv_key} in config")

    val_dataset = ValLipSyncDataset(
        metadata_csv=val_csv,
        data_root=config["data_root"],
        height=config.get("height", 512),
        width=config.get("width", 512),
        num_frames=config.get("num_frames", 81),
        mask_path=config.get("mask_path"),
        training_mode=training_mode,
    )
    logger.info(f"Loaded validation dataset: {len(val_dataset)} samples from {val_csv}")

    # Setup scheduler
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3)

    # Dispatch
    if args.mode == "full":
        run_full_inference(
            model, vae, audio_encoder, t5_encoder,
            val_dataset, scheduler, config, device,
            os.path.join(args.output_dir, "full"),
            args.max_samples,
            val_csv_mode=args.val_csv,
        )
    else:
        run_single_timestep_inference(
            model, vae, audio_encoder, t5_encoder,
            val_dataset, scheduler, config, device,
            os.path.join(args.output_dir, "single_timestep"),
            args.max_samples,
            timestep_index=args.timestep_index,
            val_csv_mode=args.val_csv,
        )


if __name__ == "__main__":
    main()
