# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This repository integrates two projects for audio-driven lipsync video generation:

- **LiveAvatar** (`LiveAvatar/`): A pretrained image-to-video (I2V) portrait animation autoregressive video diffusion model based on Wan2.2-S2V-14B (5120 hidden dim, 40 heads, 40 layers). Provides the base architecture and model checkpoint.
- **Self-Forcing LipSync StableAvatar** (external at `/home/work/.local/Self-Forcing_LipSync_StableAvatar/`): A Self-Forcing video diffusion checkpoint being adapted for the lipsync task (V2V inpainting with audio and reference conditioning). Provides the training framework and input style.

**The main goal** is to combine these: use the Self-Forcing repo's input format (49-channel: video latents + masked latents + masks + reference frame latents) and diffusion loss training style, while using LiveAvatar's architecture and model checkpoint. The result is a lipsync V2V inpainting model trained with a modified forward pass for diffusion loss.

**Additional Context**: This is a Python ML/deep learning project focused on lip-sync video generation. Key technologies: PyTorch, DDP training, VAE latent diffusion, ffmpeg for video processing. Primary repos involve LiveAvatar, Self-Forcing, LatentSync, and Wan2.1 codebases.

**Conda environment:** `hb_liveavatar`

## Temporal Compression & Block-wise Generation

The Wan VAE uses temporal stride 4: `Tzip = 1 + (N-1)/4` where N is the number of video frames.
- **81 video frames → 21 latent frames** (1 + 80/4 = 21)
- **Each block = 3 latent frames** (= 12 video frames + 1 overlap)
- **7 blocks per clip** (21 / 3 = 7), NOT 27

This applies to both LiveAvatar and Self-Forcing. All block counts, KV cache sizes, and audio slicing must use 7 blocks / 21 latent frames for 81-frame videos.

## Integration Principle: LiveAvatar-First

The integration strategy is to keep LiveAvatar's model, audio pipeline (wav2vec2), KV cache management, and inference logic **exactly as-is**. The ONLY modification is expanding the input layer from 16 to 49 channels to accept the lipsync conditioning (masked latents + mask + reference latents). Everything else — audio processing, attention, KV caching, block-wise generation — stays faithful to LiveAvatar.

## 49-Channel Input Format (from Self-Forcing LatentSync)

The training input is a 49-channel tensor `[B, 49, Tzip, H/8, W/8]` constructed by concatenating:

| Component | Channels | Description |
|-----------|----------|-------------|
| Noisy latents | 16 | VAE-encoded video frames (diffusion target) |
| Masked latents | 16 | VAE-encoded video with mouth region zeroed |
| Mask | 1 | Fixed mouth-region mask resized to latent space (trilinear interpolation) |
| Reference latents | 16 | VAE-encoded reference frames for facial identity |

Constructed in `train_lipsync.py:construct_49ch_block()`:
- `x_49 = cat([noisy_block(16), mask_block(1), masked_block(16), ref_block(16)])` → 49 channels
- Built per temporal block (3 latent frames) during the block-wise forward loop

## Self-Forcing Reference (External Repo)

The Self-Forcing LipSync StableAvatar repo is the *reference* for the 49-channel lipsync input format. Key architectural differences from LiveAvatar:

| Aspect | Self-Forcing | LiveAvatar (ours) |
|--------|-------------|-------------------|
| Audio encoder | Whisper Tiny (384-dim) | wav2vec2-large-xlsr-53 (1024-dim × 25 layers) |
| Hidden dim | 2048 | 5120 |
| Transformer layers | 32 | 40 |
| Audio projection | AudioProjModelHallo3 → cross-attn | CausalAudioEncoder (weighted layer sum → temporal conv → 5120) |
| Model size | ~1.3B | ~14B |

We do not modify the Self-Forcing repo. Its code is used only as reference for the 49-channel input construction and mouth-weighted MSE loss.

## LiveAvatar Inference Execution Path

**Entry:** `infinite_inference_single_gpu.sh` or `infinite_inference_multi_gpu.sh` → `minimal_inference/s2v_streaming_interact.py`

### Pipeline Selection

- Single GPU (80GB): `causal_s2v_pipeline.WanS2V`
- Multi-GPU (5x H800, TPP): `causal_s2v_pipeline_tpp.WanS2V` — GPUs 0-3 run DiT timesteps in parallel, GPU 4 runs VAE

### Inference Loop

```
s2v_streaming_interact.py:generate()
  → Instantiate WanS2V pipeline
  → Load LoRA weights + optional FP8 quantization
  ↓
WanS2V.generate()                                         (causal_s2v_pipeline.py:662)
  1. Audio: wav2vec2-large-xlsr-53-english → audio embeddings
  2. Image: VAE encode reference image → [1, 16, 1, H/8, W/8]
  3. Text: T5 (umt5-xxl) encode prompt → [context_len, 4096]
  ↓
  For each clip (autoregressive, infinite-length):
    Initialize noise [16, lat_target_frames, H/8, W/8]
    Initialize KV cache for all 40 transformer layers
    Prefill: run model at t=0 to cache clean KV states
    ↓
    For each temporal block (3 latent frames per block):
      For each diffusion timestep (4 steps via distillation, Euler scheduler):
        noise_pred = CausalWanModel_S2V.forward(
            latents,                    # [16, 3, H/8, W/8]
            t=timestep,
            context=text_emb,
            audio_input=audio_slice,    # wav2vec2 features for block
            motion_latents=ref_motion,  # Previous clip's last frames
            ref_latents=ref_image,
            kv_cache=...,
            crossattn_cache=...
        )
        latents = scheduler.step(noise_pred, t, latents)
    ↓
    VAE decode block latents → video frames
    Previous clip's tail → next clip's motion frames (continuity)
```

### Per-Timestep KV Caches

LiveAvatar maintains **4 independent KV caches** (`kv_cache1["1"]` through `kv_cache1["4"]`), one per denoising timestep. Both single-GPU and TPP pipelines use this pattern (see `causal_s2v_pipeline.py:995` — `for gpu_id in range(4): self._initialize_kv_cache(...)`).

- Each timestep `i` writes to its own cache: `kv_cache1[str(i+1)]`
- Block N+1 at timestep `i` attends to block N's timestep-`i` entry (same noise level)
- Prefill (`_forward_sink`) populates all 4 caches at `t=0` (conditional cache: `cond_k`/`cond_v`)
- Conditional caches can be shared across timesteps via `shared_cond_cache` (same motion/ref tokens regardless of σ)
- On single GPU, caches are swapped on/off GPU as needed (`_move_kv_cache_to_working_gpu`)

**Training simplification:** Our training uses a **single KV cache** because only one σ is sampled per training step (shared across all 7 blocks). This is consistent with inference: within any single per-timestep cache, all blocks share the same noise level.

### LiveAvatar Model Architecture (CausalWanModel_S2V)

**File:** `LiveAvatar/liveavatar/models/wan/causal_model_s2v.py` (Lines 438-1580)

- **Config:** 5120 hidden dim, 40 heads, 40 layers, 13824 FFN dim
- **Patch embedding:** Conv3d input → 5120 dim tokens
- **Audio:** CausalAudioEncoder (wav2vec2 weighted layer sum → temporal conv → 5120 dim) injected via cross-attention at layers [0,4,8,12,16,20,24,27,30,33,36,39]
- **Attention:** Causal self-attention with RoPE, Flash Attention, block masking + KV cache for streaming
- **Block-wise generation:** 3 latent frames per block, 73 motion frames as temporal context
- **Output:** head + unpatchify → [16, T, H/8, W/8] latent prediction

### Key LiveAvatar Files

| File | Purpose |
|------|---------|
| `minimal_inference/s2v_streaming_interact.py` | Inference entry point |
| `liveavatar/models/wan/causal_s2v_pipeline.py` | Single-GPU pipeline (generate loop, VAE encode/decode) |
| `liveavatar/models/wan/causal_s2v_pipeline_tpp.py` | Multi-GPU Timestep-Forcing Pipeline |
| `liveavatar/models/wan/causal_model_s2v.py` | Core 14B model: CausalWanModel_S2V (40 layers, audio injection) |
| `liveavatar/models/wan/causal_audio_encoder.py` | Wav2Vec2 audio feature extraction |
| `liveavatar/models/wan/wan_2_2/modules/s2v/audio_utils.py` | CausalAudioEncoder + AudioInjector_WAN |
| `liveavatar/models/wan/wan_2_2/modules/vae2_1.py` | VAE encoder/decoder (stride 4,8,8) |
| `liveavatar/models/wan/wan_2_2/modules/t5.py` | T5 text encoder |
| `liveavatar/models/wan/wan_2_2/configs/wan_s2v_14B_modified.py` | Model config (dims, layers, audio injection layers) |
| `liveavatar/models/wan/wan_2_2/utils/fm_solvers.py` | Flow-matching Euler scheduler |

## Training Implementation Details (train_lipsync.py)

### Audio-Video Alignment

- **Dataset always starts from frame 0** (no random temporal crop). This is required because `audio_path` points to the full source video, and `get_audio_embed_bucket_fps` extracts features starting from the beginning of the audio.
- **Audio truncated to 84 entries**: `get_audio_embed_bucket_fps(z, fps=25, batch_frames=84, m=0)` returns `min_batch_num * batch_frames` entries covering the full audio duration. We truncate to the first 84: `audio_bucket = audio_bucket[:84]`.
- **84 = 7 blocks × 12 video-frames/block**. Each block processes 3 latent frames = 12 video frames at 25 fps.

### KV Cache in Training

- **Single KV cache** (not 4 per-timestep caches like inference) — sufficient because one σ is sampled per training step, shared across all 7 blocks.
- This is consistent with inference: within any single per-timestep cache, all blocks write at the same noise level.
- Prefill via `_forward_sink` at `t=0` populates conditional caches (`cond_k`/`cond_v`), then blocks run `_forward_inference` at the sampled σ.
- Cache is reset between training samples (`reset_kv_cache` + `reset_crossattn_cache`).

### Loss Computation

- **Full-sequence backward**: All 7 blocks run forward, collecting velocity predictions into a list. `torch.cat(predictions, dim=2)` assembles the full `[B, 16, 21, H_lat, W_lat]` output. One MSE loss over all 21 latent frames, then a single `backward()` call.
- **Must use `torch.cat`, NOT in-place slice assignment.** `velocity_output[:, :, start:end] = pred` on a `torch.zeros_like` tensor severs the autograd graph — the target tensor has `requires_grad=False`, so PyTorch copies data without creating an autograd connection. This causes (a) zero gradients reaching the model (no learning) and (b) a memory leak (~32GB/iter) because `backward()` never traverses the disconnected block graphs so they are never freed.
- **Mouth-weighted MSE**: `weight = 1 + (W_mouth - 1) * mask`, default `W_mouth=5.0`. The mask is trilinearly interpolated from pixel-space mouth mask to latent space.
- Gradient checkpointing is enabled on the base model (`gradient_checkpointing = True`) to reduce memory during backward.
- Audio embeddings stored on `self` by `_forward_inference` are detached after each block to prevent graph accumulation.

### Memory Breakdown (512×512, batch=1, bf16)

| Component | Size | Notes |
|-----------|------|-------|
| Model params (14B bf16) | ~28 GB | LoRA adds ~0.5 GB trainable on top |
| KV cache (40 layers × 21504 entries × 40 heads × 128 dim × 2 bytes × k+v) | ~17.6 GB | 21504 = 21 latent frames × 1024 tokens/frame |
| Cond KV cache (40 layers × 2800 entries × same shape) | ~2.3 GB | Motion + ref + text conditioning |
| Cross-attn cache (40 layers, dynamic) | ~0.1 GB | Audio cross-attention; small |
| Latent tensors (x_0, x_t, noise, target, masked, ref, mask) | ~0.1 GB | 7 × [1, 16, 21, 64, 64] |
| **Total before forward** | **~50–56 GB** | Leaves ~24 GB for activations + gradients on 80GB GPU |

Conditioning models (VAE, wav2vec2, T5) are offloaded to CPU after preprocessing to stay within GPU budget.

## Video Stitching for Validation Outputs

When evaluating model outputs, videos are stitched side-by-side for comparison. Common patterns:

### 1. Full Videos (GT + Generated)
Stitch ground truth and generated videos from `val_outputs_full/step_*/`:

```bash
# Input: {video_id}_gt.mp4 + {video_id}_audio_{audio_id}_gen.mp4
# Output: {video_id}_full.mp4 (1024x512, 2 videos side by side)

ffmpeg -y -i gt.mp4 -i gen.mp4 \
    -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
    -map "[v]" -map "0:a" \
    -c:v libx264 -crf 18 -preset fast -c:a aac \
    output_full.mp4
```

### 2. Mixed Videos (Cross-Audio Comparison)
Stitch videos with same video_id but different audio sources from `val_outputs_full_mixed/step_*/`:

```bash
# Input: {video_id}_shot_*_audio_{audio_id1}_gen.mp4 (×3 different audio_ids)
# Output: mixed_a.mp4 (1536x512, 3 videos side by side)
# Group by video_id (extract prefix before _shot), sort files alphabetically

video_ids=($(ls *.mp4 | sed 's/_shot.*//' | sort -u))
for i in "${!video_ids[@]}"; do
    video_id="${video_ids[$i]}"
    files=($(ls "${video_id}_shot_"*"_audio_"*.mp4 | sort))
    letter=$(printf "%c" $((97 + i)))  # a, b, c, d...

    ffmpeg -y -i "${files[0]}" -i "${files[1]}" -i "${files[2]}" \
        -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
        -map "[v]" -map "0:a" \
        -c:v libx264 -crf 18 -preset fast -c:a aac \
        "mixed_${letter}.mp4"
done
```

### 3. Autoregressive Videos (GT + Recon + Frames)
Stitch ground truth, reconstruction, and generated frames from `val_outputs_autoregressive/`:

```bash
# Input: {prefix}_gt.mp4, {prefix}_recon.mp4, {prefix}_1.mp4, {prefix}_2.mp4, {prefix}_3.mp4
# Output: autoregressive_{prefix}_plus.mp4 (2560x512, 5 videos side by side)

for prefix in a b; do
    ffmpeg -y \
        -i "${prefix}_gt.mp4" -i "${prefix}_recon.mp4" \
        -i "${prefix}_1.mp4" -i "${prefix}_2.mp4" -i "${prefix}_3.mp4" \
        -filter_complex "[0:v][1:v][2:v][3:v][4:v]hstack=inputs=5[v]" \
        -map "[v]" -map "0:a" \
        -c:v libx264 -crf 18 -preset fast -c:a aac \
        "autoregressive_${prefix}_plus.mp4"
done
```

### FFmpeg Parameters
- **hstack**: Horizontal stack for side-by-side comparison
- **crf 18**: High quality encoding (lower = better, range 0-51)
- **preset fast**: Balance between speed and compression
- **Audio**: `-map "0:a"` uses audio from first (leftmost) input video

### Output Directory Structure
```
examples/wanvideo/model_training/
├── val_outputs_full/step_*/        # GT + generated pairs
├── val_outputs_full_mixed/step_*/  # Cross-audio combinations
├── val_outputs_autoregressive/     # Autoregressive generations
└── val_outputs_stitched/step_*/    # Stitched comparison videos
    ├── {video_id}_full.mp4         # Full comparisons
    ├── mixed_[a-d].mp4              # Mixed comparisons
    └── autoregressive_*_plus.mp4   # Autoregressive comparisons
```

**Note:** Use single-line ffmpeg commands when running in parallel to avoid bash variable expansion issues in multiline commands.

# Points to follow:
- Add under ## Working Style section\n\nListen carefully to what I say to ignore or keep. If I say 'ignore X' or 'keep Y', do not try to derive or modify those things. Re-read my instructions before proposing changes.
