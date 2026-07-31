"""
Precompute I2V reference latents (ref_latents_sink + motion_latents) from GT frame 0.

For each video, encodes frame 0 through VAE to produce:
  - ref_latents_sink: [16, 1, H/8, W/8] — single-frame reference for KV cache prefill
  - motion_latents:   [16, 19, H/8, W/8] — repeated-frame motion context

These are saved as separate .pt files ({video_id}_i2v_ref.pt) and referenced via
a new column "i2v_ref_latents" in metadata_precomputed.csv.

Architecture mirrors precompute_vae_latents.py (multi-GPU, prefetch pipeline, idempotent).

Usage:
    # Single GPU
    python precompute_i2v_ref_latents.py --config configs/lipsync_train_i2v.yaml

    # Multi-GPU
    CUDA_VISIBLE_DEVICES=0,1 python precompute_i2v_ref_latents.py --config configs/lipsync_train_i2v.yaml
"""

import argparse
import logging
import os
import queue
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import decord
import numpy as np
import pandas as pd
import torch
import torch.amp as amp
import torch.multiprocessing as mp
from PIL import Image
from torchvision.transforms.functional import resize, to_tensor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOTION_FRAMES_VIDEO = 73


def load_frame0(video_path, height, width):
    """Load frame 0 from a video, resize, normalize to [-1, 1].

    Returns: [3, 1, H, W] float tensor
    """
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    frame = vr[0].asnumpy()  # [H, W, 3] uint8
    img = Image.fromarray(frame)
    t = to_tensor(img)  # [3, H, W] float [0, 1]
    t = resize(t, [height, width], antialias=True)
    t = t.unsqueeze(1)  # [3, 1, H, W]
    return t * 2.0 - 1.0  # [-1, 1]


def batched_vae_encode(vae, videos, device):
    """Encode a batch of videos through VAE with B>1 support.

    Falls back to single-video encoding on OOM.
    """
    try:
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            latents = vae.model.encode(videos, vae.scale)
        return latents.float()
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"OOM with batch B={videos.shape[0]}, falling back to single encoding")
        torch.cuda.empty_cache()
        results = []
        for i in range(videos.shape[0]):
            with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
                lat = vae.model.encode(videos[i:i+1], vae.scale)
            results.append(lat.float())
        return torch.cat(results, dim=0)


def _load_single_video(entry, height, width, i2v_dir):
    """Load frame 0 for one video. CPU only, thread-safe."""
    video_id = entry["video_id"]
    output_path = os.path.join(i2v_dir, f"{video_id}_i2v_ref.pt")

    if os.path.exists(output_path):
        return {"status": "skip", "entry": entry, "output_path": output_path}

    try:
        ref_single = load_frame0(entry["video_path"], height, width)  # [3, 1, H, W]
        return {
            "status": "ok",
            "entry": entry,
            "output_path": output_path,
            "ref_single": ref_single,
        }
    except Exception as e:
        return {"status": "fail", "entry": entry, "error": str(e)}


def _prefetch_batches(entries, height, width, i2v_dir, batch_size, prefetch_queue, stats,
                      num_loader_threads=4):
    """Background thread: load frame 0 for batches, put on queue."""
    with ThreadPoolExecutor(max_workers=num_loader_threads) as pool:
        for batch_start in range(0, len(entries), batch_size):
            batch_entries = entries[batch_start:batch_start + batch_size]

            futures = {
                pool.submit(_load_single_video, e, height, width, i2v_dir): e
                for e in batch_entries
            }

            valid_items = []
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "skip":
                    stats["skipped"] += 1
                    stats["skip_entries"].append(result)
                elif result["status"] == "fail":
                    stats["failed"] += 1
                    stats["fail_errors"].append(f"{result['entry']['video_id']}: {result.get('error', '?')}")
                else:
                    valid_items.append(result)

            if valid_items:
                prefetch_queue.put(valid_items)

    prefetch_queue.put(None)  # sentinel


def process_batch(valid_items, vae, device):
    """Encode ref_latents_sink + motion_latents for a batch of videos."""
    B = len(valid_items)

    # 5-frame encodes for ref_latents_sink
    batch_5 = []
    for item in valid_items:
        ref_5 = item["ref_single"].repeat(1, 5, 1, 1)  # [3, 5, H, W]
        batch_5.append(ref_5)
    batch_5 = torch.stack(batch_5).to(device)  # [B, 3, 5, H, W]
    latents_5 = batched_vae_encode(vae, batch_5, device)  # [B, 16, 2, H/8, W/8]

    # 73-frame encodes for motion_latents
    batch_73 = []
    for item in valid_items:
        motion = item["ref_single"].repeat(1, MOTION_FRAMES_VIDEO, 1, 1)  # [3, 73, H, W]
        batch_73.append(motion)
    batch_73 = torch.stack(batch_73).to(device)  # [B, 3, 73, H, W]
    latents_73 = batched_vae_encode(vae, batch_73, device)  # [B, 16, 19, H/8, W/8]

    results = []
    for i in range(B):
        ref_latents_sink = latents_5[i][:, 1:]  # [16, 1, H/8, W/8]
        motion_latents = latents_73[i]           # [16, 19, H/8, W/8]

        save_dict = {
            "ref_latents_sink": ref_latents_sink.to(torch.bfloat16).cpu(),
            "motion_latents": motion_latents.to(torch.bfloat16).cpu(),
        }
        results.append((valid_items[i], save_dict))

    return results


def worker_process(gpu_id, worker_entries, args, config):
    """Worker: load VAE on assigned GPU and process videos with prefetch pipeline."""
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    torch.set_num_threads(4)

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    logging.basicConfig(
        level=logging.INFO,
        format=f'[GPU {gpu_id}] %(asctime)s %(message)s',
        datefmt='%H:%M:%S'
    )
    wlog = logging.getLogger(f"worker_{gpu_id}")

    if not worker_entries:
        wlog.info("No videos assigned, exiting")
        return []

    batch_size = args.batch_size
    num_batches = (len(worker_entries) + batch_size - 1) // batch_size
    wlog.info(f"Assigned {len(worker_entries)} videos (batch_size={batch_size}, ~{num_batches} batches)")

    height = config.get("height", 512)
    width = config.get("width", 512)

    # Load VAE
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "LiveAvatar"))
    from liveavatar.models.wan.wan_2_2.modules.vae2_1 import Wan2_1_VAE

    wlog.info("Loading VAE...")
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(config["checkpoint_dir"], config.get("vae_checkpoint", "Wan2.1_VAE.pth")),
        device=device,
        dtype=torch.bfloat16,
    )
    wlog.info("VAE loaded")

    i2v_dir = os.path.join(args.output_dir, "i2v_ref_latents")

    stats = {"skipped": 0, "failed": 0, "skip_entries": [], "fail_errors": []}
    processed = 0

    prefetch_q = queue.Queue(maxsize=2)
    loader_thread = threading.Thread(
        target=_prefetch_batches,
        args=(worker_entries, height, width, i2v_dir,
              batch_size, prefetch_q, stats, args.num_loader_threads),
        daemon=True,
    )
    loader_thread.start()

    pbar = tqdm(total=num_batches, desc=f"GPU {gpu_id}", position=gpu_id)
    metadata_rows = []

    while True:
        batch_data = prefetch_q.get()
        if batch_data is None:
            break

        try:
            results = process_batch(batch_data, vae, device)
        except Exception as e:
            wlog.error(f"Error encoding batch: {e}")
            traceback.print_exc()
            stats["failed"] += len(batch_data)
            torch.cuda.empty_cache()
            pbar.update(1)
            continue

        for item, save_dict in results:
            torch.save(save_dict, item["output_path"])
            metadata_rows.append({
                "video_id": item["entry"]["video_id"],
                "i2v_ref_latents": item["output_path"],
            })
            processed += 1

        pbar.update(1)
        pbar.set_postfix(done=processed, skip=stats["skipped"], err=stats["failed"])

        if processed % 200 == 0:
            torch.cuda.empty_cache()

    loader_thread.join()
    pbar.close()

    # Record skipped entries too
    for skip_result in stats["skip_entries"]:
        entry = skip_result["entry"]
        metadata_rows.append({
            "video_id": entry["video_id"],
            "i2v_ref_latents": skip_result["output_path"],
        })

    if stats["fail_errors"]:
        for err in stats["fail_errors"][:10]:
            wlog.warning(f"Failed: {err}")
        if len(stats["fail_errors"]) > 10:
            wlog.warning(f"... and {len(stats['fail_errors']) - 10} more failures")

    wlog.info(f"Done: processed={processed}, skipped={stats['skipped']}, failed={stats['failed']}")

    del vae
    torch.cuda.empty_cache()

    return metadata_rows


def _worker_wrapper(gid, entries, a, c, rq_):
    """Top-level wrapper for mp.spawn (must be picklable)."""
    rows = worker_process(gid, entries, a, c)
    rq_.put(rows)


def main():
    parser = argparse.ArgumentParser(description="Precompute I2V reference latents from GT frame 0")
    parser.add_argument("--config", type=str, required=True, help="Path to lipsync_train YAML config")
    parser.add_argument("--output_dir", type=str, default="/home/work/liveavatar_data",
                        help="Output directory (default: /home/work/liveavatar_data)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Videos per VAE batch (only 78 frames per video, so higher batch OK)")
    parser.add_argument("--num_loader_threads", type=int, default=4,
                        help="Threads per GPU for parallel video decoding")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    i2v_dir = os.path.join(args.output_dir, "i2v_ref_latents")
    os.makedirs(i2v_dir, exist_ok=True)

    # Load existing precomputed CSV to get video list
    precomputed_csv = config.get("precomputed_csv")
    if not precomputed_csv or not os.path.exists(precomputed_csv):
        raise ValueError(f"precomputed_csv not found: {precomputed_csv}. "
                         "Run precompute_vae_latents.py first.")

    df = pd.read_csv(precomputed_csv)
    data_root = config["data_root"]

    video_entries = []
    for _, row in df.iterrows():
        video_rel = row["video"]
        video_path = os.path.join(data_root, video_rel)
        video_id = os.path.splitext(os.path.basename(video_rel))[0]
        video_entries.append({
            "video_rel": video_rel,
            "video_path": video_path,
            "video_id": video_id,
        })

    logger.info(f"Total videos: {len(video_entries)}")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")
    logger.info(f"Found {num_gpus} GPUs, batch_size={args.batch_size}")

    # Split across GPUs
    splits = []
    chunk_size = len(video_entries) // num_gpus
    remainder = len(video_entries) % num_gpus
    start = 0
    for i in range(num_gpus):
        end = start + chunk_size + (1 if i < remainder else 0)
        splits.append(video_entries[start:end])
        start = end

    if num_gpus == 1:
        all_metadata = worker_process(0, splits[0], args, config)
    else:
        ctx = mp.get_context('spawn')
        result_queues = []
        processes = []

        for gpu_id in range(num_gpus):
            rq = ctx.Queue()
            result_queues.append(rq)
            p = ctx.Process(target=_worker_wrapper, args=(gpu_id, splits[gpu_id], args, config, rq))
            p.start()
            processes.append(p)
            logger.info(f"Started worker on GPU {gpu_id} with {len(splits[gpu_id])} videos")

        all_metadata = []
        for rq in result_queues:
            all_metadata.extend(rq.get())
        for p in processes:
            p.join()

    # Build lookup: video_id -> i2v_ref_latents path
    i2v_lookup = {row["video_id"]: row["i2v_ref_latents"] for row in all_metadata}

    # Update the precomputed CSV with new column
    # Derive video_id from video column (filename without extension)
    df["i2v_ref_latents"] = df["video"].apply(
        lambda v: i2v_lookup.get(os.path.splitext(os.path.basename(v))[0])
    )
    missing = df["i2v_ref_latents"].isna().sum()
    if missing > 0:
        logger.warning(f"{missing} videos missing I2V ref latents (will be dropped during training)")

    df.to_csv(precomputed_csv, index=False)
    logger.info(f"Updated {precomputed_csv} with i2v_ref_latents column ({len(all_metadata)} entries)")


if __name__ == "__main__":
    main()
