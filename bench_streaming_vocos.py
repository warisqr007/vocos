#!/usr/bin/env python3
"""
Benchmark StreamingVocos latency + RTF on CPU and GPU.

Metrics reported (per device):
  - avg_proc_per_chunk_ms
  - avg_first_chunk_proc_ms
  - avg_latency_ms = chunk_buffer_ms + avg_proc_per_chunk_ms
  - first_chunk_latency_ms = chunk_buffer_ms + avg_first_chunk_proc_ms
  - avg_rtf = avg_total_processing_time_s / audio_duration_s
  - avg_total_processing_time_ms

Notes:
  * chunk_size is in MEL frames.
  * hop_size is in waveform samples (default 320 @ 16kHz => 20ms per mel frame).
  * "chunk buffer latency" = chunk_size * hop_size / sr seconds (time you must wait to accumulate a chunk).
  * For GPU timing, torch.cuda.synchronize() is used to get accurate wall time.
"""

import argparse
import time
import numpy as np
import torch
import librosa
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Your module
from src.modules import VocosVocoderModule  # adjust if needed


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


@torch.inference_mode()
def load_model(repo_id: str, ckpt_filename: str, device: torch.device):
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_filename)
    model = VocosVocoderModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    model.to(device)
    return model


@torch.inference_mode()
def compute_mel(model, audio_1d_np: np.ndarray, device: torch.device):
    # audio_1d_np: (T,)
    x = torch.from_numpy(audio_1d_np).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)
    mel = model.feature_extractor(x)  # expected (1, 80, Tm) or similar
    return mel


@torch.inference_mode()
def run_streaming_once(model, mel: torch.Tensor, chunk_size: int, device: torch.device):
    """
    Runs streaming inference once and times:
      - total processing time
      - first chunk processing time
      - average per-chunk processing time

    Returns:
      total_s, first_chunk_s, avg_chunk_s, n_chunks
    """
    # Ensure we start fresh streaming state each run by re-entering context
    n_chunks = 0
    chunk_times = []

    _sync_if_cuda(device)
    t0 = time.perf_counter()

    with model.decoder[0].streaming(1), model.decoder[1].streaming(1):
        # split along time axis (dim=2)
        for i, mel_chunk in enumerate(mel.split(chunk_size, dim=2)):
            _sync_if_cuda(device)
            tc0 = time.perf_counter()

            _ = model(mel_chunk)  # (1,1,hop*Tchunk) usually

            _sync_if_cuda(device)
            tc1 = time.perf_counter()

            chunk_times.append(tc1 - tc0)
            n_chunks += 1

    _sync_if_cuda(device)
    t1 = time.perf_counter()

    total_s = t1 - t0
    first_chunk_s = chunk_times[0] if chunk_times else 0.0
    avg_chunk_s = float(np.mean(chunk_times)) if chunk_times else 0.0
    return total_s, first_chunk_s, avg_chunk_s, n_chunks


def benchmark_device(
    device: torch.device,
    model_repo: str,
    ckpt_filename: str,
    audio_path: str,
    sr: int,
    hop_size: int,
    chunk_size: int,
    warmup: int,
    runs: int,
    use_fp16_on_cuda: bool,
):
    # Load audio on CPU
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    audio_dur_s = len(audio) / sr

    # Load model
    model = load_model(model_repo, ckpt_filename, device=device)

    # Optional fp16 on GPU (only if your model supports it safely)
    if device.type == "cuda" and use_fp16_on_cuda:
        model = model.half()

    # Precompute mel ONCE per benchmark to avoid including feature_extractor time
    # If you want end-to-end latency including mel extraction, move mel computation into the loop.
    mel = compute_mel(model, audio, device=device)
    if device.type == "cuda" and use_fp16_on_cuda:
        mel = mel.half()

    # Buffering latency from chunk accumulation
    chunk_buffer_s = (chunk_size * hop_size) / sr
    chunk_buffer_ms = chunk_buffer_s * 1000.0

    # Warmup (not recorded)
    for _ in tqdm(range(warmup), desc=f"Warmup on {device}"):
        _ = run_streaming_once(model, mel, chunk_size, device)

    # Timed runs
    totals = []
    firsts = []
    avgs = []
    nchunks = None

    for _ in tqdm(range(runs), desc=f"Benchmarking on {device}"):
        total_s, first_s, avg_chunk_s, n_chunks = run_streaming_once(model, mel, chunk_size, device)
        totals.append(total_s)
        firsts.append(first_s)
        avgs.append(avg_chunk_s)
        nchunks = n_chunks

    totals = np.array(totals, dtype=np.float64)
    firsts = np.array(firsts, dtype=np.float64)
    avgs = np.array(avgs, dtype=np.float64)

    avg_total_s = float(totals.mean())
    avg_first_s = float(firsts.mean())
    avg_chunk_s = float(avgs.mean())

    # RTF: processing time / audio duration
    avg_rtf = avg_total_s / max(audio_dur_s, 1e-9)

    results = {
        "device": str(device),
        "sr": sr,
        "hop_size": hop_size,
        "chunk_size_mel_frames": chunk_size,
        "mel_frame_ms": (hop_size / sr) * 1000.0,
        "chunk_buffer_ms": chunk_buffer_ms,
        "n_chunks_per_run": int(nchunks) if nchunks is not None else 0,
        "audio_duration_s": audio_dur_s,
        "warmup": warmup,
        "runs": runs,
        "avg_proc_per_chunk_ms": avg_chunk_s * 1000.0,
        "avg_first_chunk_proc_ms": avg_first_s * 1000.0,
        "avg_latency_ms": chunk_buffer_ms + (avg_chunk_s * 1000.0),          # buffer + avg compute
        "first_chunk_latency_ms": chunk_buffer_ms + (avg_first_s * 1000.0),  # buffer + first compute
        "avg_total_processing_time_ms": avg_total_s * 1000.0,
        "avg_rtf": avg_rtf,
        "p50_total_ms": float(np.percentile(totals * 1000.0, 50)),
        "p90_total_ms": float(np.percentile(totals * 1000.0, 90)),
        "p99_total_ms": float(np.percentile(totals * 1000.0, 99)),
    }
    return results


def pretty_print(res: dict):
    print("\n" + "=" * 80)
    print(f"Device: {res['device']}")
    print(f"Audio duration: {res['audio_duration_s']:.3f} s")
    print(f"SR={res['sr']}  hop={res['hop_size']} samples  mel_frame={res['mel_frame_ms']:.2f} ms")
    print(f"Chunk size: {res['chunk_size_mel_frames']} mel frames  -> buffer={res['chunk_buffer_ms']:.2f} ms")
    print(f"Chunks per run: {res['n_chunks_per_run']}")
    print("-" * 80)
    print(f"Avg proc/chunk:        {res['avg_proc_per_chunk_ms']:.3f} ms")
    print(f"Avg first-chunk proc:  {res['avg_first_chunk_proc_ms']:.3f} ms")
    print(f"Avg latency:           {res['avg_latency_ms']:.3f} ms (buffer + avg proc/chunk)")
    print(f"First-chunk latency:   {res['first_chunk_latency_ms']:.3f} ms (buffer + first-chunk proc)")
    print("-" * 80)
    print(f"Avg total proc time:   {res['avg_total_processing_time_ms']:.3f} ms")
    print(f"RTF (avg):             {res['avg_rtf']:.4f}")
    print(f"Total time percentiles (ms): p50={res['p50_total_ms']:.2f}, p90={res['p90_total_ms']:.2f}, p99={res['p99_total_ms']:.2f}")
    print("=" * 80 + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_repo", type=str, default="warisqr007/StreamingVocos")
    p.add_argument("--ckpt", type=str, default="epoch=3.ckpt")
    p.add_argument("--audio", type=str, required=True, help="Path to input wav for benchmarking")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--hop_size", type=int, default=320)
    p.add_argument("--chunk_size", type=int, default=1, help="Chunk size in MEL frames")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--fp16_cuda", action="store_true", help="Use fp16 on CUDA (only if model supports it)")
    args = p.parse_args()

    # CPU
    cpu_res = benchmark_device(
        device=torch.device("cpu"),
        model_repo=args.model_repo,
        ckpt_filename=args.ckpt,
        audio_path=args.audio,
        sr=args.sr,
        hop_size=args.hop_size,
        chunk_size=args.chunk_size,
        warmup=args.warmup,
        runs=args.runs,
        use_fp16_on_cuda=False,
    )
    pretty_print(cpu_res)

    # GPU if available
    if torch.cuda.is_available():
        gpu_res = benchmark_device(
            device=torch.device("cuda:0"),
            model_repo=args.model_repo,
            ckpt_filename=args.ckpt,
            audio_path=args.audio,
            sr=args.sr,
            hop_size=args.hop_size,
            chunk_size=args.chunk_size,
            warmup=args.warmup,
            runs=args.runs,
            use_fp16_on_cuda=args.fp16_cuda,
        )
        pretty_print(gpu_res)
    else:
        print("CUDA not available; GPU benchmark skipped.")


if __name__ == "__main__":
    main()
