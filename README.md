# Streaming Vocos: Neural vocoder for fast streaming applications
Vocos was proposed as fast neural vocoder designed to synthesize audio waveforms from acoustic features.
This repo replicates the design as the origial vocos archiecture but modified to have an streaming implementation.
So all the vanilla CNNs are replaced with causal CNNs and modified to work in streaming settings with dynmically adjustable chunk size (in multiples of hop size of 320ms).

What makes vocos different from other typical GAN-based vocoders is that Vocos does not model audio samples in the time domain. Instead, it generates spectral coefficients, facilitating rapid audio reconstruction through inverse Fourier transform. This cuts down the processing time significantly and is very appropriate for streaming applications that require minimal latency.

The model takes 50Hz log-melspectrogram as input (1024 window size, 320 hopsize) and produces 16KHz audio.

Training follows the Generative Adversarial Network (GAN) objective as original but loss functions are changed to those proposed in the descript audio codec (see [repo](https://github.com/descriptinc/descript-audio-codec)).

Visit our [demo website (coming soon)]() for audio samples. 
Refer below for original paper and audio samples.
[Audio samples](https://gemelo-ai.github.io/vocos/) |
Paper [[abs]](https://arxiv.org/abs/2306.00814) [[pdf]](https://arxiv.org/pdf/2306.00814.pdf)



## ⚡ Streaming Latency & Real-Time Performance

We benchmark **Streaming Vocos** in **streaming inference mode** using chunked mel-spectrogram decoding on both CPU and GPU.

### Benchmark setup

- **Audio duration:** 3.24 s  
- **Sample rate:** 16 kHz  
- **Mel hop size:** 320 samples (20 ms per mel frame)  
- **Chunk size:** 5 mel frames (100 ms buffering latency)  
- **Runs:** 100 warm-up + 1000 timed runs  
- **Inference mode:** Streaming (stateful causal decoding)  

**Metrics**
- **Processing time per chunk**
- **End-to-end latency** = chunk buffering + processing time
- **RTF (Real-Time Factor)** = processing time / audio duration

---

### Results

#### Streaming performance (chunk size = 5 frames, 100 ms buffer)

| Device | Avg proc / chunk | First-chunk proc | End-to-end latency | Total proc (3.2 s audio) | RTF |
|------|------------------|------------------|--------------------|---------------------------|-----|
| **CPU** | 14.0 ms | 14.0 ms | **114.0 ms** | 464 ms | 0.14 |
| **GPU (CUDA)** | **3.4 ms** | **3.3 ms** | **103.3 ms** | **113 ms** | **0.035** |

> End-to-end latency includes the **100 ms chunk buffering delay** required for streaming inference.

---

### Interpretation

- **Real-time capable on CPU**  
  Streaming Vocos achieves an RTF of approximately **0.14**, corresponding to inference running ~7× faster than real time.

- **Ultra-low compute overhead on GPU**  
  Chunk processing time is reduced to **~3.4 ms**, making overall latency dominated by buffering rather than computation.

- **Streaming-friendly first-chunk behavior**  
  First-chunk latency closely matches steady-state latency, indicating **no cold-start penalty** during streaming inference.

- **Latency–quality tradeoff**  
  Smaller chunk sizes further reduce buffering latency (e.g., 1–2 frames → <40 ms), at the cost of slightly increased computational overhead.

---

With a **chunk size of 1 frame (20 ms buffering)**, GPU end-to-end latency drops below **25 ms**, making **Streaming Vocos** suitable for **interactive and conversational TTS pipelines**.




## Checkpoints
You can download the checkpoint from [here](https://huggingface.co/warisqr007/StreamingVocos/resolve/main/epoch%3D3.ckpt).

## Usage

### Installation

```bash
# Clone project
git clone https://github.com/warisqr007/vocos.git 
cd vocos

# [Optional] Create a conda virtual environment
conda create -n <env_name> python=3.10
conda activate <env_name>

# [Optional] Use mamba instead of conda to speed up
conda install mamba -n base -c conda-forge

# Install dependencies
pip install -r requirements.txt
```

### Data Preprocessing
We follow the same data-processing stage as [here](https://github.com/PSI-TAMU/streamVC). Please follow directions in the specified repo.


### Run

**Fit**

```bash
python src/main.py fit -c configs/data/resynthesis.yaml -c configs/model/vocosvocoder.yaml --trainer.logger.name debug
```

**Resume**

```bash
python src/main.py fit -c configs/data/resynthesis.yaml -c configs/model/vocosvocoder.yaml --ckpt_path <ckpt_path> --trainer.logger.id exp1_id
```


### Inference

See [notebooks/inference.ipynb](notebooks/inference.ipynb)


### Benchmarking latency
```bash
python bench_streaming_vocos.py \
        --audio test.wav \
        --chunk_size 5 \
        --warmup 100 \
        --runs 1000

```


## Acknowledgements
- [Vocos Repo](https://github.com/gemelo-ai/vocos)
- [Moshi Repo for streaming implementation](https://github.com/kyutai-labs/moshi)
- [lightning-template](https://github.com/DavidZhang73/pytorch-lightning-template)
