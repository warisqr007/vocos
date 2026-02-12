# Streaming Vocos: Neural vocoder for fast streaming applications
Vocos was proposed as fast neural vocoder designed to synthesize audio waveforms from acoustic features.
This repo replicates the design as the origial vocos archiecture but modified to have an streaming implementation.
So all the vanilla CNNs are replaced with causal CNNs and modified to work in streaming settings with dynmically adjustable chunk size (in multiples of hop size of 320ms).

What makes vocos different from other typical GAN-based vocoders is that Vocos does not model audio samples in the time domain. Instead, it generates spectral coefficients, facilitating rapid audio reconstruction through inverse Fourier transform. This cuts down the processing time significantly and is very appropriate for streaming applications that require minimal latency.


Training follows the Generative Adversarial Network (GAN) objective as original but loss functions are changed to those proposed in the descript audio codec (see [repo](https://github.com/descriptinc/descript-audio-codec)).

Visit our [demo website (coming soon)]() for audio samples. You can download the checkpoint from [here](https://huggingface.co/warisqr007/StreamingVocos/resolve/main/epoch%3D3.ckpt)

Refer below for original paper and audio samples.
[Audio samples](https://gemelo-ai.github.io/vocos/) |
Paper [[abs]](https://arxiv.org/abs/2306.00814) [[pdf]](https://arxiv.org/pdf/2306.00814.pdf)


## Checkpoints
Coming soon

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


## Acknowledgements
- [Vocos Repo](https://github.com/gemelo-ai/vocos)
- [Moshi Repo for streaming implementation](https://github.com/kyutai-labs/moshi)
- [lightning-template](https://github.com/DavidZhang73/pytorch-lightning-template)
