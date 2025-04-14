# Benchmark Multimodal Dataloaders

Original code taken from the [Lightning AI Blogpost](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries?view=public&section=featured&tab=overview) benchmarking using the ImageNet Dataset

## Setup

1. Clone this repository with submodules: 
```bash
git clone --recurse-submodules https://github.com/Nano1337/benchmark-dataloader.git
```

2. We use `uv` as dependency management for this project. Please see installation instructions in [here](https://docs.astral.sh/uv/getting-started/installation/). Once installed, please run:
```bash
uv sync
source .venv/bin/activate
```

## Data

We will be using a random 42k sample shard from DataComp (~3GB). We will get it from the s3 path stored in a `.env` file in a environment variable called `BENCHMARK_SHARD_PATH`.

Please make sure to set this environment variable before running the benchmark.

This benchmark shard will be downloaded locally into the folder `./data` to be used in the various output format datasets for benchmarking. Please run the following: 
```bash
./scripts/download_shard.sh
```
