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

3. We will need to install `litData` as a module: 
```bash
cd litData
uv pip install -e .
cd ..
```

## Data

We will be using a random 42858 sample shard from DataComp (~3GB). We will get it from the s3 path stored in a `.env` file in a environment variable called `BENCHMARK_SHARD_PATH`.

Please make sure to set this environment variable before running the benchmark.

This benchmark shard will be downloaded locally into the folder `./data` to be used in the various output format datasets for benchmarking. Please run the following: 
```bash
./scripts/download_shard.sh
```

The parquet dataset I'm working with has the following columns: 
```python
Index(['document_id', 'document_metadata', 'image', 'text', 'audio', 'video', 'raw_data'], dtype='object')
```
Of which the most important data cols are `image` and `text`. The `image.content` contains the raw bytes of the image while `text.content` contains the corresponding caption in text characters. The `document_id` is the spark-generated uuid of the data sample. 


## Benchmarking

### Data Preparation

Through `nproc`, my machine has 16 cpus. To reproduce this table below, simply run `./scripts/prepare_datasets.sh`. The resulting datasets can be found in `./shards` and be uploaded to your respective s3 bucket for streaming benchmarking.

| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files |
| --- | --- | --- | --- | --- |
| WebDataset | 18.00 | 10.33 | 1.82 | 14 |
| MDS | 19.00 | 10.35 | 1.67 | 28 |

These results roughly line up with what's reported in the original blogpost.

### Streaming

TODO: maybe make env var for s3 bucket where datasets are streamed from. 

