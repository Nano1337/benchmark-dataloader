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
| LitData (PL) | 22.00 | 17.24 | 1.64 | 28 |
| WebDataset (WDS) | 42.00 | 9.72 | 1.82 | 14 |
| MosaicML Dataset (MDS) | 18.00 | 10.45 | 1.67 | 28 |

Scaling this up and running on 6GB of parquet data (with machine RAM limited to 40GB, noting that uncompressed data can grow substantially) yields these results:

| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files |
| --- | --- | --- | --- | --- |
| LitData (PL) | 48.00 | 43.77 | 5.54 | 105 |
| WebDataset (WDS) | 51.00 | 38.73 | 6.32 | 46 |
| MosaicML Dataset (MDS) | 42.00 | 34.04 | 5.71 | 93 |

My hypothesis after running the benchmark and watching active RAM utilization metrics using `watch free -h`, LitData is using a lot more RAM, especially when the workers start to write out the .bin files. I assume this happens bc each worker is decompressing and making a copy (via writing out the raw byte sequences) of the data to put in the .bin file. What happens if we run this on a machine with much more RAM?

Using CPU Count: 192, RAM: 1TB on small benchmark shard:

| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files |
| --- | --- | --- | --- | --- |
| LitData (PL) | 31.00 | 18.94 | 1.67 | 28 |
| WebDataset (WDS) | 33.00 | 9.02 | 1.60 | 14 |
| MosaicML Dataset (MDS) | 37.00 | 9.12 | 1.73 | 28 |

Turning zstd compression off for LitData:
20GB shard: each worker reads entire parquet shard

| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files |
| --- | --- | --- | --- | --- |
| LitData (PL) | 672.00 | 655.19 | 13.11 | 499 |
| WebDataset (WDS) | 179.00 | 135.39 | 21.72 | 164 |
| MosaicML Dataset (MDS) | 117.00 | 87.65 | 20.78 | 328 |

20GB shard (test with pyarrow batched reading, batch_size=1k)

| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files |
| --- | --- | --- | --- | --- |
| LitData (PL) | 508.00 | 498.35 | 15.84 | 508 |
| WebDataset (WDS) | 175.00 | 145.12 | 21.65 | 164 |
| MosaicML Dataset (MDS) | 105.00 | 88.41 | 20.49 | 328 |

20GB shard (test with pyarrow batched reading, batch_size=2k/all)

| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files |
| --- | --- | --- | --- | --- |
| LitData (PL) | 364.00 | 353.99 | 14.17 | 508 |
| WebDataset (WDS) | 167.00 | 138.86 | 21.58 | 164 |
| MosaicML Dataset (MDS) | 105.00 | 87.20 | 20.38 | 328 |

Turning zstd compression on for LitData:

20GB shard (test with pyarrow batched reading, batch_size=2k/all)

| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files |
| --- | --- | --- | --- | --- |
| LitData (PL) | 363.00 | 351.76 | 21.18 | 508 |
| WebDataset (WDS) | 184.00 | 135.51 | 22.73 | 164 |
| MosaicML Dataset (MDS) | 114.00 | 88.02 | 21.69 | 328 |


We likely won’t need the PyArrow batched reading fallback, as it primarily addresses scenarios where all workers operate on a single machine. In the ideal case, Spark’s distributed processing will naturally distribute the workload across machines, allowing each node to independently handle its assigned Parquet shard rather than processing everything on a single node.

### Streaming

To run streaming benchmarks, please set the env var `S3_BENCHMARK_DATA_PATH` to the s3 path of the dataset you want to benchmark containing directories `webdataset`, `mds`, and `litdata`. Then run `./scripts/stream_datasets.sh`.

CPU Count: 16
Configuration: Batch Size = 256, Workers = 8

| Dataset | Throughput (img/s) | Samples | Epoch 1 (s) | Epoch 2 (s) | Processing Time (s) | Wall Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| WebDataset | 2048.81 | 85716 | 23.10 | 18.73 | 41.84 | 46.34 |
| MosaicML MDS | 2066.48 | 85716 | 25.14 | 16.34 | 41.48 | 45.98 |
| LitData | 2495.73 | 85716 | 18.89 | 15.45 | 34.35 | 39.57 |

Using the larger 6GB dataset: 

| Dataset | Throughput (img/s) | Samples | Epoch 1 (s) | Epoch 2 (s) | Processing Time (s) | Wall Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| WebDataset | 2432.81 | 352906 | 79.27 | 65.79 | 145.06 | 149.55 |
| MosaicML MDS | 2264.01 | 352906 | 91.14 | 64.74 | 155.88 | 160.44 |
| LitData | 2571.11 | 352906 | 62.64 | 74.62 | 137.26 | 142.53 |
