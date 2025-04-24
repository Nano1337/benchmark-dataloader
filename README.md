# Benchmark Multimodal Dataloaders

Benchmarking inspired by [Lightning AI Blogpost](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries?view=public&section=featured&tab=overview) originally benchmarking using the ImageNet Dataset

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

We will be using a random 88513 sample shard from DataComp (~3GB) that's been uploaded to HuggingFace. To download the data: 

1. Install git lfs
```bash
sudo apt update 
sudo apt install git-lfs
git lfs install
```
2. Download the data from HuggingFace: 
```bash
mkdir -p data
cd data
git clone https://huggingface.co/datasets/Nano1337/benchmark_dataset
mv benchmark_dataset/ benchmark_dataset.parquet/
```


The parquet dataset here has the following columns: 
```python
Index(['image', 'text'], dtype='object')
```
The `image.content` contains the raw bytes of the image while `text.content` contains the corresponding caption in text characters.


## Benchmarking

### Data Preparation

1. Run `python prepare_datasets.py` to run the dataset preparation benchmarking. The output datasets will be found in `./shards`. You can view the resource usage plots in `./results/processing/plots`.

2. Please upload `./shards` to your respective cloud storage provider. Here's an example for s3: 
```bash
aws s3 cp ./shards s3://<your-bucket>/shards --recursive
```

3. Please update the `S3_BENCHMARK_DATA_PATH` in your `.env` to the s3 path of the dataset you want to benchmark streaming with in the next section. 

Note that we benchmark only using 3GB worth of data for dataset preparation (representing potentially one data shard in the worst case) as the RAM overhead growth is not linear in some cases (e.g. LitData) but can be easily scaled up using spark distributed data processing. 

The results can be found here: 

CPU Count: 16

| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files | Peak RAM (MB) |
| --- | --- | --- | --- | --- | --- |
| LitData (PL) | 34.69 | 30.14 | 2.78 | 60 | 33225.6 |
| WebDataset (WDS) | 30.91 | 24.43 | 3.17 | 23 | 72140.9 |
| MosaicML Dataset (MDS) | 21.07 | 12.82 | 2.86 | 47 | 7745.4 |
| Energon (WDS+) | 37.60 | 48.86 | 3.18 | 51 | 72140.9 |

Total benchmark time: 94.56 seconds

### Streaming

To run streaming benchmarks, please set the env var `S3_BENCHMARK_DATA_PATH` to the s3 path of the dataset you want to benchmark containing directories `webdataset`, `mds`, and `litdata`. Then run `./scripts/stream_datasets.sh`.

### Example

Assuming you have an s3 bucket `my-bucket` and you want to benchmark the `webdataset` dataset, you can run:
```bash
export S3_BENCHMARK_DATA_PATH="s3://my-bucket/shards/webdataset"
./scripts/stream_datasets.sh
```

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
