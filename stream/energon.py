#!/usr/bin/env python

"""
Energon Streaming Benchmark

This script benchmarks streaming performance from Energon format. 
It measures how fast images and text can be loaded from Energon shards stored in S3.

The Energon shards are expected to have the following structure:
- __key__: document_id or generated index
- jpg: Raw image bytes
- txt: Caption text (encoded as UTF-8)
"""

import os
import sys
import argparse
from time import time
from tqdm import tqdm
from pytorch_lightning import seed_everything
import boto3
import torch
from dotenv import load_dotenv

# Import shared utilities
from utils import create_transforms, parse_s3_path as utils_parse_s3_path

# Load environment variables from .env file
load_dotenv()

# Check if MSC_CONFIG is set from .env or environment
msc_config = os.environ.get('MSC_CONFIG')
if msc_config:
    print(f"Using MSC_CONFIG: {msc_config}")
    os.environ['MSC_CONFIG'] = msc_config
else:
    raise ValueError("MSC_CONFIG environment variable is not set. Energon benchmark will fail.")

try:
    from megatron.energon import get_train_dataset, get_loader, WorkerConfig
except ImportError:
    print("Please install megatron-energon with extras like [s3] to stream remote datasets.")
    sys.exit(1)

# These functions are now imported from utils.py


def run_benchmark(dataloader, batch_size, num_epochs=2):
    """Run streaming benchmark for specified number of epochs"""
    print(f"Starting benchmark with batch size {batch_size}")
    
    total_samples = 0
    total_time = 0
    
    for epoch in range(num_epochs):
        num_samples = 0
        t0 = time()
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", smoothing=0, mininterval=1):
            # batch is a CaptioningSample dataclass with image and caption attributes
            batch_images = batch.image
            num_samples += batch_images.shape[0]
        
        elapsed = time() - t0
        throughput = num_samples / elapsed
        print(f'Epoch {epoch+1}: Processed {num_samples} samples in {elapsed:.2f}s ({throughput:.2f} images/sec)')
        
        total_samples += num_samples
        total_time += elapsed
    
    # Report overall statistics
    avg_throughput = total_samples / total_time
    print(f"\nBenchmark Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average throughput: {avg_throughput:.2f} images/sec")
    
    return {
        "total_samples": total_samples,
        "total_time": total_time,
        "avg_throughput": avg_throughput
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Energon Streaming Benchmark")
    
    # Data parameters
    parser.add_argument("--s3_path", type=str, help="S3 path for Energon shards (will use S3_BENCHMARK_DATA_PATH env var if not specified)")

    # Performance parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training (default: 256)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads for data loading (default: 8)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch (default: 2)")
    parser.add_argument("--shuffle_buffer", type=int, default=100, help="Size of shuffle buffer (default: 100)")
    
    # Image processing parameters
    parser.add_argument("--image_size", type=int, default=224, help="Size of images after preprocessing (default: 224)")
    
    # Benchmark parameters
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for benchmark (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--msc_config", type=str, default="default", help="MSC config name for remote dataset")
    parser.add_argument("--max_seq_len", type=int, default=100, help="Max samples per sequence for dataset")
    return parser.parse_args()


def main():
    """Main entry point for benchmark"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Build MSC URL for remote Energon dataset
    s3_base = args.s3_path or os.getenv('S3_BENCHMARK_DATA_PATH')
    if not s3_base:
        print("No S3 path specified. Please use --s3_path or set S3_BENCHMARK_DATA_PATH environment variable.")
        sys.exit(1)
        
    # Use our standardized utility to parse the S3 path
    remote_path, _, _ = utils_parse_s3_path(s3_base, "energon")
    
    # Extract the path portion (after the bucket) if any
    if remote_path.startswith('s3://'):
        parts = remote_path[5:].split('/', 1)
        if len(parts) > 1:
            # We have a path after the bucket
            relative_path = parts[1]
            dataset_name = f"msc://s3-iad-webdataset/{relative_path}"
        else:
            # Just the bucket name, no additional path
            dataset_name = "msc://s3-iad-webdataset/"
    else:
        dataset_name = "msc://s3-iad-webdataset/"

    # define worker config
    worker_config = WorkerConfig.default_worker_config()
    worker_config.num_workers = args.num_workers

    # Create remote Energon dataset via MSC
    ds = get_train_dataset(
        dataset_name,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer,
        max_samples_per_sequence=args.max_seq_len,
        virtual_epoch_length=352, # Energon dataset size
        worker_config=worker_config,
    )
    loader = get_loader(
        ds,
        prefetch_factor=args.prefetch_factor,
    )
    
    # Run benchmark
    results = run_benchmark(loader, args.batch_size, args.epochs)
    print("Benchmark complete")

    return results


if __name__ == "__main__":
    main()