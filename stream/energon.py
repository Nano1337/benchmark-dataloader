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
from utils import create_transforms, process_image, process_text, parse_s3_path as utils_parse_s3_path

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
    from megatron.energon import get_train_dataset, get_loader, WorkerConfig, DefaultTaskEncoder
except ImportError:
    print("Please install megatron-energon with extras like [s3] to stream remote datasets.")
    sys.exit(1)

class CaptioningTaskEncoder(DefaultTaskEncoder):
    """A simple task encoder for captioning."""

    def __init__(
        self,
        image_size=224,
    ):
        super().__init__()
        self.image_transform = create_transforms(image_size, torch.float16)

    def encode_sample(self, sample):
        sample.image = self.image_transform(process_image(sample.image))
        sample.caption = process_text(sample.caption)
        return sample


def run_benchmark(dataloader, batch_size, num_epochs=2):
    """Run streaming benchmark for specified number of epochs"""
    print(f"Starting benchmark with batch size {batch_size}")
    
    total_samples = 0
    time_to_first_batch = None
    epoch_metrics = []
    wall_time_start = time()
    
    for epoch in range(num_epochs):
        num_samples = 0
        t0 = time()
        
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", smoothing=0, mininterval=1)):
            # Record time to first batch (only on first epoch)
            if epoch == 0 and i == 0:
                time_to_first_batch = time() - t0
                print(f"Time to first batch: {time_to_first_batch:.4f}s")
                
            # batch is a CaptioningSample dataclass with image and caption attributes
            batch_images, batch_text = batch.image, batch.caption
            num_samples += batch_images.shape[0]
        
        elapsed = time() - t0
        throughput = num_samples / elapsed
        print(f'Epoch {epoch+1}: Processed {num_samples} samples in {elapsed:.2f}s ({throughput:.2f} images/sec)')
        
        # Store metrics for this epoch
        epoch_metrics.append({
            "samples": num_samples,
            "time": elapsed,
            "throughput": throughput
        })
        
        total_samples += num_samples
    
    wall_time = time() - wall_time_start
    
    # Get second epoch throughput (if available)
    second_epoch_throughput = epoch_metrics[1]["throughput"] if len(epoch_metrics) > 1 else 0
    
    # Report overall statistics
    print(f"\nBenchmark Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Wall time: {wall_time:.2f}s")
    print(f"  Throughput: {second_epoch_throughput:.2f} images/sec")
    print(f"  Time to first batch: {time_to_first_batch:.4f}s")
    
    return {
        "samples": total_samples,
        "wall_time": wall_time,
        "throughput": second_epoch_throughput,  # Using second epoch throughput
        "time_to_first_batch": time_to_first_batch
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

    data_size = 88514
    num_steps = (data_size // args.batch_size) + 1

    # Create remote Energon dataset via MSC
    ds = get_train_dataset(
        dataset_name,
        batch_size=args.batch_size,
        max_samples_per_sequence=args.max_seq_len,
        shuffle_buffer_size=args.shuffle_buffer,
        virtual_epoch_length=num_steps, # Energon dataset size
        worker_config=worker_config,
        task_encoder=CaptioningTaskEncoder(image_size=args.image_size),
    )
    
    # Create dataloader
    loader = get_loader(
        ds,
        prefetch_factor=args.prefetch_factor,
        worker_config=worker_config,
    )
    
    # Run benchmark
    results = run_benchmark(loader, args.batch_size, args.epochs)
    print("Benchmark complete")

    return results


if __name__ == "__main__":
    main()