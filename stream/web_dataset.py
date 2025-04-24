#!/usr/bin/env python

"""
WebDataset Streaming Benchmark

This script benchmarks streaming performance from WebDataset format. 
It measures how fast images and text can be loaded from WebDataset shards stored in S3.

The WebDataset shards are expected to have the following structure:
- __key__: generated index
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
import webdataset as wds
import torch
from dotenv import load_dotenv

# Import shared utilities
from utils import create_transforms, process_image, process_text, parse_s3_path as utils_parse_s3_path

# Load environment variables from .env file
load_dotenv()

def parse_s3_path(s3_path, webdataset_subdir="webdataset"):
    """Parse S3 path into bucket and prefix for WebDataset"""
    # Use the imported function from utils
    remote_path, bucket, dataset_path = utils_parse_s3_path(s3_path, webdataset_subdir)
    return bucket, dataset_path

def prepare_urls(bucket, prefix, split="train", file_prefix="dataset"):
    """Prepare URLs for WebDataset from S3 bucket and prefix
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix path
        split: Dataset split (e.g., 'train')
        file_prefix: Prefix for shard files (default: 'dataset')
    """
    s3_client = boto3.client('s3')
    token = None
    keys = []

    print(f"Listing objects...")
    
    try:
        while token is None or token != "":
            if token:
                objects = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                    ContinuationToken=token,
                )
            else:
                objects = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                )   

            token = objects.get("NextContinuationToken", "end") 
            if 'Contents' in objects:
                keys.extend(obj['Key'] for obj in objects['Contents'])

            if token == "end":
                break
    except Exception as e:
        print(f"Error listing S3 objects: {e}")
        return []

    print(f"Found {len(keys)} objects in S3")
    
    # Filter for tar files with format matching from webdataset_converter.py
    # Default pattern: {prefix}-{split}-%06d.tar (e.g., "dataset-train-000001.tar")
    pattern = f"{file_prefix}-{split}-"
    filtered_keys = [key for key in keys if key.endswith(".tar") and pattern in key]
    print(f"Found {len(filtered_keys)} {split} tar files matching pattern '{pattern}'")
    
    # Generate proper S3 URLs for the WebDataset to stream from
    # Using aws s3 cp for better streaming performance
    return [f"pipe:aws s3 cp s3://{os.path.join(bucket, key)} -" for key in filtered_keys]


def create_dataset(urls, batch_size, shuffle_buffer=1000, image_size=224):
    """Create WebDataset pipeline"""
    transforms = create_transforms(image_size)
    
    dataset = (
        wds.WebDataset(urls)
        .shuffle(shuffle_buffer)  # Shuffle with a buffer
        .map(lambda sample: {
            "__key__": sample["__key__"], 
            "image": process_image(sample["jpg"]), 
            "text": process_text(sample["txt"])
        })
        .map(lambda sample: {
            "__key__": sample["__key__"],
            "image": transforms(sample["image"]),
            "text": sample["text"]
        })
        .batched(batch_size, partial=True)
    )
    
    return dataset


def create_dataloader(dataset, num_workers, prefetch_factor=2):
    """Create WebDataset dataloader"""
    loader = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        batch_size=None,       # We're already batching in the dataset
        prefetch_factor=prefetch_factor,
    )
    
    return loader


def run_benchmark(dataloader, batch_size, num_epochs=2):
    """Run streaming benchmark for specified number of epochs"""
    print(f"Starting benchmark with batch size {batch_size}")
    
    total_samples = 0
    total_time = 0
    
    for epoch in range(num_epochs):
        num_samples = 0
        t0 = time()
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", smoothing=0, mininterval=1):
            images, text = batch['image'], batch['text']
            num_samples += images.shape[0]
        
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
    parser = argparse.ArgumentParser(description="WebDataset streaming benchmark")
    
    # S3 parameters
    parser.add_argument("--s3_path", type=str, help="S3 path to the benchmark dataset (e.g., s3://bucket/path)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--prefix", type=str, default="benchmark", help="Prefix for shard files (default: benchmark)")
    parser.add_argument("--webdataset_subdir", type=str, default="webdataset", help="Subdirectory name for WebDataset shards (default: webdataset)")
    
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
    
    return parser.parse_args()


def main():
    """Main entry point for benchmark"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    print(f"Seed set to {args.seed}")
    
    # Get S3 path from args or environment
    s3_path = args.s3_path or os.environ.get('S3_BENCHMARK_DATA_PATH')
    if not s3_path:
        print("No S3 path specified. Please use --s3_path or set S3_BENCHMARK_DATA_PATH environment variable.")
        sys.exit(1)
    
    # Parse S3 path
    bucket, prefix = parse_s3_path(s3_path, args.webdataset_subdir)
    
    # Get S3 path from args or environment
    s3_path = args.s3_path or os.environ.get('S3_BENCHMARK_DATA_PATH')
    if not s3_path:
        print("No S3 path specified. Please use --s3_path or set S3_BENCHMARK_DATA_PATH environment variable.")
        sys.exit(1)
    
    # Parse S3 path
    bucket, prefix = parse_s3_path(s3_path, args.webdataset_subdir)
    
    # Prepare dataset URLs
    # Use the specified prefix to match format from webdataset_converter.py 
    urls = prepare_urls(bucket, prefix, args.split, args.prefix)
    if not urls:
        print("No WebDataset tar files found in S3. Please check your S3 bucket and prefix.")
        sys.exit(1)
    
    # Limit num_workers to avoid errors when there are fewer shards than workers
    num_workers = min(args.num_workers, len(urls), os.cpu_count() or 1)
    
    # Create dataset and dataloader
    dataset = create_dataset(
        urls, 
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        image_size=args.image_size
    )
    
    dataloader = create_dataloader(
        dataset, 
        num_workers=num_workers,
        prefetch_factor=args.prefetch_factor
    )
    
    print(f"Using {num_workers} worker threads for data loading")
    
    # Run benchmark
    results = run_benchmark(dataloader, args.batch_size, args.epochs)
    print("Benchmark complete")
    
    return results


if __name__ == "__main__":
    main()