#!/usr/bin/env python

"""
WebDataset Streaming Benchmark

This script benchmarks streaming performance from WebDataset format. 
It measures how fast images and text can be loaded from WebDataset shards stored in S3.

The WebDataset shards are expected to have the following structure:
- __key__: document_id or generated index
- jpg: Raw image bytes
- txt: Caption text (encoded as UTF-8)
"""

import os
import sys
import argparse
from time import time
import torchvision.transforms.v2 as T
from tqdm import tqdm
from pytorch_lightning import seed_everything
import boto3
import webdataset as wds
import shutil
import torch
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure AWS credentials using environment variables if needed
# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY should be set in the environment
# or in ~/.aws/credentials

def setup_cache(cache_dir):
    """Set up cache directory for WebDataset"""
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    return cache_dir


def parse_s3_path(s3_path, webdataset_subdir="webdataset"):
    """Parse S3 path into bucket and prefix"""
    # Ensure path has proper format
    if not s3_path:
        print("S3 path is not set. Please check your environment variables.")
        sys.exit(1)
    
    # Remove s3:// prefix if present
    path = s3_path
    if path.startswith('s3://'):
        path = path[5:]  # Remove 's3://' prefix
        
    # Ensure path ends with a slash for consistent joining
    if not path.endswith('/'):
        path += '/'
    
    # Extract bucket and prefix
    parts = path.split('/')
    bucket = parts[0]
    
    # Create the prefix by appending webdataset_subdir to the path parts
    rest_parts = [part for part in parts[1:] if part]  # Skip empty parts
    prefix = '/'.join(rest_parts + [webdataset_subdir])
    
    return bucket, prefix

def prepare_urls(bucket, prefix, split="train"):
    """Prepare URLs for WebDataset from S3 bucket and prefix"""
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
    
    # Filter for tar files containing specified split
    filtered_keys = [key for key in keys if key.endswith(".tar") and split in key]
    print(f"Found {len(filtered_keys)} {split} tar files")
    
    # Generate proper S3 URLs for the WebDataset to stream from
    # Using aws s3 cp for better streaming performance
    return [f"pipe:aws s3 cp s3://{os.path.join(bucket, key)} -" for key in filtered_keys]

def process_image(img_bytes):
    """Process image data from WebDataset format and ensure consistent channels"""
    if isinstance(img_bytes, bytes):
        img = Image.open(io.BytesIO(img_bytes))
        # Convert grayscale images to RGB to ensure consistent channels
        if img.mode == 'L':
            img = img.convert('RGB')
        return img
    return img_bytes


def process_text(text):
    """Process text data from WebDataset format"""
    if isinstance(text, bytes):
        return text.decode('utf-8')
    return text

def create_transforms(image_size=224, to_dtype=torch.float16):
    """Create image transforms for preprocessing"""
    return T.Compose([
        T.RandomResizedCrop(image_size, antialias=True),
        T.RandomHorizontalFlip(),
        T.ToImage(),
        # Ensure all images have 3 channels (RGB)
        T.Lambda(lambda x: x if x.shape[0] == 3 else torch.cat([x]*3, dim=0) if x.shape[0] == 1 else x[:3]),
        T.ToDtype(to_dtype, scale=True),
    ])


def create_dataset(urls, cache_dir, batch_size, shuffle_buffer=1000, image_size=224):
    """Create WebDataset pipeline"""
    transforms = create_transforms(image_size)
    
    dataset = (
        wds.WebDataset(urls, cache_dir=cache_dir)
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
            # For each batch, we have a dict with keys: __key__, image, text
            if isinstance(batch, dict) and "image" in batch:
                # Get the batch size from the image tensor shape
                batch_images = batch["image"]
                if isinstance(batch_images, torch.Tensor):
                    num_samples += batch_images.shape[0]
                else:
                    num_samples += len(batch_images)
        
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
    parser = argparse.ArgumentParser(description="WebDataset Streaming Benchmark")
    
    # Data parameters
    parser.add_argument("--s3_path", type=str, help="S3 path for WebDataset shards (will use S3_BENCHMARK_DATA_PATH env var if not specified)")
    parser.add_argument("--split", type=str, default="train", help="Data split to use (default: train)")
    parser.add_argument("--webdataset_subdir", type=str, default="webdataset", help="Subdirectory name for WebDataset shards (default: webdataset)")
    parser.add_argument("--cache_dir", type=str, help="Cache directory for WebDataset (default: ./cache/webdataset_benchmark)")
    
    # Performance parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training (default: 256)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads for data loading (default: 8)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch (default: 2)")
    parser.add_argument("--shuffle_buffer", type=int, default=1000, help="Size of shuffle buffer (default: 1000)")
    
    # Image processing parameters
    parser.add_argument("--image_size", type=int, default=224, help="Size of images after preprocessing (default: 224)")
    
    # Benchmark parameters
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for benchmark (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--keep_cache", action="store_true", help="Don't delete cache directory after benchmark")
    
    return parser.parse_args()


def main():
    """Main entry point for benchmark"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    print(f"Seed set to {args.seed}")
    
    # Set up cache directory
    cache_dir = args.cache_dir
    if not cache_dir:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache/webdataset_benchmark")
    cache_dir = setup_cache(cache_dir)
    
    # Get S3 path from args or environment
    s3_path = args.s3_path or os.environ.get('S3_BENCHMARK_DATA_PATH')
    if not s3_path:
        print("No S3 path specified. Please use --s3_path or set S3_BENCHMARK_DATA_PATH environment variable.")
        sys.exit(1)
    
    # Parse S3 path
    bucket, prefix = parse_s3_path(s3_path, args.webdataset_subdir)
    
    # Prepare dataset URLs
    urls = prepare_urls(bucket, prefix, args.split)
    if not urls:
        print("No WebDataset tar files found in S3. Please check your S3 bucket and prefix.")
        sys.exit(1)
    
    # Limit num_workers to avoid errors when there are fewer shards than workers
    num_workers = min(args.num_workers, len(urls), os.cpu_count() or 1)
    
    # Create dataset and dataloader
    dataset = create_dataset(
        urls, 
        cache_dir=cache_dir, 
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
    try:
        results = run_benchmark(dataloader, args.batch_size, args.epochs)
        print("Benchmark complete")
    finally:
        # Clean up cache unless --keep_cache was specified
        if not args.keep_cache and os.path.exists(cache_dir):
            print(f"Cleaning up cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
    
    return results


if __name__ == "__main__":
    main()