#!/usr/bin/env python

"""
MosaicML Dataset Streaming Benchmark

This script benchmarks streaming performance from MosaicML Dataset format. 
It measures how fast images can be loaded from MosaicML streaming datasets stored in S3.
"""

import os
import sys
import argparse
from time import time
import torchvision.transforms.v2 as T
from tqdm import tqdm
from pytorch_lightning import seed_everything
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import torch
import shutil

# Load environment variables from .env file
load_dotenv()

# Configure AWS credentials using environment variables if needed
# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY should be set in the environment
# or in ~/.aws/credentials


def setup_cache(cache_dir):
    """Set up cache directory for MosaicML Dataset"""
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    return cache_dir


def parse_s3_path(s3_path, mds_subdir="mds"):
    """Parse S3 path into local and remote paths for MosaicML Dataset"""
    # Ensure path has proper format
    if not s3_path:
        print("S3 path is not set. Please check your environment variables.")
        sys.exit(1)
    
    # For MosaicML, we need to convert the S3 path to the format they expect
    # Remove s3:// prefix if present
    if s3_path.startswith('s3://'):
        path = s3_path[5:]
    else:
        path = s3_path
        
    # Ensure path ends with a slash for consistent joining
    if not path.endswith('/'):
        path += '/'
    
    # Split into bucket and prefix
    parts = path.split('/')
    bucket = parts[0]
    
    # Create the MDS path
    rest_parts = [part for part in parts[1:] if part]  # Skip empty parts
    mds_path = '/'.join(rest_parts + [mds_subdir])
    
    # Combine into full S3 path
    remote_path = f"s3://{bucket}/{mds_path}"
    
    return remote_path


# Create streaming dataset class
class ImageTextStreamingDataset(StreamingDataset):
    """MosaicML streaming dataset for image-text data"""
    
    def __init__(self, local, remote, batch_size, image_size=224, shuffle=True):
        super().__init__(local=local, remote=remote, shuffle=shuffle, batch_size=batch_size)
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, antialias=True),
            T.RandomHorizontalFlip(),
            T.ToImage(),
            # Ensure all images have 3 channels (RGB)
            T.Lambda(lambda x: x if x.shape[0] == 3 else torch.cat([x]*3, dim=0) if x.shape[0] == 1 else x[:3]),
            T.ToDtype(torch.float16, scale=True),
        ])

    def __getitem__(self, index):
        """
        Get a sample from the dataset
        Returns: image tensor, text string
        """
        sample = super().__getitem__(index)
        
        # Process image - assuming 'image' or 'x' contains the image data
        if 'image' in sample:
            image_data = sample['image']
        elif 'x' in sample:
            image_data = sample['x']
        else:
            # Default to the first item if keys don't match expected
            image_data = next(iter(sample.values()))
        
        # Process text - assuming 'text' or 'y' contains the text data
        if 'text' in sample:
            text = sample['text']
        elif 'y' in sample:
            text = sample['y']
        else:
            text = "No text available"
        
        # Apply transforms to the image
        try:
            # Attempt to convert image data to tensor and transform
            image = self.transform(image_data)
        except Exception as e:
            # If transformation fails, create a small random tensor as fallback
            print(f"Error transforming image: {e}")
            image = torch.randn(3, 224, 224, dtype=torch.float16)
            
        return image, text


def create_dataset(remote_path, cache_dir, batch_size, image_size=224, shuffle=True):
    """Create MosaicML streaming dataset"""
    try:
        dataset = ImageTextStreamingDataset(
            local=cache_dir,
            remote=remote_path,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle
        )
        return dataset
    except Exception as e:
        print(f"Error creating dataset: {e}")
        sys.exit(1)


def create_dataloader(dataset, batch_size=256, num_workers=8):
    """Create dataloader for MosaicML dataset"""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,  # Prefetch batches for smoother streaming
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
            # Each batch contains (image, text) tuples
            images, _ = batch
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
    parser = argparse.ArgumentParser(description="MosaicML Dataset Streaming Benchmark")
    
    # Data parameters
    parser.add_argument("--s3_path", type=str, help="S3 path for MosaicML Dataset (will use S3_BENCHMARK_DATA_PATH env var if not specified)")
    parser.add_argument("--split", type=str, default="train", help="Data split to use (default: train)")
    parser.add_argument("--mds_subdir", type=str, default="mds", help="Subdirectory name for MosaicML Dataset (default: mds)")
    parser.add_argument("--cache_dir", type=str, help="Cache directory for MosaicML Dataset (default: ./cache/mds_benchmark)")
    
    # Performance parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training (default: 256)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads for data loading (default: 8)")
    
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
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache/mds_benchmark")
    cache_dir = setup_cache(cache_dir)
    
    # Get S3 path from args or environment
    s3_path = args.s3_path or os.environ.get('S3_BENCHMARK_DATA_PATH')
    if not s3_path:
        print("No S3 path specified. Please use --s3_path or set S3_BENCHMARK_DATA_PATH environment variable.")
        sys.exit(1)
    
    # Parse S3 path to get remote MosaicML path
    remote_path = parse_s3_path(s3_path, args.mds_subdir)
    
    # Create dataset and dataloader
    try:
        dataset = create_dataset(
            remote_path=remote_path,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            shuffle=True
        )
        
        dataloader = create_dataloader(
            dataset, 
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        print(f"Using {args.num_workers} worker threads for data loading")
        
        # Run benchmark
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