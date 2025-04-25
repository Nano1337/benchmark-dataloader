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
from tqdm import tqdm
from pytorch_lightning import seed_everything
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import torch

# Import shared utilities
from utils import (create_transforms, parse_s3_path as utils_parse_s3_path, 
                 get_default_cache_dir, setup_cache, cleanup_cache, process_image, process_text)

# Load environment variables from .env file
load_dotenv()


def parse_s3_path(s3_path, mds_subdir="mds"):
    """Parse S3 path into local and remote paths for MosaicML Dataset"""
    # Use the imported function from utils
    remote_path, _, _ = utils_parse_s3_path(s3_path, mds_subdir)
    return remote_path


# Create streaming dataset class
class ImageTextStreamingDataset(StreamingDataset):
    """MosaicML streaming dataset for image-text data"""
    
    def __init__(self, remote, batch_size, image_size=224, shuffle=True, cache_dir=None):
        super().__init__(remote=remote, shuffle=shuffle, batch_size=batch_size, local=cache_dir)
        self.transform = create_transforms(image_size, torch.float16)

    def __getitem__(self, index):
        """
        Get a sample from the dataset
        Returns: image tensor, text string
        """
        sample = super().__getitem__(index)
        
        # Process image - assuming 'image' contains the image data
        image_data = sample['image']
        image_data = process_image(image_data)
        image_data = self.transform(image_data)
        
        # Process text - assuming 'text' contains the text data
        text = sample['text']
        text = process_text(text)

        return image_data, text


def create_dataset(remote_path, batch_size, image_size=224, shuffle=True, cache_dir=None):
    """Create MosaicML streaming dataset
    
    Args:
        remote_path: Path to remote MosaicML dataset
        batch_size: Batch size for the dataset
        image_size: Target image size for preprocessing
        shuffle: Whether to shuffle the dataset
        cache_dir: Directory to cache downloaded data
        
    Returns:
        MosaicML streaming dataset ready for dataloader
    """
    try:
        dataset = ImageTextStreamingDataset(
            remote=remote_path,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            cache_dir=cache_dir
        )
        return dataset
    except Exception as e:
        print(f"Error creating dataset: {e}")
        sys.exit(1)


def create_dataloader(dataset, batch_size=256, num_workers=8, prefetch_factor=2):
    """Create dataloader for MosaicML dataset"""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    return loader


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
                
            # Each batch contains (image, text) tuples
            images, texts = batch
            num_samples += images.shape[0]
        
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
    parser = argparse.ArgumentParser(description="MosaicML Dataset Streaming Benchmark")
    
    # Data parameters
    parser.add_argument("--s3_path", type=str, help="S3 path for MosaicML Dataset (will use S3_BENCHMARK_DATA_PATH env var if not specified)")
    parser.add_argument("--split", type=str, default="train", help="Data split to use (default: train)")
    parser.add_argument("--mds_subdir", type=str, default="mds", help="Subdirectory name for MosaicML Dataset (default: mds)")
    
    # Performance parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training (default: 256)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads for data loading (default: 8)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch (default: 2)")

    # Image processing parameters
    parser.add_argument("--image_size", type=int, default=224, help="Size of images after preprocessing (default: 224)")
    
    # Benchmark parameters
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for benchmark (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    # Cache parameters
    parser.add_argument("--cache_dir", type=str, help="Directory to use for caching dataset files (default: stream/cache/mds_benchmark)")
    
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
    
    # Parse S3 path to get remote MosaicML path
    remote_path = parse_s3_path(s3_path, args.mds_subdir)
    
    # Setup cache directory
    cache_dir = args.cache_dir or get_default_cache_dir('mds')
    setup_cache(cache_dir, clear_existing=True)
    print(f"Using cache directory: {cache_dir}")
    
    # Create dataset and dataloader
    dataset = create_dataset(
        remote_path=remote_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True,
        cache_dir=cache_dir
    )
        
    dataloader = create_dataloader(
        dataset, 
        prefetch_factor=args.prefetch_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
        
    print(f"Using {args.num_workers} worker threads for data loading")
    
    try:
        # Run benchmark
        results = run_benchmark(dataloader, args.batch_size, args.epochs)
        print("Benchmark complete")
    finally:
        # Clean up cache
        cleanup_cache(cache_dir)
    
    return results


if __name__ == "__main__":
    main()