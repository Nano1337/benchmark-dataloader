#!/usr/bin/env python

"""
Lightning Data Streaming Benchmark

This script benchmarks streaming performance of Lightning Data with image-text pairs.
It measures throughput in images per second and supports various configurations.
"""

import os
import sys
import time
import argparse
import torch
from tqdm import tqdm
from litdata import StreamingDataset, StreamingDataLoader
from pytorch_lightning import seed_everything
from dotenv import load_dotenv

# Import shared utilities
from utils import (create_transforms, process_image, process_text, parse_s3_path as utils_parse_s3_path,
                 get_default_cache_dir, setup_cache, cleanup_cache)

# Load environment variables from .env file
load_dotenv()


def parse_s3_path(s3_path, litdata_subdir="litdata"):
    """Parse S3 path into path for Lightning Data"""
    # Use the imported function from utils
    remote_path, _, _ = utils_parse_s3_path(s3_path, litdata_subdir)
    print(f"Using remote path: {remote_path}")
    return remote_path


# Create streaming dataset class for image-text pairs
class ImageTextStreamingDataset(StreamingDataset):
    """Lightning Data streaming dataset for image-text data"""
    
    def __init__(self, input_dir, max_cache_size="25GB", image_size=224, shuffle=True, cache_dir=None):
        self.cache_dir = cache_dir
        super().__init__(input_dir=input_dir, max_cache_size=max_cache_size)
        self.transform = create_transforms(image_size, torch.float16)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        Returns: image tensor, text string
        """
        sample = super().__getitem__(index)
        

        image_bytes = sample['image_bytes']
        image_data = process_image(image_bytes)
        image_data = self.transform(image_data)
        
        text = sample['text']
        text = process_text(text)

        return image_data, text


def create_dataset(input_dir, image_size=224, shuffle=True, cache_dir=None):
    """Create Lightning Data streaming dataset
    
    Args:
        input_dir: Path to the dataset directory
        image_size: Target image size for preprocessing
        shuffle: Whether to shuffle the dataset
        cache_dir: Directory to cache downloaded data
        
    Returns:
        Lightning Data streaming dataset instance
    """
    try:
        max_cache_size = "25GB"
        dataset = ImageTextStreamingDataset(
            input_dir=input_dir,
            max_cache_size=max_cache_size,
            image_size=image_size,
            shuffle=shuffle,
            cache_dir=cache_dir
        )
        return dataset
    except Exception as e:
        print(f"Error creating dataset: {e}")
        raise


def create_dataloader(dataset, batch_size=256, num_workers=8, prefetch_factor=2):
    """Create dataloader for Lightning Data dataset"""
    print(f"Using {num_workers} worker threads for data loading")
    dataloader = StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
    return dataloader


def run_benchmark(dataloader, batch_size, num_epochs=2):
    """Run streaming benchmark for specified number of epochs"""
    print(f"Starting benchmark with batch size {batch_size}")
    
    total_samples = 0
    time_to_first_batch = None
    epoch_metrics = []
    wall_time_start = time.time()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        num_samples = 0
        
        # Iterate through the dataloader
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", smoothing=0, mininterval=1)):
            # Record time to first batch (only on first epoch)
            if epoch == 0 and i == 0:
                time_to_first_batch = time.time() - start_time
                print(f"Time to first batch: {time_to_first_batch:.4f}s")
            
            # Get batch size from actual data
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                batch_images = batch[0]
                num_samples += batch_images.shape[0]
            else:
                num_samples += batch_size  # Fallback if we can't determine batch size
        
        elapsed = time.time() - start_time
        throughput = num_samples / elapsed
        
        print(f"Epoch {epoch+1}: Processed {num_samples} samples in {elapsed:.2f}s ({throughput:.2f} images/sec)")
        
        # Store metrics for this epoch
        epoch_metrics.append({
            "samples": num_samples,
            "time": elapsed,
            "throughput": throughput
        })
        
        total_samples += num_samples
    
    wall_time = time.time() - wall_time_start
    
    # Get second epoch throughput (if available)
    second_epoch_throughput = epoch_metrics[1]["throughput"] if len(epoch_metrics) > 1 else 0
    
    # Report overall statistics
    print(f"\nBenchmark Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Wall time: {wall_time:.2f}s")
    print(f"  Throughput: {second_epoch_throughput:.2f} images/sec")
    print(f"  Time to first batch: {time_to_first_batch:.4f}s")
    print("Benchmark complete")
    
    return {
        "samples": total_samples,
        "wall_time": wall_time,
        "throughput": second_epoch_throughput,  # Using second epoch throughput
        "time_to_first_batch": time_to_first_batch
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Lightning Data Streaming Benchmark")
    
    # Data parameters
    parser.add_argument("--s3_path", type=str, default=None, 
                        help="S3 path to the dataset. Can also be set via S3_BENCHMARK_DATA_PATH env var.")
    
    # Benchmark parameters
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of worker threads for dataloader")
    parser.add_argument("--prefetch_factor", type=int, default=2, 
                        help="Number of batches to prefetch")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs to run the benchmark")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Size to resize images to")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Cache parameters
    parser.add_argument("--cache_dir", type=str,
                        help="Directory to use for caching dataset files (default: stream/cache/litdata_benchmark)")
    
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
        return
    
    print(f"Using S3 path from environment...")
    
    # Parse S3 path for Lightning Data
    input_dir = parse_s3_path(s3_path)
    
    # Setup cache directory
    cache_dir = args.cache_dir or get_default_cache_dir('litdata')
    setup_cache(cache_dir, clear_existing=True)
    print(f"Using cache directory: {cache_dir}")
    
    # Create dataset and dataloader
    dataset = create_dataset(
        input_dir=input_dir,
        image_size=args.image_size,
        shuffle=True,
        cache_dir=cache_dir
    )
    
    dataloader = create_dataloader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )
    
    try:
        # Run benchmark
        results = run_benchmark(dataloader, args.batch_size, args.epochs)
        print("Benchmark complete")
    finally:
        # Clean up cache
        cleanup_cache(cache_dir)
    

if __name__ == "__main__":
    main()