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
from utils import create_transforms, parse_s3_path as utils_parse_s3_path

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
    
    def __init__(self, remote, batch_size, image_size=224, shuffle=True):
        super().__init__(remote=remote, shuffle=shuffle, batch_size=batch_size)
        self.transform = create_transforms(image_size, torch.float16)

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


def create_dataset(remote_path, batch_size, image_size=224, shuffle=True):
    """Create MosaicML streaming dataset"""
    try:
        dataset = ImageTextStreamingDataset(
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
    
    # Performance parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training (default: 256)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads for data loading (default: 8)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch (default: 2)")

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
    
    # Parse S3 path to get remote MosaicML path
    remote_path = parse_s3_path(s3_path, args.mds_subdir)
    
    # Create dataset and dataloader
    dataset = create_dataset(
        remote_path=remote_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True
    )
        
    dataloader = create_dataloader(
        dataset, 
        prefetch_factor=args.prefetch_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
        
    print(f"Using {args.num_workers} worker threads for data loading")
        
    # Run benchmark
    results = run_benchmark(dataloader, args.batch_size, args.epochs)
    print("Benchmark complete")
    
    return results


if __name__ == "__main__":
    main()