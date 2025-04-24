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
from utils import create_transforms, process_image, process_text, parse_s3_path as utils_parse_s3_path

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
    
    def __init__(self, input_dir, max_cache_size="25GB", image_size=224, shuffle=True):
        super().__init__(input_dir=input_dir, max_cache_size=max_cache_size)
        self.transform = create_transforms(image_size, torch.float16)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        Returns: image tensor, text string
        """
        sample = super().__getitem__(index)
        
        # Process image - LitData stores image as 'image_bytes'
        if 'image_bytes' in sample:
            # Convert raw bytes to PIL Image using our utility function
            try:
                image_bytes = sample['image_bytes']
                image_data = process_image(image_bytes)
            except Exception as e:
                print(f"Error loading image bytes: {e}")
                image_data = torch.zeros((3, 224, 224), dtype=torch.float16)
        elif 'image' in sample:
            image_data = sample['image']
        elif 'x' in sample:
            image_data = sample['x']
        else:
            # Default to the first item if keys don't match expected
            image_data = next(iter(sample.values()))
        
        # Process text - assuming 'text' contains the caption data
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


def create_dataset(input_dir, image_size=224, shuffle=True):
    """Create Lightning Data streaming dataset"""
    try:
        max_cache_size = "25GB"
        dataset = ImageTextStreamingDataset(
            input_dir=input_dir,
            max_cache_size=max_cache_size,
            image_size=image_size,
            shuffle=shuffle
        )
        return dataset
    except Exception as e:
        print(f"Error creating dataset: {e}")
        raise


def create_dataloader(dataset, batch_size=256, num_workers=8):
    """Create dataloader for Lightning Data dataset"""
    print(f"Using {num_workers} worker threads for data loading")
    dataloader = StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return dataloader


def run_benchmark(dataloader, batch_size, num_epochs=2):
    """Run streaming benchmark for specified number of epochs"""
    print(f"Starting benchmark with batch size {batch_size}")
    
    total_samples = 0
    total_time = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        num_samples = 0
        
        # Iterate through the dataloader
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", smoothing=0, mininterval=1):
            # Get batch size from actual data
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                batch_images = batch[0]
                num_samples += batch_images.shape[0]
            else:
                num_samples += batch_size  # Fallback if we can't determine batch size
        
        elapsed = time.time() - start_time
        throughput = num_samples / elapsed
        
        print(f"Epoch {epoch+1}: Processed {num_samples} samples in {elapsed:.2f}s ({throughput:.2f} images/sec)")
        
        total_samples += num_samples
        total_time += elapsed
    
    # Report overall statistics
    avg_throughput = total_samples / total_time
    print(f"\nBenchmark Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average throughput: {avg_throughput:.2f} images/sec")
    print("Benchmark complete")
    
    return {
        "total_samples": total_samples,
        "total_time": total_time,
        "avg_throughput": avg_throughput
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
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs to run the benchmark")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Size to resize images to")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
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
    
    # Create dataset and dataloader
    dataset = create_dataset(
        input_dir=input_dir,
        image_size=args.image_size,
        shuffle=True
    )
    
    dataloader = create_dataloader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Run benchmark
    results = run_benchmark(dataloader, args.batch_size, args.epochs)
    

if __name__ == "__main__":
    main()