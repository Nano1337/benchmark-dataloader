#!/usr/bin/env python

"""
Lightning Data Streaming Benchmark

This script benchmarks streaming performance of Lightning Data with image-text pairs.
It measures throughput in images per second and supports various configurations.
"""

import os
import io
import time
import argparse
import random
import shutil
import torch
import torchvision.transforms.v2 as T
from tqdm import tqdm
from PIL import Image
from litdata import StreamingDataset, StreamingDataLoader
from pytorch_lightning import seed_everything
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def setup_cache(cache_dir):
    """Set up cache directory for Lightning Data"""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    return cache_dir


def parse_s3_path(s3_path, litdata_subdir="litdata"):
    """Parse S3 path into path for Lightning Data"""
    # Ensure path has proper format
    if not s3_path:
        print("S3 path is not set. Please check your environment variables.")
        sys.exit(1)
    
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
    
    # Create the Lightning Data path
    rest_parts = [part for part in parts[1:] if part]  # Skip empty parts
    litdata_path = '/'.join(rest_parts + [litdata_subdir])
    
    # Combine into full S3 path
    remote_path = f"s3://{bucket}/{litdata_path}"
    
    print(f"Using remote path: {remote_path}")
    return remote_path


# Create streaming dataset class for image-text pairs
class ImageTextStreamingDataset(StreamingDataset):
    """Lightning Data streaming dataset for image-text data"""
    
    def __init__(self, input_dir, max_cache_size="10GB", image_size=224, shuffle=True, cache_dir=None):
        super().__init__(input_dir=input_dir, max_cache_size=max_cache_size, cache_dir=cache_dir)
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
        
        # Process image - LitData stores image as 'image_bytes'
        if 'image_bytes' in sample:
            # Convert raw bytes to PIL Image first
            try:
                image_bytes = sample['image_bytes']
                image_data = Image.open(io.BytesIO(image_bytes))
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


def create_dataset(input_dir, cache_dir, image_size=224, shuffle=True):
    """Create Lightning Data streaming dataset"""
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
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for Lightning Data. Defaults to './cache/litdata_benchmark'")
    
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
    parser.add_argument("--keep_cache", action="store_true",
                        help="Keep cache after benchmark (default: False)")
    
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
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache/litdata_benchmark")
    cache_dir = setup_cache(cache_dir)
    
    # Get S3 path from args or environment
    s3_path = args.s3_path or os.environ.get('S3_BENCHMARK_DATA_PATH')
    if not s3_path:
        print("No S3 path specified. Please use --s3_path or set S3_BENCHMARK_DATA_PATH environment variable.")
        return
    
    print(f"Using S3 path from environment...")
    
    # Parse S3 path for Lightning Data
    input_dir = parse_s3_path(s3_path)
    
    # Create dataset and dataloader
    try:
        dataset = create_dataset(
            input_dir=input_dir,
            cache_dir=cache_dir,
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
        
        # Cleanup cache if not keeping
        if not args.keep_cache and os.path.exists(cache_dir):
            print(f"Cleaning up cache directory {cache_dir}")
            shutil.rmtree(cache_dir)
            
    except Exception as e:
        print(f"Error in benchmark: {e}")
        if not args.keep_cache and os.path.exists(cache_dir):
            print(f"Cleaning up cache directory {cache_dir}")
            shutil.rmtree(cache_dir)


if __name__ == "__main__":
    main()