#
# Modified from the WebDataset library.
# Original copyright (c) 2017-2023 NVIDIA CORPORATION. All rights reserved.
#

import os
import os.path
import random
import argparse
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# First, make sure webdataset is installed
try:
    import webdataset
    from webdataset.writer import TarWriter
except ImportError:
    print("webdataset package not found. Installing...") 
    import subprocess
    subprocess.check_call(["pip", "install", "webdataset"])
    import webdataset
    from webdataset.writer import TarWriter


parser = argparse.ArgumentParser("""Generate sharded WebDataset from parquet data.""")
parser.add_argument(
    "--maxsize", 
    type=float, 
    default=2 << 26,
    help="Maximum size of each shard in bytes"
)
parser.add_argument(
    "--maxcount", 
    type=float, 
    default=10000,
    help="Maximum number of samples per shard"
)
parser.add_argument(
    "--shards", 
    default="./shards", 
    help="Directory where shards are written"
)
parser.add_argument(
    "--data",
    default="./data/benchmark_dataset.parquet",
    help="Path to the parquet file containing image-text data",
)
parser.add_argument(
    "--prefix",
    default="dataset",
    help="Prefix for the output shard files",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for shuffling",
)
args = parser.parse_args()


assert args.maxsize > 10000000, "maxsize should be at least 10MB"
assert args.maxcount < 1000000, "maxcount should be less than 1 million"


def write_dataset(df, base="./shards"):
    """Write a WebDataset from the parquet dataframe."""
    # Make sure the output directory exists
    os.makedirs(base, exist_ok=True)
    
    # Get indices in random order
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    # Create shards with proper naming pattern
    split = "train"  # Always use train
    shard_name_pattern = os.path.join(base, f"{args.prefix}-{split}-%06d.tar")
    
    # Track keys to ensure uniqueness
    all_keys = set()
    
    # Initialize counters
    shard_index = 0
    samples_in_shard = 0
    current_shard_size = 0
    current_writer = None
    
    try:
        for i in tqdm(indices, desc="Processing dataset"):
            # Create a new shard if needed
            if current_writer is None or \
               samples_in_shard >= int(args.maxcount) or \
               current_shard_size >= int(args.maxsize):
                
                # Close previous writer if it exists
                if current_writer is not None:
                    current_writer.close()
                
                # Create a new writer with the next shard index
                shard_path = shard_name_pattern % shard_index
                current_writer = TarWriter(shard_path)
                shard_index += 1
                samples_in_shard = 0
                current_shard_size = 0
            
            row = df.iloc[i]
            
            # Get image bytes and text content
            image_bytes = row['image']['content']
            text = row['text']['content']
            
            # Generate a unique key
            key = f"train_{i:07d}"
            
            # Ensure key uniqueness
            if key in all_keys:
                print(f"Warning: Duplicate key {key}, appending random suffix")
                key = f"{key}_{random.randint(0, 999999)}"
            
            all_keys.add(key)
            
            # Construct a sample with image and text
            sample = {
                "__key__": key,
                "jpg": image_bytes,  # Image bytes stored with jpg extension
                "txt": text.encode('utf-8')  # Text encoded as bytes
            }
            
            # Write the sample to the current tar archive
            current_writer.write(sample)
            
            # Update counters
            samples_in_shard += 1
            # Approximate size tracking (very rough estimate)
            current_shard_size += len(image_bytes) + len(text) + 100  # add overhead
    
    finally:
        # Close the final writer
        if current_writer is not None:
            current_writer.close()


def main():
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Read the parquet file
    print(f"Reading parquet file: {args.data}")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} samples from parquet file")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.shards, exist_ok=True)
    
    # Process all data as a single split with timing
    print(f"Processing all {len(df)} samples as train split")
    from time import time
    write_start = time()
    write_dataset(df, base=args.shards)
    write_elapsed = time() - write_start
    
    print(f"Dataset processed. Output in {args.shards}/")
    print(f"Dataset write time: {write_elapsed:.2f} seconds")


if __name__ == "__main__":
    from time import time
    script_start = time()
    main()
    script_elapsed = time() - script_start
    print(f"Total script execution time: {script_elapsed:.2f} seconds")