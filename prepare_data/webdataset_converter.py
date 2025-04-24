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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# First, make sure webdataset is installed
try:
    import webdataset as wds
    from webdataset.writer import ShardWriter
except ImportError:
    print("webdataset package not found. Installing...") 
    import subprocess
    subprocess.check_call(["pip", "install", "webdataset"])
    import webdataset as wds
    from webdataset.writer import ShardWriter


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
parser.add_argument(
    "--num_workers",
    type=int,
    default=None,
    help="Number of worker processes for parallel processing. Default: use all available CPUs",
)
args = parser.parse_args()


assert args.maxsize > 10000000, "maxsize should be at least 10MB"
assert args.maxcount < 1000000, "maxcount should be less than 1 million"


def process_batch(batch_data):
    """Process a batch of samples in a worker process
    This reduces the overhead of process creation and IPC
    """
    results = []
    for i, row in batch_data:
        try:
            # Get image bytes and text content
            image_bytes = row['image']['content']
            text = row['text']['content']
            
            # Generate a unique key
            key = f"train_{i:07d}"
            
            # Construct a sample with image and text
            sample = {
                "__key__": key,
                "jpg": image_bytes,  # Image bytes stored with jpg extension
                "txt": text.encode('utf-8')  # Text encoded as bytes
            }
            
            results.append(sample)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    
    return results


def write_dataset(df, base="./shards"):
    """Write a WebDataset from the parquet dataframe."""
    # Make sure the output directory exists
    os.makedirs(base, exist_ok=True)
    
    # Get indices in random order
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    # Determine if we should use parallel processing
    num_workers = args.num_workers if args.num_workers is not None else multiprocessing.cpu_count()
    total_samples = len(df)
    
    # Only use parallel processing if we have multiple workers and enough samples
    use_parallel = num_workers > 1 and total_samples > 1000
    
    # If parallel processing isn't beneficial, default to single process
    if not use_parallel:
        num_workers = 1
    
    print(f"Using {num_workers} worker process{'es' if num_workers > 1 else ''}")
    
    # Create a shard pattern
    split = "train"  # Always use train
    shard_pattern = os.path.join(base, f"{args.prefix}-{split}-%06d.tar")
    
    # Optimize for better parallel performance: work in batches
    batch_size = 100  # Size of batches to reduce process overhead
    
    # Create batches of indices
    batches = []
    for i in range(0, total_samples, batch_size):
        end = min(i + batch_size, total_samples)
        batch_indices = indices[i:end]
        batch = [(idx, df.iloc[idx]) for idx in batch_indices]
        batches.append(batch)
    
    print(f"Total samples: {total_samples}, divided into {len(batches)} batches")
    
    # Use ShardWriter to handle shard creation and management
    processed_count = 0
    
    # Start the ShardWriter
    with ShardWriter(shard_pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
        # Single worker case: process directly
        if num_workers == 1:
            for batch in tqdm(batches, desc="Processing batches"):
                for sample in process_batch(batch):
                    sink.write(sample)
                    processed_count += 1
        # Multi-worker case: use ProcessPoolExecutor
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit batch processing tasks
                future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
                
                # Process results as they complete
                for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
                    try:
                        # Get processed samples from the batch and write them
                        batch_results = future.result()
                        for sample in batch_results:
                            sink.write(sample)
                            processed_count += 1
                    except Exception as e:
                        batch_idx = future_to_batch[future]
                        print(f"Batch {batch_idx} generated an exception: {e}")
    
    print(f"Successfully processed and wrote {processed_count} samples")


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