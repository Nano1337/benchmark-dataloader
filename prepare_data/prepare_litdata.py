#!/usr/bin/env python

import os
import sys
from time import time
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser("""Prepare LitData dataset from parquet file""")
    parser.add_argument(
        "--data", 
        default="./data/benchmark_dataset.parquet",
        help="Path to the parquet file containing image-text data"
    )
    parser.add_argument(
        "--output_dir", 
        default="./shards/litdata",
        help="Directory to store the output LitData dataset"
    )
    parser.add_argument(
        "--prefix", 
        default="benchmark",
        help="Prefix for the output dataset"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Maximum number of samples per chunk"
    )
    parser.add_argument(
        "--chunk_bytes",
        type=str,
        default="64MB",
        help="Maximum chunk size in bytes (e.g. '64MB', '1GB')"
    )
    parser.add_argument(
        "--compression",
        default=None,
        help="Compression algorithm to use (e.g. 'zstd')"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes. Default: uses all available CPUs"
    )
    return parser.parse_args()


def main():
    script_start = time()
    args = parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use the current Python executable
    python_executable = sys.executable
    
    # Print info
    print(f"Converting {args.data} to LitData format in {args.output_dir}")
    
    # Build command
    cmd = [
        python_executable,
        "prepare_data/litdata_converter.py",  
        "--data", args.data, 
        "--out_dir", args.output_dir,
        "--prefix", args.prefix
    ]
    
    if args.compression:
        cmd.extend(["--compression", args.compression])
        
    # Add num_workers if specified
    if args.num_workers is not None:
        cmd.extend(["--num_workers", str(args.num_workers)])
    
    print(f"Running: {' '.join(cmd)}")
    
    # Execute converter with timing
    converter_start = time()
    subprocess.run(cmd, check=True)
    converter_elapsed = time() - converter_start
    
    # Calculate and report total script time
    script_elapsed = time() - script_start
    print(f"\nTiming Summary:")
    print(f"  Dataset write time: {converter_elapsed:.2f} seconds")
    print(f"  Total script time: {script_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()