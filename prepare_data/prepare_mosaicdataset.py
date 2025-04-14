import os
import sys
from time import time
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser("""Prepare MosaicML Streaming Dataset from parquet file""")
    parser.add_argument(
        "--data", 
        default="./data/benchmark_shard.parquet",
        help="Path to the parquet file containing image-text data"
    )
    parser.add_argument(
        "--output_dir", 
        default="./shards/mds",
        help="Directory to store the output MDS dataset"
    )
    parser.add_argument(
        "--compression", 
        default="",
        help="Compression algorithm to use. Default: None"
    )
    parser.add_argument(
        "--hashes", 
        default="",
        help="Hashing algorithms to apply to shard files. Default: None"
    )
    parser.add_argument(
        "--num_workers", 
        type=int,
        default=None,
        help="Number of worker processes. Default: uses all available CPUs"
    )
    return parser.parse_args()


def ensure_dependencies():
    """Make sure required packages are installed"""
    try:
        import pandas
        from streaming.base import MDSWriter
    except ImportError:
        print("Required packages not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "mosaicml-streaming"])


def main():
    args = parse_args()
    
    # Make sure dependencies are installed
    ensure_dependencies()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Optimize with MosaicML Streaming
    print(f"Converting {args.data} to MDS format in {args.output_dir}")
    t0 = time()
    
    # Use the current Python executable
    python_executable = sys.executable
    
    cmd = [
        python_executable,
        "optimize/mosaicdataset_converter.py",
        "--data", args.data, 
        "--out_dir", args.output_dir
    ]
    
    # Add optional arguments if specified
    if args.compression:
        cmd.extend(["--compression", args.compression])
    
    if args.hashes:
        cmd.extend(["--hashes", args.hashes])
    
    if args.num_workers is not None:
        cmd.extend(["--num_workers", str(args.num_workers)])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    elapsed = time() - t0
    print(f"MDS conversion completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()