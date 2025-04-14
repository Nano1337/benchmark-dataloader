import os
from time import time
import subprocess
import argparse
import sys
def parse_args():
    parser = argparse.ArgumentParser("""Prepare WebDataset from parquet file""")
    parser.add_argument(
        "--data", 
        default="./data/benchmark_shard.parquet",
        help="Path to the parquet file containing image-text data"
    )
    parser.add_argument(
        "--output_dir", 
        default="./shards/webdataset",
        help="Directory to store the output WebDataset shards"
    )
    parser.add_argument(
        "--prefix", 
        default="benchmark",
        help="Prefix for the output shard files"
    )
    parser.add_argument(
        "--use_doc_id", 
        action="store_true", 
        help="Use document_id as key (default: index)"
    )
    return parser.parse_args()


def main():
    script_start = time()
    args = parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Optimize with Webdataset
    print(f"Converting {args.data} to WebDataset format in {args.output_dir}")
    
    # Use the current Python executable
    python_executable = sys.executable
    
    cmd = [
        python_executable,
        "prepare_data/webdataset_converter.py",  
        "--data", args.data, 
        "--shards", args.output_dir,
        "--prefix", args.prefix
    ]
    
    # Add use_doc_id flag if specified
    if args.use_doc_id:
        cmd.append("--use_doc_id")
    
    print(f"Running: {' '.join(cmd)}")
    
    # Execute converter with timing
    converter_start = time()
    subprocess.run(cmd, check=True)
    converter_elapsed = time() - converter_start
    
    # Calculate and report total script time
    script_elapsed = time() - script_start
    print(f"\nTiming Summary:")
    print(f"  Converter execution time: {converter_elapsed:.2f} seconds")
    print(f"  Total script time: {script_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()