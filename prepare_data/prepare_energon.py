import os
from time import time
import subprocess
import argparse
import sys
import glob
import shutil
def parse_args():
    parser = argparse.ArgumentParser("""Prepare WebDataset from parquet file""")
    parser.add_argument(
        "--data", 
        default="./shards/webdataset",
        help="Path to the WDS containing image-text data"
    )
    parser.add_argument(
        "--output_dir", 
        default="./shards/energon",
        help="Directory to store the output energon shards"
    )
    parser.add_argument(
        "--dataset_yaml", 
        default="./prepare_data/dataset.yaml",
        help="Path to the dataset yaml file"
    )
    parser.add_argument(
        "--prefix", 
        default="benchmark",
        help="Prefix for the output shard files"
    )
    parser.add_argument(
        "--webdataset_time", 
        type=float,
        default=0.0,
        help="Total time taken by WebDataset preparation (seconds)"
    )
    parser.add_argument(
        "--webdataset_write_time", 
        type=float,
        default=0.0,
        help="Write time taken by WebDataset preparation (seconds)"
    )
    parser.add_argument(
        "--num_workers", 
        type=int,
        default=None,
        help="Number of worker processes. Default: uses all available CPUs"
    )
    return parser.parse_args()


def main():
    # Start timing for the entire script
    script_start = time()
    args = parse_args()
    
    print(f"Starting Energon conversion at {time()}")
    print(f"CPU count: {os.cpu_count()}")
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Optimize with Energon
    print(f"\n==========================================")
    print(f"   ENERGON BENCHMARK")
    print(f"   {time()}")
    print(f"==========================================")
    print(f"\nConverting {args.data} to Energon format in {args.output_dir}")

    # Copy data contents into output directory using glob pattern matching
    source_files = glob.glob(os.path.join(args.data, "*"))
    if not source_files:
        print(f"Error: No files found in {args.data}")
        sys.exit(1)
        
    print(f"Copying {len(source_files)} files from {args.data} to {args.output_dir}")
    for file in source_files:
        dest = os.path.join(args.output_dir, os.path.basename(file))
        print(f"Copying {file} to {dest}")
        if os.path.isdir(file):
            shutil.copytree(file, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(file, dest)

    # Run energon prepare with automatic input responses
    cmd = [
        "energon", "prepare", "./"
    ]
    
    # Create process with pipe for stdin so we can send inputs
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=args.output_dir
    )
    
    print(f"Running: {' '.join(cmd)}")
    print("Automatically sending inputs: '1,0,0' and 'n'")
    
    # Execute converter with timing
    converter_start = time()
    
    # Send first input: "1,0,0"
    process.stdin.write("1,0,0\n")
    process.stdin.flush()
    
    # Send second input: "n"
    process.stdin.write("n\n")
    process.stdin.flush()
    
    # Get output and wait for process to complete
    stdout, stderr = process.communicate()
    
    # Print output
    if stdout:
        print("Output:", stdout)
    if stderr:
        print("Errors:", stderr)
    
    # Check return code
    if process.returncode != 0:
        print(f"Error: Energon prepare failed with code {process.returncode}")
        sys.exit(process.returncode)

    # Copy dataset.yaml into output_dir/.nv-meta/
    copy_cmd = ["cp", args.dataset_yaml, os.path.join(args.output_dir, ".nv-meta", "dataset.yaml")]
    subprocess.run(copy_cmd, check=True)
    
    converter_elapsed = time() - converter_start
    
    # Calculate and report dataset write time and total script time
    script_elapsed = time() - script_start
    
    # Add WebDataset times to Energon times
    total_write_time = converter_elapsed + args.webdataset_write_time
    total_time = script_elapsed + args.webdataset_time
    
    print(f"\nTiming Summary:")
    print(f"WebDataset preparation time: {args.webdataset_time:.2f} seconds")
    print(f"WebDataset write time: {args.webdataset_write_time:.2f} seconds")
    print(f"Energon converter execution time: {converter_elapsed:.2f} seconds")
    print(f"Energon dataset write time: {converter_elapsed:.2f} seconds")  # Using converter time as the write time
    print(f"Energon script time: {script_elapsed:.2f} seconds")
    print(f"\nCombined Metrics:")
    print(f"Converter execution time: {converter_elapsed:.2f} seconds")  # For backwards compatibility
    print(f"Dataset write time: {total_write_time:.2f} seconds")  # Combined write time
    print(f"Total script time: {total_time:.2f} seconds")  # Combined total time


if __name__ == "__main__":
    main()