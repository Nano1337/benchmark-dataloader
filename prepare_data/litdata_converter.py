#!/usr/bin/env python
# Converter script for LitData format

import os
import sys
import io
import random
import argparse
import pandas as pd
import pyarrow.parquet as pq
from time import time
from tqdm import tqdm
from PIL import Image

from litdata.processing.functions import optimize
from pytorch_lightning import seed_everything


def parse_args():
    """Parse command-line arguments for LitData converter."""
    parser = argparse.ArgumentParser(description="Convert parquet dataset to Lightning.ai's LitData format")
    parser.add_argument(
        '--data',
        type=str,
        default='./data/benchmark_shard.parquet',
        help='Path to the parquet file or directory containing image-text data',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='./shards/litdata',
        help='Directory path to store the output LitData dataset',
    )
    parser.add_argument(
        '--chunk_bytes',
        type=str,
        default='67mb',  # Match 1<<26 (67MB) used by WebDataset/MDS
        help='Maximum chunk size in bytes (e.g. "67mb", "1gb")',
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',  # Enable compression by default
        help='Compression algorithm to use (e.g. "zstd", "lz4")',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of worker processes. Default: uses all available CPUs',
    )
    parser.add_argument(
        '--use_doc_id',
        action='store_true',
        help='Use document_id as key (default: index)',
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='benchmark',
        help='Prefix for dataset identification',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    return parser.parse_args()


def get_parquet_files(data_path):
    """Get list of parquet files from the provided path.
    If data_path is a directory, finds all parquet files within it.
    If data_path is a file, returns it as a single-item list.
    """
    if os.path.isdir(data_path):
        # Find all parquet files in directory
        files = []
        for root, _, filenames in os.walk(data_path):
            for filename in filenames:
                if filename.endswith('.parquet'):
                    files.append(os.path.join(root, filename))
        print(f"Found {len(files)} parquet files in directory: {data_path}")
        return sorted(files)
    elif os.path.isfile(data_path) and data_path.endswith('.parquet'):
        # Single parquet file
        print(f"Using single parquet file: {data_path}")
        return [data_path]
    else:
        raise ValueError(f"Invalid data path: {data_path}. Must be a parquet file or directory containing parquet files.")


def count_samples_in_parquet_files(parquet_files):
    """Efficiently count total number of samples across all parquet files.
    Uses PyArrow to get row counts without loading data into memory.
    """
    print("Counting total number of samples...")
    total_samples = 0
    for file_path in tqdm(parquet_files, desc="Counting samples"):
        try:
            # Get metadata only to efficiently count rows
            metadata = pq.read_metadata(file_path)
            total_samples += metadata.num_rows
        except Exception as e:
            print(f"Error counting samples in {file_path}: {e}")
    return total_samples


def optimize_fn(parquet_file, use_doc_id=False):
    """Processing function for LitData optimize.
    Takes a parquet file path, processes rows, and yields formatted data samples.
    Like WebDataset and MosaicML, we store raw bytes directly instead of PIL objects.
    """
    try:
        # Read parquet file within worker process
        print(f"Worker processing parquet file: {parquet_file}")
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # Process each row
        for idx, row in enumerate(df.itertuples()):
            try:
                # Extract image and text content
                image_bytes = row.image['content']
                text = row.text['content']
                doc_id = row.document_id if hasattr(row, 'document_id') and use_doc_id else f"sample_{idx}"
                
                # Yield formatted data - no decompression needed
                yield {
                    "id": doc_id,
                    "image_bytes": image_bytes,  # Raw bytes, more efficient
                    "text": text
                }
            except Exception as e:
                print(f"Error processing sample {idx} in {parquet_file}: {e}")
                continue
    except Exception as e:
        print(f"Error processing file {parquet_file}: {e}")


def count_output_files(output_dir):
    """Count the number of files in the output directory."""
    file_count = 0
    for _, _, files in os.walk(output_dir):
        file_count += len(files)
    return file_count


if __name__ == '__main__':
    # Record script start time for benchmark
    script_start = time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    random.seed(args.seed)
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Get parquet file paths instead of loading data
    parquet_files = get_parquet_files(args.data)
    print(f"Processing {len(parquet_files)} parquet files...")
    
    # Efficiently count samples across all parquet files
    total_samples = count_samples_in_parquet_files(parquet_files)
    print(f"Found {total_samples} total samples across all parquet files")
    
    # Set number of workers if not specified
    num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
    num_workers = min(num_workers, len(parquet_files))  # Don't use more workers than files
    
    # Create partial function with use_doc_id parameter
    from functools import partial
    process_fn = partial(optimize_fn, use_doc_id=args.use_doc_id)
    
    # Start dataset write timing
    dataset_write_start = time()
    
    # Pass file paths to optimize() and let workers handle data loading/processing
    # NOTE: Must call optimize() directly in __main__ otherwise we get multiprocessing errors
    # related issue: https://github.com/Lightning-AI/litData/issues/305
    optimize(
        fn=process_fn,
        inputs=parquet_files, 
        output_dir=args.out_dir,
        chunk_bytes=args.chunk_bytes, 
        compression=args.compression, 
        num_workers=num_workers,
        reorder_files=False,
        fast_dev_run=False,
    )
    
    # End dataset write timing
    dataset_write_time = time() - dataset_write_start
    
    # Calculate total script time
    script_time = time() - script_start
    
    # Report statistics and timing
    file_count = count_output_files(args.out_dir)
    
    print(f"\nLitData statistics:")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Number of samples: {total_samples}")
    print(f"  Number of files created: {file_count}")
    
    print(f"\nTiming Summary:")
    print(f"  Dataset write time: {dataset_write_time:.2f} seconds")
    print(f"  Total script time: {script_time:.2f} seconds")