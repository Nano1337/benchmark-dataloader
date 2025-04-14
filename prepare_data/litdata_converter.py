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
        help='Path to the parquet file containing image-text data',
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


def load_parquet_data(parquet_file):
    """Load data from parquet file and prepare for conversion."""
    print(f"Reading parquet file: {parquet_file}")
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    print(f"Loaded {len(df)} samples from parquet file")
    return df


def prepare_input_data(df, use_doc_id=False):
    """Prepare input data for LitData optimization."""
    inputs = []
    print("Preparing data samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Extract image and text content
            image_bytes = row['image']['content']
            text = row['text']['content']
            doc_id = row['document_id'] if 'document_id' in row and use_doc_id else f"sample_{idx}"
            
            # Add to inputs list
            inputs.append({
                "idx": idx,
                "doc_id": doc_id,
                "image_bytes": image_bytes,
                "text": text
            })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    return inputs


def optimize_fn(sample):
    """Processing function for LitData optimize.
    Takes a sample dictionary and returns the formatted data.
    Like WebDataset and MosaicML, we store raw bytes directly instead of PIL objects.
    """
    try:
        # Store raw image bytes directly - no need to decompress/recompress
        # This matches how WebDataset and MosaicML store the data
        return {
            "id": sample['doc_id'],
            "image_bytes": sample['image_bytes'],  # Raw bytes, more efficient
            "text": sample['text']
        }
    except Exception as e:
        print(f"Error in optimize_fn: {e}")
        return None


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
    
    # Load and prepare data
    df = load_parquet_data(args.data)
    inputs = prepare_input_data(df, args.use_doc_id)
    print(f"Converting {len(inputs)} samples to LitData format...")
    
    # Set number of workers if not specified
    num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
    
    # Start dataset write timing
    dataset_write_start = time()
    
    # NOTE: Must call optimize() directly in __main__ otherwise we get multiprocessing errors
    # related issue: https://github.com/Lightning-AI/litData/issues/305
    optimize(
        fn=optimize_fn,
        inputs=inputs,
        output_dir=args.out_dir,
        chunk_bytes=args.chunk_bytes,  # Match WebDataset/MDS chunk size
        compression=args.compression,  # Enable compression
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
    print(f"  Number of samples: {len(inputs)}")
    print(f"  Number of files created: {file_count}")
    
    print(f"\nTiming Summary:")
    print(f"  Dataset write time: {dataset_write_time:.2f} seconds")
    print(f"  Total script time: {script_time:.2f} seconds")