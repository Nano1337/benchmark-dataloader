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
from lightning import seed_everything

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
        '--chunk_size',
        type=int,
        default=1000,
        help='Maximum number of samples per chunk',
    )
    parser.add_argument(
        '--chunk_bytes',
        type=str,
        default='67MB',  # Similar to WebDataset default of 2<<26
        help='Maximum chunk size in bytes (e.g. "67MB", "1GB")',
    )
    parser.add_argument(
        '--compression',
        type=str,
        default=None,
        help='Compression algorithm to use (e.g. "zstd")',
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


def convert_parquet_to_litdata(args):
    """Convert a parquet file with image-text pairs to LitData format."""
    print(f"Reading parquet file: {args.data}")
    table = pq.read_table(args.data)
    df = table.to_pandas()
    print(f"Loaded {len(df)} samples from parquet file")
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    random.seed(args.seed)
    
    # Create input data for optimization
    inputs = []
    
    print("Preparing data samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Extract image and text content
            image_bytes = row['image']['content']
            text = row['text']['content']
            doc_id = row['document_id'] if 'document_id' in row and args.use_doc_id else f"sample_{idx}"
            
            # Add to inputs list
            inputs.append({
                "idx": idx,
                "doc_id": doc_id,
                "image_bytes": image_bytes,
                "text": text
            })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    # Make output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Converting {len(inputs)} samples to LitData format...")
    
    # Define the processing function
    def process_sample(sample):
        """Process a single sample for LitData format."""
        try:
            # Open image from bytes
            image_stream = io.BytesIO(sample['image_bytes'])
            image = Image.open(image_stream).convert('RGB')
            
            # Return formatted sample
            return {
                "id": sample['doc_id'],
                "image": image,
                "text": sample['text']
            }
        except Exception as e:
            print(f"Error in process_sample: {e}")
            return None

    # Start dataset write timing
    dataset_write_start = time()
    
    # Use LitData's optimize function to create the dataset
    optimize(
        fn=process_sample,
        inputs=inputs,
        output_dir=args.out_dir,
        chunk_size=args.chunk_size,
        chunk_bytes=args.chunk_bytes,
        compression=args.compression,
        num_workers=args.num_workers,
    )
    
    # End dataset write timing
    dataset_write_time = time() - dataset_write_start
    print(f"Dataset write time: {dataset_write_time:.2f} seconds")
    
    # Report statistics
    print(f"\nLitData statistics:")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Number of samples: {len(inputs)}")
    
    # Count files in the output directory
    file_count = 0
    for _, _, files in os.walk(args.out_dir):
        file_count += len(files)
    print(f"  Number of files created: {file_count}")
    
    return dataset_write_time


def main():
    # Parse arguments
    args = parse_args()
    
    # Set up timing
    script_start = time()
    
    # Execute conversion
    dataset_write_time = convert_parquet_to_litdata(args)
    
    # Calculate total script execution time
    script_time = time() - script_start
    
    # Report timing information
    print(f"\nTiming Summary:")
    print(f"  Dataset write time: {dataset_write_time:.2f} seconds")
    print(f"  Total script time: {script_time:.2f} seconds")
    
    return script_time, dataset_write_time


if __name__ == '__main__':
    script_time, dataset_write_time = main()