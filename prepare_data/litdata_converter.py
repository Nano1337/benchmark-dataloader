#!/usr/bin/env python
# Optimized converter script for LitData format

import os
import io
import argparse
import random
import pyarrow.parquet as pq
from time import time
from tqdm import tqdm
from PIL import Image
import numpy as np

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
        default='64MB',
        help='Maximum chunk size in bytes (e.g. "64MB", "1GB")',
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=1000,
        help='Maximum number of samples per chunk',
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',
        help='Compression algorithm to use (e.g. "zstd", "lz4")',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,  # Default to all available CPUs
        help='Number of worker processes',
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
        '--resize',
        action='store_true',
        help='Whether to resize images to (224, 224)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    return parser.parse_args()


def get_parquet_files(data_path):
    """Get list of parquet files from the provided path efficiently."""
    if os.path.isdir(data_path):
        # Find all parquet files in directory
        files = []
        for root, _, filenames in os.walk(data_path):
            files.extend([
                os.path.join(root, filename)
                for filename in filenames
                if filename.endswith('.parquet')
            ])
        print(f"Found {len(files)} parquet files in directory: {data_path}")
        return sorted(files)
    elif os.path.isfile(data_path) and data_path.endswith('.parquet'):
        # Single parquet file
        print(f"Using single parquet file: {data_path}")
        return [data_path]
    else:
        raise ValueError(f"Invalid data path: {data_path}. Must be a parquet file or directory.")


def optimize_fn(parquet_file, args):
    """Processing function for LitData optimize.
    Takes an entire parquet file path and processes it efficiently.
    All work is done within the worker process.
    """
    try:
        # Read just the file metadata to get row count
        metadata = pq.read_metadata(parquet_file)
        total_rows = metadata.num_rows
        print(f"Worker processing parquet file: {parquet_file} ({total_rows} rows)")
        
        # Open the file as a ParquetFile object to enable true batch-based reading
        parquet_object = pq.ParquetFile(parquet_file)
        
        # Process in batches using iter_batches so we don't load the whole shard into RAM
        batch_size = 2000
        row_index = 0
        
        # Iterate through batches directly
        for batch in parquet_object.iter_batches(batch_size=batch_size):
            # Convert batch to table and then to Python objects
            batch_table = batch.to_pandas()
            
            # Process each row in the batch
            for _, row in batch_table.iterrows():
                try:
                    # Extract image and text content
                    image_bytes = row['image']['content']
                    text = row['text']['content'] if 'text' in row else None
                    
                    # Use document_id if available and requested
                    doc_id = None
                    if args.use_doc_id and 'document_id' in row:
                        doc_id = row['document_id']
                    else:
                        doc_id = f"sample_{row_index}"
                    
                    # Process image if resizing is enabled
                    if args.resize:
                        # Convert bytes to PIL image, resize, and back to bytes
                        img = Image.open(io.BytesIO(image_bytes))
                        img = img.resize((224, 224))
                        buff = io.BytesIO()
                        img.convert('RGB').save(buff, format="JPEG")
                        image_bytes = buff.getvalue()
                    
                    # Yield the processed sample
                    sample_data = {
                        "image_bytes": image_bytes,
                        "text": text
                    }
                    
                    # Add ID if using document_id
                    if args.use_doc_id:
                        sample_data["id"] = doc_id
                    
                    yield sample_data
                    row_index += 1
                    
                except Exception as e:
                    print(f"Error processing sample {row_index} in {parquet_file}: {str(e)}")
                    row_index += 1
                    continue
                    
    except Exception as e:
        print(f"Error processing file {parquet_file}: {str(e)}")


def count_output_files(output_dir):
    """Count the number of files in the output directory."""
    file_count = 0
    for _, _, files in os.walk(output_dir):
        file_count += len(files)
    return file_count


def count_total_samples(parquet_files):
    """Count total number of samples in all parquet files efficiently."""
    print("Counting total samples in parquet files...")
    total_samples = 0
    for file_path in tqdm(parquet_files, desc="Counting samples"):
        try:
            # Get metadata only to efficiently count rows
            metadata = pq.read_metadata(file_path)
            total_samples += metadata.num_rows
        except Exception as e:
            print(f"Error counting samples in {file_path}: {e}")
    
    return total_samples


if __name__ == '__main__':
    # Record script start time for benchmark
    script_start = time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Set number of workers
    max_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
    print(f"Using {max_workers} worker processes")
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    np.random.seed(args.seed)
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Get parquet file paths
    parquet_files = get_parquet_files(args.data)
    
    # Create partially applied function with args
    from functools import partial
    process_fn = partial(optimize_fn, args=args)
    
    print(f"Processing {len(parquet_files)} parquet files with {max_workers} workers...")
    
    # Start dataset write timing
    dataset_write_start = time()
    
    # Run the optimization with improved parallelism
    optimize(
        fn=process_fn,
        inputs=parquet_files,
        output_dir=args.out_dir,
        chunk_bytes=args.chunk_bytes,
        compression=args.compression,
        num_workers=max_workers,
        reorder_files=False,
    )
    
    # End dataset write timing
    dataset_write_time = time() - dataset_write_start
    
    # Count total samples from parquet files (outside of dataset write timing)
    sample_count_start = time()
    total_samples = count_total_samples(parquet_files)
    sample_count_time = time() - sample_count_start
    
    # Calculate total script time
    script_time = time() - script_start
    
    # Report statistics and timing
    file_count = count_output_files(args.out_dir)
    
    print(f"\nLitData statistics:")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Number of samples in parquet files: {total_samples}")
    print(f"  Number of files created: {file_count}")
    print(f"  Prefix: {args.prefix}")
    
    # Format timing exactly as prepare_datasets.sh expects to extract with grep
    print(f"\nTiming Summary:")
    print(f"  Dataset write time: {dataset_write_time:.2f} seconds")
    print(f"  Total script time: {script_time:.2f} seconds")