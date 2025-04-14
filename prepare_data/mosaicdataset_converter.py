from time import time
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Check and install streaming package if needed
try:
    from streaming.base import MDSWriter
    from streaming.base.util import get_list_arg
except ImportError:
    print("MosaicML streaming package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mosaicml-streaming"])
    from streaming.base import MDSWriter
    from streaming.base.util import get_list_arg


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert parquet dataset to MosaicML streaming format")
    parser.add_argument(
        '--data',
        type=str,
        default='./data/benchmark_shard.parquet',
        help='Path to the parquet file containing image-text data',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='./shards/mds',
        help='Directory path to store the output MDS dataset',
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='',
        help='Compression algorithm to use. Default: None',
    )
    parser.add_argument(
        '--hashes',
        type=str,
        default='',
        help='Hashing algorithms to apply to shard files. Default: None',
    )
    parser.add_argument(
        '--size_limit',
        type=int,
        default=1 << 26,  # 67MB
        help='Shard size limit, after which point to start a new shard. Default: 1 << 26',
    )
    parser.add_argument(
        '--progress_bar',
        type=int,
        default=1,
        help='Use tqdm progress bar. Default: 1 (True)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of worker processes. Default: uses all available CPUs',
    )
    return parser.parse_args()


def main(args):
    """Convert parquet dataset to MosaicML streaming format."""
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Read parquet data
    print(f"Reading parquet file: {args.data}")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} samples from parquet file")
    
    # Shuffle indices
    indices = np.random.permutation(len(df))
    
    # Set up the columns for the MDS dataset
    # We'll store image bytes, text content, and document ID
    columns = {
        'doc_id': 'str',      # document ID 
        'image': 'jpeg',      # image bytes
        'text': 'str',        # caption text
        'idx': 'int'          # sample index
    }
    
    # Get hashes if specified
    hashes = get_list_arg(args.hashes)
    
    # Set number of workers
    max_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
    print(f"Using {max_workers} worker processes")
    
    # Write the dataset
    print(f"Writing {len(df)} samples to MDS format at: {args.out_dir}")
    with MDSWriter(
        out=args.out_dir,
        columns=columns,
        compression=args.compression,
        hashes=hashes,
        size_limit=args.size_limit,
        progress_bar=bool(args.progress_bar),
        max_workers=max_workers,
    ) as out:
        for i in tqdm(indices):
            row = df.iloc[i]
            
            # Get document_id, image bytes and text
            doc_id = str(row['document_id'])
            image_bytes = row['image']['content']
            text = row['text']['content']
            
            # Write the sample
            out.write({
                'doc_id': doc_id,
                'image': image_bytes,
                'text': text,
                'idx': int(i)  # Cast to int to avoid numpy type issues
            })
    
    print(f"Successfully wrote MDS dataset to {args.out_dir}")


if __name__ == '__main__':
    t0 = time()
    main(parse_args())
    elapsed = time() - t0
    print(f"Conversion completed in {elapsed:.2f} seconds")