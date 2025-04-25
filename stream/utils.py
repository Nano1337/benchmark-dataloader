"""
Common utilities for dataset streaming benchmarks

This module provides standardized functions for image processing, S3 path parsing,
and cache directory management used by all streaming benchmark implementations.
"""

import os
import sys
import io
import shutil
import time
from PIL import Image
import torch
import torchvision.transforms.v2 as T


def parse_s3_path(s3_path, subdir):
    """Parse S3 path into appropriate format
    
    Args:
        s3_path: Base S3 path (e.g., s3://bucket/path or bucket/path)
        subdir: Subdirectory for dataset format (e.g., 'webdataset', 'mds', 'litdata')
    
    Returns:
        Formatted path for the specific dataset format
    """
    # Ensure path has proper format
    if not s3_path:
        print("S3 path is not set. Please check your environment variables.")
        sys.exit(1)
    
    # Remove s3:// prefix if present
    if s3_path.startswith('s3://'):
        path = s3_path[5:]
    else:
        path = s3_path
        
    # Ensure path ends with a slash for consistent joining
    if not path.endswith('/'):
        path += '/'
    
    # Split into bucket and prefix
    parts = path.split('/')
    bucket = parts[0]
    
    # Create the dataset-specific path
    rest_parts = [part for part in parts[1:] if part]  # Skip empty parts
    dataset_path = '/'.join(rest_parts + [subdir])
    
    # Combine into full S3 path
    remote_path = f"s3://{bucket}/{dataset_path}"
    
    print(f"Using remote path: {remote_path}")
    return remote_path, bucket, dataset_path


def process_image(img_bytes):
    """Process image data from bytes and ensure consistent format
    
    Args:
        img_bytes: Raw image bytes or PIL Image
        
    Returns:
        PIL Image with consistent format
    """
    if isinstance(img_bytes, bytes):
        img = Image.open(io.BytesIO(img_bytes))
        # Convert grayscale images to RGB to ensure consistent channels
        if img.mode == 'L':
            img = img.convert('RGB')
        return img
    return img_bytes


def process_text(text):
    """Process text data from dataset
    
    Args:
        text: Text data as bytes or string
        
    Returns:
        Text as UTF-8 string
    """
    if isinstance(text, bytes):
        return text.decode('utf-8')
    return text


def create_transforms(image_size=224, to_dtype=torch.float16):
    """Create standardized image transforms for preprocessing
    
    Args:
        image_size: Target image size (default: 224)
        to_dtype: Target tensor dtype (default: torch.float16)
        
    Returns:
        torchvision.transforms composition
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, antialias=True),
        T.RandomHorizontalFlip(),
        T.ToImage(),
        # Ensure all images have 3 channels (RGB)
        T.Lambda(lambda x: x if x.shape[0] == 3 else torch.cat([x]*3, dim=0) if x.shape[0] == 1 else x[:3]),
        T.ToDtype(to_dtype, scale=True),
    ])


def to_rgb(img):
    """Convert image to RGB format
    
    This is a legacy function that is kept for backward compatibility.
    New code should use process_image or the transforms pipeline.
    
    Args:
        img: Image as tensor or PIL Image
        
    Returns:
        RGB version of the image
    """
    if isinstance(img, torch.Tensor):
        if img.shape[0] == 1:
            img = img.repeat((3, 1, 1))
        if img.shape[0] == 4:
            img = img[:3]
    else:
        if img.mode == "L":
            img = img.convert('RGB')
    return img


def get_default_cache_dir(dataset_name):
    """Get the default cache directory for a specific dataset format
    
    Args:
        dataset_name: Name of the dataset format (e.g., 'webdataset', 'mds', 'litdata', 'energon')
        
    Returns:
        Default path to the cache directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, f"cache/{dataset_name}_benchmark")


def setup_cache(cache_dir, clear_existing=True):
    """Set up cache directory for dataset streaming
    
    Args:
        cache_dir: Path to cache directory
        clear_existing: Whether to clear the existing cache directory if it exists (default: True)
        
    Returns:
        Path to the cache directory
    """
    # Remove the directory if it exists and clear_existing is True
    if clear_existing and os.path.exists(cache_dir):
        print(f"Cleaning existing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
        
    # Create the directory
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    return cache_dir


def cleanup_cache(cache_dir, keep_cache=False):
    """Clean up a cache directory after benchmark
    
    Args:
        cache_dir: Path to cache directory
        keep_cache: Whether to keep the cache (default: False)
        
    Returns:
        None
    """
    if not keep_cache and os.path.exists(cache_dir):
        print(f"Cleaning up cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
    elif keep_cache and os.path.exists(cache_dir):
        print(f"Keeping cache directory: {cache_dir}")
    else:
        print(f"Cache directory not found: {cache_dir}")