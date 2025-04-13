from time import time
import os
import io
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Any
import numpy as np
from lightning.data import optimize, walk
import json
from lightning import seed_everything
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from utils import load_imagenet_val_class_names, class_names_to_index_map


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Args:
        Namespace: command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--input_dir',
        default="/teamspace/studios/imagenet-1m/data/train",
        type=str,
        required=True,
        help='Input directory where the data lives',
    )
    args.add_argument(
        '--output_dir',
        default="/teamspace/datasets/imagenet/train",
        type=str,
        required=True,
        help='Output directory where the optimized dataset will be stored',
    )
    args.add_argument(
        '--resize',
        default="False",
        action='store_true',
        help='Whether to resize the images to (224, 244).',
    )
    args.add_argument(
        '--filepath_only',
        default="False",
        action='store_true',
        help='Whether to store only the image and the original filepath.',
    )
    args.add_argument(
        '--use_jpeg_90',
        default="False",
        action='store_true',
        help='Whether to store the images as JPEG.',
    )
    return args.parse_args()


def get_classes(input_dir: str) -> Tuple[List[int], List[str]]:
    """Get the classes for a dataset split of sample image filenames."""
    with open(os.path.join(os.path.expanduser('~'), "imagenet_class_index.json"), "r") as f:
        data = json.load(f)

    return {v[0]: int(k) for k, v in data.items()}

def get_class_from_filepath(filepath: str, classes) -> int:
    class_name = filepath.split("/")[-2]
    return classes[class_name]


def get_inputs(input_dir: str) -> Any:
    classes = get_classes(input_dir)
    filepaths = np.random.permutation([
        os.path.join(root, filename)
        for root, _, filenames in tqdm(walk(input_dir), smoothing=0)
        for filename in filenames]
    )
    if "train" in input_dir:
        return [(filepath, get_class_from_filepath(filepath, classes)) for filepath in filepaths]

    class_names = load_imagenet_val_class_names()
    return [(filepath, class_names_to_index_map[class_name]) for filepath, class_name in zip(filepaths, class_names)]


def optimize_fn(data):
    filepath, class_index = data 
    img = Image.open(filepath)
    if args.resize:
        img = img.resize((224, 224))

    if not args.filepath_only:
        return img, class_index

    # Used only to build optimized search indexes
    filepath = "/".join(filepath.split("/")[3:])

    if args.use_jpeg_90:
        buff = io.BytesIO()
        img.convert('RGB').save(buff, format="JPEG", quality=90)
        buff.seek(0)
        img = Image.open(buff)

    return img, filepath


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    optimize(
        fn=optimize_fn,
        inputs=get_inputs(args.input_dir),
        output_dir=args.output_dir,
        chunk_bytes="64MB",
        reorder_files=False,
        num_downloaders=10,
    )