from time import time
import os
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from lightning.data import walk
from streaming.base import MDSWriter
from streaming.base.util import get_list_arg
from multiprocessing import Pool
from functools import partial
from utils import check_extensions, get_classes


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Args:
        Namespace: command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument(
        '--in_root',
        type=str,
        required=True,
        help='Local directory path of the input raw dataset',
    )
    args.add_argument(
        '--out_root',
        type=str,
        required=True,
        help='Directory path to store the output dataset',
    )
    args.add_argument(
        '--splits',
        type=str,
        default='train,val',
        help='Split to use. Default: train,val',
    )
    args.add_argument(
        '--compression',
        type=str,
        default='',
        help='Compression algorithm to use. Default: None',
    )
    args.add_argument(
        '--hashes',
        type=str,
        default='',
        help='Hashing algorithms to apply to shard files. Default: None',
    )
    args.add_argument(
        '--size_limit',
        type=int,
        default=1 << 26,
        help='Shard size limit, after which point to start a new shard. Default: 1 << 26',
    )
    args.add_argument(
        '--progress_bar',
        type=int,
        default=1,
        help='tqdm progress bar. Default: 1 (True)',
    )
    args.add_argument(
        '--leave',
        type=int,
        default=1,
        help='Keeps all traces of the progressbar upon termination of iteration. Default: 0 ' +
        '(False)',
    )
    args.add_argument(
        '--validate',
        type=int,
        default=1,
        help='Validate that it is an Image. Default: 1 (True)',
    )
    args.add_argument(
        '--extensions',
        type=str,
        default='jpeg',
        help='Validate filename extensions. Default: jpeg',
    )
    args.add_argument(
        '--resize',
        default="False",
        type=bool,
        action='store_true',
        help='Local directory path of the input raw dataset',
    )
    return args.parse_args()


def init_worker():
    pass


def get_data(i, args=None, filepaths=None, classes=None):
    if args.validate:
        x = Image.open(filepaths[i])
    x = open(filepaths[i], 'rb').read()
    y = classes[i]
    return {
            'i': i.item(),
            'x': x,
            'y': y,
        }

def main(args: Namespace) -> None:
    """Main: create streaming ImageNet dataset.

    Args:
        args (Namespace): command-line arguments.
    """
    splits = get_list_arg(args.splits)
    columns = {'i': 'int', 'x': 'jpeg', 'y': 'int'}
    hashes = get_list_arg(args.hashes)
    extensions = set(get_list_arg(args.extensions))
    class_names = None
    classes = None
    for split in splits:
        data_dir = os.path.join(args.in_root, split)
        cache_dir = os.path.join(args.out_root, split)
        if split == "train":
            pattern = os.path.join(data_dir, '*', '*')
        else:
            pattern = os.path.join(data_dir, '*')

        filepaths = [
            os.path.join(root, filename)
            for root, _, filepaths in tqdm(walk(data_dir), smoothing=0)
            for filename in filepaths
        ]

        check_extensions(filepaths, extensions)

        if not classes:
            classes, class_names = get_classes(filepaths, class_names)

        if os.path.exists(os.path.join(cache_dir, "index.json")):
            continue

        indices = np.random.permutation(len(filepaths))

        out_split_dir = os.path.join(args.out_root, split)

        with MDSWriter(
            out=out_split_dir,
            columns=columns,
            compression=args.compression,
            hashes=hashes,
            size_limit=args.size_limit,
            progress_bar=args.progress_bar,
            max_workers=os.cpu_count(),
        ) as out:
            for i in tqdm(indices):
                if args.validate:
                    x = Image.open(filepaths[i])
                x = open(filepaths[i], 'rb').read()
                y = classes[i]
                out.write({
                    'i': i.item(),
                    'x': x,
                    'y': y,
                })


if __name__ == '__main__':
    t0 = time()
    main(parse_args())
    print(time() - t0)