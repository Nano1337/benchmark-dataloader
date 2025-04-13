#
# Copyright (c) 2017-2023 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

import sys
import os
import os.path
import random
import argparse
from tqdm import tqdm
from lightning.data import walk
from utils import get_classes, check_extensions
import webdataset as wds


parser = argparse.ArgumentParser("""Generate sharded dataset from original ImageNet data.""")
parser.add_argument("--splits", default="train,val", help="which splits to write")
parser.add_argument(
    "--filekey", action="store_true", help="use file as key (default: index)"
)
parser.add_argument("--maxsize", type=float, default=2 << 26)
parser.add_argument("--maxcount", type=float, default=100000)
parser.add_argument(
    "--shards", default="./shards", help="directory where shards are written"
)
parser.add_argument(
    "--data",
    default="./data",
    help="directory containing ImageNet data distribution suitable for torchvision.datasets",
)
args = parser.parse_args()


assert args.maxsize > 10000000
assert args.maxcount < 1000000


splits = args.splits.split(",")


def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()


all_keys = set()


def write_dataset(imagenet, base="./shards", split="train"):
    filepaths = [
        os.path.join(root, filename)
        for root, _, filenames in tqdm(walk(os.path.join(imagenet, split)), smoothing=0)
        for filename in filenames
    ]

    classes, class_names = get_classes(filepaths, None)

    indexes = list(range(len(filepaths)))
    random.shuffle(indexes)

    # This is the output pattern under which we write shards.
    pattern = os.path.join(base, f"imagenet-{split}-%06d.tar")

    with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
        for i in tqdm(indexes):
            fname = filepaths[i]

            # Read the JPEG-compressed image file contents.
            image = readfile(fname)
            cls = classes[i]

            # Construct a unique key from the filename.
            key = os.path.splitext(os.path.basename(fname))[0]

            # Useful check.
            assert key not in all_keys
            all_keys.add(key)

            # Construct a sample.
            xkey = key if args.filekey else "%07d" % i
            sample = {"__key__": xkey, "jpg": image, "cls": cls}

            # Write the sample to the sharded tar archives.
            sink.write(sample)


for split in splits:
    print("# split", split)
    write_dataset(args.data, base=args.shards, split=split)