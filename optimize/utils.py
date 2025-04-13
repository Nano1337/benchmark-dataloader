import os
import json
import numpy as np
import torchvision.transforms as T
import os
from PIL import Image
from typing import List, Optional, Tuple

def load_imagenet_class_names_to_index_map():
    # download ImageNet class mapping
    os.system("curl https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json -o ~/imagenet_class_index.json")
    
    # Find the associate class for each filepath 
    with open(os.path.join(os.path.expanduser('~'), "optimize", "imagenet_class_index.json"), "r") as f:
        data = json.load(f)

    return {v[0]: int(k) for k, v in data.items()}

def get_class_index_from_filepath(filepath):
    class_name = filepath.split("/")[-2]
    return int(class_names_to_index_map[class_name])

def shuffle(x):
    return np.random.permutation(x).tolist()


def to_rgb(img):
    if img.shape[0] == 1:
        img = img.repeat((3, 1, 1))
    if img.shape[0] == 4:
        img = img[:3]
    return img


def load_imagenet_val_class_names():
    os.system("curl https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt -o ~/imagenet_val_labels.txt")

    with open(os.path.join(os.path.expanduser('~'), "optimize", "imagenet_val_labels.txt"), "r") as f:
        return f.read().split("\n")

class_names_to_index_map = load_imagenet_class_names_to_index_map()


def check_extensions(filenames, extensions) -> None:
    """Validate filename extensions.

    Args:
        filenames (List[str]): List of files.
        extensions (Set[str]): Acceptable extensions.
    """
    for f in filenames:
        idx = f.rindex('.')
        ext = f[idx + 1:]
        assert ext.lower() in extensions


def get_classes(filenames: List[str],
                class_names: Optional[List[str]] = None) -> Tuple[List[int], List[str]]:
    """Get the classes for a dataset split of sample image filenames.

    Args:
        filenames (List[str]): Files, in the format ``"root/split/class/sample.jpeg"``.
        class_names (List[str], optional): List of class names from the other splits that we must
            match. Defaults to ``None``.

    Returns:
        Tuple[List[int], List[str]]: Class ID per sample, and the list of unique class names.
    """
    classes = []
    dirname2class = {}
    for f in filenames:
        d = f.split(os.path.sep)[-2]
        c = dirname2class.get(d)
        if c is None:
            c = len(dirname2class)
            dirname2class[d] = c
        classes.append(c)
    new_class_names = sorted(dirname2class)
    if class_names is not None:
        assert class_names == new_class_names
    return classes, new_class_names