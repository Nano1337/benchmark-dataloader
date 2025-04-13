import os
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from time import time
from utils import to_rgb
import torchvision.transforms.v2 as T
from tqdm import tqdm
from lightning_cloud.utils.data_connection import add_s3_connection
from lightning import seed_everything
from lightning.data.streaming.resolver import _resolve_dir
import torch
import shutil

# 1. Clean cache
cache_dir = "/tmp/mosaic_ml_imagenet"
if os.path.isdir(cache_dir):
    shutil.rmtree(cache_dir)
os.makedirs(cache_dir, exist_ok=True)

# 2. Fixed the seed across packages
seed_everything(42)

# 3. Get the optimized dataset
add_s3_connection("optimized-imagenet-1m")


# 4. Create streaming dataset
class ImageNetStreamingDataset(StreamingDataset):
    
    def __init__(self, local, remote):
       super().__init__(local=local, remote=remote, shuffle=True)
       self.transform = T.Compose(
            [   
                T.RandomResizedCrop(224, antialias=True),
                T.RandomHorizontalFlip(),
                T.ToImage(),
                T.ToDtype(torch.float16, scale=True),
            ]
        )

    def __getitem__(self, index):
        obj = super().__getitem__(index)
        return self.transform(to_rgb(obj['x']))[:3], obj["y"]


# 5. Define the dataset and dataLoader
dataset = ImageNetStreamingDataset(
    local=cache_dir,
    remote=_resolve_dir("/teamspace/s3_connections/optimized-imagenet-1m/mosaic_ml_imagenet/train").url,
)
dataloader = DataLoader(
        dataset, 
        batch_size=256,
        num_workers=os.cpu_count(),
)

# 6. Iterate over the datasets for 2 epochs
for epoch in range(2):
    num_samples = 0
    t0 = time()
    for data in tqdm(dataloader, smoothing=0, mininterval=1):
        num_samples += data[0].squeeze(0).shape[0]
    print(f'For {__file__} on {epoch}, streamed over {num_samples} samples in {time() - t0} or {num_samples / (time() - t0)} images/sec.')

# 7. Cleanup cache
shutil.rmtree(cache_dir)