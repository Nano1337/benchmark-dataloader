import os
from lightning.data import StreamingDataset, StreamingDataLoader
from torch.utils.data import DataLoader
from utils import to_rgb
from time import time
import torchvision.transforms.v2 as T
from tqdm import tqdm
from lightning_cloud.utils.data_connection import add_s3_connection
from lightning import seed_everything
import shutil
import torch

# 1. Clean cache
cache_dir = "/cache/chunks"
if os.path.isdir(cache_dir):
    shutil.rmtree(cache_dir)

# 2. Fixed the seed across packages
seed_everything(42)

# 3. Get the optimized dataset
add_s3_connection("optimized-imagenet-1m")

# 4. Create a custom streaming dataset for Imagenet
class ImageNetStreamingDataset(StreamingDataset):

    def __init__(self, *args, **kwargs):
        self.transform = T.Compose([
                T.RandomResizedCrop(224, antialias=True),
                T.RandomHorizontalFlip(),
                T.ToDtype(torch.float16, scale=True),
        ])
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Note: If torchvision is installed, we return a tensor image instead of a pil image as it is much faster. 
        img, class_index = super().__getitem__(index) # <- Whatever you returned from the DatasetOptimizer prepare_item method.
        return self.transform(to_rgb(img)), int(class_index)    
        

# 5. Define the DataLoader
dataloader = StreamingDataLoader(
    ImageNetStreamingDataset(
        input_dir="/teamspace/s3_connections/optimized-imagenet-1m/lightning_data_imagenet/train",
        max_cache_size="200GB",
    ), 
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