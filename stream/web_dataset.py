import os
from lightning.data import StreamingDataset, StreamingDataLoader
from time import time
import torchvision.transforms.v2 as T
from tqdm import tqdm
from lightning_cloud.utils.data_connection import add_s3_connection
from lightning import seed_everything
import boto3
import webdataset as wds
import shutil
import torch

# 1. Clean cache
cache_dir = "/cache/webdataset_imagenet"
if os.path.isdir(cache_dir):
    shutil.rmtree(cache_dir)
os.makedirs(cache_dir, exist_ok=True)

# 2. Fixed the seed across packages
seed_everything(42)

# 3. Get the optimized dataset
add_s3_connection("optimized-imagenet-1m")

# 4. Prepare URLs
def prepare_urls():
    s3_client = boto3.client('s3')
    token = None
    keys = []

    while token is None or token != "":
        if token:
            objects = s3_client.list_objects_v2(
                Bucket="optimized-imagenet-1m",
                Prefix="webdataset_imagenet",
                ContinuationToken=token,
            )
        else:
             objects = s3_client.list_objects_v2(
                Bucket="optimized-imagenet-1m",
                Prefix="webdataset_imagenet",
            )   

        token = objects.get("NextContinuationToken", "end") 
        keys.extend(obj['Key'] for obj in objects['Contents'])

        if token == "end":
            break

    return [f"pipe:aws s3 cp s3://{os.path.join('optimized-imagenet-1m', key)}  -" for key in keys if "train" in key]
        

# 5. Define the dataset and dataLoader
urls = prepare_urls()
dataset = (
    wds.WebDataset(urls, cache_dir=cache_dir)
    .shuffle(True)
    .decode("pil")
    .to_tuple("jpg;png;jpeg cls")
    .map_tuple(
        T.Compose(
            [
                T.RandomResizedCrop(224, antialias=True),
                T.RandomHorizontalFlip(),
                T.ToImage(), 
                T.ToDtype(torch.float16, scale=True),
            ]
        ))
    .batched(256, partial=True)
)

dataloader = wds.WebLoader(
    dataset,
    num_workers=os.cpu_count(),
)

# 6. Iterate over the datasets for 2 epochs
for epoch in range(2):
    num_samples = 0
    t0 = time()
    for data in tqdm(dataloader, smoothing=0, mininterval=1):
        num_samples += data[0].squeeze(0).shape[0]
    print(f'For {__file__} on {epoch}, streamed over {num_samples} samples in {time() - t0} or {num_samples / (time() - t0)} images/sec.')

# 7. Cleanup the cache
shutil.rmtree(cache_dir)