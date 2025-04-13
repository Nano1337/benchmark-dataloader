import os
from time import time
from lightning_sdk import Studio, Machine

studio = Studio(name="imagenet-1m-01", create_ok=True)
studio.start(machine=Machine.DATA_PREP)

# Upload files
studio.upload_file("requirements.txt", "requirements.txt")
studio.run("pip install -r requirements.txt")
studio.upload_file("imagenet_class_index.json", "imagenet_class_index.json")

# Optimize with Webdataset
t0 = time()
studio.run("python optimize/prepare_webdataset.py --data ./data/ --shards webdataset_imagenet")
print(time() - t0)