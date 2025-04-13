import os
from time import time
from lightning_sdk import Studio, Machine

studio = Studio(name="imagenet-1m-01", create_ok=True)
studio.start(machine=Machine.DATA_PREP)

# Optimize with Mosaic ML
t0 = time()
studio.run("mkdir mosaic_ml_imagenet")
studio.run("python optimize/mosaic_ml.py --in_root ./data --out_root mosaic_ml_imagenet")
print(time() - t0)