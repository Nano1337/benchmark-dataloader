import os
from time import time
from lightning_sdk import Studio, Machine
from lightning_cloud.utils.data_connection import add_s3_connection
from datetime import datetime

# 1. Add the optimized datasets
add_s3_connection('optimized-imagenet-1m')

# 2. Create a new Studio
studio_name = f"imagenet-benchmark-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
studio = Studio(name=studio_name)
studio.start(machine=Machine.A10G)

try:
    print(" ")
    print(f"A10G Studio created: {studio_name}")
    print(" ")
    print("######################")
    print(" ")

    print("Setting up...")
    print(" ")

    # 3. Setup: Upload files + install dependencies
    for folder, _, filenames in os.walk("./stream"):
        for filename in filenames:
            path = os.path.join(folder, filename)
            studio.upload_file(path, path.replace("./", ""))

    studio.upload_file("requirements.txt", "requirements.txt")
    studio.run("pip install -r requirements.txt")
    studio.run("pip install --no-cache-dir git+https://github.com/Lightning-AI/pytorch-lightning.git@master")
    studio.run("pip install -U lightning-cloud")
    studio.run("sudo apt-get install file")
    studio.run("conda config --add channels conda-forge")
    studio.run("conda config --set channel_priority strict")
    studio.run("conda install s5cmd > /dev/null 2>&1")

    print()
    print("######################")
    print()


    # 4. Benchmark Web Dataset
    t0 = time()
    print("WebDataset Logs:")
    print()
    print(studio.run("python stream/web_dataset.py"))
    print()
    print(f"WebDataset executed in {time() - t0}")

    print()
    print("######################")
    print()

    # 5. Benchmark Lightning Data
    t0 = time()
    print("PyTorch Lightning Data Logs:")
    print()
    print(studio.run("python stream/lightning_data.py"))
    print()
    print(f"PyTorch Lightning Data  executed in {time() - t0}")

    print()
    print("######################")
    print()

    # 6. Benchmark Mosaic ML
    t0 = time()
    print("Mosaic ML Logs:")
    print()
    print(studio.run("python stream/mosaic_ml.py"))
    print()
    print(f"Mosaic ML executed in {time() - t0}")

    print()
    print("######################")
    print()

    print("Stopping Studio...")
    studio.stop()
    print("Done.")

except Exception as e:
    print(e)

studio.delete()