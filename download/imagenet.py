import os
from time import time
from lightning_sdk import Studio, Machine
from datetime import datetime

studio_name = f"imagenet-1m-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
studio = Studio(name=studio_name, create_ok=True)
studio.start(machine=Machine.DATA_PREP)

try:
    # Upload files
    studio.upload_file("/teamspace/studios/this_studio/.kaggle/kaggle.json", ".kaggle/kaggle.json")
    studio.run("chmod 600 /teamspace/studios/this_studio/.kaggle/kaggle.json")

    studio.upload_file("requirements.txt", "requirements.txt")
    studio.run("pip install -r requirements.txt")

    for folder, _, filenames in os.walk("./optimize"):
        for filename in filenames:
            path = os.path.join(folder, filename)
            studio.upload_file(path, path.replace("./", ""))

    # Bump the latest Lightning
    studio.run("pip install --no-cache-dir git+https://github.com/Lightning-AI/pytorch-lightning.git@master")
    studio.run("pip install -U lightning-cloud")

    # Download Imagenet 1M
    t0 = time()
    studio.run("chmod 600 /teamspace/studios/this_studio/.kaggle/kaggle.json")
    studio.run("pip install kaggle")
    studio.run("mkdir data")
    studio.run("rm main.py")
    studio.run("cd /cache && kaggle competitions download -c imagenet-object-localization-challenge")
    studio.run("cd /cache && unzip -qq imagenet-object-localization-challenge.zip '*.JPEG' -d ./data")
    studio.run("cp -r /cache/data/ILSVRC/Data/CLS-LOC/* /teamspace/studios/this_studio/data/ 2>/dev/null")
    print(time() - t0)

except Exception:
    studio.stop()