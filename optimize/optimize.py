from lightning_sdk import Machine, Studio, JobsPlugin, MultiMachineTrainingPlugin

# Start the studio
s = Studio()
print("starting Studio...")
s.start()

# Install plugin if not installed (in this case, it is already installed)
s.install_plugin("jobs")

jobs_plugin = s.installed_plugins["jobs"]

# --------------------------
# Launch job to optimize the data
# --------------------------
scripts = ['prepare_lightning', 'prepare_mosaic_ml', 'prepare_webdataset']
for script in scripts:
    job_cmd = f'python {script}.py'
    jobs_plugin.run(job_cmd, name=script, machine=Machine.DATA_PREP)