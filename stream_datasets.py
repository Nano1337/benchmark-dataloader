#!/usr/bin/env python

"""
Dataset Streaming Benchmark Script

This script runs streaming benchmarks for different dataset formats:
- WebDataset
- MosaicML Streaming Dataset
- Lightning Data
- Energon

It measures streaming performance in terms of throughput (images/sec),
and generates a comparative summary report.
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import threading
import psutil
import re
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_results_dir(results_dir="results/streaming"):
    """Create results directory if it doesn't exist"""
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def format_number(value):
    """Format number to 2 decimal places"""
    return f"{float(value):.2f}"

def print_header(title):
    """Print a formatted header with timestamp"""
    print("\n\n=========================================")
    print(f"   {title}")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=========================================\n")

def extract_metric(log_content, pattern, default="0"):
    """Extract metric from log content based on pattern"""
    match = re.search(pattern, log_content)
    if match:
        return match.group(1)
    return default

def extract_throughput(log_content):
    """Extract throughput from log content"""
    # Try different patterns
    pattern1 = r"Average throughput: ([0-9.]+) images/sec"
    pattern2 = r"([0-9.]+) images/sec"
    
    value = extract_metric(log_content, pattern1)
    if value == "0":
        # Try the second pattern and get the last match (final result)
        matches = re.findall(pattern2, log_content)
        if matches:
            value = matches[-1]
    
    return value

def extract_samples(log_content):
    """Extract total samples from log content"""
    # Try different patterns
    pattern1 = r"Total samples: ([0-9]+)"
    pattern2 = r"Processed ([0-9]+) samples"
    
    value = extract_metric(log_content, pattern1)
    if value == "0":
        # Try the second pattern and get the last match (final result)
        matches = re.findall(pattern2, log_content)
        if matches:
            value = matches[-1]
    
    return value

def extract_time(log_content):
    """Extract processing time from log content"""
    # Try different patterns
    pattern1 = r"Total time: ([0-9.]+)s"
    pattern2 = r"Processed .* in ([0-9.]+)s"
    
    value = extract_metric(log_content, pattern1)
    if value == "0":
        # Try the second pattern and get the last match (final result)
        matches = re.findall(pattern2, log_content)
        if matches:
            value = matches[-1]
    
    return value

def extract_epoch_times(log_content):
    """Extract epoch times from log content"""
    pattern = r"Epoch ([0-9]+): Processed [0-9]+ samples in ([0-9.]+)s"
    matches = re.findall(pattern, log_content)
    
    # Create a dictionary with epoch numbers as keys
    epoch_times = {}
    for match in matches:
        epoch_num = int(match[0])
        epoch_time = float(match[1])
        epoch_times[epoch_num] = epoch_time
    
    # Get times for epochs 1 and 2 (or 0 if not found)
    epoch1 = epoch_times.get(1, 0)
    epoch2 = epoch_times.get(2, 0)
    
    return epoch1, epoch2

def check_prerequisites():
    """Check prerequisites"""
    # Create results directory
    results_dir = setup_results_dir()
    
    # Check if S3_BENCHMARK_DATA_PATH environment variable is set
    s3_path = os.environ.get('S3_BENCHMARK_DATA_PATH')
    if not s3_path:
        print("Error: S3_BENCHMARK_DATA_PATH environment variable is not set.")
        print("Please set it in the .env file or export it in your shell.")
        sys.exit(1)
    
    return results_dir

def cleanup_cache(cache_dir):
    """Clean up a specific cache directory"""
    if os.path.isdir(cache_dir):
        print(f"Cleaning up cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)

class ResourceMonitor:
    """Monitor system resource usage over time"""
    def __init__(self, name):
        self.name = name
        self.monitoring = False
        self.data = {"timestamps": [], "ram_mb": [], "ram_percent": [], "cpu_percent": []}
        self.start_time = None
        self.peak_ram = 0  # in MB
        self.monitoring_thread = None
        
        # Initialize process CPU tracking
        self.process = psutil.Process(os.getpid())
        # Call once to initialize CPU measurement
        self.process.cpu_percent()
    
    def _monitor_resources(self):
        """Continuously monitor system resources"""
        self.start_time = time.time()
        cpu_count = psutil.cpu_count()
        
        # Define patterns for benchmark-related processes
        benchmark_patterns = [
            "web_dataset.py", 
            "lightning_data.py", 
            "mosaic_ml.py", 
            "energon.py"
        ]
        
        while self.monitoring:
            try:
                # Record timestamp relative to start
                current_time = time.time() - self.start_time
                
                # Get all related processes: direct children and anything matching our patterns
                all_processes = [self.process] + self.process.children(recursive=True)
                
                # Also find any processes that match our benchmark patterns
                # This catches processes started by subprocess that might not be direct children
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        # Skip if already in our list
                        if proc in all_processes:
                            continue
                            
                        # Check command line args for our benchmark patterns
                        cmdline = ' '.join(proc.cmdline())
                        if any(pattern in cmdline for pattern in benchmark_patterns):
                            all_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Get memory usage in MB (sum across all related processes)
                total_ram_mb = sum(p.memory_info().rss for p in all_processes) / (1024 * 1024)
                ram_percent = (total_ram_mb / psutil.virtual_memory().total) * 100 * 1024
                
                # Update peak RAM
                self.peak_ram = max(self.peak_ram, total_ram_mb)
                
                # Get CPU usage with a small interval for more accurate measurement
                # First collect all process objects
                measured_processes = []
                for p in all_processes:
                    try:
                        # Skip zombie processes
                        if p.status() == 'zombie':
                            continue
                        measured_processes.append(p)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Measure CPU across all processes (with a very small interval)
                total_cpu_percent = sum(p.cpu_percent(interval=0.1) for p in measured_processes)
                # Cap at 100% per CPU
                total_cpu_percent = min(total_cpu_percent, 100 * cpu_count)
                
                # Store the data
                self.data["timestamps"].append(current_time)
                self.data["ram_mb"].append(total_ram_mb)
                self.data["ram_percent"].append(ram_percent)
                self.data["cpu_percent"].append(total_cpu_percent)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Handle process termination gracefully
                pass
            
            # Slightly longer sample interval since we're doing more work
            # We already used 0.1s for CPU measurement, so 0.4s more for total of 0.5s
            time.sleep(0.4)
    
    def start(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        print(f"Started resource monitoring for {self.name}")
    
    def stop(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        print(f"Resource monitoring for {self.name} complete. Peak RAM: {self.peak_ram:.1f} MB")


def run_benchmark(script_path, benchmark_name, batch_size, num_workers, prefetch_factor, epochs=2, 
                  image_size=224, additional_args=None, enable_profiling=True):
    """Run a streaming benchmark and capture its output"""
    print_header(f"{benchmark_name} Streaming Benchmark")
    print(f"Configuration: batch_size={batch_size}, num_workers={num_workers}, prefetch_factor={prefetch_factor}")
    
    # Common arguments for all benchmarks
    cmd = [
        "python", script_path,
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--epochs", str(epochs),
        "--image_size", str(image_size)
    ]
    
    # Add prefetch_factor if the script supports it
    if script_path != "stream/lightning_data.py":  # LitData doesn't use prefetch_factor
        cmd.extend(["--prefetch_factor", str(prefetch_factor)])
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    start_time = time.time()
    
    # Create resource monitor if profiling is enabled
    monitor = None
    if enable_profiling:
        monitor = ResourceMonitor(benchmark_name)
        monitor.start()
    
    # Run the benchmark with real-time output handling
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect the full output
        full_output = []
        
        # Read and process output line by line as it becomes available
        for line in iter(process.stdout.readline, ''):
            # Print to console
            print(line, end='')
            
            # Save for return
            full_output.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Check if the command was successful
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
            
        # Join output lines
        output = ''.join(full_output)
    except subprocess.CalledProcessError as e:
        print(f"Error running {benchmark_name} benchmark: {e}")
        output = e.stdout if e.stdout else ""
        output += e.stderr if e.stderr else ""
    finally:
        # Stop resource monitoring
        if monitor:
            monitor.stop()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Extract metrics from the output
    throughput = extract_throughput(output)
    samples = extract_samples(output)
    benchmark_time = extract_time(output)
    epoch1, epoch2 = extract_epoch_times(output)
    
    # Print a summary
    print("")
    print(f"Results for {benchmark_name}:")
    print(f"  Throughput: {throughput} images/sec")
    print(f"  Total samples processed: {samples}")
    print(f"  Benchmark time: {benchmark_time} seconds")
    print(f"  Wall clock time: {format_number(total_time)} seconds")
    
    # Return the metrics
    result = {
        "name": benchmark_name,
        "throughput": float(throughput),
        "samples": int(samples),
        "benchmark_time": float(benchmark_time),
        "wall_time": total_time,
        "epoch1": epoch1,
        "epoch2": epoch2,
        "output": output
    }
    
    # Add resource monitoring data if available
    if monitor:
        result["monitor"] = monitor
    
    return result

def run_webdataset_benchmark(batch_size, num_workers, prefetch_factor, enable_profiling=True, **kwargs):
    """Run WebDataset streaming benchmark"""
    return run_benchmark(
        script_path="stream/web_dataset.py",
        benchmark_name="WebDataset",
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        enable_profiling=enable_profiling,
        **kwargs
    )

def run_mds_benchmark(batch_size, num_workers, prefetch_factor, enable_profiling=True, **kwargs):
    """Run MosaicML MDS streaming benchmark"""
    return run_benchmark(
        script_path="stream/mosaic_ml.py",
        benchmark_name="MosaicML MDS",
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        enable_profiling=enable_profiling,
        **kwargs
    )

def run_litdata_benchmark(batch_size, num_workers, prefetch_factor, enable_profiling=True, **kwargs):
    """Run LitData streaming benchmark"""
    return run_benchmark(
        script_path="stream/lightning_data.py",
        benchmark_name="LitData",
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,  # This will be ignored by run_benchmark for LitData
        enable_profiling=enable_profiling,
        **kwargs
    )

def run_energon_benchmark(batch_size, num_workers, prefetch_factor, enable_profiling=True, **kwargs):
    """Run Energon streaming benchmark"""
    return run_benchmark(
        script_path="stream/energon.py",
        benchmark_name="Energon",
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        enable_profiling=enable_profiling,
        **kwargs
    )

def generate_resource_plots(results, timestamp, worker_count=None, prefetch_factor=None, batch_size=None):
    """Generate plots for RAM and CPU utilization
    
    Args:
        results: Dictionary with benchmark results
        timestamp: Timestamp to use in filenames
        worker_count: Number of workers used for this run (optional)
        prefetch_factor: Prefetch factor used for this run (optional)
        batch_size: Batch size used for this run (optional)
    """
    # Set a nice style for plots
    plt.style.use('ggplot')
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define colors for each dataset format
    colors = {
        'WebDataset': '#1f77b4',  # blue
        'MosaicML MDS': '#ff7f0e',  # orange
        'LitData': '#2ca02c',  # green
        'Energon': '#d62728'  # red
    }
    
    # RAM utilization plot (in MB)
    for name, result in results.items():
        if 'monitor' in result:
            monitor = result['monitor']
            if monitor.data['timestamps'] and monitor.data['ram_mb']:
                ax1.plot(
                    monitor.data['timestamps'], 
                    monitor.data['ram_mb'],
                    label=name,
                    color=colors.get(name, None),
                    linewidth=2
                )
    
    ax1.set_title('RAM Usage Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('RAM Usage (MB)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Ensure y-axis starts at 0 for better comparison
    ax1.set_ylim(bottom=0)
    
    # CPU utilization plot
    for name, result in results.items():
        if 'monitor' in result:
            monitor = result['monitor']
            if monitor.data['timestamps'] and monitor.data['cpu_percent']:
                ax2.plot(
                    monitor.data['timestamps'], 
                    monitor.data['cpu_percent'],
                    label=name,
                    color=colors.get(name, None),
                    linewidth=2
                )
    
    ax2.set_title('CPU Utilization Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('CPU Usage (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    
    # Ensure y-axis starts at 0 and has a reasonable upper limit
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save the plots
    plots_dir = "results/streaming/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Build filename with configuration parameters
    filename_parts = []
    if worker_count is not None:
        filename_parts.append(f"{worker_count}_workers")
    if prefetch_factor is not None:
        filename_parts.append(f"{prefetch_factor}_prefetch")
    if batch_size is not None:
        filename_parts.append(f"{batch_size}_batch")
    
    if filename_parts:
        plot_file = f"{plots_dir}/resource_usage_{'_'.join(filename_parts)}_{timestamp}.png"
    else:
        plot_file = f"{plots_dir}/resource_usage_{timestamp}.png"
        
    plt.savefig(plot_file, dpi=120, bbox_inches='tight')
    print(f"\nResource usage plots saved to: {plot_file}")

def generate_summary(results, results_file, summary_file, batch_size, num_workers, prefetch_factor, total_time):
    """Generate summary report of benchmark results"""
    # Write summary to file
    with open(summary_file, 'w') as f:
        f.write("DATASET STREAMING BENCHMARK SUMMARY\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CPU Count: {os.cpu_count()}\n")
        f.write(f"Configuration: Batch Size = {batch_size}, Workers = {num_workers}, Prefetch Factor = {prefetch_factor}\n\n")
        
        f.write("| Dataset | Throughput (img/s) | Samples | Epoch 1 (s) | Epoch 2 (s) | Processing Time (s) | Wall Time (s) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        
        for result in results:
            throughput_fmt = format_number(result["throughput"])
            epoch1_fmt = format_number(result["epoch1"])
            epoch2_fmt = format_number(result["epoch2"])
            proc_time_fmt = format_number(result["benchmark_time"])
            wall_time_fmt = format_number(result["wall_time"])
            
            f.write(f"| {result['name']} | {throughput_fmt} | {result['samples']} | {epoch1_fmt} | {epoch2_fmt} | {proc_time_fmt} | {wall_time_fmt} |\n")
        
        f.write(f"\nTotal benchmark time: {format_number(total_time)} seconds\n")
    
    # Print the summary to console
    print_header("DATASET STREAMING BENCHMARK SUMMARY")
    with open(summary_file, 'r') as f:
        print(f.read())
    
    print("")
    print(f"Streaming benchmarks complete! Summary saved to {summary_file}")
    print(f"Full logs available at {results_file}")

def main():
    """Main entry point for benchmark script"""
    parser = argparse.ArgumentParser(description="Run dataset streaming benchmarks")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker threads for data loading (default: 8)")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of batches to prefetch (default: 2)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for dataloader (default: 256)")
    parser.add_argument("--disable_profiling", action="store_true",
                        help="Disable resource profiling (CPU and RAM monitoring)")
    args = parser.parse_args()
    
    # Whether to enable resource profiling
    enable_profiling = not args.disable_profiling
    
    # Print header
    print_header("DATASET STREAMING BENCHMARKS")
    
    # Check prerequisites and set up results directory
    results_dir = check_prerequisites()
    
    # Set up results files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{results_dir}/benchmark_stream_{timestamp}.log"
    summary_file = f"{results_dir}/summary_stream_{timestamp}.txt"
    
    # Log basic system info
    print(f"Starting streaming benchmarks at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU count: {os.cpu_count()}")
    print("")
    
    # Start the overall timer
    total_start_time = time.time()
    
    # Store results
    results = []
    
    # Run WebDataset benchmark
    print("\n\nRunning WebDataset benchmark...\n")
    wds_result = run_webdataset_benchmark(
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        epochs=2, 
        image_size=224, 
        enable_profiling=enable_profiling
    )
    results.append(wds_result)
    
    # Run MosaicML benchmark
    print("\n\nRunning MosaicML Dataset benchmark...\n")
    mds_result = run_mds_benchmark(
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        epochs=2, 
        image_size=224, 
        enable_profiling=enable_profiling
    )
    results.append(mds_result)
    
    # Run LitData benchmark
    print("\n\nRunning LitData benchmark...\n")
    lit_result = run_litdata_benchmark(
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        epochs=2, 
        image_size=224, 
        enable_profiling=enable_profiling
    )
    results.append(lit_result)
    
    # Run Energon benchmark
    print("\n\nRunning Energon benchmark...\n")
    energon_result = run_energon_benchmark(
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        epochs=2, 
        image_size=224, 
        enable_profiling=enable_profiling
    )
    results.append(energon_result)
    
    # Calculate total time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Write all outputs to the log file
    with open(results_file, 'w') as f:
        f.write(f"DATASET STREAMING BENCHMARKS\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: batch_size={args.batch_size}, num_workers={args.num_workers}, prefetch_factor={args.prefetch_factor}\n\n")
        
        for result in results:
            f.write(f"\n\n========== {result['name']} ==========\n")
            f.write(result['output'])
    
    # Convert results list to a dictionary keyed by benchmark name
    results_dict = {result["name"]: result for result in results}
    
    # Generate resource usage plots if profiling was enabled
    if enable_profiling:
        generate_resource_plots(
            results_dict, 
            timestamp, 
            worker_count=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            batch_size=args.batch_size
        )
    
    # Generate summary
    generate_summary(
        results, 
        results_file, 
        summary_file, 
        args.batch_size, 
        args.num_workers,
        args.prefetch_factor,
        total_time
    )

if __name__ == "__main__":
    main()
