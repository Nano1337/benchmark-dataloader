#!/usr/bin/env python

import os
import sys
import time
import subprocess
import argparse
import glob
import shutil
import threading
import psutil
import matplotlib.pyplot as plt
from datetime import datetime


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
        
        while self.monitoring:
            try:
                # Record timestamp relative to start
                current_time = time.time() - self.start_time
                
                # Get all child processes too
                all_processes = [self.process] + self.process.children(recursive=True)
                
                # Get memory usage in MB (sum across all related processes)
                total_ram_mb = sum(p.memory_info().rss for p in all_processes) / (1024 * 1024)
                ram_percent = (total_ram_mb / psutil.virtual_memory().total) * 100 * 1024
                
                # Update peak RAM
                self.peak_ram = max(self.peak_ram, total_ram_mb)
                
                # Get CPU usage (percent of all CPUs)
                # Interval=None gets usage since last call
                total_cpu_percent = sum(p.cpu_percent(interval=None) for p in all_processes)
                # Cap at 100% per CPU
                cpu_count = psutil.cpu_count()
                total_cpu_percent = min(total_cpu_percent, 100 * cpu_count)
                
                # Store the data
                self.data["timestamps"].append(current_time)
                self.data["ram_mb"].append(total_ram_mb)
                self.data["ram_percent"].append(ram_percent)
                self.data["cpu_percent"].append(total_cpu_percent)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Handle process termination gracefully
                pass
            
            # Sample every 500ms for smoother curves and less overhead
            time.sleep(0.5)
    
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


def format_time(seconds):
    """Format time with 2 decimal places"""
    return f"{seconds:.2f}"


def format_size_gb(size_mb):
    """Convert MB to GB with 2 decimal places"""
    return f"{float(size_mb)/1024:.2f}"


def print_header(title):
    """Print header with timestamp"""
    print(f"\n\n=========================================")
    print(f"   {title}")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=========================================\n")


def check_prerequisites():
    """Check if data file exists and create necessary directories"""
    if not os.path.exists("./data/benchmark_dataset.parquet"):
        print("Error: Benchmark data not found. Please see README.md for instructions on how to download the data.")
        sys.exit(1)

    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/processing", exist_ok=True)
    os.makedirs("shards/webdataset", exist_ok=True)
    os.makedirs("shards/mds", exist_ok=True)
    os.makedirs("shards/litdata", exist_ok=True)
    os.makedirs("shards/energon", exist_ok=True)


def get_directory_stats(directory):
    """Calculate directory size in MB and count files"""
    if not os.path.exists(directory):
        return 0, 0
        
    # Get size in KB and convert to MB
    cmd = ["du", "-sk", directory]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    size_kb = float(result.stdout.split()[0])
    size_mb = size_kb / 1024
    
    # Count files
    cmd = ["find", directory, "-type", "f"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    file_count = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
    
    return size_mb, file_count


def extract_timing(log_content, pattern):
    """Extract timing information from logs"""
    import re
    match = re.search(pattern + r'\s*([0-9.]+)', log_content)
    return float(match.group(1)) if match else 0


def run_benchmark(cmd, log_file):
    """Run a benchmark command and display its output in real-time"""
    start_time = time.time()
    
    # Open log file for writing
    with open(log_file, 'w') as log_f:
        # Run process with real-time output handling
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
            # Write to log file
            log_f.write(line)
            log_f.flush()
            
            # Print to console
            print(line, end='')
            
            # Save for return
            full_output.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Check if the command was successful
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Join output lines
    output = ''.join(full_output)
    
    return elapsed_time, output


def run_webdataset_benchmark(num_workers):
    """Run WebDataset benchmark"""
    print_header("WEBDATASET BENCHMARK")
    cmd = [sys.executable, "prepare_data/prepare_webdataset.py"]
    
    # Add num_workers if it's supported
    if num_workers > 0:
        cmd.extend(["--num_workers", str(num_workers)])
    
    # Start resource monitoring
    monitor = ResourceMonitor("webdataset")
    monitor.start()
    
    # Run the benchmark
    log_file = "results/processing/webdataset_benchmark.log"
    webdataset_time, log_output = run_benchmark(cmd, log_file)
    
    # Stop resource monitoring
    monitor.stop()
    
    # Extract specific timing values
    webdataset_write_time = extract_timing(log_output, "Dataset write time:")
    webdataset_total_time = extract_timing(log_output, "Total script time:")
    webdataset_converter_time = extract_timing(log_output, "Converter execution time:")
    
    return {
        "time": webdataset_time,
        "write_time": webdataset_write_time,
        "total_time": webdataset_total_time,
        "converter_time": webdataset_converter_time,
        "log_output": log_output,
        "peak_ram_mb": monitor.peak_ram,
        "monitor": monitor
    }


def run_mds_benchmark(num_workers):
    """Run MosaicML MDS benchmark"""
    print_header("MOSAICML MDS BENCHMARK")
    cmd = [sys.executable, "prepare_data/prepare_mosaicdataset.py"]
    
    # Add num_workers if specified
    if num_workers > 0:
        cmd.extend(["--num_workers", str(num_workers)])
    
    # Start resource monitoring
    monitor = ResourceMonitor("mds")
    monitor.start()
    
    # Run the benchmark
    log_file = "results/processing/mds_benchmark.log"
    mds_time, log_output = run_benchmark(cmd, log_file)
    
    # Stop resource monitoring
    monitor.stop()
    
    # Extract specific timing values
    mds_write_time = extract_timing(log_output, "Conversion completed in")
    mds_converter_time = extract_timing(log_output, "MDS conversion completed in")
    
    return {
        "time": mds_time,
        "write_time": mds_write_time,
        "converter_time": mds_converter_time,
        "log_output": log_output,
        "peak_ram_mb": monitor.peak_ram,
        "monitor": monitor
    }


def run_litdata_benchmark(num_workers):
    """Run LitData benchmark"""
    print_header("LITDATA BENCHMARK")
    cmd = [sys.executable, "prepare_data/prepare_litdata.py"]
    
    # Add num_workers if it's supported
    if num_workers > 0:
        cmd.extend(["--num_workers", str(num_workers)])
    
    # Start resource monitoring
    monitor = ResourceMonitor("litdata")
    monitor.start()
    
    # Run the benchmark
    log_file = "results/processing/litdata_benchmark.log"
    litdata_time, log_output = run_benchmark(cmd, log_file)
    
    # Stop resource monitoring
    monitor.stop()
    
    # Extract specific timing values
    litdata_write_time = extract_timing(log_output, "Dataset write time:")
    litdata_total_time = extract_timing(log_output, "Total script time:")
    
    return {
        "time": litdata_time,
        "write_time": litdata_write_time,
        "total_time": litdata_total_time,
        "log_output": log_output,
        "peak_ram_mb": monitor.peak_ram,
        "monitor": monitor
    }


def run_energon_benchmark(webdataset_results, num_workers):
    """Run Energon benchmark"""
    print_header("ENERGON BENCHMARK")
    cmd = [
        sys.executable, 
        "prepare_data/prepare_energon.py",
        "--webdataset_time", str(webdataset_results["time"]),
        "--webdataset_write_time", str(webdataset_results["write_time"])
    ]
    
    # Add num_workers if it's supported
    if num_workers > 0:
        cmd.extend(["--num_workers", str(num_workers)])
    
    # Start resource monitoring
    monitor = ResourceMonitor("energon")
    monitor.start()
    
    # Run the benchmark
    log_file = "results/processing/energon_benchmark.log"
    energon_time, log_output = run_benchmark(cmd, log_file)
    
    # Stop resource monitoring
    monitor.stop()
    
    # Extract specific timing values
    energon_write_time = extract_timing(log_output, "Dataset write time:")
    energon_total_time = extract_timing(log_output, "Total script time:")
    energon_converter_time = extract_timing(log_output, "Converter execution time:")
    
    return {
        "time": energon_time,
        "write_time": energon_write_time,
        "total_time": energon_total_time,
        "converter_time": energon_converter_time,
        "log_output": log_output,
        "peak_ram_mb": monitor.peak_ram,
        "monitor": monitor
    }


def generate_resource_plots(results, timestamp):
    """Generate plots for RAM and CPU utilization"""
    # Set a nice style for plots
    plt.style.use('ggplot')
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define colors for each dataset format
    colors = {
        'webdataset': '#1f77b4',  # blue
        'mds': '#ff7f0e',         # orange
        'litdata': '#2ca02c',      # green
        'energon': '#d62728'       # red
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
    plots_dir = "results/processing/plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_file = f"{plots_dir}/resource_usage_{timestamp}.png"
    plt.savefig(plot_file, dpi=120, bbox_inches='tight')
    print(f"\nResource usage plots saved to: {plot_file}")
    
    return plot_file


def generate_summary(results, total_time):
    """Generate benchmark summary"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"results/processing/summary_{timestamp}.txt"
    
    # Get directory stats
    webdataset_size, webdataset_files = get_directory_stats("shards/webdataset")
    mds_size, mds_files = get_directory_stats("shards/mds")
    litdata_size, litdata_files = get_directory_stats("shards/litdata")
    energon_size, energon_files = get_directory_stats("shards/energon")
    
    # Format numbers
    webdataset_time_fmt = format_time(results["webdataset"]["time"])
    webdataset_write_time_fmt = format_time(results["webdataset"]["write_time"])
    webdataset_size_gb = format_size_gb(webdataset_size)
    webdataset_ram_mb = results["webdataset"]["peak_ram_mb"]
    
    mds_time_fmt = format_time(results["mds"]["time"])
    mds_write_time_fmt = format_time(results["mds"]["write_time"])
    mds_size_gb = format_size_gb(mds_size)
    mds_ram_mb = results["mds"]["peak_ram_mb"]
    
    litdata_time_fmt = format_time(results["litdata"]["time"])
    litdata_write_time_fmt = format_time(results["litdata"]["write_time"])
    litdata_size_gb = format_size_gb(litdata_size)
    litdata_ram_mb = results["litdata"]["peak_ram_mb"]
    
    energon_time_fmt = format_time(results["energon"]["time"])
    energon_write_time_fmt = format_time(results["energon"]["write_time"])
    energon_size_gb = format_size_gb(energon_size)
    energon_ram_mb = results["energon"]["peak_ram_mb"]
    
    # Combine WDS and Energon metrics
    combined_energon_time_raw = results["webdataset"]["time"] + results["energon"]["time"]
    combined_energon_write_raw = results["webdataset"]["write_time"] + results["energon"]["write_time"]
    combined_energon_time_fmt = format_time(combined_energon_time_raw)
    combined_energon_write_fmt = format_time(combined_energon_write_raw)
    combined_energon_ram_mb = max(webdataset_ram_mb, energon_ram_mb)
    
    # Generate summary content
    summary_content = [
        "BENCHMARK SUMMARY",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"CPU Count: {os.cpu_count()}",
        "",
        "| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files | Peak RAM (MB) |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| LitData (PL) | {litdata_time_fmt} | {litdata_write_time_fmt} | {litdata_size_gb} | {litdata_files} | {litdata_ram_mb:.1f} |",
        f"| WebDataset (WDS) | {webdataset_time_fmt} | {webdataset_write_time_fmt} | {webdataset_size_gb} | {webdataset_files} | {webdataset_ram_mb:.1f} |",
        f"| MosaicML Dataset (MDS) | {mds_time_fmt} | {mds_write_time_fmt} | {mds_size_gb} | {mds_files} | {mds_ram_mb:.1f} |",
        f"| Energon (WDS+) | {combined_energon_time_fmt} | {combined_energon_write_fmt} | {energon_size_gb} | {energon_files} | {combined_energon_ram_mb:.1f} |",
        "",
        f"Total benchmark time: {format_time(total_time)} seconds"
    ]
    
    # Write summary to file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_content))
    
    print_header("BENCHMARK SUMMARY")
    print('\n'.join(summary_content))
    
    # Generate resource usage plots
    generate_resource_plots(results, timestamp)
    
    return summary_file


def clean_shards_directory():
    """Clean up or create shards directory"""
    shards_dir = "./shards"
    if os.path.exists(shards_dir):
        print("Cleaning up existing shards directory...")
        shutil.rmtree(shards_dir)
    
    print("Creating fresh shards directory...")
    os.makedirs(f"{shards_dir}/webdataset", exist_ok=True)
    os.makedirs(f"{shards_dir}/mds", exist_ok=True)
    os.makedirs(f"{shards_dir}/litdata", exist_ok=True)
    os.makedirs(f"{shards_dir}/energon", exist_ok=True)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run dataset conversion benchmarks")
    parser.add_argument(
        "num_workers",
        type=int,
        nargs='?',
        default=0,
        help="Number of worker processes to use (default: 0, meaning use all available CPUs)"
    )
    return parser.parse_args()


def main():
    """Main function to run all benchmarks"""
    args = parse_args()
    num_workers = args.num_workers
    
    # Set up logging directory and file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "results/processing"
    results_file = f"{log_dir}/benchmark_run_{timestamp}.log"
    
    os.makedirs(log_dir, exist_ok=True)
    
    print_header("DATASET BENCHMARKING SCRIPT")
    
    # Check prerequisites
    check_prerequisites()
    
    # Clean up shards directory
    clean_shards_directory()
    
    print(f"Starting benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU count: {os.cpu_count()}")
    print(f"Using {num_workers if num_workers > 0 else 'all available'} worker processes")
    print("")
    
    # Start the overall timer
    total_start_time = time.time()
    
    # Run all benchmarks
    webdataset_results = run_webdataset_benchmark(num_workers)
    mds_results = run_mds_benchmark(num_workers)
    litdata_results = run_litdata_benchmark(num_workers)
    energon_results = run_energon_benchmark(webdataset_results, num_workers)
    
    # Calculate total benchmark time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Collect all results
    all_results = {
        "webdataset": webdataset_results,
        "mds": mds_results,
        "litdata": litdata_results,
        "energon": energon_results
    }
    
    # Generate and display summary
    summary_file = generate_summary(all_results, total_time)
    
    print(f"\nBenchmark completed. Results saved to {results_file} and summary to {summary_file}")


if __name__ == "__main__":
    main()
