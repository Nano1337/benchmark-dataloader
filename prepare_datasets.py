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
import re
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
        cpu_count = psutil.cpu_count()
        
        # Define patterns for benchmark-related processes
        benchmark_patterns = [
            "prepare_webdataset", 
            "prepare_litdata", 
            "prepare_mds", 
            "prepare_energon",
            "webdataset_converter",
            "litdata_converter",
            "mosaicml_converter",
            "energon_converter"
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
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
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


def run_webdataset_benchmark(num_workers, log_dir="results/processing", enable_profiling=True):
    """Run WebDataset benchmark"""
    print_header("WEBDATASET BENCHMARK")
    cmd = [sys.executable, "prepare_data/prepare_webdataset.py"]
    
    # Add num_workers if it's supported
    if num_workers > 0:
        cmd.extend(["--num_workers", str(num_workers)])
    
    # Start resource monitoring if profiling is enabled
    monitor = None
    if enable_profiling:
        monitor = ResourceMonitor("webdataset")
        monitor.start()
    
    # Run the benchmark
    log_file = f"{log_dir}/webdataset_benchmark.log"
    webdataset_time, log_output = run_benchmark(cmd, log_file)
    
    # Stop resource monitoring if started
    if monitor:
        monitor.stop()
    
    # Extract specific timing values
    webdataset_write_time = extract_timing(log_output, "Dataset write time:")
    webdataset_total_time = extract_timing(log_output, "Total script time:")
    webdataset_converter_time = extract_timing(log_output, "Converter execution time:")
    
    result = {
        "time": webdataset_time,
        "write_time": webdataset_write_time,
        "total_time": webdataset_total_time,
        "converter_time": webdataset_converter_time,
        "log_output": log_output,
    }
    
    # Add profiling data if available
    if monitor:
        result["peak_ram_mb"] = monitor.peak_ram
        result["monitor"] = monitor
    
    return result


def run_mds_benchmark(num_workers, log_dir="results/processing", enable_profiling=True):
    """Run MosaicML MDS benchmark"""
    print_header("MOSAICML MDS BENCHMARK")
    cmd = [sys.executable, "prepare_data/prepare_mosaicdataset.py"]
    
    # Add num_workers if specified
    if num_workers > 0:
        cmd.extend(["--num_workers", str(num_workers)])
    
    # Start resource monitoring if profiling is enabled
    monitor = None
    if enable_profiling:
        monitor = ResourceMonitor("mds")
        monitor.start()
    
    # Run the benchmark
    log_file = f"{log_dir}/mds_benchmark.log"
    mds_time, log_output = run_benchmark(cmd, log_file)
    
    # Stop resource monitoring if started
    if monitor:
        monitor.stop()
    
    # Extract specific timing values
    mds_write_time = extract_timing(log_output, "Conversion completed in")
    mds_converter_time = extract_timing(log_output, "MDS conversion completed in")
    
    result = {
        "time": mds_time,
        "write_time": mds_write_time,
        "converter_time": mds_converter_time,
        "log_output": log_output,
    }
    
    # Add profiling data if available
    if monitor:
        result["peak_ram_mb"] = monitor.peak_ram
        result["monitor"] = monitor
    
    return result


def run_litdata_benchmark(num_workers, log_dir="results/processing", enable_profiling=True):
    """Run LitData benchmark"""
    print_header("LITDATA BENCHMARK")
    cmd = [sys.executable, "prepare_data/prepare_litdata.py"]
    
    # Add num_workers if it's supported
    if num_workers > 0:
        cmd.extend(["--num_workers", str(num_workers)])
    
    # Start resource monitoring if profiling is enabled
    monitor = None
    if enable_profiling:
        monitor = ResourceMonitor("litdata")
        monitor.start()
    
    # Run the benchmark
    log_file = f"{log_dir}/litdata_benchmark.log"
    litdata_time, log_output = run_benchmark(cmd, log_file)
    
    # Stop resource monitoring if started
    if monitor:
        monitor.stop()
    
    # Extract specific timing values
    litdata_write_time = extract_timing(log_output, "Dataset write time:")
    litdata_total_time = extract_timing(log_output, "Total script time:")
    
    result = {
        "time": litdata_time,
        "write_time": litdata_write_time,
        "total_time": litdata_total_time,
        "log_output": log_output,
    }
    
    # Add profiling data if available
    if monitor:
        result["peak_ram_mb"] = monitor.peak_ram
        result["monitor"] = monitor
    
    return result


def run_energon_benchmark(webdataset_results, num_workers, log_dir="results/processing", enable_profiling=True):
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
    
    # Start resource monitoring if profiling is enabled
    monitor = None
    if enable_profiling:
        monitor = ResourceMonitor("energon")
        monitor.start()
    
    # Run the benchmark
    log_file = f"{log_dir}/energon_benchmark.log"
    energon_time, log_output = run_benchmark(cmd, log_file)
    
    # Stop resource monitoring if started
    if monitor:
        monitor.stop()
    
    # Extract specific timing values
    energon_write_time = extract_timing(log_output, "Dataset write time:")
    energon_total_time = extract_timing(log_output, "Total script time:")
    energon_converter_time = extract_timing(log_output, "Converter execution time:")
    
    result = {
        "time": energon_time,
        "write_time": energon_write_time,
        "total_time": energon_total_time,
        "converter_time": energon_converter_time,
        "log_output": log_output,
    }
    
    # Add profiling data if available
    if monitor:
        result["peak_ram_mb"] = monitor.peak_ram
        result["monitor"] = monitor
    
    return result


def generate_resource_plots(results, timestamp, worker_count=None):
    """Generate plots for RAM and CPU utilization
    
    Args:
        results: Dictionary with benchmark results
        timestamp: Timestamp to use in filenames
        worker_count: Number of workers used for this run (optional)
    """
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
    
    # Include worker count in filename if available
    if worker_count is not None:
        plot_file = f"{plots_dir}/resource_usage_{worker_count}_workers_{timestamp}.png"
    else:
        plot_file = f"{plots_dir}/resource_usage_{timestamp}.png"
        
    plt.savefig(plot_file, dpi=120, bbox_inches='tight')
    print(f"\nResource usage plots saved to: {plot_file}")
    
    return plot_file


def generate_summary(results, total_time, log_dir="results/processing"):
    """Generate benchmark summary"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"{log_dir}/summary_{timestamp}.txt"
    
    # Get directory stats
    webdataset_size, webdataset_files = get_directory_stats("shards/webdataset")
    mds_size, mds_files = get_directory_stats("shards/mds")
    litdata_size, litdata_files = get_directory_stats("shards/litdata")
    energon_size, energon_files = get_directory_stats("shards/energon")
    
    # Format numbers
    webdataset_time_fmt = format_time(results["webdataset"]["time"])
    webdataset_write_time_fmt = format_time(results["webdataset"]["write_time"])
    webdataset_size_gb = format_size_gb(webdataset_size)
    
    mds_time_fmt = format_time(results["mds"]["time"])
    mds_write_time_fmt = format_time(results["mds"]["write_time"])
    mds_size_gb = format_size_gb(mds_size)
    
    litdata_time_fmt = format_time(results["litdata"]["time"])
    litdata_write_time_fmt = format_time(results["litdata"]["write_time"])
    litdata_size_gb = format_size_gb(litdata_size)
    
    energon_time_fmt = format_time(results["energon"]["time"])
    energon_write_time_fmt = format_time(results["energon"]["write_time"])
    energon_size_gb = format_size_gb(energon_size)
    
    # Combine WDS and Energon metrics
    combined_energon_time_raw = results["webdataset"]["time"] + results["energon"]["time"]
    combined_energon_write_raw = results["webdataset"]["write_time"] + results["energon"]["write_time"]
    combined_energon_time_fmt = format_time(combined_energon_time_raw)
    combined_energon_write_fmt = format_time(combined_energon_write_raw)
    
    # Check if profiling data is available
    profiling_enabled = "peak_ram_mb" in results["webdataset"]
    
    # Generate summary content
    if profiling_enabled:
        # Get RAM values if profiling is enabled
        webdataset_ram_mb = results["webdataset"]["peak_ram_mb"]
        mds_ram_mb = results["mds"]["peak_ram_mb"]
        litdata_ram_mb = results["litdata"]["peak_ram_mb"]
        energon_ram_mb = results["energon"]["peak_ram_mb"]
        combined_energon_ram_mb = max(webdataset_ram_mb, energon_ram_mb)
        
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
    else:
        # Without profiling, don't include RAM column
        summary_content = [
            "BENCHMARK SUMMARY",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"CPU Count: {os.cpu_count()}",
            "",
            "| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files |",
            "| --- | --- | --- | --- | --- |",
            f"| LitData (PL) | {litdata_time_fmt} | {litdata_write_time_fmt} | {litdata_size_gb} | {litdata_files} |",
            f"| WebDataset (WDS) | {webdataset_time_fmt} | {webdataset_write_time_fmt} | {webdataset_size_gb} | {webdataset_files} |",
            f"| MosaicML Dataset (MDS) | {mds_time_fmt} | {mds_write_time_fmt} | {mds_size_gb} | {mds_files} |",
            f"| Energon (WDS+) | {combined_energon_time_fmt} | {combined_energon_write_fmt} | {energon_size_gb} | {energon_files} |",
            "",
            f"Total benchmark time: {format_time(total_time)} seconds"
        ]
    
    # Write summary to file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_content))
    
    print_header("BENCHMARK SUMMARY")
    print('\n'.join(summary_content))
    
    # Generate resource usage plots if all monitors are available
    if all('monitor' in result for result in results.values()):
        # Try to extract worker count from log_dir path
        worker_count_match = re.search(r'(\d+)_workers', log_dir)
        worker_count = int(worker_count_match.group(1)) if worker_count_match else None
        generate_resource_plots(results, timestamp, worker_count)
    
    return summary_file


def clean_shards_directory():
    """Clean up or create shards directory"""
    shards_dir = "./shards"
    if os.path.exists(shards_dir):
        print("Cleaning up existing shards directory...")
        # Use force=True to handle any permission issues
        try:
            shutil.rmtree(shards_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Error while cleaning shards directory: {e}")
            # Try with a small delay in case of filesystem issues
            time.sleep(1)
            shutil.rmtree(shards_dir, ignore_errors=True)
    
    print("Creating fresh shards directory...")
    os.makedirs(f"{shards_dir}/webdataset", exist_ok=True)
    os.makedirs(f"{shards_dir}/mds", exist_ok=True)
    os.makedirs(f"{shards_dir}/litdata", exist_ok=True)
    os.makedirs(f"{shards_dir}/energon", exist_ok=True)


def clean_previous_sweep_results():
    """Clean up previous worker sweep results"""
    results_dir = "results/processing"
    if not os.path.exists(results_dir):
        return
        
    print("Cleaning up previous sweep results...")
    for dirname in os.listdir(results_dir):
        full_path = os.path.join(results_dir, dirname)
        if os.path.isdir(full_path) and "workers" in dirname:
            print(f"  Removing previous sweep directory: {dirname}")
            try:
                shutil.rmtree(full_path, ignore_errors=True)
            except Exception as e:
                print(f"  Warning: Error while removing {dirname}: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run dataset conversion benchmarks")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes to use (default: 0, meaning use all available CPUs)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable CPU and RAM profiling and generate resource usage plots"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run benchmarks with different worker counts (1, 2, 4, 8, 16)"
    )
    return parser.parse_args()


def run_single_benchmark(num_workers, log_dir, enable_profiling):
    """Run a single benchmark with the specified number of workers"""
    # Make directories
    os.makedirs(log_dir, exist_ok=True)
    
    print_header(f"DATASET BENCHMARK WITH {num_workers} WORKER{'S' if num_workers != 1 else ''}")
    
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
    webdataset_results = run_webdataset_benchmark(num_workers, log_dir, enable_profiling)
    mds_results = run_mds_benchmark(num_workers, log_dir, enable_profiling)
    litdata_results = run_litdata_benchmark(num_workers, log_dir, enable_profiling)
    energon_results = run_energon_benchmark(webdataset_results, num_workers, log_dir, enable_profiling)
    
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
    summary_file = generate_summary(all_results, total_time, log_dir)
    
    print(f"\nBenchmark completed. Results saved to {log_dir}")
    
    return all_results, total_time


def generate_sweep_summary(sweep_results, sweep_workers):
    """Generate a summary table comparing results across different worker counts"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"results/processing/sweep_summary_{timestamp}.txt"
    
    # Extract results for different worker counts
    table_rows = []
    
    # Header rows for the table
    table_rows.append("WORKER COUNT SWEEP SUMMARY")
    table_rows.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    table_rows.append(f"CPU Count: {os.cpu_count()}")
    table_rows.append("")
    
    # Format table for dataset formats
    for dataset_format in ["litdata", "webdataset", "mds", "energon"]:
        table_rows.append(f"\n{dataset_format.upper()} RESULTS:")
        table_rows.append("| Workers | Total Time (s) | Dataset Write (s) | Peak RAM (MB) |")
        table_rows.append("| --- | --- | --- | --- |")
        
        for workers in sweep_workers:
            results = sweep_results[workers][dataset_format]
            time_fmt = format_time(results["time"])
            write_time_fmt = format_time(results["write_time"])
            
            # Peak RAM is only available if profiling was enabled
            peak_ram = results.get("peak_ram_mb", "N/A")
            if isinstance(peak_ram, (int, float)):
                peak_ram_fmt = f"{peak_ram:.1f}"
            else:
                peak_ram_fmt = "N/A"
                
            table_rows.append(f"| {workers} | {time_fmt} | {write_time_fmt} | {peak_ram_fmt} |")
    
    # Write summary to file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(table_rows))
    
    print_header("WORKER COUNT SWEEP SUMMARY")
    print('\n'.join(table_rows))
    
    return summary_file


def main():
    """Main function to run all benchmarks"""
    args = parse_args()
    
    # Set base log directory
    base_log_dir = "results/processing"
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Run in sweep mode if requested
    if args.sweep:
        print_header("WORKER COUNT SWEEP")
        print("Running benchmarks with different worker counts (1, 2, 4, 8, 16)")
        
        # Make absolutely sure we're starting with a clean state
        print("Ensuring clean state before starting sweep...")
        # Clean previous sweep results first
        clean_previous_sweep_results()
        # Then clean shards directory
        clean_shards_directory()
        
        sweep_workers = [1, 2, 4, 8, 16]
        sweep_results = {}
        
        for workers in sweep_workers:
            # Use separate log directories for each worker count
            log_dir = f"{base_log_dir}/{workers}_workers"
            
            # Run benchmark with current worker count
            results, total_time = run_single_benchmark(
                num_workers=workers,
                log_dir=log_dir,
                enable_profiling=args.profile
            )
            
            sweep_results[workers] = results
            
            # Brief pause between benchmarks
            if workers != sweep_workers[-1]:  # If not the last one
                print("\nWaiting 5 seconds before the next benchmark...")
                print("Ensuring clean state before next worker count...")
                # Clean up shards directory between different worker counts
                clean_shards_directory()
                time.sleep(5)
        
        # Generate summary comparing all worker counts
        generate_sweep_summary(sweep_results, sweep_workers)
    else:
        # Run a single benchmark with specified worker count
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"{base_log_dir}/{timestamp}"
        
        run_single_benchmark(
            num_workers=args.num_workers,
            log_dir=log_dir,
            enable_profiling=args.profile
        )


if __name__ == "__main__":
    main()
