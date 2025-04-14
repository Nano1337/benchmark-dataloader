#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="results/benchmark_run_${TIMESTAMP}.log"
SUMMARY_FILE="results/summary_${TIMESTAMP}.txt"
TMP_LOG="results/tmp_$TIMESTAMP.log"

# Format numbers according to specifications
format_time() {
    printf "%.2f" $1  # 2 decimal places for time
}

format_size_gb() {
    # Convert MB to GB with 2 decimal places, no scientific notation
    echo "scale=2; $1/1024" | bc
}

# Print header with timestamp
print_header() {
    printf "\n\n%s\n" "========================================="
    printf "%s\n" "   $1"
    printf "%s\n" "   $(date)"
    printf "%s\n\n" "========================================="
}

# Check if data file exists
check_prerequisites() {
    if [ ! -f "./data/benchmark_shard.parquet" ]; then
        echo "Error: Benchmark data not found. Please run ./scripts/download_shard.sh first."
        exit 1
    fi

    # Create output directories
    mkdir -p results
    mkdir -p shards/webdataset
    mkdir -p shards/mds
}

# Calculate directory size in MB and count files
get_directory_stats() {
    local dir=$1
    local size_kb=$(du -sk "$dir" 2>/dev/null | cut -f1)
    local size_mb=$(echo "scale=3; $size_kb/1024" | bc)
    local file_count=$(find "$dir" -type f | wc -l)
    echo "$size_mb $file_count"
}

# Extract timing info from logs
extract_timing() {
    local file=$1
    local pattern=$2
    grep -o "$pattern" "$file" | grep -o "[0-9.]*" || echo "0"
}

# Start main script
print_header "DATASET BENCHMARKING SCRIPT"

# Check prerequisites
check_prerequisites

# Log outputs to results
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "Starting benchmark at $(date)"
echo "CPU count: $(nproc)"
echo "" 

# Start the overall timer
TOTAL_START_TIME=$(date +%s)

# WebDataset benchmark
print_header "WEBDATASET BENCHMARK"
webdataset_start=$(date +%s)
# Capture output to parse timing information
python prepare_data/prepare_webdataset.py | tee "$TMP_LOG"
webdataset_end=$(date +%s)
webdataset_time=$((webdataset_end - webdataset_start))

# Extract specific timing values from WebDataset logs
webdataset_write_time=$(extract_timing "$TMP_LOG" "Dataset write time: [0-9.]* seconds")
webdataset_total_time=$(extract_timing "$TMP_LOG" "Total script execution time: [0-9.]* seconds")
webdataset_converter_time=$(extract_timing "$TMP_LOG" "Converter execution time: [0-9.]* seconds")

# Clean up any previous data if needed and ensure directory exists
mkdir -p shards/mds

# MosaicML MDS benchmark
print_header "MOSAICML MDS BENCHMARK"
mds_start=$(date +%s)
# Capture output to parse timing information
python prepare_data/prepare_mosaicdataset.py | tee "$TMP_LOG"
mds_end=$(date +%s)
mds_time=$((mds_end - mds_start))

# Extract specific timing values from MDS logs
mds_write_time=$(extract_timing "$TMP_LOG" "Conversion completed in [0-9.]* seconds")
mds_converter_time=$(extract_timing "$TMP_LOG" "MDS conversion completed in [0-9.]* seconds")

# Run LitData benchmark
print_header "LITDATA BENCHMARK"
litdata_start=$(date +%s)
python prepare_data/prepare_litdata.py | tee "$TMP_LOG"
litdata_end=$(date +%s)
litdata_time=$((litdata_end - litdata_start))
# Extract specific timing values from LitData logs
litdata_write_time=$(extract_timing "$TMP_LOG" "Dataset write time: [0-9.]* seconds")
litdata_total_time=$(extract_timing "$TMP_LOG" "Total script time: [0-9.]* seconds")

# Calculate total benchmark time
TOTAL_END_TIME=$(date +%s)
TOTAL_TIME=$((TOTAL_END_TIME - TOTAL_START_TIME))

# Get directory stats
webdataset_stats=$(get_directory_stats "shards/webdataset")
webdataset_size=$(echo $webdataset_stats | cut -d' ' -f1)
webdataset_files=$(echo $webdataset_stats | cut -d' ' -f2)
webdataset_size_gb=$(format_size_gb $webdataset_size)

mds_stats=$(get_directory_stats "shards/mds")
mds_size=$(echo $mds_stats | cut -d' ' -f1)
mds_files=$(echo $mds_stats | cut -d' ' -f2)
mds_size_gb=$(format_size_gb $mds_size)

litdata_stats=$(get_directory_stats "shards/litdata")
litdata_size=$(echo $litdata_stats | cut -d' ' -f1)
litdata_files=$(echo $litdata_stats | cut -d' ' -f2)
litdata_size_gb=$(format_size_gb $litdata_size)

# Format numbers according to specifications
webdataset_time_fmt=$(format_time $webdataset_time)
webdataset_write_time_fmt=$(format_time $webdataset_write_time)

mds_time_fmt=$(format_time $mds_time)
mds_write_time_fmt=$(format_time $mds_write_time)

litdata_time_fmt=$(format_time $litdata_time)
litdata_write_time_fmt=$(format_time $litdata_write_time)

# Generate summary
echo "BENCHMARK SUMMARY" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "CPU Count: $(nproc)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "| Format | Total Time (s) | Dataset Write (s) | Size (GB) | # Files |" >> "$SUMMARY_FILE"
echo "| --- | --- | --- | --- | --- |" >> "$SUMMARY_FILE"
echo "| LitData (PL) | $litdata_time_fmt | $litdata_write_time_fmt | $litdata_size_gb | $litdata_files |" >> "$SUMMARY_FILE"
echo "| WebDataset (WDS) | $webdataset_time_fmt | $webdataset_write_time_fmt | $webdataset_size_gb | $webdataset_files |" >> "$SUMMARY_FILE"
echo "| MosaicML Dataset (MDS) | $mds_time_fmt | $mds_write_time_fmt | $mds_size_gb | $mds_files |" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Total benchmark time: $(format_time $TOTAL_TIME) seconds" >> "$SUMMARY_FILE"

# Print summary to console
print_header "BENCHMARK SUMMARY"
cat "$SUMMARY_FILE"

echo ""
echo "Benchmark complete! Summary saved to $SUMMARY_FILE"
echo "Full logs available at $RESULTS_FILE"

# Clean up temp file
rm -f "$TMP_LOG"
