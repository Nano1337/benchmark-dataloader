#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/streaming"
RESULTS_FILE="${RESULTS_DIR}/benchmark_stream_${TIMESTAMP}.log"
SUMMARY_FILE="${RESULTS_DIR}/summary_stream_${TIMESTAMP}.txt"
TMP_LOG="${RESULTS_DIR}/tmp_$TIMESTAMP.log"

# Create directories if they don't exist
mkdir -p "${RESULTS_DIR}"

# Format numbers according to specifications
format_number() {
    printf "%.2f" $1  # 2 decimal places
}

# Print header with timestamp
print_header() {
    printf "\n\n%s\n" "========================================="
    printf "%s\n" "   $1"
    printf "%s\n" "   $(date)"
    printf "%s\n\n" "========================================="
}

# Extract throughput from logs
extract_throughput() {
    local file=$1
    # Try multiple possible patterns
    local throughput=$(grep -o "Average throughput: [0-9.]\+ images/sec" "$file" | grep -o "[0-9.]\+" || echo "")
    if [ -z "$throughput" ]; then
        throughput=$(grep -o "[0-9.]\+ images/sec" "$file" | grep -o "[0-9.]\+" | tail -1 || echo "0")
    fi
    echo "$throughput"
}

# Extract total samples processed from logs
extract_samples() {
    local file=$1
    # Try multiple possible patterns
    local samples=$(grep -o "Total samples: [0-9]*" "$file" | grep -o "[0-9]*" || echo "")
    if [ -z "$samples" ]; then
        # Find any processed samples line
        samples=$(grep -o "Processed [0-9]\+ samples" "$file" | grep -o "[0-9]\+" | tail -1 || echo "0")
    fi
    echo "$samples"
}

# Extract timing info from logs
extract_time() {
    local file=$1
    # Try multiple possible patterns
    local time=$(grep -o "Total time: [0-9.]\+s" "$file" | grep -o "[0-9.]\+" || echo "")
    if [ -z "$time" ]; then
        # Find time in seconds from the last epoch
        time=$(grep -o "Processed .* in [0-9.]\+s" "$file" | grep -o "[0-9.]\+s" | grep -o "[0-9.]\+" | tail -1 || echo "0")
    fi
    echo "$time"
}

# Check prerequisites and clean cache directories
check_prerequisites() {
    # Check if required tools are available
    command -v bc >/dev/null 2>&1 || { echo "bc is required but not installed. Aborting."; exit 1; }
    
    # Check if directories exist
    mkdir -p "$RESULTS_DIR"
    
    # Create empty log and summary files
    touch "$RESULTS_FILE"
    touch "$SUMMARY_FILE"
    
    # Clean up cache directories if they exist
    echo "Cleaning up cache directories..."
    local cache_dirs=(
        "stream/cache/webdataset_benchmark"
        "stream/cache/mds_benchmark"
        "stream/cache/litdata_benchmark"
    )
    
    for dir in "${cache_dirs[@]}"; do
        if [ -d "$dir" ]; then
            echo "Removing cache directory: $dir"
            rm -rf "$dir"
        fi
    done
    
    # Load environment variables from .env if it exists
    if [ -f ".env" ]; then
        echo "Sourcing environment variables from .env file"
        source .env
    fi
    
    # Check if environment variables are set after sourcing .env
    if [ -z "$S3_BENCHMARK_DATA_PATH" ]; then
        echo "Error: S3_BENCHMARK_DATA_PATH environment variable is not set."
        echo "Please set it in the .env file or export it in your shell."
        exit 1
    fi
}

# Extract epoch times from logs
extract_epoch_times() {
    local file=$1
    local epoch1=$(grep -o "Epoch 1: Processed [0-9]\+ samples in [0-9.]\+s" "$file" | grep -o "[0-9.]\+s" | grep -o "[0-9.]\+" || echo "0")
    local epoch2=$(grep -o "Epoch 2: Processed [0-9]\+ samples in [0-9.]\+s" "$file" | grep -o "[0-9.]\+s" | grep -o "[0-9.]\+" || echo "0")
    echo "$epoch1,$epoch2"
}

# Clean up a specific cache directory
cleanup_cache() {
    local cache_dir=$1
    if [ -d "$cache_dir" ]; then
        echo "Cleaning up cache directory: $cache_dir"
        rm -rf "$cache_dir"
    fi
}

# Run WebDataset streaming benchmark
run_webdataset_benchmark() {
    local batch_size=$1
    local num_workers=$2
    
    print_header "WebDataset Streaming Benchmark"
    echo "Configuration: batch_size=$batch_size, num_workers=$num_workers"
    
    local start_time=$(date +%s.%N)
    
    # Run the benchmark with standard parameters
    python stream/web_dataset.py \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --epochs 2 \
        --image_size 224 \
        --keep_cache \
        | tee "$TMP_LOG"
    
    local end_time=$(date +%s.%N)
    local total_time=$(echo "$end_time - $start_time" | bc)
    
    # Extract metrics from the logs
    local throughput=$(extract_throughput "$TMP_LOG")
    local samples=$(extract_samples "$TMP_LOG")
    local benchmark_time=$(extract_time "$TMP_LOG")
    local epoch_times=$(extract_epoch_times "$TMP_LOG")
    
    # Print a summary
    echo ""
    echo "Results for WebDataset:"
    echo "  Throughput: $throughput images/sec"
    echo "  Total samples processed: $samples"
    echo "  Benchmark time: $benchmark_time seconds"
    echo "  Wall clock time: $(format_number $total_time) seconds"
    
    # Return the metrics for the summary table
    echo "WebDataset,$throughput,$samples,$benchmark_time,$total_time,$epoch_times"
}

# Run MosaicML MDS streaming benchmark
run_mds_benchmark() {
    local batch_size=$1
    local num_workers=$2
    
    print_header "MosaicML Dataset Streaming Benchmark"
    echo "Configuration: batch_size=$batch_size, num_workers=$num_workers"
    
    local start_time=$(date +%s.%N)
    
    # Run the benchmark with standard parameters
    python stream/mosaic_ml.py \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --epochs 2 \
        --image_size 224 \
        --keep_cache \
        | tee "$TMP_LOG"
    
    local end_time=$(date +%s.%N)
    local total_time=$(echo "$end_time - $start_time" | bc)
    
    # Extract metrics from the logs
    local throughput=$(extract_throughput "$TMP_LOG")
    local samples=$(extract_samples "$TMP_LOG")
    local benchmark_time=$(extract_time "$TMP_LOG")
    local epoch_times=$(extract_epoch_times "$TMP_LOG")
    
    # Print a summary
    echo ""
    echo "Results for MosaicML Dataset:"
    echo "  Throughput: $throughput images/sec"
    echo "  Total samples processed: $samples"
    echo "  Benchmark time: $benchmark_time seconds"
    echo "  Wall clock time: $(format_number $total_time) seconds"
    
    # Return the metrics for the summary table
    echo "MosaicML MDS,$throughput,$samples,$benchmark_time,$total_time,$epoch_times"
}

# Run LitData streaming benchmark
run_litdata_benchmark() {
    local batch_size=$1
    local num_workers=$2
    
    print_header "LitData Streaming Benchmark"
    echo "Configuration: batch_size=$batch_size, num_workers=$num_workers"
    
    local start_time=$(date +%s.%N)
    
    # Run the benchmark with standard parameters
    python stream/lightning_data.py \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --epochs 2 \
        --image_size 224 \
        --keep_cache \
        | tee "$TMP_LOG"
    
    local end_time=$(date +%s.%N)
    local total_time=$(echo "$end_time - $start_time" | bc)
    
    # Extract metrics from the logs
    local throughput=$(extract_throughput "$TMP_LOG")
    local samples=$(extract_samples "$TMP_LOG")
    local benchmark_time=$(extract_time "$TMP_LOG")
    local epoch_times=$(extract_epoch_times "$TMP_LOG")
    
    # Print a summary
    echo ""
    echo "Results for LitData:"
    echo "  Throughput: $throughput images/sec"
    echo "  Total samples processed: $samples"
    echo "  Benchmark time: $benchmark_time seconds"
    echo "  Wall clock time: $(format_number $total_time) seconds"
    
    # Return the metrics for the summary table
    echo "LitData,$throughput,$samples,$benchmark_time,$total_time,$epoch_times"
}

# Start main script
print_header "DATASET STREAMING BENCHMARKS"

# Check prerequisites
check_prerequisites

# Log outputs to results
exec > >(tee -a "$RESULTS_FILE") 2>&1

echo "Starting streaming benchmarks at $(date)"
echo "CPU count: $(nproc)"
echo ""

# Start the overall timer
TOTAL_START_TIME=$(date +%s)

# Set standard parameters for all benchmarks
BATCH_SIZE=256
NUM_WORKERS=8

# Array to store results
declare -a results

# Print announcement before running each benchmark
echo -e "\n\nRunning WebDataset benchmark...\n"

# Run WebDataset streaming benchmark
wds_result=$(run_webdataset_benchmark $BATCH_SIZE $NUM_WORKERS)
echo "DEBUG: WebDataset result: $wds_result"
results+=("$wds_result")
cleanup_cache "stream/cache/webdataset_benchmark"

# Print announcement before running MosaicML benchmark
echo -e "\n\nRunning MosaicML Dataset benchmark...\n"

# Run MosaicML MDS streaming benchmark
mds_result=$(run_mds_benchmark $BATCH_SIZE $NUM_WORKERS)
echo "DEBUG: MosaicML result: $mds_result"
results+=("$mds_result")
cleanup_cache "stream/cache/mds_benchmark"

# Print announcement before running LitData benchmark
echo -e "\n\nRunning LitData benchmark...\n"

# Run LitData streaming benchmark
lit_result=$(run_litdata_benchmark $BATCH_SIZE $NUM_WORKERS)
echo "DEBUG: LitData result: $lit_result"
results+=("$lit_result")
cleanup_cache "stream/cache/litdata_benchmark"

# Calculate total benchmark time
TOTAL_END_TIME=$(date +%s)
TOTAL_TIME=$((TOTAL_END_TIME - TOTAL_START_TIME))

# Generate summary
echo "DATASET STREAMING BENCHMARK SUMMARY" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "CPU Count: $(nproc)" >> "$SUMMARY_FILE"
echo "Configuration: Batch Size = $BATCH_SIZE, Workers = $NUM_WORKERS" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "| Dataset | Throughput (img/s) | Samples | Processing Time (s) | Wall Time (s) |" >> "$SUMMARY_FILE"
echo "| --- | --- | --- | --- | --- |" >> "$SUMMARY_FILE"

# Update the table header to include epoch times
sed -i 's/| Dataset | Throughput (img\/s) | Samples | Processing Time (s) | Wall Time (s) |/| Dataset | Throughput (img\/s) | Samples | Epoch 1 (s) | Epoch 2 (s) | Processing Time (s) | Wall Time (s) |/' "$SUMMARY_FILE"
sed -i 's/| --- | --- | --- | --- | --- |/| --- | --- | --- | --- | --- | --- | --- |/' "$SUMMARY_FILE"

# Add results to the summary
echo "DEBUG: Results array length: ${#results[@]}"
for result in "${results[@]}"; do
    # Skip empty results or entire benchmark outputs (take only the last line which contains CSV)
    if [ -z "$result" ]; then
        continue
    fi
    
    # Extract last line which should contain our CSV data
    csv_line=$(echo "$result" | tail -n 1)
    echo "DEBUG: CSV line: $csv_line"
    
    # Parse the CSV result
    IFS=',' read -r name throughput samples proc_time wall_time epoch1 epoch2 <<< "$csv_line"
    
    # Handle any missing values by setting defaults
    name=${name:-"Unknown Dataset"}
    throughput=${throughput:-0}
    samples=${samples:-0}
    epoch1=${epoch1:-0}
    epoch2=${epoch2:-0}
    proc_time=${proc_time:-0}
    wall_time=${wall_time:-0}
    
    # Format numbers
    throughput_fmt=$(format_number $throughput)
    epoch1_fmt=$(format_number $epoch1)
    epoch2_fmt=$(format_number $epoch2)
    proc_time_fmt=$(format_number $proc_time)
    wall_time_fmt=$(format_number $wall_time)
    
    # Add to summary file
    echo "| $name | $throughput_fmt | $samples | $epoch1_fmt | $epoch2_fmt | $proc_time_fmt | $wall_time_fmt |" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "Total benchmark time: $(format_number $TOTAL_TIME) seconds" >> "$SUMMARY_FILE"

# Print summary to console
print_header "DATASET STREAMING BENCHMARK SUMMARY"
cat "$SUMMARY_FILE"

echo ""
echo "Streaming benchmarks complete! Summary saved to $SUMMARY_FILE"
echo "Full logs available at $RESULTS_FILE"

# Clean up temp file
rm -f "$TMP_LOG"
