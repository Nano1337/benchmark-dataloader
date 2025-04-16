#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Script to reduce the size of the data directory to below 20GB
# by removing files from the end of the directory listing

DATA_DIR="./data/benchmark_shard.parquet"
MAX_SIZE_MB=20000  # 20GB threshold

# Check if directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory $DATA_DIR does not exist."
    exit 1
fi

# Get initial size in MB
initial_size=$(du -sm "$DATA_DIR" | cut -f1)
echo "Initial directory size: ${initial_size}MB"

# If already under threshold, no need to proceed
if [ "$initial_size" -lt "$MAX_SIZE_MB" ]; then
    echo "Directory is already smaller than 20GB (${initial_size}MB), no need to cut down."
    exit 0
fi

echo "Reducing directory size to below ${MAX_SIZE_MB}MB..."

# Count total files
total_files=$(ls -1 "$DATA_DIR" | wc -l)
echo "Total files before reduction: $total_files"

# Remove files one by one from the end of the listing until size is below threshold
while [ "$(du -sm "$DATA_DIR" | cut -f1)" -ge "$MAX_SIZE_MB" ]; do
    # Get the last file in the sorted listing
    last_file=$(ls -1 "$DATA_DIR" | sort | tail -n 1)
    
    if [ -z "$last_file" ]; then
        echo "No more files to remove. Unable to reduce size below threshold."
        break
    fi
    
    echo "Removing file: $last_file"
    rm "$DATA_DIR/$last_file"
    
    # Update current size
    current_size=$(du -sm "$DATA_DIR" | cut -f1)
    echo "Current directory size: ${current_size}MB"
done

# Final report
final_size=$(du -sm "$DATA_DIR" | cut -f1)
final_files=$(ls -1 "$DATA_DIR" | wc -l)

echo -e "\nReduction complete:"
echo "  Initial size: ${initial_size}MB"
echo "  Final size: ${final_size}MB"
echo "  Removed: $((total_files - final_files)) files"
echo "  Remaining: $final_files files"

if [ "$final_size" -lt "$MAX_SIZE_MB" ]; then
    echo "Successfully reduced directory size to below ${MAX_SIZE_MB}MB"
else
    echo "Warning: Directory is still larger than ${MAX_SIZE_MB}MB. Manual intervention may be needed."
fi
