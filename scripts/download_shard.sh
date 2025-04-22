#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Load environment variables from .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Check if BENCHMARK_SHARD_PATH is set
if [ -z "$BENCHMARK_SHARD_PATH" ]; then
    echo "Error: BENCHMARK_SHARD_PATH environment variable is not set in .env file"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p ./data

# Clean the S3 path
CLEAN_PATH=$(echo $BENCHMARK_SHARD_PATH | sed 's/\/*$//')
echo "Looking for files in: $CLEAN_PATH"

# Check if the path exists
aws s3 ls $CLEAN_PATH/ &> /dev/null
if [ $? -ne 0 ]; then
    echo "Error: S3 path does not exist at $CLEAN_PATH"
    echo "Please check the BENCHMARK_SHARD_PATH in your .env file"
    exit 1
fi

# download all contents --recursive of path
aws s3 cp $CLEAN_PATH ./data/ --recursive
