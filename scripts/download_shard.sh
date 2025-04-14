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

# Check if it's a directory by looking for objects inside it
echo "Checking for parquet files in the directory..."
PARQUET_FILES=$(aws s3 ls $CLEAN_PATH/ | grep ".parquet" || true)

if [ -z "$PARQUET_FILES" ]; then
    # Try downloading as a file directly
    echo "No parquet files found in directory. Trying to download as a single file..."
    aws s3 cp $CLEAN_PATH ./data/benchmark_shard.parquet
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded benchmark shard from $CLEAN_PATH to ./data/benchmark_shard.parquet"
        exit 0
    else
        echo "Failed to download $CLEAN_PATH directly."
        echo "Trying to download the entire directory..."
        aws s3 cp $CLEAN_PATH/ ./data/ --recursive
        if [ $? -eq 0 ]; then
            echo "Successfully downloaded all files from $CLEAN_PATH/ to ./data/"
            exit 0
        else
            echo "Failed to download benchmark shard. Please check the S3 path."
            exit 1
        fi
    fi
else
    # Extract the first parquet file name
    FIRST_FILE=$(echo "$PARQUET_FILES" | head -n 1 | awk '{print $NF}')
    echo "Found parquet file: $FIRST_FILE"
    
    # Download the file
    aws s3 cp $CLEAN_PATH/$FIRST_FILE ./data/benchmark_shard.parquet
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded benchmark shard from $CLEAN_PATH/$FIRST_FILE to ./data/benchmark_shard.parquet"
    else
        echo "Failed to download the benchmark shard"
        exit 1
    fi
fi
