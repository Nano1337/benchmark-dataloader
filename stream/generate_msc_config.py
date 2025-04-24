#!/usr/bin/env python

import os
import yaml
from dotenv import load_dotenv
import sys
import boto3

# Load environment variables from .env
load_dotenv()

# Get S3 path from environment variables
s3_path = os.environ.get('S3_BENCHMARK_DATA_PATH')
if not s3_path:
    print("Error: S3_BENCHMARK_DATA_PATH not set in .env file")
    sys.exit(1)

# Append energon subdirectory to the S3 path
full_s3_path = os.path.join(s3_path, "energon")
print(f"Full Energon dataset path: {full_s3_path}")

# Parse bucket from s3://bucket/path format
if full_s3_path.startswith('s3://'):
    # Remove s3:// prefix
    path_without_scheme = full_s3_path[5:]
    # Split to get bucket name and rest of path
    parts = path_without_scheme.split('/', 1)
    bucket_name = parts[0]
    # Get the path after the bucket
    dataset_relative_path = parts[1] if len(parts) > 1 else ""
else:
    print(f"Error: S3_BENCHMARK_DATA_PATH '{full_s3_path}' does not start with s3://")
    sys.exit(1)

# Get AWS credentials from environment variables or boto3 session
try:
    # Try to get from environment variables first
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    session_token = os.environ.get('AWS_SESSION_TOKEN')
    
    # If not in env vars, try to get from boto3 session
    if not (access_key and secret_key):
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials:
                access_key = credentials.access_key
                secret_key = credentials.secret_key
                session_token = credentials.token
                print("Using AWS credentials from boto3 session")
        except Exception as e:
            print(f"Warning: Could not get AWS credentials from boto3: {e}")
    
    # Check if we have credentials
    if not (access_key and secret_key):
        print("Warning: No AWS credentials found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for authenticated access.")
        # Using null for authenticated access through instance profile or similar mechanisms
        access_key = None
        secret_key = None
        session_token = None

except Exception as e:
    print(f"Warning: Error getting credentials: {e}")
    access_key = None
    secret_key = None
    session_token = None

# Get other AWS config from environment variables
region = os.environ.get('AWS_REGION', 'us-east-1')  # Default to us-east-1 if not specified
endpoint_url = os.environ.get('AWS_ENDPOINT_URL', '')  # Optional endpoint for S3-compatible stores

# Create MSC config with the S3StorageProvider parameters
msc_config = {
    'profiles': {
        's3-iad-webdataset': {
            'credentials_provider': {
                'type': 'S3Credentials',
                'options': {
                    'access_key': access_key,
                    'secret_key': secret_key,
                }
            } if access_key is not None else None,  # Only include if we have credentials
            'storage_provider': {
                'type': 's3',
                'options': {
                    'base_path': bucket_name,
                    'region_name': region,
                    # Add these advanced options if they are set in environment variables
                    **(({'endpoint_url': endpoint_url}) if endpoint_url else {}),
                    'max_pool_connections': int(os.environ.get('AWS_MAX_POOL_CONNECTIONS', 32)),
                    **(({'signature_version': os.environ.get('AWS_SIGNATURE_VERSION')}) 
                       if os.environ.get('AWS_SIGNATURE_VERSION') else {}),
                }
            }
        }
    }
}

# Write to .msc_config.yaml in the current directory
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.msc_config.yaml')
with open(config_path, 'w') as f:
    yaml.dump(msc_config, f, default_flow_style=False)

print(f"MSC config written to {config_path}")
print(f"Using bucket: {bucket_name}")