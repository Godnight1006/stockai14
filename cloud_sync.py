import boto3
from google.cloud import storage
import argparse
import os

def upload_to_cloud(provider, bucket_name, checkpoint_dir):
    if provider == "aws":
        s3 = boto3.client('s3')
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                local_path = os.path.join(root, file)
                s3_path = os.path.relpath(local_path, checkpoint_dir)
                s3.upload_file(local_path, bucket_name, f"{checkpoint_dir}/{s3_path}")
                
    elif provider == "gcp":
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                local_path = os.path.join(root, file)
                blob_path = os.path.relpath(local_path, checkpoint_dir)
                blob = bucket.blob(f"{checkpoint_dir}/{blob_path}")
                blob.upload_from_filename(local_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["aws", "gcp"], required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    
    upload_to_cloud(args.provider, args.bucket, args.checkpoint_dir)
