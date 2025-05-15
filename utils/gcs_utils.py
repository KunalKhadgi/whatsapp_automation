import os
from google.cloud import storage

def download_from_gcs(bucket_name, blob_name, destination_file_name):
    if os.path.exists(destination_file_name) and os.path.getsize(destination_file_name) > 0:
        return destination_file_name
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_file_name)
    return destination_file_name

def download_meta_jsons_from_gcs(bucket_name, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith("meta.json"):
            path = os.path.join(local_dir, os.path.relpath(blob.name, prefix))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            blob.download_to_filename(path)

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")
