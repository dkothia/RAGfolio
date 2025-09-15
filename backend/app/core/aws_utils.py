import boto3
from botocore.exceptions import NoCredentialsError
from app.core.config import settings

s3_client = boto3.client("s3", region_name=settings.AWS_REGION)

def upload_file_to_s3(file_path, s3_key):
    try:
        s3_client.upload_file(file_path, settings.S3_BUCKET_NAME, s3_key)
        return True
    except NoCredentialsError:
        print("AWS credentials not found.")
        return False

def download_file_from_s3(s3_key, local_path):
    try:
        s3_client.download_file(settings.S3_BUCKET_NAME, s3_key, local_path)
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False
