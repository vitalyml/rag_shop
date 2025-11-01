import pandas as pd
import boto3
from botocore.config import Config
from pathlib import Path

from config import settings


def load_from_s3() -> pd.DataFrame:
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
        use_ssl=False,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

    response = s3.get_object(Bucket=settings.S3_BUCKET, Key=settings.S3_KEY)
    df = pd.read_csv(response['Body'])

    return df


def load_products(rag_version: bool = True) -> pd.DataFrame:
    csv_path = settings.PROCESSED_RAG_CSV if rag_version else settings.PROCESSED_CSV
    return pd.read_csv(csv_path)
