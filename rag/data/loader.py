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


def get_available_shops() -> list[str]:
    """список доступных магазинов"""
    shops = []
    if settings.DATA_DIR.exists():
        for item in settings.DATA_DIR.iterdir():
            if item.is_dir() and (item / "processed").exists():
                shops.append(item.name)
    return sorted(shops)


def load_products(rag_version: bool = True, shop_name: str = None) -> pd.DataFrame:
    """Загрузка данных о товарах"""
    shop = shop_name or settings.DEFAULT_SHOP
    shop_paths = settings.get_shop_paths(shop)

    csv_filename = "products_rag.csv" if rag_version else "products.csv"
    csv_path = shop_paths['processed'] / csv_filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Файл {csv_path} не найден для магазина '{shop}'")

    return pd.read_csv(csv_path)
