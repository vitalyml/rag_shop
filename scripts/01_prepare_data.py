
import sysfrom pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.data.loader import load_from_s3
from rag.preprocessing.text import parse_price, build_text
from config import settings


def main():
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Загрузка данных из S3...")
    df = load_from_s3()

    print("\n2. Парсинг цен...")
    df['price_num'] = df['price'].apply(parse_price)
    df['old_price_num'] = df['old_price'].apply(parse_price)
    print(f"   Обработано {df['price_num'].notna().sum()} цен")

    print("\n3. Создание полнотекстового поля...")
    df['doc_text'] = df.apply(build_text, axis=1)
    print(f"\n4. Сохранение в {settings.PROCESSED_CSV}...")
    df.to_csv(settings.PROCESSED_CSV, index=False)

    print(f"\nСохранено: {settings.PROCESSED_CSV}")


if __name__ == "__main__":
    main()
