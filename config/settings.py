import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

# S3 настройки
S3_ENDPOINT = "http://prod.easy-profiler.org:9000"
S3_BUCKET = "rag"
S3_KEY = "nikitaefremov_products_all.csv"
AWS_ACCESS_KEY = os.getenv("AWS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET")

# AI Model настройки
# Поддерживаемые провайдеры: "vllm", "deepseek", "gemini"
AI_PROVIDER = os.getenv("AI_PROVIDER", "vllm")

# vLLM настройки
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1/")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-AWQ")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "test-key")  # Dummy key для локального сервера

# DeepSeek настройки
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_NAME = "deepseek-reasoner"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Google Gemini настройки
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Embeddings модель
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# RAG параметры
REWRITES_COUNT = 1
HITS_PER_QUERY = 5
FINAL_TOP_K = 20
RRF_CONSTANT = 60
CONTEXT_DOCS = 20
MAX_SOURCES = 5

# Пути к артефактам
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.index"
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25_index.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.pkl"
PROCESSED_CSV = PROCESSED_DATA_DIR / "products.csv"
PROCESSED_RAG_CSV = PROCESSED_DATA_DIR / "products_rag.csv"
