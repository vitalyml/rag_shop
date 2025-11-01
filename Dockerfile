FROM python:3.10-slim

# Установка инструментов для компиляции (необходимы для сборки пакетов с C расширениями)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     g++ \
#     make \
#     && rm -rf /var/lib/apt/lists/*

# Установка Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Настройка Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_INSTALLER_MAX_WORKERS=10 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5

WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock ./

# Устанавливаем зависимости через Poetry
# Используем несколько попыток для надежности при сетевых проблемах
RUN poetry install --no-interaction --no-ansi --without dev || \
    (sleep 10 && poetry install --no-interaction --no-ansi --without dev) || \
    (sleep 20 && poetry install --no-interaction --no-ansi --without dev) && \
    rm -rf $POETRY_CACHE_DIR

# Копируем код приложения
COPY . .

ENV PYTHONPATH=/app

# Убеждаемся, что директория data существует
RUN mkdir -p data/artifacts data/processed data/raw

# Открываем порт для Streamlit (по умолчанию 8501)
EXPOSE 8501

# Healthcheck для проверки работоспособности
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import streamlit; print('OK')" || exit 1

# Команда запуска Streamlit через python -m (более надежно)
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
