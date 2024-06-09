FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Установка Python 3.11
RUN apt-get update && \
    apt-get install python3.11 --no-install-recommends -y && \
    apt-get install python3.11-venv --no-install-recommends -y

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Создание виртуального окружения для проекта
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /code
# Перенос файла зависимостей
COPY requirements.txt requirements.txt
# Перенос приложения
COPY app.py app.py

# Установка Python зависимостей
RUN python -m pip install -U pip && \
    python -m pip install -U setuptools && \
    python -m pip install -U wheel && \
    python -m pip install -r requirements.txt &&  \
    rm -f requirements.txt

# Устновка LlamaCpp
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
RUN pip install llama-cpp-python

# Открыть порт приложения
EXPOSE 8000
# Точка входа в образ
CMD ["chainlit", "run", "app.py"]
