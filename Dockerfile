# CPU image. Python >=3.11 as required by the package.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

# libgomp1: OpenMP runtime for xgboost/lightgbm; graphviz: optional, for catboost plots
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# CPU-only torch (the default Linux wheel bundles CUDA libraries, ~2 GB larger)
RUN uv pip install "torch>=2.2,<2.3" --index-url https://download.pytorch.org/whl/cpu

# Install dependencies first for layer caching
COPY requirements.txt ./
RUN uv pip install -r requirements.txt

# Install the package itself (dependencies are already satisfied)
COPY pyproject.toml README.md LICENSE ./
COPY callmefair ./callmefair
RUN uv pip install --no-deps .

# Users mount their project here
WORKDIR /workspace
CMD ["python"]
