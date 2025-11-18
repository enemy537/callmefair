# Base image aligned with package requirement (Python >=3.11)
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by some Python packages
# - libgomp1: needed by xgboost and some numerical libraries
# - graphviz: used by catboost (optional), harmless if not used
# - build-essential: for any wheels needing compilation
# - git: in case users install extras from repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source and install the package
COPY . ./
RUN pip install --no-cache-dir .

# Default workdir for users; they can mount a host folder here
WORKDIR /workspace

# By default, start a Python REPL; users can override with `docker run ... python script.py`
CMD ["python"]