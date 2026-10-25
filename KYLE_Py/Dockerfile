# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies required by OpenCV, Torch etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Setup a non-root user for better security
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /code

# Configure uv behavior
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=1

# 1. Install dependencies first (leverages Docker caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# 2. Copy application source code
COPY . /code

# 3. Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Put the uv virtual environment on the system PATH
ENV PATH="/code/.venv/bin:$PATH"

# Pre-download huggingface models during the build step
RUN python app/Utils/huggingFaceModels.py

# Switch to non-root user
USER nonroot

EXPOSE 80

# Run the application in production mode, respecting the dynamic $PORT
CMD fastapi run app/main.py --host 0.0.0.0 --port ${PORT:-10000}

# CMD ["sh", "-c", "fastapi run app/main.py --host 0.0.0.0 --port ${PORT:-10000}"]