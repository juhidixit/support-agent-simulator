# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Set INSTALL_LOCAL=1 to compile llama.cpp for running GGUF models inside the container.
ARG INSTALL_LOCAL=0
RUN if [ "$INSTALL_LOCAL" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
      && rm -rf /var/lib/apt/lists/*; fi

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[server]" && if [ "$INSTALL_LOCAL" = "1" ]; then pip install ".[local]"; fi

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
# Example: -e SUPPORTSIM_AGENT=openai:gpt-4.1-mini -e OPENAI_API_KEY=...
#      or: -e SUPPORTSIM_AGENT=llamacpp:/models/qwen3-4b-2507-q4km.gguf -v $PWD/models:/models
CMD ["supportsim", "serve", "--host", "0.0.0.0", "--port", "8000"]
