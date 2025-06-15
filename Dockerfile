# Stage 1: pull the uv binary from Astral's UV image
FROM ghcr.io/astral-sh/uv:latest AS uvbin

# Stage 2: build your Python app image
FROM python:3.12-slim

# Install certificates (needed for pip) and clean up
RUN apt-get update \
    && apt-get install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy uv into the image
COPY --from=uvbin /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv
RUN mkdir -p /tds_project_1/app
# Set working directory
WORKDIR tds_project_1
# Copy lock files first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies via uv
RUN uv sync --frozen --no-cache --no-dev

# Copy the rest of your application code
COPY app/ ./app
COPY embeddings.npz .
# Expose port (adjust if needed)
EXPOSE 8000

# Default command — adjust as per your app
CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]