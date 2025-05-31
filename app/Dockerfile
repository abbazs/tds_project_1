# Use a slim Python base image
FROM python:3.13-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY app ./app

# Create virtual environment and install dependencies with uv
RUN uv sync --frozen --no-cache --no-dev

# Expose the port (optional, Railway handles port mapping)
EXPOSE 80

# Run the application using $PORT from Railway
CMD [".venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$PORT"]